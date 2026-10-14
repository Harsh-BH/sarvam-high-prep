# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""`translate` pipeline stage (W6): sentences[].hi_text -> en_text (+ en_alt for in-segment self-corrections),
translation_method, `_qa.w6`, `_archive.translate`.

    uv run w6/translate.py --in RECORDS.jsonl --audio-dir DIR --out OUT_DIR [--engine qwen3] [--judge gptoss]
                           [--no-repair] [--workers 8]

Per record: one LLM call translates all segments (w6/STYLE.md "Rules" is the system prompt), automatic QA (w6/qa.py)
checks every segment, a judge LLM of another family lists errors per segment, and one repair round (same engine,
fed the issues) replaces the translation only if the re-judged result has fewer critical errors. Every model call goes
through pipeline/archive.archive_call (service `bedrock`, or `it2` for IndicTrans2, written by w6/it2.py), so a re-run
or rebuild hits the archive and reproduces the same output byte for byte, with no new spend.

Stage contract (docs/superpowers/plans/2026-10-14-s3-pipeline.md): ids already in OUT_DIR/records.jsonl or
rejects.jsonl are skipped. A record is rejected (`no_translation`) only when no take passes the hard checks.
`--audio-dir` is accepted for the contract; this stage reads no audio. Human fixes: w6/patches.jsonl (w6/review.py).
"""
import argparse, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "w6")]
import archive, qa, s3 as store  # noqa: E402

STYLE = (ROOT / "w6/STYLE.md").read_text(encoding="utf-8")
RULES = STYLE.split("## Rules\n", 1)[1].strip()
VERSION = re.search(r"Version: (\S+)", STYLE).group(1)
PATCHES = ROOT / "w6/patches.jsonl"
# short name -> (Bedrock model id in ap-south-1, USD per 1M input tokens, per 1M output tokens). Prices: AWS Pricing API,
# ap-south-1 on-demand, 2026-10-14 (DeepSeek: 2x the listed batch price). Licences from the HF cards: Qwen3, gpt-oss
# and Mistral Large 3 are Apache-2.0 (translation engines); DeepSeek V3.2 is MIT (benchmark judge only).
MODELS = {"qwen3": ("qwen.qwen3-235b-a22b-2507-v1:0", .26, 1.04),
          "gptoss": ("openai.gpt-oss-120b-1:0", .18, .71),
          "mistral3": ("mistral.mistral-large-3-675b-instruct", .59, 1.76),
          "dsv32": ("deepseek.v3.2", .74, 2.22)}
IT2 = {"model": "prajdabre/rotary-indictrans2-indic-en-dist-200M", "rev": "00213ee82929050694b162123bb26a8dc177cdae"}
USD_INR = 88.0
CAP_INR = 1250.0  # W6 AWS cap $15 = SageMaker + Bedrock; W6 uses no SageMaker, the archive estimate over-counts ~1.5x
TAKES = 3  # hard-check failures (bad JSON, wrong segment count, Devanagari) retry up to this many takes
JUDGE_CRITICAL = {"meaning", "omission", "addition", "number", "negation", "intent", "correction", "name"}
JUDGE_CODES = JUDGE_CRITICAL | {"loanword", "style", "fluency"}

TRANSLATE_SYS = RULES + """

Input: JSON {"segments": [Hindi segment, ...]}.
Output: only JSON, no prose: {"segments": [{"en": "..."}, {"en": "...", "alt": "..."}, ...]}, one object per input
segment in the same order; "alt" only where rule 5 applies."""

JUDGE_SYS = """You are a careful bilingual Hindi-English reviewer. You check English translations of transcribed Hindi
speech against the Hindi. The transcripts are verbatim: they may contain non-standard spellings, fillers and fragments.
The translations were written to the style guide below; anything the guide asks for (dropped fillers and repetitions,
a self-correction rendered as its corrected meaning, numbers in words, lakh/crore kept) is correct, not an error.

=== STYLE GUIDE ===
""" + RULES + """
=== END ===

Input: JSON {"segments": [{"i": 0, "hi": "...", "en": "...", "alt": "..."?}, ...]}; segments are consecutive parts of
one recording, so use the neighbours for context.
Report only real errors that change what a listener understands, or that break the style guide. A different but
equally correct wording is not an error. Also not errors: a unit or currency the context implies, made explicit
("rupees" after an amount of money); a natural English word for अरे, तो, यार, भाई or जी, or leaving it out; a tense or
word order that keeps the meaning; in a segment that corrects an earlier segment, an explicit correction ("Sorry, not
Thursday, Friday."). Read Hindi numbers carefully before calling a number wrong. Codes:
- meaning: mistranslated
- omission: content of the Hindi missing (dropped fillers, repetitions and abandoned values are not omissions)
- addition: content not in the Hindi
- number: a number, amount, unit, date, time or digit string is wrong, or lakh/crore was converted
- negation: a negation lost, added or misplaced
- intent: a question, request, condition, refusal or confirmation changed, or who does what to whom changed
- correction: the abandoned value of a self-correction stated as fact in "en", or the correction lost
- name: a person, place, organisation or product name wrong
- loanword: an English word the speaker used was replaced by a synonym or restored wrongly
- style: breaks another style-guide rule (digits, symbols, million, filler kept, content moved across segments)
- fluency: ungrammatical English
Every error quotes evidence: "field" ("en" or "alt"), "en_span" copied exactly from that field (empty only for an
omission), "hi_span" copied exactly from the Hindi (empty only for an addition), and "fix", the corrected English for
en_span.
Output only JSON: {"segments": [{"i": 0, "errors": [{"code": "...", "field": "en", "en_span": "...", "hi_span": "...",
"fix": "..."}]}, ...]} with one object per input segment, "errors": [] when it is right."""

REPAIR_USER = """{hi}

Your previous translation:
{prev}

A reviewer and an automatic checker raised these issues. Either may be wrong: fix only real problems, keep segments
that are already right unchanged, and follow the style guide.
{issues}

Return the full output again in the same JSON format."""

_brt = None


def _converse(req):
    global _brt
    if _brt is None:
        _brt = store.session().client("bedrock-runtime")
    kw = {"modelId": req["model"], "system": [{"text": req["system"]}],
          "messages": [{"role": "user", "content": [{"text": req["user"]}]}],
          "inferenceConfig": {"temperature": req["temperature"], "maxTokens": req["max_tokens"]}}
    if req.get("reasoning_effort"):
        kw["additionalModelRequestFields"] = {"reasoning_effort": req["reasoning_effort"]}
    r = _brt.converse(**kw)
    blocks = r["output"]["message"]["content"]
    return {"text": "".join(b.get("text", "") for b in blocks), "stop": r["stopReason"], "usage": r["usage"],
            "reasoning_chars": sum(len(b.get("reasoningContent", {}).get("reasoningText", {}).get("text", "")) for b in blocks),
            "request_id": r["ResponseMetadata"]["RequestId"]}, None


def llm(name, system, user, take, max_tokens):
    """-> (response text, archive key). Takes > 0 sample at a higher temperature, so a retry gives a different answer."""
    mid, pin, pout = MODELS[name]
    req = {"model": mid, "system": system, "user": user, "temperature": 0.0 if take == 0 else 0.7, "take": take,
           "max_tokens": max_tokens + (6000 if name == "gptoss" else 0)}
    if name == "gptoss":
        req["reasoning_effort"] = "medium"
    # spend estimate written to the archive before the call (archive_call's cap); real usage is in the response
    est_out = len(user) / 2 + (1500 if name == "gptoss" else 0)
    inr = ((len(system) + len(user)) / 2.5 * pin + est_out * pout) / 1e6 * USD_INR
    resp, _ = archive.archive_call("bedrock", req, _converse, stream="w6", inr=round(inr, 4), cap_inr=CAP_INR)
    return resp["text"], "archive/api/bedrock/" + archive.request_key(req)


def prefill_spent(stream="w6"):
    """archive.spent() GETs every archived call one by one: minutes once there are a few thousand. Same sum, 32 GETs
    at a time, written into its cache before the worker threads start. ponytail: report to master for a spend index."""
    if stream not in archive._spent:
        ks = [k for k in store.keys("archive/api/") if k.endswith(".json")]
        metas = ThreadPoolExecutor(32).map(lambda k: json.loads(store.get(k)), ks)
        archive._spent[stream] = sum(m.get("inr", 0) for m in metas if m.get("stream") == stream)
    return archive._spent[stream]


def it2_request(his):
    return {**IT2, "src_lang": "hin_Deva", "tgt_lang": "eng_Latn", "num_beams": 5, "segments": his}


def _it2_missing(req):
    raise SystemExit("IndicTrans2 output not archived: run `uv run w6/it2.py --in RECORDS.jsonl` first (GPU)")


def parse(text, n):
    """LLM JSON -> list of n {"en", "alt"?} dicts, or raise ValueError (hard failure: retry)."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON")
    segs = json.loads(m.group(0))["segments"]
    if len(segs) != n:
        raise ValueError(f"{len(segs)} segments for {n}")
    out = []
    for s in segs:
        en, alt = s.get("en"), s.get("alt")
        if not isinstance(en, str) or not clean(en):
            raise ValueError("empty en")
        d = {"en": clean(en, filler=True)}
        if isinstance(alt, str) and clean(alt) and clean(alt) != d["en"]:
            d["alt"] = clean(alt)
        out.append(d)
    return out


ASCII = str.maketrans({"\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"', "\u2013": "-", "\u2014": "-", "\u2026": "..."})
FILLER = re.compile(r"^(?:uh+|um+|hmm+|ah+|er+)[,.]\s+(?=\w)", re.I)


def clean(x, filler=False):
    """Plain ASCII punctuation, single spaces (STYLE rule 10); `filler`: drop a leading "Uh," (rule 3)."""
    x = " ".join(x.translate(ASCII).split())
    if filler and FILLER.match(x):
        x = FILLER.sub("", x)
        x = x[:1].upper() + x[1:]
    return x


def auto_qa(his, segs, gl):
    """-> per segment {"en": flags, "alt": flags}."""
    res = []
    for h, s in zip(his, segs):
        g = qa.sentence_glosses(h, gl)
        r = {"en": qa.check(h, s["en"], g)}
        if "alt" in s:
            r["alt"] = qa.check(h, s["alt"], g, spoken=True)
        res.append(r)
    return res


def hard(auto):
    return any(qa.HARD & set(a) for r in auto for a in r.values())


def translate(engine, his, gl, take0=0, user=None):
    """-> (segments, auto, archive keys, takes used) or (None, error, keys, takes)."""
    user = user or json.dumps({"segments": his}, ensure_ascii=False)
    keys, err = [], None
    for take in range(take0, take0 + TAKES):
        if engine == "it2":
            resp, _ = archive.archive_call("it2", it2_request(his), _it2_missing, stream="w6", inr=0, cap_inr=CAP_INR)
            keys.append("archive/api/it2/" + archive.request_key(it2_request(his)))
            segs = [{"en": " ".join(t.split()) or "."} for t in resp["segments"]]
        else:
            u = user if not err or "segments for" not in err else \
                user + f"\n\nReturn exactly {len(his)} objects in \"segments\", one per input segment, even for a long one."
            text, k = llm(engine, TRANSLATE_SYS, u, take, 1500 + 150 * len(his))
            keys.append(k)
            try:
                segs = parse(text, len(his))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                err = f"take {take}: {e}"
                continue
        auto = auto_qa(his, segs, gl)
        if not hard(auto) or engine == "it2":  # IndicTrans2 is deterministic: a retry would return the same
            return segs, auto, keys, take + 1
        err = f"take {take}: hard QA {[sorted(qa.HARD & set(a)) for r in auto for a in r.values()]}"
    return None, err, keys, take0 + TAKES


def _norm(x):
    return " ".join(re.sub(r"[.,!?;:'\"()\[\]\-\u2013\u2014\u0964]", " ", qa.nfc(x).lower()).split())


def verified(e, hi, seg):
    """A judge error counts only if its quotes exist and its fix changes something (drops hallucinated errors)."""
    f, en, hs, fix = e.get("field", "en"), _norm(e.get("en_span") or ""), _norm(e.get("hi_span") or ""), _norm(e.get("fix") or "")
    if e.get("code") not in JUDGE_CODES or f not in seg:
        return False
    if (en and en not in _norm(seg[f])) or (hs and hs not in _norm(hi)) or not (en or hs) or (en and fix == en):
        return False
    return bool((en or e["code"] == "omission") and (hs or e["code"] == "addition"))


def judge(name, his, segs):
    """-> (per segment {"en": codes, "alt": codes, "errors": verified errors, "dropped": n}, archive key);
    None if no take parses."""
    user = json.dumps({"segments": [{"i": i, "hi": h, **s} for i, (h, s) in enumerate(zip(his, segs))]}, ensure_ascii=False)
    for take in range(TAKES):
        text, k = llm(name, JUDGE_SYS, user, take, 800 + 200 * len(his))
        try:
            out = json.loads(re.search(r"\{.*\}", text, re.S).group(0))["segments"]
            if len(out) != len(his):
                continue
            res = []
            for o, h, s in zip(out, his, segs):
                errs = [e for e in o.get("errors") or [] if isinstance(e, dict)]
                ok = [e for e in errs if verified(e, h, s)]
                res.append({"en": sorted({e["code"] for e in ok if e.get("field", "en") == "en"}),
                            "alt": sorted({e["code"] for e in ok if e.get("field") == "alt"}),
                            "errors": ok, "dropped": len(errs) - len(ok)})
            return res, k
        except (AttributeError, KeyError, TypeError, json.JSONDecodeError):
            pass
    return None, k


NUM_FLAGS = {"num_extra", "num_missing", "num_corr", "unit"}


def arbitrate(his, auto, jud):
    """The number parser outranks the judge on values: a judge `number` error on a field whose Hindi numbers all appear
    in it, with no extra one (no qa num/unit flag, at least one Hindi number), is overruled and counted."""
    for h, a, v in zip(his, auto, jud or []):
        ok = {f for f in a if qa.hi_numbers(h) and not NUM_FLAGS & set(a[f])}
        keep = [e for e in v["errors"] if not (e["code"] == "number" and e.get("field", "en") in ok)]
        v["overruled"], v["errors"] = len(v["errors"]) - len(keep), keep
        for f in ("en", "alt"):
            v[f] = sorted({e["code"] for e in keep if e.get("field", "en") == f})
    return jud


def critical(auto, jud):
    """Score to minimise over `en`: segments the judge marks critical + automatic critical/hard flags. A flagged `alt`
    is dropped instead (drop_bad_alts), so it never drives a repair."""
    j = sum(1 for x in jud or [] if JUDGE_CRITICAL & set(x["en"]))
    return j + sum(len((qa.CRITICAL | qa.HARD) & set(r["en"])) for r in auto)


def drop_bad_alts(segs, auto, jud):
    """`alt` is optional (§5.2): keep it only when neither the checks nor the judge find a critical error in it. -> dropped idx"""
    out = []
    for i, s in enumerate(segs):
        if "alt" in s and ((qa.CRITICAL | qa.HARD) & set(auto[i].get("alt", {})) or (jud and JUDGE_CRITICAL & set(jud[i]["alt"]))):
            del s["alt"]
            out.append(i)
    return out


def issues_text(segs, auto, jud):
    lines = []
    for i, s in enumerate(segs):
        for e in (x for x in (jud[i]["errors"] if jud else []) if x.get("field", "en") == "en"):
            lines.append(f"- segment {i} {e.get('field', 'en')}: reviewer says {e['code']}: "
                         f"\"{e.get('en_span', '')}\" (Hindi \"{e.get('hi_span', '')}\") -> \"{e.get('fix', '')}\"")
        for part in ("en",):
            for k, v in auto[i].get(part, {}).items():
                if k in qa.CRITICAL | qa.HARD | {"digits", "unit", "neg"}:
                    lines.append(f"- segment {i} {part}: checker says {k}: {v}")
    return "\n".join(lines)


def process(rec, a):
    his = [s["hi_text"] for s in rec["sentences"]]
    gl = qa.glosses(rec.get("_w1", {}).get("segments", []))
    segs, auto, keys, takes = translate(a.engine, his, gl)
    if segs is None:
        return None, {"id": rec["id"], "stage": "translate", "rule": "no_translation", "detail": auto}
    jud, jk = judge(a.judge, his, segs) if a.judge else (None, None)
    jud = arbitrate(his, auto, jud)
    keys += [jk] * bool(jk)
    score, repaired = critical(auto, jud), False
    if a.engine != "it2" and not a.no_repair and score:
        user = REPAIR_USER.format(hi=json.dumps({"segments": his}, ensure_ascii=False),
                                  prev=json.dumps({"segments": segs}, ensure_ascii=False), issues=issues_text(segs, auto, jud))
        s2, a2, k2, t2 = translate(a.engine, his, gl, take0=0, user=user)
        keys += k2
        if s2 is not None:
            j2, jk2 = judge(a.judge, his, s2) if a.judge else (None, None)
            j2 = arbitrate(his, a2, j2)
            keys += [jk2] * bool(jk2)
            if critical(a2, j2) < score:
                segs, auto, jud, score, repaired = s2, a2, j2, critical(a2, j2), True
    alt_dropped = drop_bad_alts(segs, auto, jud)
    model = IT2["model"].split("/")[1] if a.engine == "it2" else MODELS[a.engine][0]
    rec["translation_method"] = f"{VERSION}/{model}" + ("+repair" if repaired else "")
    for s, t in zip(rec["sentences"], segs):
        s.pop("en_alt", None)
        s["en_text"] = t["en"]
        if "alt" in t:
            s["en_alt"] = [t["alt"]]
    rec.setdefault("_qa", {})["w6"] = {"engine": a.engine, "judge": a.judge, "takes": takes, "repaired": repaired,
                                       "critical": score, "alt_dropped": alt_dropped, "auto": auto, "verdict": jud}
    rec.setdefault("_archive", {})["translate"] = keys
    return rec, None


def apply_patches(rec, patches):
    """w6/patches.jsonl: {"id","i","en_text","was"}; refused (kept auto) if the auto text changed since the review."""
    for p in patches.get(rec["id"], []):
        s = rec["sentences"][p["i"]]
        if s["en_text"] == p["was"]:
            s["en_text"] = p["en_text"]
            if not rec["translation_method"].endswith("+human"):
                rec["translation_method"] += "+human"
        else:
            rec["_qa"]["w6"].setdefault("stale_patch", []).append(p["i"])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine", default="mistral3", choices=sorted(MODELS) + ["it2"])  # bench: tasks/W6-translation.md
    ap.add_argument("--judge", default="gptoss", choices=sorted(MODELS) + [""])
    ap.add_argument("--no-repair", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.judge == a.engine:
        raise SystemExit("the judge must be another model family than the engine")
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("records.jsonl", "rejects.jsonl"):
        p = a.out / f
        if p.exists() and (b := p.read_bytes()) and not b.endswith(b"\n"):
            p.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: drop it, that id is redone
    done = {json.loads(l)["id"] for f in ("records.jsonl", "rejects.jsonl") if (a.out / f).exists() for l in open(a.out / f)}
    todo = [r for r in map(json.loads, open(a.inp, encoding="utf-8")) if r["id"] not in done]
    prefill_spent()
    patches = {}
    if PATCHES.exists():
        for l in open(PATCHES, encoding="utf-8"):
            p = json.loads(l)
            patches.setdefault(p["id"], [x for x in patches.get(p["id"], []) if x["i"] != p["i"]]).append(p)  # last wins
    with open(a.out / "records.jsonl", "a", encoding="utf-8") as ok, open(a.out / "rejects.jsonl", "a", encoding="utf-8") as bad, \
            ThreadPoolExecutor(a.workers) as ex:
        for n, (rec, rej) in enumerate(ex.map(lambda r: process(r, a), todo), 1):
            if rec:
                ok.write(json.dumps(apply_patches(rec, patches), ensure_ascii=False) + "\n")
            else:
                bad.write(json.dumps(rej, ensure_ascii=False) + "\n")
            ok.flush(); bad.flush()
            if n % 25 == 0:
                print(f"{n}/{len(todo)}", file=sys.stderr)


if __name__ == "__main__":
    main()
