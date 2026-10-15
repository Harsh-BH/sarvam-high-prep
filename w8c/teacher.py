# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Teacher judge: Claude (run as Claude Code subagents) checks English against the Hindi source.

    uv run w8c/teacher.py build RUN [--targets 150] [--calib 40] [--batch 70]   items + blind batches
    uv run w8c/teacher.py score RUN                                             validate judgments, write report
    uv run w8c/teacher.py selftest

Judged against the Hindi, never the reference (our references are themselves LLM output). W6's error codes and
quote rule (w6/translate.py: JUDGE_CODES, verified) are reused: an error counts only if its English and Hindi quotes
exist. Two modes: "target" = a dataset translation, judged with W6's style guide; "output" = what a model said (ASR of
its speech), judged on meaning only. Items are shuffled and keyed anonymously, so the teacher never knows the system.
Calibration: clean items plus copies with one injected error each (number, negation, omission, name) measure the
teacher's catch rate and false-flag rate before anything else is trusted.
Work lives in data/w8c/teacher/<RUN>/: items.jsonl (key -> truth), batches/bNN.json (what the teacher sees),
out/bNN.jsonl (the teacher's judgments), judged.jsonl + report.md.
"""
import argparse, json, random, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "w6")]
import translate as W6  # noqa: E402  (JUDGE_CODES, JUDGE_CRITICAL, verified, RULES)
import s3 as store  # noqa: E402
from run import _main_checkout  # noqa: E402

WORK = _main_checkout() / "data/w8c/teacher"
EVAL = {"base": "stages/w8c_eval/20261014T222707Z-e398964", "ft": "stages/w8c_eval/20261014T222712Z-e398964"}
RELEASE = "release/pilot-v2/manifest.jsonl"
NUMS = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twelve", "fifteen", "twenty",
        "thirty", "forty", "fifty", "hundred", "thousand", "lakh", "crore"]
NAMES = ["Rahul", "Priya", "Mumbai", "Lucknow"]
STARTERS = set("The This That These Those They Then There Their What When Where Which While With Your Yours Also Okay Please Sorry Hello Thanks Thank Yes Just Because After Before Since Sometimes Today Tomorrow Yesterday Every Some Many Most Once First Next Last".split())

RUBRIC = """You are a careful bilingual Hindi-English reviewer acting as a teacher. For each item, check whether the
English conveys what the Hindi says. The Hindi is a verbatim transcript of speech (fillers, repetitions, fragments).
Report only real errors that change what a listener understands. A different but equally correct wording is not an
error. Dropped fillers and repetitions are not omissions. A self-correction rendered as its corrected meaning is right.
Mode "output": the English is an ASR transcript of a model's synthesised speech. Ignore spelling, punctuation, casing
and number formatting (digits vs words); judge meaning, numbers' values, names, negation and intent only. Do not use
the codes style or fluency unless the English is unintelligible.
Mode "target": the English is a dataset translation written to the style guide below; style-guide breaks count.

=== STYLE GUIDE (mode "target" only) ===
""" + W6.RULES + """
=== END ===

Codes: meaning, omission, addition, number, negation, intent, correction, name (these are critical); loanword, style,
fluency (minor). Definitions: meaning = mistranslated; omission = Hindi content missing; addition = content not in the
Hindi; number = a number, amount, unit, date, time or digit string wrong; negation = lost, added or misplaced; intent
= a question, request, condition, refusal or confirmation changed, or who-does-what-to-whom changed; correction = the
abandoned value of a self-correction stated as fact, or the correction lost; name = a person, place, organisation or
product name wrong; loanword = an English word the speaker used replaced or restored wrongly.
Every error quotes evidence copied EXACTLY: "en_span" from the English (empty only for an omission), "hi_span" from the
Hindi (empty only for an addition), and "fix", the corrected English for en_span. Read Hindi numbers carefully.
Output: one JSON object per line, one line per item, in any order, nothing else:
{"k": "<item key>", "errors": [{"code": "...", "en_span": "...", "hi_span": "...", "fix": "..."}]}
Use "errors": [] when the English is right."""


def _lines(key):
    return [json.loads(l) for l in store.get(key).decode().splitlines() if l.strip()]


def items_outputs():
    out = []
    for system, run in EVAL.items():
        for r in _lines(f"{run}/test/per_record.jsonl"):
            if r.get("asr") and r.get("src"):
                out.append({"id": r["id"], "system": system, "kind": r.get("kind", ""), "mode": "output",
                            "hi": r["src"], "en": r["asr"], "comet": float(r.get("asr_comet") or "nan")})
    return out


def items_targets(recs):
    return [{"id": r["id"], "system": "w6-target", "kind": "synthetic" if r["source"]["synthetic"] else "real", "mode": "target",
             "hi": " ".join(s["hi_text"] for s in r["sentences"]), "en": " ".join(s["en_text"] for s in r["sentences"])}
            for r in recs]


def corrupt(item, kind, rng):
    """One injected error of `kind` into a copy of a target item, or None if this item can't carry it."""
    en = item["en"]
    if kind == "number":
        hits = [m for m in re.finditer(r"\b(" + "|".join(NUMS) + r")\b", en, flags=re.I)]
        if not hits:
            return None
        m = rng.choice(hits)
        new = rng.choice([n for n in NUMS[:13] if n != m.group(0).lower()])
        en = en[:m.start()] + new + en[m.end():]
    elif kind == "negation":
        m = re.search(r"\b(is|are|was|were|will|can|do|does|did|have|has) not\b|n't\b", en)
        if m:
            en = en[:m.start()] + (m.group(1) if m.group(1) else "") + en[m.end():]
        else:
            m = re.search(r"\b(is|are|was|were|will|can)\b", en)
            if not m:
                return None
            en = en[:m.end()] + " not" + en[m.end():]
    elif kind == "omission":
        sents = re.split(r"(?<=[.?!])\s+", en)
        if len(sents) < 3:
            return None
        i = rng.randrange(1, len(sents))
        en = " ".join(sents[:i] + sents[i + 1:])
    elif kind == "name":
        caps = [m for m in re.finditer(r"\b[A-Z][a-z]{3,}\b", en) if m.group(0) not in STARTERS]
        if not caps:
            return None
        m = rng.choice(caps)
        en = en[:m.start()] + rng.choice([n for n in NAMES if n != m.group(0)]) + en[m.end():]
    return None if en == item["en"] else {**item, "en": en, "system": f"calib-{kind}", "expect": kind}


def build(run, n_targets=150, n_calib=40, batch=70, seed=7, recs=None, outputs=None):
    rng = random.Random(seed)
    recs = recs if recs is not None else [json.loads(l) for l in store.get(RELEASE).decode().splitlines() if l.strip()]
    rng.shuffle(recs)
    real = [r for r in recs if not r["source"]["synthetic"]]
    syn = [r for r in recs if r["source"]["synthetic"]]
    targets = items_targets(real[: n_targets // 2] + syn[: n_targets - n_targets // 2])
    pool = items_targets(real[n_targets // 2: n_targets // 2 + 4 * n_calib] + syn[n_targets // 2: n_targets // 2 + 4 * n_calib])
    rng.shuffle(pool)
    clean = [{**x, "system": "calib-clean", "expect": None} for x in pool[:n_calib]]
    bad, kinds = [], ["number", "negation", "omission", "name"]
    for x in pool[n_calib:]:
        if len(bad) >= n_calib:
            break
        c = corrupt(x, kinds[len(bad) % 4], rng)
        if c:
            bad.append(c)
    items = (outputs if outputs is not None else items_outputs()) + targets + clean + bad
    rng.shuffle(items)
    for n, x in enumerate(items):
        x["k"] = f"t{n:05d}"  # anonymous: the teacher never sees system or id
    d = WORK / run
    (d / "batches").mkdir(parents=True, exist_ok=True)
    (d / "out").mkdir(exist_ok=True)
    (d / "items.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in items), encoding="utf-8")
    for b in range(0, len(items), batch):
        shown = [{"k": x["k"], "mode": x["mode"], "hi": x["hi"], "en": x["en"]} for x in items[b: b + batch]]
        (d / "batches" / f"b{b // batch:02d}.json").write_text(json.dumps({"rubric": RUBRIC, "items": shown}, ensure_ascii=False, indent=0), encoding="utf-8")
    return items


def verdict(errors):
    codes = {e["code"] for e in errors}
    return "critical" if codes & W6.JUDGE_CRITICAL else ("minor" if codes else "correct")


def validate(item, raw_errors):
    """Keep only errors whose quotes exist (W6's rule). -> (kept, dropped count)."""
    kept = [e for e in raw_errors if isinstance(e, dict) and W6.verified({**e, "field": "en"}, item["hi"], {"en": item["en"]})]
    return kept, len(raw_errors) - len(kept)


def score(run):
    d = WORK / run
    items = {x["k"]: x for x in map(json.loads, open(d / "items.jsonl", encoding="utf-8"))}
    got = {}
    for p in sorted((d / "out").glob("b*.jsonl")):
        for l in open(p, encoding="utf-8"):
            l = l.strip()
            if l.startswith("{"):
                try:
                    j = json.loads(l)
                except ValueError:
                    continue
                if j.get("k") in items:
                    got[j["k"]] = j.get("errors") or []
    judged = []
    for k, x in items.items():
        if k not in got:
            continue
        kept, dropped = validate(x, got[k])
        judged.append({**x, "errors": kept, "dropped": dropped, "verdict": verdict(kept)})
    (d / "judged.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in judged), encoding="utf-8")
    rep = report(judged, len(items))
    (d / "report.md").write_text(rep, encoding="utf-8")
    return judged, rep


def report(judged, total):
    by = defaultdict(list)
    for x in judged:
        by[x["system"] if not x["system"].startswith("calib-") else "calibration"].append(x)
        if x["mode"] == "output":
            by[f'{x["system"]} · {x["kind"]}'].append(x)
    lines = [f"Judged {len(judged)} of {total} items.\n",
             "| set | n | correct | minor only | any critical | critical errors / 100 Hindi words | top error codes |",
             "|---|---|---|---|---|---|---|"]
    for name in sorted(by):
        xs = by[name]
        if name == "calibration":
            continue
        v = Counter(x["verdict"] for x in xs)
        codes = Counter(e["code"] for x in xs for e in x["errors"]).most_common(4)
        crit = sum(e["code"] in W6.JUDGE_CRITICAL for x in xs for e in x["errors"])
        words = sum(len(x["hi"].split()) for x in xs)
        lines.append(f"| {name} | {len(xs)} | {v['correct'] / len(xs):.0%} | {v['minor'] / len(xs):.0%} | "
                     f"{v['critical'] / len(xs):.0%} | {100 * crit / max(words, 1):.2f} | {', '.join(f'{c} {n}' for c, n in codes)} |")
    cal = by.get("calibration", [])
    if cal:
        clean = [x for x in cal if x["expect"] is None]
        bad = [x for x in cal if x["expect"]]
        caught = [x for x in bad if any(e["code"] == x["expect"] or (x["expect"] == "omission" and e["code"] == "omission")
                                         for e in x["errors"])]
        flagged_any = [x for x in bad if x["verdict"] == "critical"]
        lines += ["", "**Calibration (is the teacher trustworthy?)**", "",
                  f"- injected errors caught with the right code: {len(caught)}/{len(bad)}"
                  f" ({len(caught) / max(len(bad), 1):.0%}); flagged critical at all: {len(flagged_any)}/{len(bad)}",
                  f"- unmodified dataset targets flagged critical (real errors in our data, or teacher false alarms; inspect): "
                  f"{sum(x['verdict'] == 'critical' for x in clean)}/{len(clean)}"]
        per = Counter(x["expect"] for x in bad)
        hit = Counter(x["expect"] for x in caught)
        lines.append("- by kind: " + ", ".join(f"{k} {hit[k]}/{per[k]}" for k in sorted(per)))
    lines.append(f"\nQuotes that didn't match the texts (teacher hallucinations, discarded): {sum(x['dropped'] for x in judged)}")
    return "\n".join(lines) + "\n"


def selftest():
    rng = random.Random(1)
    t = {"id": "rec-1", "system": "w6-target", "kind": "real", "mode": "target", "hi": "मेरे पास पाँच लाख रुपये नहीं हैं। कल रमेश आएगा। फिर बात करेंगे।",
         "en": "I do not have five lakh rupees. Ramesh will come tomorrow. We will talk then."}
    n = corrupt(t, "number", rng)
    assert n and "five" not in n["en"] and n["expect"] == "number"
    g = corrupt(t, "negation", rng)
    assert g and " not " not in g["en"]
    assert corrupt(t, "omission", rng)["en"].count(".") == 2
    assert "Ramesh" not in corrupt(t, "name", rng)["en"]
    good = {"code": "number", "en_span": "seven lakh", "hi_span": "पाँच लाख", "fix": "five lakh"}
    fake = {"code": "number", "en_span": "nine crore", "hi_span": "पाँच लाख", "fix": "five lakh"}
    x = {**t, "en": t["en"].replace("five", "seven")}
    kept, dropped = validate(x, [good, fake])
    assert kept == [good] and dropped == 1 and verdict(kept) == "critical" and verdict([]) == "correct"
    assert verdict([{"code": "style"}]) == "minor"
    judged = [{**x, "system": "calib-number", "expect": "number", "errors": kept, "dropped": 1, "verdict": "critical"},
              {**t, "system": "calib-clean", "expect": None, "errors": [], "dropped": 0, "verdict": "correct"},
              {**t, "mode": "output", "system": "ft", "errors": [], "dropped": 0, "verdict": "correct"}]
    rep = report(judged, 3)
    assert "caught with the right code: 1/1" in rep and "inspect): 0/1" in rep and "| ft · real |" in rep
    assert "| calib" not in rep and "/ 100 Hindi words" in rep
    print("ok: teacher selftest")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["build", "score", "selftest"])
    ap.add_argument("run", nargs="?")
    ap.add_argument("--targets", type=int, default=150)
    ap.add_argument("--calib", type=int, default=40)
    ap.add_argument("--batch", type=int, default=70)
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    if a.cmd == "build":
        items = build(a.run, a.targets, a.calib, a.batch)
        print(f"{len(items)} items -> {len(list((WORK / a.run / 'batches').glob('b*.json')))} batches in {WORK / a.run}")
        print(Counter(x["system"] for x in items))
    else:
        print(score(a.run)[1])


if __name__ == "__main__":
    main()
