# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""W4 gate rules, audit sample, and audit import (design §3.3: one human, no annotators; suspect clips are dropped).

    uv run w4/review_list.py build     # stages/{transcribe,asr_check} -> data/w4/pool.jsonl, rejects.jsonl, review.tsv
    uv run w4/review_list.py import    # filled review.tsv rows -> archive/human/transcribe/<date>-harsh.jsonl (S3,
                                       # write-once) + w4/patches.jsonl (the git copy, same lines)

`check(rec, hyps)` is the per-record gate the `asr_check` stage (w4/asr_check.py) applies. First rule that fires:
  P0  converter flag left on the record (the `transcribe` stage already rejects these; kept as a guard)
  P1  both independent ASRs (vakyansh, Whisper-turbo; not trained on IndicVoices) disagree with the reference
      (best CER >= CER_HI) or both hear more than it (length ratio >= LEN_HI); plus W1's ear items
  P2  entity check against the IndicConformer CTC draft (`ic-ctc`): a reference number word whose aligned draft word
      is a different number (value conflict), or >= ENT_MIN entity words (numbers + English-in-Devanagari) without a
      close draft match making up > ENT_FRAC of the clip's entity words
An audit `ok`/`fix` (a human listened; `_w4.review`, set by `transcribe` from w4/patches.jsonl) overrides P1/P2.
Human work = one frozen, stratified ~1 h audit sample (w4/audit_ids.txt, drawn once from accepted clips, sha1 order
within scenario x band strata); review.tsv holds only that sample.
"""
import csv, hashlib, json, re, sys
from collections import defaultdict
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from asr_score import NUMVAL, align, edits, english_words, lenient, strict  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STAGES = ROOT / "data/w4/stages"  # local runs of the two W4 stages: <stage>/{records,rejects}.jsonl
POOL, REJ = ROOT / "data/w4/pool.jsonl", ROOT / "data/w4/rejects.jsonl"
TSV, PATCHES, AUDIT = ROOT / "data/w4/review.tsv", ROOT / "w4/patches.jsonl", ROOT / "w4/audit_ids.txt"
MODELS = ("vakyansh-w2v2", "whisper-turbo", "ic-ctc")  # two independent routers + the draft
CER_HI, LEN_HI = 0.35, 1.25  # P1
ENT_MIN, ENT_FRAC = 2, 1 / 3  # P2
AUDIT_S, STRATUM_MIN_S = 3600, 600  # ~1 h; every stratum gets >= 10 min (or all it has)
W1_NOTES = {"rec-000001": "W1: ASR loses the final phrase", "rec-000022": "W1: ASR loses the final phrase",
            "rec-000021": "W1: फैन vs फ्रेंड (both ASRs hear फ्रेंड)", "rec-000434": "W1: possible mismatch after sentence 1",
            "rec-000177": "W1: Whisper loops on noise", "rec-000046": "W1: reference omits a menu list"}
COLS = ["id", "stratum", "duration", "audio", "current", "asr_hint", "action", "corrected", "minutes", "note"]


def text_of(rec):
    return " ".join(s["hi_text"] for s in rec.get("sentences", []))


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def cer(ref, hyp):
    r, h = "".join(lenient(w) for w in strict(ref)), "".join(lenient(w) for w in strict(hyp))
    return edits(r, h) / max(1, len(r)), len(h) / max(1, len(r))


def near(a, b):
    """Same entity word up to spelling (अठारह/अठ्ठारह, ओप्पो/ओपो) or word split (मिक्रो in मिक्रोफाइबर)."""
    # a conjunct (ठ्ठ) is 2 code points, so 2 edits from 5 code points up; short words (कलर/और) get 1
    return edits(a, b) <= max(2 if len(a) >= 5 else 1, len(a) // 3) or (len(a) >= 3 and a in b)


def same(w, x):
    """Two number words match only by value (सात 7 vs साठ 60); anything else by `near`."""
    return NUMVAL[w] == NUMVAL[x] if w in NUMVAL and x in NUMVAL else near(w, x)


def entities(rec, draft):
    """-> (entity words, value conflicts, misses) of the reference against the IndicConformer draft.
    Conflict: the aligned draft word is a *different* number word (सात vs साठ). Miss: no draft word within +-2 of the
    aligned position (or two adjacent ones joined: स्टॉप वॉच) is `same` as it. Entities: number words (asr_score.NUMVAL,
    Hindi and English-in-Devanagari) and IV-glossed English words."""
    ref, hyp = [lenient(w) for w in strict(text_of(rec))], [lenient(w) for w in strict(draft)]
    english, amap = english_words(rec), align(ref, hyp)[1]
    n = conflict = miss = j = 0
    for k, w in enumerate(ref):
        if not (w in NUMVAL or w in english):
            j = amap.get(k, j)
            continue
        n += 1
        j = amap.get(k, j)
        h = hyp[amap[k]] if k in amap else None
        win = hyp[max(0, j - 2): j + 3]
        if w in NUMVAL and h in NUMVAL and NUMVAL[h] != NUMVAL[w]:
            conflict += 1
        elif not any(same(w, x) for x in win + [a + b for a, b in zip(win, win[1:])]):
            miss += 1
    return n, conflict, miss


def stratum(rec):
    return f"{rec['_w1']['scenario']}/{'8k' if rec['_w1']['orig_sample_rate'] == 8000 else 'wide'}"


def check(rec, hyps):
    """One record + {model: hypothesis} -> (rule or None, detail). Detail goes to `_qa.asr_check` or the reject."""
    ref = text_of(rec)
    c = {m: cer(ref, hyps[m]) for m in MODELS[:2]}
    n, conflict, miss = entities(rec, hyps["ic-ctc"])
    best, longest = min(x[0] for x in c.values()), min(x[1] for x in c.values())
    detail = {"cer": {m: round(x[0], 3) for m, x in c.items()}, "len_ratio": {m: round(x[1], 3) for m, x in c.items()},
              "entities": n, "number_conflicts": conflict, "entity_misses": miss}
    if rec["id"] in W1_NOTES:
        detail["note"] = W1_NOTES[rec["id"]]
    if rec.get("_w4", {}).get("flags") or not rec.get("sentences"):
        return "P0", detail
    if rec["_w4"].get("review", {}).get("action") in ("ok", "fix"):
        return None, detail
    if best >= CER_HI or longest >= LEN_HI or rec["id"] in W1_NOTES:
        return "P1", detail
    if conflict or (miss >= ENT_MIN and miss > ENT_FRAC * n):
        return "P2", detail
    return None, detail


def build(stages=STAGES):
    """Convenience outputs of the local stage runs: pool.jsonl (= asr_check records), rejects.jsonl (both stages),
    and the audit TSV."""
    acc = [json.loads(l) for l in open(stages / "asr_check/records.jsonl")]
    rej = [json.loads(l) for st in ("transcribe", "asr_check") if (stages / st / "rejects.jsonl").exists()
           for l in open(stages / st / "rejects.jsonl")]
    POOL.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in acc))
    REJ.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rej))
    by = defaultdict(int)
    for x in rej:
        by[(x["stage"], x["rule"])] += 1
    print(f"accepted {len(acc)} clips {sum(r['duration'] for r in acc) / 3600:.3f} h -> {POOL}")
    print("rejected:", ", ".join(f"{s}/{r} {n}" for (s, r), n in sorted(by.items())) or "none", "->", REJ)
    draft = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(stages / "asr_check/asr/ic-ctc.jsonl")}
    audit(acc, draft)


def audit(accepted, draft):
    """Freeze the stratified ~1 h sample on first run; afterwards only re-list it."""
    if not AUDIT.exists():
        strata = defaultdict(list)
        for r in accepted:
            strata[stratum(r)].append(r)
        total = sum(r["duration"] for r in accepted)
        ids = []
        for s, xs in sorted(strata.items()):
            have = sum(r["duration"] for r in xs)
            quota, got = min(have, max(STRATUM_MIN_S, AUDIT_S * have / total)), 0.0
            for r in sorted(xs, key=lambda r: sha(r["id"])):
                if got >= quota:
                    break
                ids.append(r["id"])
                got += r["duration"]
        AUDIT.write_text(f"# W4 audit sample, drawn {date.today()} from {len(accepted)} accepted clips; frozen, never redraw\n"
                         + "".join(i + "\n" for i in sorted(ids)))
    ids = [l.strip() for l in open(AUDIT) if l.strip() and not l.startswith("#")]
    recs = {r["id"]: r for r in accepted}
    rows = [{"id": i, "stratum": stratum(recs[i]), "duration": recs[i]["duration"], "audio": f"data/w1/pool_pinned/audio/{i}.wav",
             "current": text_of(recs[i]), "asr_hint": draft.get(i, ""), "action": "", "corrected": "", "minutes": "", "note": ""}
            for i in ids if i in recs]
    old = {r["id"]: r for r in csv.DictReader(open(TSV), delimiter="\t")} if TSV.exists() else {}
    for r in rows:  # keep what the listener already typed
        r.update({k: old[r["id"]][k] for k in ("action", "corrected", "minutes", "note") if r["id"] in old and k in old[r["id"]]})
    with open(TSV, "w", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    per = defaultdict(float)
    for r in rows:
        per[r["stratum"]] += r["duration"]
    print(f"audit sample {len(rows)} of {len(ids)} frozen clips, {sum(per.values()) / 3600:.2f} h:",
          ", ".join(f"{k} {v / 60:.0f} min" for k, v in sorted(per.items())), "->", TSV)


def archive(lines, day, s3=None):
    """Write one patch batch write-once to archive/human/transcribe/<day>-harsh[-n].jsonl (design §2.3). Each line
    names its own key, so the git copy and the archived object are the same bytes. Idempotent: a key already
    holding exactly this batch is reused. -> (key, the lines as stored)."""
    sys.path.insert(0, str(ROOT / "pipeline"))
    import s3 as store
    for n in range(1, 100):
        key = f"archive/human/transcribe/{day}-harsh{'' if n == 1 else f'-{n}'}.jsonl"
        body = "".join(json.dumps({**p, "archive": key}, ensure_ascii=False) + "\n" for p in lines)
        if store.put_once(key, body.encode(), s3) or store.get(key, s3) == body.encode():
            return key, body
    raise SystemExit(f"99 patch batches on {day}?")


def import_(tsv=TSV, records=STAGES / "transcribe/records.jsonl", patches=PATCHES, s3=None, day=None):
    """Filled audit rows -> one patch batch, archived to S3 first, then appended to the git copy."""
    auto = {json.loads(l)["id"]: json.loads(l) for l in open(records)}
    have = {json.loads(l)["id"] for l in open(patches)} if Path(patches).exists() else set()
    new, words, errs = [], 0, 0
    for row in csv.DictReader(open(tsv), delimiter="\t"):
        a = row["action"].strip()
        if not a or row["id"] in have:
            continue
        assert a in ("ok", "fix", "drop"), f"{row['id']}: action {a!r}"
        rec = auto[row["id"]]
        assert "review" not in rec["_w4"], f"{row['id']}: carries patched text; a new patch needs the auto text"
        p = {"id": row["id"], "base": sha(text_of(rec)), "action": a, "reviewer": "harsh",
             "date": (day or date.today().isoformat()), "minutes": float(row["minutes"] or 0), "note": row["note"].strip()}
        ref = strict(text_of(rec))
        words += len(ref)
        if a == "fix":
            t = " ".join(row["corrected"].split())
            assert t and t[-1] in "।?!", f"{row['id']}: corrected text must end with । ? !"
            p["sentences"] = [s.strip() + m for s, m in re.findall(r"([^।?!]+)([।?!])", t)]
            errs += align(ref, strict(t))[0]
        new.append(p)
    if not new:
        print("nothing new to import")
        return None
    key, body = archive(new, day or date.today().isoformat(), s3)
    with open(patches, "a", encoding="utf-8") as f:
        f.write(body)
    print(f"{len(new)} patches -> s3://.../{key} and {patches}; audit WER of the auto text on these clips: "
          f"{100 * errs / max(1, words):.2f}% ({errs}/{words} words). Re-run the transcribe and asr_check stages.")
    return key


if __name__ == "__main__":
    assert near("अठारह", "अठ्ठारह") and near("मिक्रो", "मिक्रोफाइबर") and not near("कलर", "कॉलोनी") and not near("कलर", "और")
    {"build": build, "import": import_}[sys.argv[1]]()
