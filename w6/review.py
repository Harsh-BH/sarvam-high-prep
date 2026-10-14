# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""W6's one review list for Harsh (≤ 30 min, optional), and its import.

    uv run w6/review.py build DIR... --judge2 J [--ab DIR_A DIR_B] [--out data/w6/review.tsv]
    uv run w6/review.py import data/w6/review.tsv [--who harsh]

build: DIR = translate stage outputs (production). Every record's final English is judged a second time by J (another
family than the stage's judge; archived), then three parts are sampled, deterministically:
  A  20 segments a judge marked critical (half flagged by both judges): is the error real? -> judge precision, fixes
  B  15 segments nobody flagged: is anything wrong? -> residual error rate the automatic QA misses
  C  10 blind A/B pairs from two bench engine outputs (--ab): which English is better? -> human preference check
Answer columns: A/B `answer` = y (real error) | n (fine); C `answer` = a | b | = ; `fix` = corrected English, optional.
The A/B side of part C is in review_key.json next to the TSV.

import: fixes -> w6/patches.jsonl (applied by translate.py on the next run, refused if the English changed since)
and archive/human/translate/<date>-<who>.jsonl (write-once); prints the three measurements.
"""
import argparse, csv, json, random, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "w6")]
import qa, s3 as store, translate as tr  # noqa: E402

COLS = ["n", "part", "id", "i", "context_before", "hindi", "english_a", "english_b", "judges_say", "answer", "fix"]


def load(dirs):
    return [json.loads(l) for d in dirs for l in open(Path(d) / "records.jsonl", encoding="utf-8")]


def second_judge(recs, j2):
    def one(r):
        his = [s["hi_text"] for s in r["sentences"]]
        segs = [{"en": s["en_text"], **({"alt": s["en_alt"][0]} if "en_alt" in s else {})} for s in r["sentences"]]
        return tr.judge(j2, his, segs)[0]
    return list(ThreadPoolExecutor(8).map(one, recs))


def claims(v):
    return "; ".join(f"{e['code']}: \"{e.get('en_span', '')}\" -> \"{e.get('fix', '')}\"" for e in v.get("errors", [])
                     if e.get("field", "en") == "en")


def build(a):
    recs = load(a.dirs)
    v2 = second_judge(recs, a.judge2)
    both, one, none = [], [], []
    for r, j2 in zip(recs, v2):
        j1 = r["_qa"]["w6"]["verdict"]
        for i, s in enumerate(r["sentences"]):
            c1 = bool(j1 and tr.JUDGE_CRITICAL & set(j1[i]["en"]))
            c2 = bool(j2 and tr.JUDGE_CRITICAL & set(j2[i]["en"]))
            auto = (qa.CRITICAL | qa.HARD) & set(r["_qa"]["w6"]["auto"][i]["en"])
            row = {"id": r["id"], "i": i, "context_before": r["sentences"][i - 1]["hi_text"] if i else "",
                   "hindi": s["hi_text"], "english_a": s["en_text"], "english_b": "",
                   "judges_say": " | ".join(x for x in (claims(j1[i]) if j1 else "", claims(j2[i]) if j2 else "",
                                                        f"checker: {sorted(auto)}" * bool(auto)) if x)}
            (both if c1 and c2 else one if c1 or c2 or auto else none).append(row)
    rng = random.Random(15)
    nb = min(10, len(both))
    part_a = rng.sample(both, nb) + rng.sample(one, min(20 - nb, len(one)))
    part_b = [{**x, "judges_say": ""} for x in rng.sample(none, min(15, len(none)))]
    part_c, key = [], {}
    if a.ab:
        ea, eb = (load([d]) for d in a.ab)
        pairs = [(r1, i, s1["en_text"], s2["en_text"]) for r1, r2 in zip(ea, eb) for i, (s1, s2) in
                 enumerate(zip(r1["sentences"], r2["sentences"])) if s1["en_text"] != s2["en_text"]
                 and len(set(s1["en_text"].lower().split()) ^ set(s2["en_text"].lower().split())) >= 3]
        for r1, i, t1, t2 in rng.sample(pairs, min(10, len(pairs))):
            flip = rng.random() < .5
            key[f"{r1['id']}:{i}"] = {"a": a.ab[flip], "b": a.ab[not flip]}
            part_c.append({"id": r1["id"], "i": i, "context_before": r1["sentences"][i - 1]["hi_text"] if i else "",
                           "hindi": r1["sentences"][i]["hi_text"], "english_a": t2 if flip else t1,
                           "english_b": t1 if flip else t2, "judges_say": ""})
    out = Path(a.out)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        n = 0
        for part, rows in (("A", part_a), ("B", part_b), ("C", part_c)):
            for x in rows:
                n += 1
                w.writerow({**x, "n": n, "part": part, "answer": "", "fix": ""})
    (out.parent / "review_key.json").write_text(json.dumps(key, indent=1))
    print(f"{out}: A {len(part_a)} (both judges {nb}), B {len(part_b)}, C {len(part_c)}; "
          f"pool: both {len(both)}, one {len(one)}, none {len(none)} segments")


def do_import(a):
    rows = list(csv.DictReader(open(a.tsv, encoding="utf-8"), delimiter="\t"))
    key = json.loads((Path(a.tsv).parent / "review_key.json").read_text())
    patches = [{"id": r["id"], "i": int(r["i"]), "en_text": " ".join(r["fix"].split()), "was": r["english_a"]}
               for r in rows if r["part"] in "AB" and r["fix"].strip()]
    if patches:
        body = "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in patches)
        k = f"archive/human/translate/{date.today()}-{a.who}.jsonl"
        if not store.put_once(k, body.encode()):
            raise SystemExit(f"{k} exists: patches are write-once; import under another --who")
        with open(tr.PATCHES, "a", encoding="utf-8") as f:
            f.write(body)
    ans = lambda p: [r["answer"].strip().lower() for r in rows if r["part"] == p and r["answer"].strip()]
    A, B, C = ans("A"), ans("B"), [(r, r["answer"].strip().lower()) for r in rows if r["part"] == "C" and r["answer"].strip()]
    wins = {}
    for r, x in C:
        k = key[f"{r['id']}:{r['i']}"]
        winner = "tie" if x == "=" else Path(k[x]).name
        wins[winner] = wins.get(winner, 0) + 1
    print(f"A judge precision: {A.count('y')}/{len(A)} flagged segments are real errors")
    print(f"B residual errors: {B.count('y')}/{len(B)} unflagged segments have an error")
    print(f"C preference: {wins}")
    print(f"patches: {len(patches)} -> {tr.PATCHES} (re-run the translate stage to apply)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("dirs", nargs="+")
    b.add_argument("--judge2", required=True, choices=sorted(tr.MODELS))
    b.add_argument("--ab", nargs=2)
    b.add_argument("--out", default="/home/harsh1/github-repos/sarvam-high-prep/data/w6/review.tsv")
    i = sub.add_parser("import")
    i.add_argument("tsv")
    i.add_argument("--who", default="harsh")
    a = ap.parse_args()
    build(a) if a.cmd == "build" else do_import(a)


if __name__ == "__main__":
    main()
