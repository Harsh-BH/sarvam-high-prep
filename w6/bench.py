# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""W6 engine benchmark on the pilot (tasks/W6-translation.md "Engine benchmark").

    uv run w6/bench.py report DIR...            # DIR = a translate.py --out dir named <engine>.<judge>; prints markdown
    uv run w6/bench.py inject DIR --judge J     # judge recall on known injected errors (synthetic controls)
    uv run w6/bench.py agree DIR DIR            # same engine, two judges: agreement on critical segments

`report`: per engine x judge: segments, judge critical-error rate (+ per code), automatic entity flags (number value,
unit, digits), negation and length flags, loanword restoration rate (IndicVoices L2 glosses, real clips only).
`inject`: takes segments of DIR that are clean (no judge error, no automatic flag), corrupts exactly one segment per
record (number, negation, unit, name, omission, abandoned value stated as fact), re-judges the record and reports, per
corruption type, how often the judge flags the corrupted segment (any critical code) and how often the automatic
checks catch it; false positives = critical codes on the untouched segments.
"""
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "w6")]
import qa, translate as tr  # noqa: E402

AUTO_ENTITY = {"num_extra", "num_missing", "unit"}


def load(d):
    return [json.loads(l) for l in open(Path(d) / "records.jsonl", encoding="utf-8")]


def segs(recs):
    """-> one row per segment: synthetic?, auto flags (en), judge codes (en), glosses expected / restored."""
    for r in recs:
        w = r["_qa"]["w6"]
        gl = qa.glosses(r.get("_w1", {}).get("segments", []))
        for i, s in enumerate(r["sentences"]):
            g = qa.sentence_glosses(s["hi_text"], gl)
            yield {"id": r["id"], "i": i, "syn": r["source"]["synthetic"], "auto": w["auto"][i]["en"],
                   "judge": (w["verdict"] or [{}] * 999)[i],
                   "gl": len(g), "gl_ok": len(g) - len(w["auto"][i]["en"].get("loanword", [])), "alt": "en_alt" in s}


def pct(a, b):
    return f"{100 * a / b:.1f}" if b else "-"


def report(dirs):
    print("| engine | judge | set | segs | judge critical % | any judge error % | auto entity % | digits % | neg flag % | "
          "length flag % | loanwords restored % | alt segs | judge errors dropped | top judge codes |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs:
        eng, jud = Path(d).name.split(".")
        rows = list(segs(load(d)))
        for name, sub in (("all", rows), ("real", [x for x in rows if not x["syn"]]), ("synthetic", [x for x in rows if x["syn"]])):
            n = len(sub)
            crit = sum(1 for x in sub if tr.JUDGE_CRITICAL & set(x["judge"].get("en", [])))
            anyj = sum(1 for x in sub if x["judge"].get("en"))
            codes = Counter(c for x in sub for c in x["judge"].get("en", []))
            ent = sum(1 for x in sub if AUTO_ENTITY & set(x["auto"]))
            f = lambda k: sum(1 for x in sub if k in x["auto"])
            gl, glok = sum(x["gl"] for x in sub), sum(x["gl_ok"] for x in sub)
            print(f"| {eng} | {jud} | {name} | {n} | {pct(crit, n)} | {pct(anyj, n)} | {pct(ent, n)} | {pct(f('digits'), n)} | "
                  f"{pct(f('neg'), n)} | {pct(f('length'), n)} | {pct(glok, gl)} ({gl}) | {sum(x['alt'] for x in sub)} | "
                  f"{sum(x['judge'].get('dropped', 0) for x in sub)} | "
                  f"{', '.join(f'{c} {k}' for c, k in ((c, k) for c, k in codes.most_common(4)))} |")


SMALL = {v: k for k, v in qa.EN_SMALL.items() if 2 <= v <= 9 and not k.endswith(("th", "st", "nd", "rd"))}


def corrupt(kind, en, alt=None):
    """-> corrupted text or None if this kind doesn't apply."""
    if kind == "number":
        m = re.search(r"\b(two|three|four|five|six|seven|eight)\b", en, re.I)
        return m and en[:m.start()] + SMALL[qa.EN_SMALL[m.group(1).lower()] + 1] + en[m.end():]
    if kind == "negation":
        for pat, rep in ((r"\b(do|does|did|is|was|are|were|could|would|should|has|have|had)n't\b", r"\1"),
                         (r"\b(?:don't|won't|can't)\b", None), (r"\bnot\s+", ""), (r"\bnever\b", "always")):
            m = re.search(pat, en, re.I)
            if m and rep is None:
                return en[:m.start()] + {"don't": "do", "won't": "will", "can't": "can"}[m.group(0).lower()] + en[m.end():]
            if m:
                return en[:m.start()] + m.expand(rep) + en[m.end():]
        return None
    if kind == "unit":
        return re.sub(r"\blakh\b", "million", en, count=1) if re.search(r"\blakh\b", en) else None
    if kind == "name":
        m = re.search(r"(?<=\w )([A-Z][a-z]{3,})\b", en)
        return m and en[:m.start()] + ("Ramesh" if m.group(1) != "Ramesh" else "Suresh") + en[m.end():]
    if kind == "omission":
        w = en.split()
        cut = en.rfind(", ")
        return en[:cut] + en[-1] if len(w) >= 10 and cut > len(en) / 3 else None
    if kind == "correction":
        m = alt and re.search(r",\s*(?:no|sorry|I mean|wait)\b", alt, re.I)
        return m and alt[:m.start()].rstrip(",") + "."
    raise ValueError(kind)


KINDS = ["number", "negation", "unit", "name", "omission", "correction"]


def inject(d, judge, per_kind):
    recs = load(d)
    jobs = []
    for kind in KINDS:
        n = 0
        for r in recs:
            w = r["_qa"]["w6"]
            if not w["verdict"]:
                continue
            for i, s in enumerate(r["sentences"]):
                clean = not w["verdict"][i]["en"] and not w["auto"][i]["en"]
                bad = clean and corrupt(kind, s["en_text"], (s.get("en_alt") or [None])[0])
                if bad and n < per_kind:
                    jobs.append((kind, r, i, bad))
                    n += 1
                    break
    res = defaultdict(lambda: [0, 0, 0, 0, 0])  # n, judge caught, auto caught, fp segments, other segments
    from concurrent.futures import ThreadPoolExecutor

    def run(job):
        kind, r, i, bad = job
        his = [s["hi_text"] for s in r["sentences"]]
        sg = [{"en": s["en_text"], **({"alt": s["en_alt"][0]} if "en_alt" in s else {})} for s in r["sentences"]]
        sg[i] = {"en": bad}
        v, _ = tr.judge(judge, his, sg)
        return kind, i, sg, his, v

    for kind, i, sg, his, v in ThreadPoolExecutor(6).map(run, jobs):
        x = res[kind]
        x[0] += 1
        x[1] += bool(v and tr.JUDGE_CRITICAL & set(v[i]["en"]))
        x[2] += bool((qa.CRITICAL | {"neg"}) & set(qa.check(his[i], sg[i]["en"])))
        if v:
            x[3] += sum(1 for k, o in enumerate(v) if k != i and tr.JUDGE_CRITICAL & set(o["en"]))
            x[4] += len(v) - 1
    print(f"| corruption | n | judge {judge} recall % | auto recall % | judge false-positive % (other segs) |")
    print("|---|---|---|---|---|")
    for kind in KINDS:
        n, j, a, fp, o = res[kind]
        print(f"| {kind} | {n} | {pct(j, n)} | {pct(a, n)} | {pct(fp, o)} |")


def agree(d1, d2):
    """Two judges on the same translations: P(both critical | either), Cohen's kappa on the critical flag."""
    crit = lambda x: bool(tr.JUDGE_CRITICAL & set(x["judge"].get("en", [])))
    a = {(x["id"], x["i"]): crit(x) for x in segs(load(d1))}
    b = {(x["id"], x["i"]): crit(x) for x in segs(load(d2))}
    k = a.keys() & b.keys()
    n, both, either = len(k), sum(a[x] and b[x] for x in k), sum(a[x] or b[x] for x in k)
    pa, pb = sum(a[x] for x in k) / n, sum(b[x] for x in k) / n
    po = sum(a[x] == b[x] for x in k) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    print(f"| {Path(d1).name} vs {Path(d2).name} | {n} | {100 * pa:.1f} | {100 * pb:.1f} | {pct(both, either)} | "
          f"{(po - pe) / (1 - pe):.2f} |")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("dirs", nargs="+")
    i = sub.add_parser("inject")
    i.add_argument("dir")
    i.add_argument("--judge", required=True, choices=sorted(tr.MODELS))
    i.add_argument("--per-kind", type=int, default=25)
    g = sub.add_parser("agree")
    g.add_argument("dirs", nargs=2)
    a = ap.parse_args()
    if a.cmd == "report":
        report(a.dirs)
    elif a.cmd == "agree":
        agree(*a.dirs)
    else:
        inject(a.dir, a.judge, a.per_kind)


if __name__ == "__main__":
    main()
