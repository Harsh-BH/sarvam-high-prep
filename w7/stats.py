# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Data-card statistics for a release, from the bucket alone (W7 task: stats for the report; data_card.md).

    uv run w7/stats.py <tag> > stats.md

Reads release/<tag>/ (manifest + frozen stage files, which keep the `_` fields), the gate funnel (each origin run's
run.json `input` chain and rejects.jsonl), and W3's archived QA table. Prints markdown.
"""
import collections, csv, io, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "pipeline")]
import s3 as store, validate  # noqa: E402
from release import merge  # noqa: E402

FILLERS = {"अ", "अं", "उम्म", "हम्म"}  # SPEC R9 canonical; ए and आ are left out (also ordinary words)
BUCKETS = [0, 5, 10, 15, 20, 30, 45, 60]
GLOSS = re.compile(r"\[[A-Za-z][^\]]*\]")  # IndicVoices normalized text: English word -> देवनागरी [latin]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def funnel(origin, s3=None):
    """origin 'stages/<s>/<run>/records.jsonl' -> [(stage run, records, rejects, {rule: n})], newest first."""
    out, run = [], origin.rsplit("/", 1)[0]
    while run:
        meta = json.loads(store.get(f"{run}/run.json", s3))
        rej = (store.get(f"{run}/rejects.jsonl", s3) or b"").decode().splitlines()
        out.append((run, meta.get("records"), meta.get("rejects"), collections.Counter(json.loads(l).get("rule") for l in rej if l.strip())))
        nxt = meta.get("input", "")
        run = nxt.rsplit("/", 1)[0] if nxt.startswith("stages/") else None
    return out


def tts_qa(s3=None):
    """W3's archived QA verdicts -> {engine: (kept, rendered)} for the bake-off set (Devanagari input, default params)."""
    qa = next(k for k in store.keys("archive/w3/qa-", s3))
    ren = next(k for k in store.keys("archive/w3/renders-", s3))
    rend = {r["key"]: r for r in map(json.loads, store.get(ren, s3).decode().splitlines())}
    c = collections.defaultdict(lambda: [0, 0])
    for row in csv.DictReader(io.StringIO(store.get(qa, s3).decode()), delimiter="\t"):
        r = rend[row["key"]]
        if r["variant"] == "dev" and not r["params"] and "+" not in r["pid"]:
            c[r["engine"]][0] += row["verdict"] == "keep"
            c[r["engine"]][1] += 1
    return dict(c), rend


def stats(tag, s3=None):
    man = store.get(f"release/{tag}/manifest.jsonl", s3).decode().splitlines()
    prov = [json.loads(l) for l in store.get(f"release/{tag}/provenance.jsonl", s3).decode().splitlines()]
    work = Path("/tmp") / f"w7stats-{tag}"
    work.mkdir(exist_ok=True)
    files = []
    for k in prov[0]["stage_files"]:
        (p := work / Path(k).name).write_bytes(store.get(k, s3))
        files.append(p)
    full = {r["id"]: r for r in merge(files)}  # with `_` fields
    recs = [json.loads(l) for l in man]
    _, hours = validate.validate(man, work, partial=True)  # hours only; audio isn't here
    syn = lambda r: r["source"]["synthetic"]
    out = [f"# Statistics: release/{tag}", "",
           f"{len(recs)} records; **{hours['total']:.3f} h** = real {hours['real']:.3f} h over {hours['real_files']} source files "
           f"(union of intervals, SPEC §7) + synthetic {hours['synthetic']:.3f} h over {hours['synthetic_files']} files.", ""]

    by = collections.defaultdict(lambda: [0, 0.0])
    for r in recs:
        f = full[r["id"]]
        k = (r["source"]["corpus"], "synthetic" if syn(r) else "real",
             f"{r['source']['model']}" if syn(r) else f"{f.get('_w1', {}).get('scenario', '?')}, {'8 kHz' if f.get('_qa', {}).get('audio_qc', {}).get('telephone') else 'wideband'}")
        by[k][0] += 1
        by[k][1] += r["duration"]
    out += ["## Hours by source", "", table(["corpus", "kind", "scenario / engine", "clips", "hours (sum of durations)"],
                                            [(*k, n, f"{h / 3600:.3f}") for k, (n, h) in sorted(by.items())]), ""]

    hist = collections.Counter()
    for r in recs:
        b = max(i for i, x in enumerate(BUCKETS[:-1]) if r["duration"] > x)
        hist[(b, syn(r))] += 1
    out += ["## Duration histogram (s)", "", table(["bucket", "real", "synthetic"],
            [(f"{BUCKETS[i]}–{BUCKETS[i + 1]}", hist[(i, False)], hist[(i, True)]) for i in range(len(BUCKETS) - 1)]),
            "", f"p50 {q([r['duration'] for r in recs], .5):.1f} s, min {min(r['duration'] for r in recs):.1f} s, max {max(r['duration'] for r in recs):.1f} s.", ""]

    spk, w1 = collections.defaultdict(set), {}
    for r in recs:
        spk[syn(r)].add(r.get("speaker_id"))
        if not syn(r):
            w1[r.get("speaker_id")] = full[r["id"]]["_w1"]
    gender = collections.Counter(x.get("gender") for x in w1.values())
    states = {x.get("state") for x in w1.values()}
    out += ["## Speakers", "", f"Real: {len(spk[False])} speakers ({', '.join(f'{k} {v}' for k, v in sorted(gender.items()))}) "
            f"from {len(states)} states. Synthetic: {len(spk[True])} TTS voices.", ""]

    _, rend = tts_qa(s3)
    ph = collections.defaultdict(collections.Counter)
    for r in recs:
        f, k = full[r["id"]], "synthetic" if syn(r) else "real"
        toks = [w["w"] for w in r["hi_words"]]
        c = ph[k]
        c["words"] += len(toks)
        c["sentences"] += len(r["sentences"])
        c["questions"] += sum(s["hi_text"].endswith("?") for s in r["sentences"])
        c["fillers"] += sum(t in FILLERS for t in toks)
        c["immediate repeats"] += sum(a == b for a, b in zip(toks, toks[1:]))
        if syn(r):
            c["entities (W3 probe tags)"] += len(f["_w3"]["entities"])
            c["English spans (W3 probe tags)"] += len(rend[f["_w3"]["key"]]["en_words"])
        else:
            c["English words (IndicVoices [latin] glosses)"] += sum(len(GLOSS.findall(s.get("normalized", ""))) for s in f["_w1"]["segments"])
        c["en_alt present"] += sum("en_alt" in s for s in r["sentences"])
    keys = sorted({k for c in ph.values() for k in c}, key=lambda k: (k != "words", k))
    out += ["## Text phenomena", "", table(["count", "real", "synthetic"], [(k, ph["real"][k], ph["synthetic"][k]) for k in keys]), "",
            f"Code-mix rate (English words / all words): real {ph['real']['English words (IndicVoices [latin] glosses)'] / max(1, ph['real']['words']):.1%} "
            "(transcriber-marked, a lower bound); synthetic: see English spans.", ""]

    kept, _ = tts_qa(s3)
    out += ["## TTS QA (W3 bake-off: Devanagari input, default params)", "",
            table(["engine", "rendered", "kept", "reject rate"], [(e, n, k, f"{1 - k / n:.1%}") for e, (k, n) in sorted(kept.items())]), ""]

    reg = collections.defaultdict(list)
    crit = collections.Counter()
    for r in recs:
        f = full[r["id"]]
        reg[syn(r)].append(f["_qa"]["align"]["regret"])
        crit[syn(r)] += f["_qa"].get("w6", {}).get("critical", 0) > 0
    out += ["## Aligner and translation QA on the released records", "",
            table(["", "real", "synthetic"], [
                ("aligner regret p50 / p90", f"{q(reg[False], .5):.3f} / {q(reg[False], .9):.3f}", f"{q(reg[True], .5):.3f} / {q(reg[True], .9):.3f}"),
                ("records with ≥ 1 critical flag after repair (judge or automatic check; `_qa.w6.critical` > 0)", crit[False], crit[True])]), "",
            "translation_method: " + ", ".join(f"`{k}` {v}" for k, v in collections.Counter(r["translation_method"] for r in recs).most_common()), ""]

    out += ["## Gate funnel (each run, newest first)", ""]
    for o in prov[0]["origins"]:
        out += [table(["run", "records out", "rejects", "rules"], [(run, n, rj, ", ".join(f"{k} {v}" for k, v in rules.most_common()) or "–")
                                                                 for run, n, rj, rules in funnel(o, s3)]), ""]
    return "\n".join(out)


if __name__ == "__main__":
    print(stats(sys.argv[1]))
