"""W2 reports, as markdown on stdout (pasted into tasks/W2-scripts.md):

    uv run --no-project python w2/report.py bake       # generator bake-off: constraints, naturalness vs IndicVoices, cost
    uv run --no-project python w2/report.py coverage   # data/w2/scripts.jsonl cell counts vs the coverage-matrix targets

Naturalness proxies are computed on hi_text the same way for scripts and for real IndicVoices transcripts
(data/w4/pool.jsonl, W4's SPEC §2 conversion of IV Level 1). IV doesn't mark English words, so English share uses a
lexicon of words the generators marked as English in at least 2 scripts: a lower bound, the same for both sides.
"""
import json
import statistics as st
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen  # noqa: E402
from banks import nfc, FRAC, NUMWORDS as NUMISH  # noqa: E402

DATA = gen.DATA
IV = DATA.parent / "w4/pool.jsonl"
FILL = {"अ", "अं", "ए", "उम्म", "हम्म"}
DISC = set(nfc("मतलब यानी अच्छा देखिए देखो बस एक्चुअली लाइक वैसे अरे हाँ यार").split())
WRITTEN = set(nfc("एवं तथा हेतु अतः उक्त किंतु परंतु यदि अथवा कृपया द्वारा अवश्य प्रदान सूचित निवेदन अनुरोध संपर्क").split())
STOP = set(nfc("बस ना तो का की के है में से को और पे भी ही").split())
LICENCE = {"qwen3-235b": "Apache-2.0", "gpt-oss-120b": "Apache-2.0", "mistral-large-3": "Apache-2.0", "deepseek-v3.2": "MIT"}


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")] if Path(p).exists() else []


def toks(text):
    return text.replace("।", " ").replace("?", " ").replace("!", " ").split()


def lexicon(recs):
    seen = Counter()
    for r in recs:
        seen.update({w for e in r.get("en_words", []) for w in e.split()})
    return {w for w, c in seen.items() if c >= 2} - STOP - NUMISH


def style(texts, lex):
    """texts: list of (hi_text, sentences). -> naturalness proxies, per 100 words unless noted."""
    W = [toks(t) for t, _ in texts]
    n = sum(map(len, W)) or 1
    rep = sum(a == b and a not in NUMISH for w in W for a, b in zip(w, w[1:]))
    sl = [len(toks(s)) for _, ss in texts for s in ss]
    grams = [tuple(w[i:i + 2]) for w in W for i in range(len(w) - 1)]
    return {"texts": len(texts), "words/text": st.mean(map(len, W)), "words/sent": st.mean(sl),
            "words/sent p90": sorted(sl)[9 * len(sl) // 10], "fillers": 100 * sum(x in FILL for w in W for x in w) / n,
            "discourse": 100 * sum(x in DISC for w in W for x in w) / n, "repeats": 100 * rep / n,
            "written": 100 * sum(x in WRITTEN for w in W for x in w) / n,
            "English (lexicon) %": 100 * sum(x in lex for w in W for x in w) / n,
            "distinct-2": len(set(grams)) / max(1, len(grams))}


def jacc(a, b):
    return len(a & b) / max(1, len(a | b))


def tri(t):
    w = toks(t)
    return {tuple(w[i:i + 3]) for i in range(len(w) - 2)}


def diversity(recs):
    T = [tri(r["hi_text"]) for r in recs]
    shots = [tri(gen.probe.parse(f"s\td\t-\t{gen.TYPED.sub(r'[\1]', a[8:-9])}")["hi_text"]) for _, a in gen.SHOTS]
    pairs = [jacc(a, b) for a, b in combinations(T, 2)]
    openers = Counter(" ".join(toks(r["hi_text"])[:2]) for r in recs)
    return {"max pair 3-gram Jaccard": max(pairs, default=0), "pairs > 0.2": sum(p > 0.2 for p in pairs),
            "max overlap with few-shot": max((jacc(t, s) for t in T for s in shots), default=0),
            "top opener share": openers.most_common(1)[0][1] / len(recs) if recs else 0}


def unmarked(recs, lex):
    """Share of lexicon (English) tokens that the script left outside {..|..} markup."""
    tot = mk = 0
    for r in recs:
        tot += sum(x in lex for x in toks(r["hi_text"]))
        mk += sum(x in lex for e in r["en_words"] for x in e.split())
    return 100 * max(0, tot - mk) / max(1, tot)


def table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{x:.3g}" if isinstance(x, float) else str(x) for x in r) + " |")
    return "\n".join(out)


def iv_texts():
    return [(" ".join(s["hi_text"] for s in r["sentences"]), [s["hi_text"] for s in r["sentences"]]) for r in load(IV)]


def bake():
    runs = {p.stem: load(p) for p in sorted((DATA / "bake").glob("*.jsonl"))}
    allrecs = [r for rs in runs.values() for r in rs if "hi_text" in r]
    lex = lexicon(allrecs)
    rules = ["format", "spec_text", "entity_missing", "entity_type", "correction", "gender", "filler", "length", "intent", "disfluency"]
    rows = []
    for name, rs in runs.items():
        n = len(rs)
        f = Counter(x for r in rs for x in r["fails"])
        lo_hi = [gen.LENGTH[r["cell"]["length"]][1] for r in rs if "words" in r]
        inband = sum(lo <= r["words"] <= hi for r, (lo, hi) in zip([r for r in rs if "words" in r], lo_hi))
        usd = sum(r["_gen"]["usd"] for r in rs)
        ms = st.median(t["ms"] for r in rs for t in r["_gen"]["tries"])
        rows.append([name, LICENCE[rs[0]["model"]], n, f"{100 * sum(not r['fails'] for r in rs) / n:.0f}%"]
                    + [f.get(x, 0) for x in rules] + [f"{100 * inband / max(1, len(lo_hi)):.0f}%", f"{usd / n * 1000:.2f}", f"{ms / 1000:.1f}"])
    print("### Constraint adherence (first take only)\n")
    print(table(rows, ["model", "licence", "n", "pass"] + rules + ["words in band", "USD per 1k scripts", "latency s p50"]))
    print("\n### Naturalness proxies vs real IndicVoices (per 100 words unless noted)\n")
    ivs = style(iv_texts(), lex)
    rows = [["IndicVoices L1 (W4 pool)"] + list(ivs.values()) + ["—", "—"]]
    for name, rs in runs.items():
        ok = [r for r in rs if "hi_text" in r]
        s = style([(r["hi_text"], r["sentences"]) for r in ok], lex)
        rows.append([name] + list(s.values()) + [unmarked(ok, lex), f"{100 * sum(len(' '.join(r['en_words']).split()) for r in ok) / max(1, sum(r['words'] for r in ok)):.1f}"])
    print(table(rows, ["text"] + list(ivs) + ["unmarked English %", "marked English %"]))
    print("\n### Diversity (scripts that parsed)\n")
    rows = [[name] + list(diversity([r for r in rs if "hi_text" in r]).values()) for name, rs in runs.items()]
    print(table(rows, ["model", "max pair 3-gram Jaccard", "pairs > 0.2", "max overlap with few-shot", "top opener share"]))
    print(f"\nEnglish lexicon: {len(lex)} words marked English in ≥ 2 scripts across all models.")


def coverage():
    recs = load(DATA / "scripts.jsonl")
    rej = load(DATA / "rejects.jsonl")
    n_all = len(recs) + len(rej)
    cells = gen.design(n_all, "pilot-v1")
    tgt = lambda key: Counter(c[key] for c in cells)
    got = lambda key: Counter(r["cell"][key] for r in recs)
    print(f"**{len(recs)} kept of {n_all} designed** ({len(rej)} rejected after all takes: "
          f"{dict(Counter(x for r in rej for x in r['fails']))}).\n")
    for key, title in (("domain", "Domain"), ("intent", "Intent phenomenon (requested)"), ("corr", "Self-correction"),
                       ("marker", "Correction marker"), ("mix", "Code-mix level"), ("length", "Length"), ("gender", "Speaker gender"),
                       ("filler", "Filler asked"), ("repeat", "Repetition asked")):
        t, g = tgt(key), got(key)
        print(f"#### {title}\n")
        print(table([[k if k is not None else "—", t[k], g[k]] for k in t], ["cell", "target", "kept"]) + "\n")
    print("#### Domain × primary number form (kept / target)\n")
    t = Counter((c["domain"], c["numform"]) for c in cells)
    g = Counter((r["cell"]["domain"], r["cell"]["numform"]) for r in recs)
    rows = [[d] + [f"{g[d, f]}/{t[d, f]}" if t[d, f] else "·" for f in gen.FORMS] for d in gen.DOMAINS]
    rows.append(["**total**"] + [f"{sum(g[d, f] for d in gen.DOMAINS)}/{sum(t[d, f] for d in gen.DOMAINS)}" for f in gen.FORMS])
    print(table(rows, ["domain"] + gen.FORMS) + "\n")
    print("#### Realised content (scripts containing it, any entity, measured on the text)\n")
    tags = Counter(x for r in recs for x in r["tags"])
    etoks = lambda r: [w for e in r["entities"] for w in e.split()]
    extra = {f"फ़्रैक्शन {w}": sum(w in etoks(r) for r in recs) for w in FRAC}
    extra["डबल/ट्रिपल"] = sum(any(x in ("डबल", "ट्रिपल") for x in etoks(r)) for r in recs)
    rows = [[k, v] for k, v in sorted(tags.items()) if not k.startswith("i:")] + [[k, v] for k, v in extra.items()]
    print(table(rows, ["content tag", "scripts"]) + "\n")
    print("#### Intent phenomena: requested vs detected in the text\n")
    rows = []
    for k in gen.INTENTS:
        req = [r for r in recs if r["cell"]["intent"] == k]
        rows.append([k, tgt("intent")[k], len(req), sum(k in r["detected"] for r in req) if k in gen.DETECT else "n/a (needs a reader)",
                     sum(k in r["detected"] for r in recs) if k in gen.DETECT else "n/a"])
    print(table(rows, ["phenomenon", "target", "kept (requested)", "requested and detected", "detected in any script"]) + "\n")
    print("#### Entities\n")
    ty = Counter(s["type"] for r in recs for s in r["entity_spans"])
    giv = sum(s["given"] for r in recs for s in r["entity_spans"])
    tot = sum(len(r["entity_spans"]) for r in recs)
    print(f"{tot} entity spans in {len(recs)} scripts ({tot / max(1, len(recs)):.1f} per script); {giv} pre-verbalised by "
          f"banks.py (value known), {tot - giv} added by the LLM (value unknown); "
          f"{sum(s['retracted'] for r in recs for s in r['entity_spans'])} retracted (self-corrected away).\n")
    print(table([[k, v] for k, v in ty.most_common()], ["type", "spans"]) + "\n")
    print("#### Length and code-mix, realised\n")
    rows = []
    for k, (_, (lo, hi)) in gen.LENGTH.items():
        rs = sorted(r["est_sec"] for r in recs if r["cell"]["length"] == k)
        ws = [r["words"] for r in recs if r["cell"]["length"] == k]
        if rs:
            rows.append([k, f"{lo}–{hi} words", len(rs), f"{rs[len(rs) // 10]}/{rs[len(rs) // 2]}/{rs[9 * len(rs) // 10]}",
                         f"{100 * sum(lo <= w <= hi for w in ws) / len(ws):.0f}%"])
    print(table(rows, ["length", "asked", "kept", "est. seconds p10/p50/p90", "words in band"]) + "\n")
    rows = []
    for k in gen.MIX:
        sh = sorted(100 * len(" ".join(r["en_words"]).split()) / r["words"] for r in recs if r["cell"]["mix"] == k)
        if sh:
            rows.append([k, len(sh), f"{sh[len(sh) // 10]:.0f}/{sh[len(sh) // 2]:.0f}/{sh[9 * len(sh) // 10]:.0f}"])
    print(table(rows, ["code-mix", "kept", "marked English % of words p10/p50/p90"]) + "\n")
    hours = sum(r["est_sec"] for r in recs) / 3600
    gm = sum(sum(r["gender_marks"]) > 0 for r in recs)
    print(f"Estimated audio: {hours:.2f} h at one render per script ({gen.S_PER_CHAR} s/char, W3's median). "
          f"{gm}/{len(recs)} scripts carry a first-person gender marker the voice must match.\n")
    lex = lexicon(recs)
    print("#### Naturalness vs IndicVoices, and diversity\n")
    s, ivs = style([(r["hi_text"], r["sentences"]) for r in recs], lex), style(iv_texts(), lex)
    print(table([["IndicVoices L1"] + list(ivs.values()), ["W2 pilot"] + list(s.values())], ["text"] + list(s)) + "\n")
    d = diversity(recs)
    print(table([list(d.values()) + [f"{unmarked(recs, lex):.1f}"]], list(d) + ["unmarked English %"]))


def review():
    """data/w2/review.tsv: Harsh's one optional list (≈15 min). A: every Hindi number word and fraction pattern the
    verbaliser can emit (a typo here would be in every script). B: 10 pilot scripts and 10 IndicVoices transcripts
    of similar length, shuffled; mark each real/generated and natural 1-3."""
    import random
    import banks
    rows = [("A", f"{i}", w, "spelling ok? (y/fix)") for i, w in enumerate(banks.H)]
    r = random.Random(0)
    for f in ("lakh_crore", "frac", "small", "time", "date", "idcode"):
        for i in range(3):
            e = banks.NUMFORMS[f](random.Random(f"review{f}{i}"), r.choice(banks.DOMAINS))
            rows.append(("A", e.value, gen.dev(e.text), "says this value? (y/fix)"))
    recs = load(DATA / "scripts.jsonl")
    ivs = [t for t, _ in iv_texts() if 45 <= len(t.split()) <= 90]
    blind = [("gen", x["hi_text"]) for x in r.sample(recs, 10)] + [("iv", t) for t in r.sample(ivs, 10)]
    r.shuffle(blind)
    rows += [("B", f"b{i + 1:02d}", t, "real or generated? natural 1-3?") for i, (_, t) in enumerate(blind)]
    out = DATA / "review.tsv"
    out.write_text("section\tkey\ttext\task\tanswer\n" + "".join("\t".join(x) + "\t\n" for x in rows), encoding="utf-8")
    (DATA / "review_key.tsv").write_text("".join(f"b{i + 1:02d}\t{s}\n" for i, (s, _) in enumerate(blind)))
    print(f"{out}: {len(rows)} rows; answers for B in review_key.tsv")


if __name__ == "__main__":
    assert toks("मैं मैं ठीक हूँ।") == ["मैं", "मैं", "ठीक", "हूँ"]
    assert style([("मैं मैं उम्म ठीक हूँ।", ["मैं मैं उम्म ठीक हूँ।"])], set())["repeats"] == 20.0
    {"bake": bake, "coverage": coverage, "review": review}[sys.argv[1]]()
