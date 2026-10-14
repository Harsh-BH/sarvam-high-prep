"""Score ASR hypotheses against W4's converted IndicVoices L1 reference (data/w4/pool.jsonl).

    python3 -I w4/asr_score.py [--set ivvalid] [--tag NAME] [MODEL ...]
    default set `bench`: the frozen 1 h benchmark, data/w4/bench/{manifest,ref}.jsonl (rec-000001..174 of W1 pool v1,
    W4 conversion sha1 794f1019), hypotheses data/w4/asr/*.jsonl (vakyansh-w2v2.jsonl = W1's asr_check TSVs, copied). Output scores[-NAME].tsv

Reference = W4 `hi_text` as converted (tags already mapped to fillers / deleted). Clips flagged by w4/convert.py are excluded:
their reference has a known hole. No Whisper normaliser (research T6: it strips matras). Two text forms:
- strict: NFC, punctuation removed. Digits and Latin stay in, and count as errors: §3A forbids them.
- lenient: strict + nukta dropped, chandrabindu -> anusvara, half nasal -> anusvara. Spelling noise only.
Columns: WER/CER (lenient), content WER (fillers removed from both sides), digit/Latin share of hyp words,
filler and repetition retention, English-word and number-word recall (lenient match on an aligned position).
"""
import csv, glob, json, re, sys, unicodedata
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILLERS = {"अ", "अं", "हम्म", "उम्म"}  # R9 fillers W4 writes for [uhh] [hmm] [umm]; आ is also a verb, so left out
_HI = ("सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस "
       "सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस "
       "तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन "
       "उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर "
       "छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे "
       "इक्यानबे बानबे तिरानबे चौरानबे पंचानबे छियानबे सत्तानबे अट्ठानबे निन्यानबे").split()  # 7..99 in order
_EN = {"वन": 1, "टू": 2, "थ्री": 3, "फोर": 4, "फाइव": 5, "सिक्स": 6, "सेवन": 7, "एट": 8, "नाइन": 9, "टेन": 10, "इलेवन": 11,
       "ट्वेल्व": 12, "थर्टीन": 13, "फोर्टीन": 14, "फिफ्टीन": 15, "सिक्सटीन": 16, "सेवनटीन": 17, "सेवंटीन": 17, "एटीन": 18,
       "नाइनटीन": 19, "ट्वेंटी": 20, "थर्टी": 30, "फोर्टी": 40, "फिफ्टी": 50, "सिक्सटी": 60, "सिक्स्टी": 60, "सेवनटी": 70,
       "सेवंटी": 70, "एटी": 80, "एट्टी": 80, "नाइंटी": 90, "हंड्रेड": 100, "थाउजेंड": 1000, "थाउसेंड": 1000}  # in Devanagari
LATIN_OR_DIGIT = re.compile(r"[A-Za-z0-9\u0966-\u096f]")


def strict(t):
    t = unicodedata.normalize("NFC", re.sub("[\u200b-\u200f\ufeff]", "", t))
    return re.sub(r"[^\u0900-\u0963\u0966-\u097fA-Za-z0-9 ]", " ", t).split()


def lenient(w):
    w = unicodedata.normalize("NFD", w).replace("़", "")  # nukta
    w = unicodedata.normalize("NFC", w).replace("ँ", "ं")
    return re.sub(r"[ङञणनम]्(?=[क-ह])", "ं", w)


# word -> value; modifiers and repeaters are their own value. Spelling variants of one number share it (छः/छह/छे/छै)
NUMVAL = {lenient(w): v for w, v in {
    "शून्य": 0, "जीरो": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "छह": 6, "छः": 6, "छे": 6, "छै": 6,
    **{w: 7 + k for k, w in enumerate(_HI)}, "सौ": 100, "हजार": 1e3, "लाख": 1e5, "करोड़": 1e7, "डेढ़": 1.5, "ढाई": 2.5,
    "साढ़े": "साढ़े", "सवा": "सवा", "पौने": "पौने", "डबल": "डबल", "ट्रिपल": "ट्रिपल", **_EN}.items()}
NUMBERS = set(NUMVAL)




def english_words(w1rec):
    """Words IV glosses as English in `normalized` (`साइज़ [size]`). A bracket name that also occurs in the segment's
    `verbatim` is a tag (`[inhaling]` looks like a gloss), not a gloss: verbatim (L1) carries tags but no glosses."""
    out = set()
    for seg in w1rec["_w1"]["segments"]:
        tags = {t.lower() for t in re.findall(r"[\[<]([^\]>]*)[\]>]", seg["verbatim"])}
        out |= {lenient(unicodedata.normalize("NFC", d)) for d, g in re.findall(r"(\S+)\s*\[([A-Za-z][^\]]*)\]", seg["normalized"])
                if g.lower() not in tags}
    return out - FILLERS - {"आ"}  # `अ [a]`: a filler, never an entity


def align(r, h):
    """Word-level Levenshtein; returns (edits, ref index -> hyp index or None)."""
    n, m = len(r), len(h)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (r[i - 1] != h[j - 1]))
    i, j, amap = n, m, {}
    while i and j:
        if d[i][j] == d[i - 1][j - 1] + (r[i - 1] != h[j - 1]):
            amap[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return d[n][m], amap


def edits(a, b):
    return align(list(a), list(b))[0]


def load_hyps(paths):
    hyps = {}
    for p in paths:
        hyps[Path(p).stem] = {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(p)}
    return hyps


def main():
    args = sys.argv[1:]
    s = args[args.index("--set") + 1] if "--set" in args else "bench"
    tag = args[args.index("--tag") + 1] if "--tag" in args else ""
    want = [a for a in args if a not in ("--set", s, "--tag", tag)]
    sub = "" if s == "bench" else f"{s}/"
    ref_f, man_f = ROOT / f"data/w4/{s}/ref.jsonl", ROOT / f"data/w4/{s}/manifest.jsonl"
    hyps = load_hyps(sorted(glob.glob(str(ROOT / f"data/w4/asr/{sub}*.jsonl"))))
    if want:
        hyps = {k: v for k, v in hyps.items() if k in want}
    pool = {json.loads(l)["id"]: json.loads(l) for l in open(ref_f)}
    w1 = {json.loads(l)["id"]: json.loads(l) for l in open(man_f)}
    bench = {i for i in pool if all(i in h for h in hyps.values()) and not pool[i]["_w4"]["flags"]}  # every model has it
    rows = []
    for name, H in hyps.items():
        acc = defaultdict(lambda: defaultdict(float))
        for i in sorted(bench):
            ref_s = strict(" ".join(s["hi_text"] for s in pool[i]["sentences"]))
            hyp_s = strict(H[i])
            ref, hyp = [lenient(w) for w in ref_s], [lenient(w) for w in hyp_s]
            english = english_words(w1[i])
            e, amap = align(ref, hyp)
            hit = lambda k: k in amap and hyp[amap[k]] == ref[k]
            cond = "tel-8k" if w1[i]["_w1"]["orig_sample_rate"] == 8000 else "wide"
            for c in (cond, w1[i]["_w1"]["scenario"].lower(), "all"):
                a = acc[c]
                a["clips"] += 1
                a["ref_w"] += len(ref)
                a["err_w"] += e
                a["ref_c"] += len("".join(ref))
                a["err_c"] += edits("".join(ref), "".join(hyp))
                rc, hc = [w for w in ref if w not in FILLERS], [w for w in hyp if w not in FILLERS]
                a["ref_cw"] += len(rc)
                a["err_cw"] += align(rc, hc)[0]
                a["hyp_w"] += len(hyp)
                a["latdig"] += sum(bool(LATIN_OR_DIGIT.search(w)) for w in hyp_s)
                for k, w in enumerate(ref):
                    rep = k and ref[k - 1] == w and w not in FILLERS
                    for key, test in (("fill", w in FILLERS), ("rep", rep), ("eng", w in english), ("num", w in NUMBERS)):
                        if test:
                            a[key] += 1
                            a[key + "_hit"] += hit(k) and (key != "rep" or hit(k - 1))
                a["hyp_fill"] += sum(w in FILLERS for w in hyp)
        for c, a in acc.items():
            pct = lambda x, y: round(100 * a[x] / a[y], 1) if a[y] else None
            rows.append({"model": name, "cond": c, "clips": int(a["clips"]), "ref_words": int(a["ref_w"]),
                         "WER": pct("err_w", "ref_w"), "CER": pct("err_c", "ref_c"),
                         "content_WER": pct("err_cw", "ref_cw"), "latin_digit_%hyp": pct("latdig", "hyp_w"),
                         "filler_kept_%": pct("fill_hit", "fill"), "fillers_ref": int(a["fill"]),
                         "fillers_hyp": int(a["hyp_fill"]), "repeat_kept_%": pct("rep_hit", "rep"),
                         "repeats_ref": int(a["rep"]), "english_%": pct("eng_hit", "eng"), "english_ref": int(a["eng"]),
                         "number_%": pct("num_hit", "num"), "numbers_ref": int(a["num"])})
    rows.sort(key=lambda r: (r["cond"], r["WER"]))
    out = ROOT / f"data/w4/asr/{sub}scores{'-' + tag if tag else ''}.tsv"
    with open(out, "w") as f:
        w = csv.DictWriter(f, rows[0].keys(), delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    cols = list(rows[0])
    print("\t".join(cols))
    for r in rows:
        print("\t".join(str(r[c]) for c in cols))
    print(f"-> {out}  ({len(bench)} unflagged clips common to all models)")


if __name__ == "__main__":
    assert align("a b c".split(), "a x c d".split())[0] == 2 and lenient("हिन्दी") == lenient("हिंदी") == "हिंदी"
    assert lenient("साइज़") == "साइज" and lenient("हाँ") == "हां"
    assert NUMVAL[lenient("निन्यानबे")] == 99 and NUMVAL["साठ"] == 60 and NUMVAL[lenient("छः")] == NUMVAL["छह"] == 6 and len(_HI) == 93
    main()
