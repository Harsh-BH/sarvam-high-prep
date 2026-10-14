# /// script
# requires-python = ">=3.12"
# dependencies = ["punctuators==0.0.7", "huggingface_hub"]
# ///
"""`transcribe` pipeline stage (design C6): IndicVoices L1 `verbatim` -> SPEC §2 `sentences[].hi_text`.

    uv run w4/convert.py --in RECORDS.jsonl --audio-dir DIR --out OUT_DIR [--no-sbd] [--patches w4/patches.jsonl]
    # --no-sbd: one sentence per hard chunk (tests; no 1.1 GB punctuation model)

Stage contract (docs/superpowers/plans/2026-10-14-s3-pipeline.md): OUT_DIR/records.jsonl = input records + `sentences`
+ `_w4`; OUT_DIR/rejects.jsonl = {"id","stage","rule","detail"} per excluded record; ids already in either file are
skipped, so a killed run resumes. Input records need `_w1.segments` (W1's cut: `verbatim` L1 + `normalized` L2 per
segment). `--audio-dir` is accepted for the contract; this stage reads no audio. HF weights: $HF_HUB_CACHE.

Per SPEC §2.4: tags -> R9 fillers or deleted, ZWJ/ZWNJ out, NFC (decomposes nukta letters, R2), `:` -> visarga,
dotted and glued letter-by-letter acronyms -> spaced letter names (R8), hyphens -> word break (§2.2 echo words),
R12 nasal spelling, then sentence split + final mark (R3). Source `।` `?` `.` are kept as hard boundaries; inside
those, 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase (Apache-2.0, ONNX, CPU) places boundaries and picks
`।` / `?`. Its commas are dropped (SPEC §2.2).

Never guesses what needs ears: `[stammers]`, `[uh-huh]`, `[ugh]` would leave a heard fragment untranscribed, so the
clip is rejected (rule `ear_tag`, design §3.3 P0; W1 recuts around the tag), as are second-voice/unintelligible tags
(`second_voice`), Latin/digits (`chars`), and empty text. Audit patches (w4/patches.jsonl, the git copy of
archive/human/transcribe/) are applied here; a patch made against different auto text is refused (`stale_patch`).
"""
import hashlib, json, re, sys, unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PUNCT_MODEL = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
PUNCT_REV = "d1769a597ce8dfaa070d436bc67d4ee761f58884"  # pinned: the sentence split is part of the rebuild (§3A)
PATCHES = ROOT / "w4/patches.jsonl"  # audit decisions, append-only, last line per id wins (w4/review_list.py import)

FILLER = {"hmm": "हम्म", "umm": "उम्म", "uhh": "अ", "ahh": "आ"}  # R9; [uhh] -> अ, nasal अं only by ear
EARS = {"stammers", "uh_huh", "ugh"}  # SPEC §2.4: listen and write what was said
DROP = {"unintelligible", "talking", "children_talking", "whisper", "yelling", "singing"}  # recut or drop
# a tag is Latin inside [..] or <..>; IV also has broken ones: `[Inhaling>`, `rustling>`, a trailing `<Persistent-noise`
TAG = re.compile(r"[\[<]\s*([A-Za-z][A-Za-z_\- ]*?)\s*[\]>]|\b([A-Za-z][A-Za-z_\-]*)>|<([A-Za-z][A-Za-z_\-]*)\s*$")
GLOSS = re.compile(r"(\S+)\s*\[([A-Z][A-Z. ]+)\]")  # L2 `normalized`: `आईडी [ID]`
LETTERS = "ए बी सी डी ई एफ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड जेड".split()  # R8
LETTER_RE = re.compile("(?:" + "|".join(sorted(map(re.escape, LETTERS), key=len, reverse=True)) + ")")
DEV = r"\u0900-\u097f"
OK_CHAR = re.compile(r"[\u0900-\u0963\u0971-\u097f]")  # SPEC R1 (validate.py HI_CHAR)
MARKS = "।?!"
STOP = "\x00"  # internal hard sentence boundary, followed by its mark


def nasal(w):
    """R12: nasal consonant + stop -> anusvara; word-final anusvara after ा/ू -> chandrabindu; never ँ over a matra."""
    w = re.sub(r"ङ्(?=[क-घ])|ञ्(?=[च-झ])|ण्(?=[ट-ढ])|न्(?=[क-घच-झट-ढत-ध])|म्(?=[प-भ])", "ं", w)
    w = re.sub(r"(?<=[ाू])ं$", "ँ", w)
    return re.sub(r"(?<=[िीेैोौ])ँ", "ं", w)


def spell_acronyms(words, glossed):
    """Glued letter names (`आईडी [ID]`) -> spaced (R8), only where the L2 gloss confirms an all-caps acronym."""
    out = []
    for w in words:
        parts = LETTER_RE.findall(w)
        if w in glossed and len(parts) == len(glossed[w]) > 1 and "".join(parts) == w:
            out += parts
        else:
            out.append(w)
    return out


def clean(text, flags, nwords):
    """One segment of IV verbatim -> words and STOP markers. Tag flags carry the index of the next word."""
    text = re.sub("[\u200b-\u200f\ufeff]", "", text).replace("\u00a0", " ")
    text = unicodedata.normalize("NFC", text)

    def tag(m):
        name = re.sub(r"[\s-]+", "_", (m.group(1) or m.group(2) or m.group(3)).lower().strip())
        return f" \x01{name}\x01 "
    text = TAG.sub(tag, text)
    text = re.sub(rf"(?<=[{DEV}]):", "ः", text)  # `छ:` = छः
    text = re.sub(rf"((?:[{DEV}]+\.){{2,}})", lambda m: " " + m.group(1).replace(".", " ") + " ", text)  # पी.आर.ए.एन.
    text = re.sub(r"[.|।]", f" {STOP} । ", text)
    text = re.sub(r"([?!])", rf" {STOP} \1 ", text)
    text = re.sub(r"[-,;\"'()\n\t]", " ", text)
    words = []
    for tok in text.split():
        if tok.startswith("\x01"):
            name = tok.strip("\x01")
            if name in FILLER:
                words.append(FILLER[name])
            elif name in EARS:
                flags.append(f"{name}@{nwords + len(words)}")
            elif name in DROP:
                flags.append(f"drop:{name}")
            continue
        bad = {c for c in tok if c not in MARKS and c != STOP and not OK_CHAR.match(c)}
        if bad:  # Latin, digits, symbols: none in the IV pool; never silently dropped
            flags.append(f"chars:{''.join(sorted(bad))}@{nwords + len(words)}")
            tok = "".join(c for c in tok if c not in bad)
        if tok:
            words.append(tok)
    return words


def split(chunks, model):
    """chunks: word lists between hard boundaries -> per chunk [(words, model mark or None, flag)].
    The model sometimes re-spaces a word (`इस्कूल` -> `इ स्कूल`), so its sentence ends are mapped back by character
    offset and land on the end of the word that contains them. Words always come from the input."""
    outs = model.infer([" ".join(c) for c in chunks], apply_sbd=True) if model else [[" ".join(c)] for c in chunks]
    res = []
    for words, out in zip(chunks, outs):
        got = [re.sub(r"[^\u0900-\u0963\u0966-\u097f ]", "", s).replace(" ", "") for s in out]  # model punct out
        if "".join(got) != "".join(words):  # model changed characters: keep the chunk whole rather than trust it
            res.append([(words, None, "sbd_mismatch")])
            continue
        ends, n = [], 0
        for w in words:
            n += len(w)
            ends.append(n)
        parts, k, c = [], 0, 0
        for g, s in zip(got, out):
            c += len(g)
            if not g:
                continue
            j = next(i for i, e in enumerate(ends) if e >= c)
            if j >= k:
                parts.append((words[k:j + 1], {"?": "?", "!": "!"}.get(s.rstrip()[-1]), None))
                k = j + 1
        res.append(parts)
    return res


def merge_short(sents):
    """A `।` sentence with <= 1 non-filler word (`अ।`, `जिसका।`) is no translation unit: join it to the next
    sentence, or to the previous one when it is last. Words and order never change, so W5 timings are unaffected."""
    out, carry = [], []
    for ws, m in sents:
        ws = carry + ws
        if m == "।" and sum(w not in FILLER.values() for w in ws) <= 1:
            carry = ws
            continue
        out.append((ws, m))
        carry = []
    if carry:
        if out:
            out[-1] = (out[-1][0] + carry, out[-1][1] if out[-1][1] == "?" else "।")
        else:
            out.append((carry, "।"))
    return out


def convert(rec, model):
    flags, words, chunks, marks = [], [], [], []
    glossed = {}
    for seg in rec["_w1"]["segments"]:
        glossed.update({d: g.replace(".", "").replace(" ", "")
                        for d, g in GLOSS.findall(unicodedata.normalize("NFC", seg["normalized"]))})
        words += clean(seg["verbatim"], flags, len([w for w in words if w != STOP and w not in MARKS]))
    # words -> chunks at STOP markers
    cur = []
    for i, w in enumerate(words):
        if w == STOP:
            continue
        if w in MARKS and i and words[i - 1] == STOP:
            if cur:
                chunks.append(cur)
                marks.append(w)
            cur = []
            continue
        cur.append(w)
    if cur:
        chunks.append(cur)
        marks.append(None)
    chunks = [spell_acronyms([nasal(w) for w in c], glossed) for c in chunks]
    return chunks, marks, flags


def rule(flags):
    """First converter flag -> reject rule."""
    kind = flags[0].split("@")[0].split(":")
    return {"stammers": "ear_tag", "uh_huh": "ear_tag", "ugh": "ear_tag", "chars": "chars", "stale_patch": "stale_patch"}.get(
        kind[0], {"empty": "empty", "drop": "audit_drop", "recut": "audit_recut"}.get(kind[-1], "second_voice"))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-sbd", action="store_true")
    ap.add_argument("--patches", type=Path, default=PATCHES, help="audit patches (default: the git copy)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("records.jsonl", "rejects.jsonl"):
        p = a.out / f
        if p.exists() and (b := p.read_bytes()) and not b.endswith(b"\n"):
            p.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: drop it, that id is redone
    done = {json.loads(l)["id"] for f in ("records.jsonl", "rejects.jsonl") if (a.out / f).exists() for l in open(a.out / f)}
    recs = [r for r in (json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()) if r["id"] not in done]
    if not recs:
        return
    model = None
    if not a.no_sbd:
        from huggingface_hub import snapshot_download
        from punctuators.models import PunctCapSegModelONNX
        from punctuators.models.punc_cap_seg_model import PunctCapSegConfigONNX
        path = snapshot_download(PUNCT_MODEL, revision=PUNCT_REV, allow_patterns=["sp.model", "model.onnx", "config.yaml"])
        model = PunctCapSegModelONNX(PunctCapSegConfigONNX(directory=path))
    patches = {json.loads(l)["id"]: json.loads(l) for l in open(a.patches)} if a.patches.exists() else {}
    conv = [convert(r, model) for r in recs]
    # one batch for the whole input; each chunk's split doesn't depend on the batch (608/608 identical, pool v1 vs v2)
    flat_sents = iter(split([c for chunks, _, _ in conv for c in chunks], model))
    with open(a.out / "records.jsonl", "a", encoding="utf-8") as ok, open(a.out / "rejects.jsonl", "a", encoding="utf-8") as bad:
        for r, (chunks, marks, flags) in zip(recs, conv):
            sents = []
            for chunk, hard in zip(chunks, marks):
                parts = next(flat_sents)
                for k, (ws, m, flag) in enumerate(parts):
                    if flag:
                        flags.append(flag)
                    last = k == len(parts) - 1
                    sents.append((ws, (hard if last and hard else m) or "।"))
            sentences = [{"hi_text": " ".join(ws) + m} for ws, m in merge_short(sents)]
            if not sentences:
                flags.append("drop:empty")
            sbd = "sbd_mismatch" in flags  # the split fell back to whole chunks: a note, not a reason to reject
            flags = [f for f in flags if f != "sbd_mismatch"]
            w4, arch = {"flags": flags, **({"sbd": "fallback"} if sbd else {})}, None
            p = patches.get(r["id"])
            if p and p["base"] != hashlib.sha1(" ".join(x["hi_text"] for x in sentences).encode()).hexdigest()[:12]:
                flags.append("stale_patch")  # converter output changed since review: re-review, never auto-apply
            elif p:
                w4 = {"flags": [f"drop:{p['action']}"] if p["action"] in ("drop", "recut") else [],
                      "review": {k: p[k] for k in ("action", "reviewer", "date")}}
                arch = p.get("archive", "w4/patches.jsonl")
                if p["action"] == "fix":
                    sentences = [{"hi_text": t} for t in p["sentences"]]
            if w4["flags"]:
                bad.write(json.dumps({"id": r["id"], "stage": "transcribe", "rule": rule(w4["flags"]),
                                      "detail": {"flags": w4["flags"]}}, ensure_ascii=False) + "\n")
                bad.flush()
                continue
            # SPEC §6 key order: the record's public keys (W1 writes them in §5 order), sentences, then `_` keys
            out = {k: v for k, v in r.items() if not k.startswith("_") and k != "sentences"}
            out["sentences"] = sentences
            out.update({k: v for k, v in r.items() if k.startswith("_") and k not in ("_w4", "_archive")})
            out["_w4"] = w4
            if arch:
                out["_archive"] = {**r.get("_archive", {}), "transcribe": arch}
            ok.write(json.dumps(out, ensure_ascii=False) + "\n")
            ok.flush()


if __name__ == "__main__":
    assert merge_short([(["अ"], "।"), (["जी", "हाँ"], "।"), (["सर"], "।")]) == [(["अ", "जी", "हाँ", "सर"], "।")]
    assert merge_short([(["ठीक"], "?"), (["अ"], "।")]) == [(["ठीक", "अ"], "?")]
    assert nasal("हिन्दी") == "हिंदी" and nasal("यहां") == "यहाँ" and nasal("उन्हें") == "उन्हें"
    main()
