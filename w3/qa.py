"""TTS QA over data/w3/renders.jsonl (PS §3A: catch mispronounced names/numbers, skipped/repeated words, artefacts).

    uv run --no-project --with numpy --with torch --with transformers python -I w3/qa.py asr w2v     # CPU CTC
    uv run --no-project --python 3.13 --with torch --with transformers --with onnxruntime \
        python -I w3/qa.py asr ic                                                                     # IndicConformer, CPU
    uv run --no-project --with numpy python -I w3/qa.py score
    uv run --no-project --with numpy python -I w3/qa.py controls     # then: asr w2v controls; asr ic controls; report

`asr` caches one hypothesis per render in data/w3/asr/<model>.jsonl (resumable). `score` writes data/w3/qa.tsv:
round-trip CER per ASR, entity hits, skip/repeat runs, duration per character, clipping and silence, and a verdict.
Both ASRs are from different families than every TTS under test (Bulbul, Magpie, Parler). A content defect
(entity miss, skip, repeat) counts only when EVERY available ASR shows it: a TTS error is heard by all ASRs,
an ASR error usually by one.
"""
import json
import re
import sys
import unicodedata
import wave
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
from run import _main_checkout  # noqa: E402

ROOT = _main_checkout()  # data lives in the main checkout; worktrees have none
DATA = ROOT / "data/w3"
ASR_DIR = DATA / "asr"

# Thresholds, tuned on positive controls (`controls` + `report`; tasks/W3-hindi-tts.md "QA thresholds").
# Recall on controls: skip 0.88, repeat 0.85, number swap 0.97, English-word swap 0.97, clip 1.0, pause 1.0.
ENT_MAX = 0.25        # entity hit if some hyp substring is within this normalised edit distance
CER_REJECT = 0.06     # reject when the best ASR's CER exceeds this (0.15 caught only half the repeats)
SUBS_REJECT = 2       # reject at this many words every ASR heard as something else; 1 goes to human review
SPC_LO, SPC_HI = 0.6, 1.6  # seconds-per-char outside [lo, hi] x the engine-voice median
PAUSE_MAX = 1.5       # longest internal pause (s)
EDGE_MAX = 1.0        # leading or trailing silence (s)
CLIP_MAX = 1e-4       # fraction of samples at full scale
FILLERS = {"अ", "अं", "आ", "ए", "उम्म", "हम्म"}
# Colloquial or spelling variants that both ASRs rewrite to the standard form (measured: the agreed substitutions in the
# round-2 takes were ये->यह 5, पे->पर 5, मे->में 4, न->ना, हैलो->हेलो, and ऐ-initial English words, folded in canon;
# a TTS reading the script would not swap them).
# Not a substitution, like same-value number spellings.
SAME = {frozenset(p) for p in (("ये", "यह"), ("पे", "पर"), ("मे", "में"), ("न", "ना"), ("वो", "वह"), ("हैलो", "हेलो"))}  # SPEC R9; ASRs drop them (W4: 0% kept), so a missing filler is unverifiable


def canon(t):
    """Spelling variants the ear can't separate (nukta, candrabindu/anusvara, final marks) -> one form."""
    t = unicodedata.normalize("NFC", t)
    t = t.replace("\u093c", "").replace("\u0901", "\u0902").replace("\u0903", "")  # nukta, candrabindu, visarga
    t = t.replace("\u0910", "\u090f")  # independent ऐ -> ए: both ASRs write एन/एड/एट for English an/add/at
    return " ".join(re.sub(r"[^\u0900-\u0963\u0971-\u097f ]", " ", t).split())


def lev(a, b):
    d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(b, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return d[-1]


def cer(ref, hyp):
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    return lev(r, h) / max(1, len(r))


def substring_dist(needle, hay):
    """min edit distance of needle to any substring of hay, / len(needle)."""
    n, h = needle.replace(" ", ""), hay.replace(" ", "")
    d = [0] * (len(h) + 1)
    for i, x in enumerate(n, 1):
        prev, d[0] = d[0], i
        for j, y in enumerate(h, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (x != y))
    return min(d) / max(1, len(n))


# Number words -> value labels, so one wrong digit in a long string is a miss, not a 2/35 edit distance.
# 0-99 in order; then multipliers, fractions, English digits, repeaters; plus spellings the ASRs emit (छः छे नो जीरो).
_N099 = ("शून्य एक दो तीन चार पाँच छह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह सोलह सत्रह अठारह उन्नीस बीस "
         "इक्कीस बाईस तेईस चौबीस पच्चीस छब्बीस सत्ताईस अट्ठाईस उनतीस तीस इकतीस बत्तीस तैंतीस चौंतीस पैंतीस छत्तीस "
         "सैंतीस अड़तीस उनतालीस चालीस इकतालीस बयालीस तैंतालीस चवालीस पैंतालीस छियालीस सैंतालीस अड़तालीस उनचास पचास "
         "इक्यावन बावन तिरपन चौवन पचपन छप्पन सत्तावन अट्ठावन उनसठ साठ इकसठ बासठ तिरसठ चौंसठ पैंसठ छियासठ सड़सठ अड़सठ "
         "उनहत्तर सत्तर इकहत्तर बहत्तर तिहत्तर चौहत्तर पचहत्तर छिहत्तर सतहत्तर अठहत्तर उन्यासी अस्सी इक्यासी बयासी "
         "तिरासी चौरासी पचासी छियासी सत्तासी अट्ठासी नवासी नब्बे इक्यानवे बानवे तिरानवे चौरानवे पंचानवे छियानवे सत्तानवे "
         "अट्ठानवे निन्यानवे").split()
_NUM = {**{w: str(i) for i, w in enumerate(_N099)},
        "सौ": "e2", "हज़ार": "e3", "लाख": "e5", "करोड़": "e7", "डेढ़": "1.5", "ढाई": "2.5", "सवा": "+.25", "साढ़े": "+.5",
        "पौने": "-.25", "ज़ीरो": "0", "वन": "1", "टू": "2", "थ्री": "3", "फ़ोर": "4", "फ़ाइव": "5", "सिक्स": "6",
        "सेवन": "7", "एट": "8", "नाइन": "9", "टेन": "10", "नाइंटी": "90", "डबल": "x2", "ट्रिपल": "x3",
        "छः": "6", "छ": "6", "छे": "6", "नो": "9", "पन्द्रह": "15", "चवालीस": "44", "चौवालीस": "44", "ऐट": "8"}
def _nk(w):  # number lookup ignores anusvara: ASRs write चौसठ for चौंसठ
    return w.replace("\u0902", "")


NUM = {_nk(canon(k)): v for k, v in _NUM.items()}
assert len(_N099) == 100


def num_runs(text):
    out, cur = [], []
    for w in text.split():
        if _nk(w) in NUM:
            cur.append(NUM[_nk(w)])
        elif cur:
            out.append(cur)
            cur = []
    return out + ([cur] if cur else [])


def num_hit(ent, hyp):
    """Every number run of the entity appears, label for label, inside one number run of the hyp."""
    hr = num_runs(hyp)
    return all(any(any(h[i:i + len(e)] == e for i in range(len(h) - len(e) + 1)) for h in hr) for e in num_runs(ent))


def ent_hit(ent, hyp):
    return substring_dist(ent, hyp) <= ENT_MAX and num_hit(ent, hyp)


def word_ops(ref, hyp):
    """Word alignment -> (ref words the hyp lacks, ref positions where the hyp has extra words, ref words heard as
    something else). Unequal 'replace' blocks whose concatenations match are segmentation (यूपीआई vs यू पी आई)."""
    r, h = ref.split(), hyp.split()
    gone, extra, subs = set(), set(), set()
    for op, i1, i2, j1, j2 in SequenceMatcher(None, r, h, autojunk=False).get_opcodes():
        rs, hs = "".join(r[i1:i2]), "".join(h[j1:j2])
        if op == "delete" or (op == "replace" and i2 - i1 > j2 - j1 and len(rs) - len(hs) >= 2 and cer(rs, hs) > 0.34):
            gone.update(range(i1, i2))
        elif op == "insert" or (op == "replace" and j2 - j1 > i2 - i1 and len(hs) - len(rs) >= 2 and cer(rs, hs) > 0.34):
            extra.add(i1)
        elif op == "replace" and i2 - i1 == j2 - j1:  # word for word: same-value number spellings (छह/छः) are fine
            subs.update(i for i, x in zip(range(i1, i2), h[j1:j2])
                        if cer(r[i], x) >= 0.5 and not (_nk(r[i]) in NUM and NUM[_nk(r[i])] == NUM.get(_nk(x)))
                        and frozenset((r[i], x)) not in SAME)
        elif op == "replace" and cer(rs, hs) >= 0.5:
            subs.update(range(i1, i2))
    return gone, extra, subs


def agreed(ref, hyps):
    """Skipped ref words / insertion points / substituted ref words that EVERY ASR shows (insertions +-1 word)."""
    ops = [word_ops(ref, x) for x in hyps]
    w = ref.split()
    gone = {i for i in set.intersection(*(g for g, _, _ in ops)) if w[i] not in FILLERS}
    extra = {p for p in ops[0][1] if all(any(abs(p - q) <= 1 for q in e) for _, e, _ in ops[1:])}
    subs = {i for i in set.intersection(*(s for _, _, s in ops)) - gone if w[i] not in FILLERS}
    return ([w[i] for i in sorted(gone)], [f"@{p}:{' '.join(w[max(0, p - 1):p + 1])}" for p in sorted(extra)],
            [w[i] for i in sorted(subs)])


def audio_stats(path):
    with wave.open(str(path)) as w:
        sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32)
    clip = float(np.mean(np.abs(x) >= 32766))
    hop = int(0.02 * sr)
    fr = x[: len(x) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) / 32768 + 1e-9)
    thr = max(np.percentile(db, 90) - 35, -60)
    on = np.flatnonzero(db > thr)
    if not len(on):
        return {"clip": clip, "lead": len(x) / sr, "trail": 0.0, "pause": 0.0, "speech": 0.0, "peak_db": float(db.max())}
    gaps = np.diff(on) * 0.02
    return {"clip": clip, "lead": on[0] * 0.02, "trail": (len(db) - 1 - on[-1]) * 0.02,
            "pause": float(gaps.max()) if len(gaps) else 0.0, "speech": (on[-1] - on[0] + 1) * 0.02,
            "peak_db": float(20 * np.log10(np.abs(x).max() / 32768 + 1e-9))}


def load16k(path):
    import subprocess
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768


def asr_w2v():
    """Vakyansh wav2vec2 Hindi CTC (MIT; same checkpoint and blank fix as w1/asr_check.py). CPU, greedy."""
    import torch
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    name = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"
    proc, model = Wav2Vec2Processor.from_pretrained(name), Wav2Vec2ForCTC.from_pretrained(name).eval()
    vocab = {v: k for k, v in proc.tokenizer.get_vocab().items()}
    blank = {proc.tokenizer.convert_tokens_to_ids(t) for t in ("<s>", "<pad>", "</s>", "<unk>")}
    delim = proc.tokenizer.word_delimiter_token
    torch.set_num_threads(6)

    def decode(x):
        with torch.inference_mode():
            ids = model(proc(x, sampling_rate=16000, return_tensors="pt").input_values).logits[0].argmax(-1).tolist()
        out, prev = [], None
        for k in ids:
            if k != prev and k not in blank:
                out.append(" " if vocab[k] == delim else vocab[k])
            prev = k
        return "".join(out)

    def run(path):
        x = load16k(path)
        return " ".join(decode(x[i:j]) for i, j in quiet_chunks(x))
    return run


def quiet_chunks(x, sr=16000):
    """vakyansh on whole clips > ~10 s silently loses everything after some pause (measured: 4-23 % of renders per
    engine, CER 0.3-0.6 -> 0.01-0.08 when chunked). So decode <= 7 s chunks cut at the quietest 20 ms frame."""
    hop = sr // 50
    e = (x[: len(x) // hop * hop].reshape(-1, hop) ** 2).mean(1)
    cuts = [0]
    while len(x) - cuts[-1] > 7 * sr:
        a, b = (cuts[-1] + 4 * sr) // hop, (cuts[-1] + 7 * sr) // hop
        cuts.append((a + int(np.argmin(e[a:b]))) * hop)
    cuts.append(len(x))
    return list(zip(cuts, cuts[1:]))


def asr_ic():
    """AI4Bharat IndicConformer 600M multilingual (MIT, gated), RNNT decoding, CPU onnxruntime; loader as w4/asr_run.py.
    Uses W4's local copy (real files: onnxruntime rejects external data behind cache symlinks); same files as
    snapshot_download("ai4bharat/indic-conformer-600m-multilingual", revision="2a77d305", local_dir=...).
    The Hindi .nemo needs AI4Bharat's NeMo fork (W4)."""
    import importlib.util
    import torch
    path = ROOT / "data/w4/ic600m"
    spec = importlib.util.spec_from_file_location("model_onnx", f"{path}/model_onnx.py")  # = trust_remote_code
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {k: v for k, v in json.load(open(f"{path}/config.json")).items() if k != "auto_map"}
    model = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=path, **cfg))
    torch.set_num_threads(6)

    def run(p):
        with torch.inference_mode():
            return model(torch.from_numpy(load16k(p))[None], "hi", "rnnt")
    return run


def renders(index="renders"):
    return [json.loads(l) for l in open(DATA / f"{index}.jsonl", encoding="utf-8")]


def cmd_asr(name, index="renders"):
    out = ASR_DIR / f"{name}.jsonl"
    ASR_DIR.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["key"] for l in open(out)} if out.exists() else set()
    todo = [r for r in renders(index) if r["key"] not in done and "asr_key" not in r]
    if not todo:
        return print(name, "nothing to do")
    run = {"w2v": asr_w2v, "ic": asr_ic}[name]()
    with open(out, "a", encoding="utf-8") as f:
        for r in todo:
            f.write(json.dumps({"key": r["key"], "hyp": run(ROOT / r["wav"])}, ensure_ascii=False) + "\n")
            f.flush()
    print(name, len(todo), "->", out)


def measure(r, hyps, wav):
    """One render vs its ASR hypotheses {model: raw hyp} -> QA row (everything but the speed rule's spc_rel/why).
    r needs hi_text, entities, en_words, chars."""
    ref = canon(r["hi_text"])
    h = {m: canon(x) for m, x in hyps.items()}
    cers = {m: cer(ref, x) for m, x in h.items()}  # all ASRs, for the record
    cov = {m: len(x.replace(" ", "")) / max(1, len(ref.replace(" ", ""))) for m, x in h.items()}
    if max(cov.values()) >= 0.85:  # an ASR that stopped early is an ASR failure while another one heard it all
        h = {m: x for m, x in h.items() if cov[m] >= 0.7}
    a = audio_stats(wav)
    ents = [canon(e) for e in r["entities"]]
    miss = [e for e in ents if not any(ent_hit(e, x) for x in h.values())]  # missed only if every ASR misses it
    en = [canon(e) for e in r["en_words"]]
    en_miss = [e for e in en if not any(ent_hit(e, x) for x in h.values())]
    skips, adds, subs = agreed(ref, list(h.values()))
    return {**a, "cer": cers, "best_cer": min(cer(ref, x) for x in h.values()), "ent_n": len(ents), "ent_miss": miss,
            "en_n": len(en), "en_miss": en_miss, "skips": skips, "adds": adds, "subs": subs, "asrs": sorted(h),
            "spc": a["speech"] / max(1, r["chars"])}


def rules(x, dur):
    """QA row (with spc_rel, None = no speed reference yet) -> reasons to reject; [] = keep."""
    why = []
    if x["best_cer"] > CER_REJECT:
        why.append(f"cer{x['best_cer']:.2f}")
    if x["ent_miss"]:
        why.append(f"ent{len(x['ent_miss'])}")
    if x["en_miss"]:
        why.append(f"en{len(x['en_miss'])}")
    if len(x["subs"]) >= SUBS_REJECT:
        why.append(f"subs{len(x['subs'])}")
    if x["skips"]:
        why.append("skip")
    if x["adds"]:
        why.append("add")
    if x["spc_rel"] is not None and not SPC_LO <= x["spc_rel"] <= SPC_HI:
        why.append(f"spc{x['spc_rel']:.2f}")
    if x["pause"] > PAUSE_MAX:
        why.append(f"pause{x['pause']:.1f}")
    if max(x["lead"], x["trail"]) > EDGE_MAX:
        why.append("edge")
    if x["clip"] > CLIP_MAX:
        why.append("clip")
    if dur > 60:
        why.append("over60")
    return why


def cmd_score(index="renders"):
    hyps = {}
    for p in sorted(ASR_DIR.glob("*.jsonl")):
        for l in open(p, encoding="utf-8"):
            d = json.loads(l)
            hyps.setdefault(d["key"], {})[p.stem] = d["hyp"]
    rows = [{**r, **measure(r, hyps[k], ROOT / r["wav"])} for r in renders(index) if (k := r.get("asr_key", r["key"])) in hyps]
    # duration-per-char relative to the engine-voice median
    med = {}
    for x in rows:
        med.setdefault((x["engine"], x["voice"]), []).append(x["spc"])
    med = {k: float(np.median(v)) for k, v in med.items()}
    if index != "renders":  # controls are scored against the medians of the real renders
        med = json.loads((DATA / "spc_median.json").read_text())
        med = {tuple(k.split("|")): v for k, v in med.items()}
    else:
        (DATA / "spc_median.json").write_text(json.dumps({"|".join(k): v for k, v in med.items()}))
    for x in rows:
        x["spc_rel"] = x["spc"] / med[(x["engine"], x["voice"])]
        x["why"] = rules(x, x["dur"])
        x["verdict"] = "reject" if x["why"] else "keep"
    cols = ["key", "pid", "variant", "engine", "voice", "params", "dur", "chars", "spc", "spc_rel", "lead", "trail",
            "pause", "clip", "peak_db", "best_cer", "cer", "ent_n", "ent_miss", "en_n", "en_miss", "skips", "adds",
            "subs", "asrs", "verdict", "why"]
    out = DATA / ("qa.tsv" if index == "renders" else f"qa_{index}.tsv")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\t".join(cols) + "\n")
        for x in rows:
            f.write("\t".join(json.dumps(x[c], ensure_ascii=False) if isinstance(x[c], (dict, list)) else
                              (f"{x[c]:.4g}" if isinstance(x[c], float) else str(x[c])) for c in cols) + "\n")
    print(f"{len(rows)} scored -> {out}")
    return rows


def cmd_controls(n=40):
    """Known-bad copies of clean renders (positive controls for threshold tuning). Never dataset records."""
    import random
    import shutil
    rng = random.Random(0)
    lines = open(DATA / "qa.tsv", encoding="utf-8").read().splitlines()
    head = lines[0].split("\t")
    rows = {d["key"]: (d["verdict"], json.loads(d["cer"])) for d in (dict(zip(head, l.split("\t"))) for l in lines[1:])}
    clean = [r for r in renders() if rows.get(r["key"], ("", {}))[0] == "keep"
             and max(rows[r["key"]][1].values()) <= 0.05 and r["variant"] == "dev"]
    rng.shuffle(clean)
    out = DATA / "controls"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir()
    idx = []

    def audio_ctl(r, kind):
        with wave.open(str(ROOT / r["wav"])) as w:
            sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").copy()
        a = audio_stats(ROOT / r["wav"])
        mid = int((a["lead"] + a["speech"] * rng.uniform(0.3, 0.7)) * sr)
        if kind == "skip":    # ~one word gone
            x = np.concatenate([x[:mid], x[mid + int(0.6 * sr):]])
        elif kind == "repeat":  # ~a word or two said twice
            x = np.concatenate([x[:mid], x[mid - int(0.9 * sr):]])
        elif kind == "pause":
            x = np.concatenate([x[:mid], np.zeros(int(2.5 * sr), "<i2"), x[mid:]])
        elif kind == "clip":
            x = np.clip(x.astype(np.int32) * 6, -32768, 32767).astype("<i2")
        p = out / f"{kind}-{r['key']}.wav"
        with wave.open(str(p), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(x.tobytes())
        return {**r, "key": f"ctl-{kind}-{r['key']}", "variant": f"ctl-{kind}", "wav": str(p.relative_to(ROOT))}

    def text_ctl(r, kind):
        if kind == "num":  # TTS said the original number, the script asked for a neighbouring one
            ents = [e for e in r["entities"] if any(w in _N099 for w in e.split())]
            if not ents:
                return None
            e = rng.choice(ents)
            ws = e.split()
            i = rng.choice([j for j, w in enumerate(ws) if w in _N099])
            k = _N099.index(ws[i])
            ws[i] = _N099[k + 1 if k < 99 else k - 1]
            e2 = " ".join(ws)
            ents2 = [e2 if x == e else x for x in r["entities"]]
            hi = r["hi_text"].replace(e, e2, 1)
        else:  # an English word swapped for another English word
            pool = sorted({w for x in renders() for w in x["en_words"] if len(w) >= 4})
            ens = [w for w in r["en_words"] if len(w) >= 4 and w in r["hi_text"]]
            if not ens:
                return None
            e = rng.choice(ens)
            e2 = rng.choice([w for w in pool if w != e and abs(len(w) - len(e)) <= 2] or pool)
            hi, ents2 = r["hi_text"].replace(e, e2, 1), r["entities"]
            r = {**r, "en_words": [e2 if w == e else w for w in r["en_words"]]}
        return {**r, "key": f"ctl-{kind}-{r['key']}", "variant": f"ctl-{kind}", "asr_key": r["key"],
                "hi_text": hi, "entities": ents2, "_swap": [e, e2]}

    for i, r in enumerate(clean[:n]):
        for kind in ("skip", "repeat", "pause", "clip"):
            idx.append(audio_ctl(r, kind))
        for kind in ("num", "en"):
            c = text_ctl(r, kind)
            if c:
                idx.append(c)
    with open(DATA / "controls.jsonl", "w", encoding="utf-8") as f:
        for x in idx:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print(f"{len(idx)} controls from {min(n, len(clean))} clean renders -> {DATA / 'controls.jsonl'}")


def cmd_report(index="controls"):
    """Detection rate per control kind (recall) and reject rate on the clean renders (false positives)."""
    rows = cmd_score(index)
    kinds = {}
    for x in rows:
        kinds.setdefault(x["variant"], []).append(x)
    for k, g in sorted(kinds.items()):
        caught = [x for x in g if x["verdict"] == "reject"]
        why = {}
        for x in caught:
            for w in x["why"]:
                w = re.sub(r"[\d.]+$", "", w)
                why[w] = why.get(w, 0) + 1
        print(f"{k}: caught {len(caught)}/{len(g)}  by {why}")


def cmd_summary():
    """Markdown tables for the TTS-choice memo, from qa.tsv + renders + probe tags."""
    lines = open(DATA / "qa.tsv", encoding="utf-8").read().splitlines()
    head = lines[0].split("\t")
    q = {d["key"]: d for d in (dict(zip(head, l.split("\t"))) for l in lines[1:])}
    tags = {json.loads(l)["pid"]: json.loads(l)["tags"] for l in open(DATA / "probe.jsonl", encoding="utf-8")}
    rs = [r for r in renders() if r["key"] in q]
    for r in rs:
        d = q[r["key"]]
        r["q"] = d
        r["tags"] = sorted({t for p in r["pid"].split("+") for t in tags.get(p, [])})
        r["cers"] = json.loads(d["cer"])
        r["miss"], r["enmiss"] = json.loads(d["ent_miss"]), json.loads(d["en_miss"])
        r["meta"] = json.loads((ROOT / r["wav"]).with_suffix(".json").read_text())

    def pct(a, b):
        return f"{100 * a / b:.1f}%" if b else "–"

    def table(groups, title):
        print(f"\n**{title}**\n")
        print("| group | renders | audio h | CER ic p50/p90 | CER best p50 | entity hit | number-entity hit | English-span hit "
              "| agreed skip | agreed insert | agreed subst (review) | pause/edge/clip | QA reject | s/char p50 | compute s per audio s |")
        print("|" + "---|" * 15)
        for name, g in groups:
            if not g:
                continue
            ic = np.array([r["cers"].get("ic", np.nan) for r in g])
            best = np.array([float(r["q"]["best_cer"]) for r in g])
            en_n = sum(len(r["entities"]) for r in g)
            en_m = sum(len(r["miss"]) for r in g)
            nn = sum(1 for r in g for e in r["entities"] if num_runs(canon(e)))
            nm = sum(1 for r in g for e in r["miss"] if num_runs(e))
            sp_n, sp_m = sum(len(r["en_words"]) for r in g), sum(len(r["enmiss"]) for r in g)
            sk = sum(1 for r in g if json.loads(r["q"]["skips"]))
            ad = sum(1 for r in g if json.loads(r["q"]["adds"]))
            su = sum(1 for r in g if json.loads(r["q"]["subs"]))
            ac = sum(1 for r in g if any(w.startswith(("pause", "edge", "clip")) for w in json.loads(r["q"]["why"])))
            rej = sum(1 for r in g if r["q"]["verdict"] == "reject")
            dur = sum(r["dur"] for r in g)
            secs = sum(r["meta"].get("secs") or 0 for r in g)
            spc = np.median([float(r["q"]["spc"]) for r in g])
            print(f"| {name} | {len(g)} | {dur / 3600:.2f} | {np.nanmedian(ic):.3f}/{np.nanpercentile(ic, 90):.3f} | "
                  f"{np.median(best):.3f} | {pct(en_n - en_m, en_n)} | {pct(nn - nm, nn)} | {pct(sp_n - sp_m, sp_n)} | "
                  f"{pct(sk, len(g))} | {pct(ad, len(g))} | {pct(su, len(g))} | {pct(ac, len(g))} | {pct(rej, len(g))} | {spc:.3f} | "
                  f"{secs / dur:.2f} |")

    engines = sorted({r["engine"] for r in rs})
    dev = [r for r in rs if r["variant"] == "dev" and not r["params"] and "+" not in r["pid"]]
    table([(e, [r for r in dev if r["engine"] == e]) for e in engines], "By engine (Devanagari input, default params, single items)")
    table([(f"{e} {v}", [r for r in dev if r["engine"] == e and r["voice"] == v])
           for e in engines for v in sorted({r["voice"] for r in dev if r["engine"] == e})], "By voice")
    table([(f"{e} {t}", [r for r in dev if r["engine"] == e and t in r["tags"]])
           for t in ("digits", "enum", "lakh", "crore", "frac", "num2199", "date", "time", "name", "brand", "acronym", "filler", "selfcorr")
           for e in engines], "By phenomenon tag")
    # paired comparisons: same pid + voice, one factor changed
    def pairs(cond_a, cond_b, label):
        by = {}
        for r in rs:
            by.setdefault((r["engine"], r["pid"], r["voice"]), []).append(r)
        out = []
        for k, g in by.items():
            a = [r for r in g if cond_a(r)]
            b = [r for r in g if cond_b(r)]
            if a and b:
                out.append((a[0], b[0]))
        if not out:
            return
        print(f"\n**{label}** ({len(out)} pairs)\n")
        print("| engine | pairs | English-span hit A → B | CER ic p50 A → B | QA reject A → B |")
        print("|---|---|---|---|---|")
        for e in engines:
            pe = [p for p in out if p[0]["engine"] == e]
            if not pe:
                continue
            def hit(rr):
                n = sum(len(r["en_words"]) for r in rr)
                return pct(n - sum(len(r["enmiss"]) for r in rr), n)
            A, B = [p[0] for p in pe], [p[1] for p in pe]
            print(f"| {e} | {len(pe)} | {hit(A)} → {hit(B)} | {np.nanmedian([r['cers'].get('ic', np.nan) for r in A]):.3f} → "
                  f"{np.nanmedian([r['cers'].get('ic', np.nan) for r in B]):.3f} | "
                  f"{pct(sum(r['q']['verdict'] == 'reject' for r in A), len(A))} → {pct(sum(r['q']['verdict'] == 'reject' for r in B), len(B))} |")
    pairs(lambda r: r["variant"] == "dev" and not r["params"], lambda r: r["variant"] == "lat", "English words: Devanagari (A) vs Latin (B) input")
    pairs(lambda r: r["variant"] == "dev" and not r["params"], lambda r: r["params"].get("temperature") == 0.3, "Temperature 0.6 (A) vs 0.3 (B)")
    lg = [r for r in rs if "+" in r["pid"]]
    if lg:
        table([("long-form", lg)], "Long-form inputs (3 items joined)")
        print("durations:", sorted(round(r["dur"], 1) for r in lg))


def cmd_spotcheck():
    """data/w3/spotcheck.tsv: what a human must listen to, because no ASR can decide it."""
    lines = open(DATA / "qa.tsv", encoding="utf-8").read().splitlines()
    head = lines[0].split("\t")
    q = {d["key"]: d for d in (dict(zip(head, l.split("\t"))) for l in lines[1:])}
    rs = {r["key"]: r for r in renders()}
    pilot = [json.loads(l) for l in open(DATA / "pilot/manifest.jsonl", encoding="utf-8")]
    out, seen = [], set()
    for r in rs.values():  # 1. one clip per voice: gender (F0 contradicts some names) and accent
        if (r["engine"], r["voice"]) not in seen and r["variant"] == "dev":
            seen.add((r["engine"], r["voice"]))
            out.append(("voice", r["key"], "", r["wav"], f"{r['engine']} {r['voice']}: gender? natural Hindi accent?"))
    for p in pilot:
        k, w = p["_w3"]["key"], " ".join(s["hi_text"] for s in p["sentences"])
        fil = [x for x in w[:-1].replace("।", " ").replace("?", " ").split() if x in FILLERS]
        if fil:  # 2. fillers: every ASR drops them (W4), so only ears can say the TTS spoke them
            out.append(("filler", k, p["id"], rs[k]["wav"], "spoken? " + " ".join(fil)))
        if p["_w3"]["variant"] == "lat":
            out.append(("latin-input", k, p["id"], rs[k]["wav"], "English words as the Devanagari transcript says?"))
    for k, d in q.items():  # 3. one word every ASR heard as something else (below the reject threshold)
        if d["verdict"] == "keep" and d["subs"] != "[]":
            out.append(("subst", k, "", rs[k]["wav"], "said as written? " + " ".join(json.loads(d["subs"]))))
    with open(DATA / "spotcheck.tsv", "w", encoding="utf-8") as f:
        f.write("kind\tkey\tid\twav\tquestion\n")
        f.writelines("\t".join(x) + "\n" for x in out)
    kinds = {}
    for x in out:
        kinds[x[0]] = kinds.get(x[0], 0) + 1
    print(f"{len(out)} spot checks -> {DATA / 'spotcheck.tsv'}: {kinds}")


if __name__ == "__main__":
    assert canon("फ़ोन पाँच, OK।") == "फोन पांच" and cer("abc", "abd") == 1 / 3
    assert substring_dist("तीन लाख", "साढ़े तीन लाख रुपये") == 0 and substring_dist("abcd", "xxabxdxx") == 0.25
    assert num_hit(canon("नौ एक छह"), canon("चार नौ एक छः शून्य")) and not num_hit("नौ एक छह", "नौ आठ छह")
    assert ent_hit(canon("साढ़े तीन लाख"), canon("कल साढे तीन लाख रुपये")) and not ent_hit("साढे तीन लाख", "साढे चार लाख")
    assert agreed("एक दो तीन चार पाँच", ["एक दो पाँच", "एक दो पांच"])[0] == ["तीन", "चार"]
    assert agreed("चार आठ एक", ["चार ए आठ एक", "चार टी आठ एक"])[1] == ["@1:चार आठ"]
    assert agreed("यू पी आई नंबर", ["यूपीआई नंबर"]) == ([], [], []) == agreed("अ मैंने भेजे", ["मैंने भेजे"])
    assert agreed("और ब्रांच लुधियाना", ["और बांस लुधियाना"]) == ([], [], ["ब्रांच"])
    assert num_hit(canon("चौंसठ साल"), canon("चौसठ साल"))
    assert agreed(canon("पाँच छह चार"), [canon("पांच छः चार")]) == ([], [], [])
    assert agreed(canon("मेरे नाम पे चेक"), [canon("मेरे नाम पर चेक")]) == ([], [], [])
    assert agreed(canon("हैलो इट वाज़ ऐन ऐड"), [canon("हेलो इट वाज एन एड")]) == ([], [], [])
    ch = quiet_chunks(np.r_[np.ones(80000), np.zeros(320), np.ones(160000)].astype(np.float32))
    assert ch[0] == (0, 80000) and ch[-1][1] == 240320 and all(j - i <= 7 * 16000 for i, j in ch), ch
    a = sys.argv[1:]
    {"asr": lambda: cmd_asr(*a[1:]), "score": lambda: cmd_score(*a[1:]), "controls": cmd_controls,
     "report": lambda: cmd_report(*a[1:]), "summary": cmd_summary,
     "spotcheck": cmd_spotcheck}[a[0]]()
