# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "torchaudio", "transformers", "numpy", "huggingface_hub", "uroman", "onnxruntime", "boto3"]
# ///
"""W5 aligner benchmark (not a pipeline stage). Data under data/w5/ (gitignored).

    uv run w5/bench.py ctrl                       # render the known-truth control set (MMS-TTS, CC BY-NC: never released)
    uv run w5/bench.py run ALIGNER SET [COND...]  # -> data/w5/bench/<ALIGNER>/<SET>.<COND>.jsonl
    uv run w5/bench.py score                      # tables for tasks/W5-alignment.md

SET: ctrl | real (transcribe LATEST, 642 IndicVoices clips) | syn (W3 pilot, 50 Bulbul/Magpie clips).
COND: clean | tel (8 kHz, 300 Hz high-pass, 8-bit mu-law) | noise (white, 10 dB SNR) | shift (0.37 s of silence
prepended: every time must move by exactly 0.37 s) | cut (one word's audio removed: the transcript word is unspoken)
| dup (one word's audio repeated: speech the transcript lacks).
"""
import functools, json, random, re, sys, time, unicodedata, wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = Path("/home/harsh1/github-repos/sarvam-high-prep/data")  # main checkout: worktrees have no data/
W5 = DATA / "w5"
sys.path.insert(0, str(HERE))
import align as A  # noqa: E402

SETS = {"ctrl": (W5 / "ctrl/records.jsonl", W5 / "ctrl"),
        "real": (DATA / "cache/transcribe/20261014T084235Z-733cbdc/out/records.jsonl",
                 DATA / "cache/audio/stages_cut_20261014T083143Z-68a7f35_audio"),
        "syn": (DATA / "master/tts_pilot/records.jsonl", DATA / "master/tts_pilot")}
TTS, TTS_REV = "facebook/mms-tts-hin", "1d83b223ec78e30b944f7d96bd117eb3d7023303"  # CC BY-NC 4.0: benchmark only
SHIFT = 0.37
GLOSS = re.compile(r"(\S+)\s*[\[<]\s*([A-Za-z][A-Za-z .'_\-]*?)\s*[\]>]")


def recs(name):
    return [json.loads(l) for l in open(SETS[name][0], encoding="utf-8") if l.strip()]


def read24(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == 24000
        return torch.from_numpy(np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768)


def write24(path, x):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000)
        w.writeframes((x.clamp(-1, 32767 / 32768) * 32768).round().short().numpy().tobytes())


def english(rs):
    """Devanagari words IndicVoices glossed with Latin in `normalized` (minus noise tags, which use the same brackets)."""
    tags = {t.lower() for r in rs for s in r["_w1"]["segments"] for t in re.findall(r"[\[<]\s*([^\]>]+?)\s*[\]>]", s["verbatim"])}
    return {d for r in rs for s in r["_w1"]["segments"]
            for d, g in GLOSS.findall(unicodedata.normalize("NFC", s["normalized"])) if g.lower() not in tags}


def category(w, eng):
    return "filler" if w in A.FILLERS else "english" if w in eng else "hindi"


# ---------------------------------------------------------------- control set

def ctrl():
    from transformers import VitsModel
    torch.set_num_threads(8)
    model = VitsModel.from_pretrained(TTS, revision=TTS_REV, cache_dir=W5 / "hf").eval()
    vocab = json.loads(Path(next((W5 / "hf").glob("models--facebook--mms-tts-hin/snapshots/*/vocab.json"))).read_text())
    got = {}
    model.duration_predictor.register_forward_hook(lambda m, i, o: got.__setitem__("logd", o))
    real, syn = recs("real"), recs("syn")
    eng = english(real)
    rnd = random.Random(5)
    def nwords(r, cat):
        return sum(category(w, eng) == cat for ws in A.words_of(r) for w in ws)
    pick = sorted(real, key=lambda r: -nwords(r, "filler"))[:35] + sorted(real, key=lambda r: -nwords(r, "english"))[:35]
    pick = list({r["id"]: r for r in pick}.values())
    pick += rnd.sample([r for r in real if r not in pick], 100 - len(pick)) + syn
    out, n = [], 950000
    for r in pick:
        words = [w for ws in A.words_of(r) for w in ws]
        ids, owner = [], []  # text tokens (spaces included) and the word each belongs to
        for k, w in enumerate(words):
            if k:
                ids.append(vocab[" "]), owner.append(None)
            cs = [c for c in w if c in vocab]
            ids += [vocab[c] for c in cs]
            owner += [k] * len(cs)
        if len(set(o for o in owner if o is not None)) != len(words):
            continue  # a word with no renderable character
        x = torch.tensor([[t for i in ids for t in (0, i)] + [0]])  # VITS add_blank: 0 between and around tokens
        torch.manual_seed(n)
        with torch.inference_mode():
            wav = model(input_ids=x).waveform[0]
        dur = torch.ceil(torch.exp(got["logd"][0, 0]) / model.config.speaking_rate).long().tolist()
        assert sum(dur) * 256 == len(wav), (sum(dur) * 256, len(wav))
        cum = np.cumsum([0] + dur) * 256 / 16000  # cum[p] = start of token p
        truth, edge = [], []
        for k in range(len(words)):
            ps = [2 * j + 1 for j, o in enumerate(owner) if o == k]
            truth.append([round(cum[ps[0]], 4), round(cum[ps[-1] + 1], 4)])
            edge.append([dur[ps[0] - 1], dur[ps[-1] + 1]])  # blank frames just outside the word (ambiguity, 16 ms each)
        x24 = AF.resample(wav, 16000, 24000)
        if len(x24) / 24000 > 58:
            continue
        rid = f"rec-{n}"
        write24(W5 / "ctrl/audio" / f"{rid}.wav", x24)
        out.append({"id": rid, "audio": f"audio/{rid}.wav", "duration": round(len(x24) / 24000, 3),
                    "sentences": [{"hi_text": s["hi_text"]} for s in r["sentences"]],
                    "_w5": {"src": r["id"], "truth": truth, "edge_blank": edge,
                            "cat": [category(w, eng) for w in words]}})
        n += 1
    # long clips: 3-5 short controls joined with 0.4 s gaps, up to 58 s
    base, k = list(out), 0
    while k < 12:
        group, tot = [], 0.0
        for r in rnd.sample(base, len(base)):
            if tot + r["duration"] + 0.4 <= 58:
                group.append(r)
                tot += r["duration"] + 0.4
            if tot > 45:
                break
        rid, xs, truth, cats, off = f"rec-{n}", [], [], [], 0.0
        for r in group:
            xs += [read24(W5 / "ctrl" / r["audio"]), torch.zeros(int(0.4 * 24000))]
            truth += [[round(s + off, 4), round(e + off, 4)] for s, e in r["_w5"]["truth"]]
            cats += r["_w5"]["cat"]
            off += r["duration"] + 0.4
        x = torch.cat(xs[:-1])
        write24(W5 / "ctrl/audio" / f"{rid}.wav", x)
        out.append({"id": rid, "audio": f"audio/{rid}.wav", "duration": round(len(x) / 24000, 3),
                    "sentences": [s for r in group for s in r["sentences"]],
                    "_w5": {"src": "+".join(r["id"] for r in group), "truth": truth, "cat": cats, "long": True}})
        n, k = n + 1, k + 1
    with open(W5 / "ctrl/records.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in out)
    print(f"{len(out)} control records, {sum(r['duration'] for r in out) / 60:.1f} min")


# ---------------------------------------------------------------- aligners: (x24, words) -> result dict or None

def w2v():
    m = A.W2V()

    def run(x24, words):
        lp = m(AF.resample(x24, 24000, A.SR))
        res = A.viterbi(lp, words, m.index)
        if res is None:
            return None
        spans, forced = res
        return {"frames": lp.shape[0], "regret": (lp.max(-1).values.sum().item() - forced) / lp.shape[0],
                "words": [[s * A.FRAME, e * A.FRAME, c] for s, e, c, *_ in A.word_spans(spans, m.index["|"])],
                "spans": [[m.tokens[t], s, e, round(c, 4)] for t, s, e, c in spans]}
    return run


def mms():
    """torchaudio MMS_FA (CC BY-NC weights: benchmark only, research T3) on uroman-romanised words, as in the
    torchaudio multilingual tutorial: no word delimiter, words split by token count."""
    import os
    os.environ.setdefault("TORCH_HOME", str(W5 / "torch"))
    import torchaudio
    import uroman
    b = torchaudio.pipelines.MMS_FA
    model, d, ur = b.get_model(with_star=False).eval(), b.get_dict(star=None), uroman.Uroman()
    torch.set_num_threads(A.THREADS)

    def run(x24, words):
        with torch.inference_mode():
            lp = model(AF.resample(x24, 24000, A.SR)[None])[0][0].log_softmax(-1)
        toks = [[d[c] for c in ur.romanize_string(w).lower() if c in d] or [d["a"]] for w in words]
        ids = [t for w in toks for t in w]
        if len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) > lp.shape[0]:
            return None
        path, sc = AF.forced_align(lp[None], torch.tensor([ids], dtype=torch.int32), blank=0)
        spans = AF.merge_tokens(path[0], sc[0].exp())
        out, k = [], 0
        for w in toks:
            ss = spans[k:k + len(w)]
            out.append([ss[0].start * A.FRAME, ss[-1].end * A.FRAME, sum(s.score for s in ss) / len(ss)])
            k += len(w)
        return {"frames": lp.shape[0], "regret": (lp.max(-1).values.sum().item() - sc.sum().item()) / lp.shape[0], "words": out}
    return run


IC_S3 = "raw/_models/indic-conformer-600m-multilingual/2a77d305038dfc502fd9464e7d028b90381976cb/"


def bpe(w, vocab):
    """Greedy longest-match into IndicConformer's Hindi pieces (the HF repo ships no SentencePiece model)."""
    out, i = [], 0
    while i < len(w):
        p = "▁" if not out else ""
        j = next((j for j in range(len(w), i, -1) if p + w[i:j] in vocab), None)
        out.append(vocab[p] if j is None else vocab[p + w[i:j]])  # no `▁x` piece: a bare `▁`, then `x`
        i = i if j is None else j
    return out


def ic():
    """IndicConformer 600M multilingual (MIT), CTC head on the Hindi mask, as the HF wrapper's model_onnx.py does:
    80 ms frames, blank = 256. Its 2.6 GB of weights exceed W5's laptop budget: runs on SageMaker, fetched from
    the bucket's pinned raw/_models copy."""
    import os
    import onnxruntime as ort
    d = Path(os.environ.get("W5_IC_DIR", "/opt/ml/input/data/_ic600m"))
    if not (d / "assets/encoder.onnx").exists():
        sys.path.insert(0, str(ROOT / "pipeline"))
        import s3 as store
        store.download(IC_S3, d)
    torch.set_num_threads(A.THREADS)
    pre = torch.jit.load(str(d / "assets/preprocessor.ts"), map_location="cpu")
    opt = ort.SessionOptions()
    opt.intra_op_num_threads = A.THREADS
    enc, ctc = (ort.InferenceSession(str(d / f"assets/{n}.onnx"), opt, providers=["CPUExecutionProvider"]) for n in ("encoder", "ctc_decoder"))
    vocab = {t: i for i, t in enumerate(json.load(open(d / "assets/vocab.json"))["hi"][:256])}
    mask = json.load(open(d / "assets/language_masks.json"))["hi"]
    frame = 0.08

    def run(x24, words):
        x = AF.resample(x24, 24000, A.SR)
        with torch.inference_mode():
            sig, n = pre(input_signal=x[None], length=torch.tensor([len(x)]))
        out, lens = enc.run(["outputs", "encoded_lengths"], {"audio_signal": sig.numpy(), "length": n.numpy()})
        lp = torch.from_numpy(ctc.run(["logprobs"], {"encoder_output": out})[0][:, :, mask]).log_softmax(-1)[0, :int(lens[0])]
        assert abs(lp.shape[0] * frame - len(x) / A.SR) < 0.25, (lp.shape, len(x))
        toks = [bpe(w, vocab) for w in words]
        ids = [t for w in toks for t in w]
        if len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) > lp.shape[0]:
            return None
        path, sc = AF.forced_align(lp[None], torch.tensor([ids], dtype=torch.int32), blank=256)
        spans = AF.merge_tokens(path[0], sc[0].exp(), blank=256)
        res, k = [], 0
        for w in toks:
            ss = spans[k:k + len(w)]
            res.append([ss[0].start * frame, ss[-1].end * frame, sum(s.score for s in ss) / len(ss)])
            k += len(w)
        return {"frames": lp.shape[0], "regret": (lp.max(-1).values.sum().item() - sc.sum().item()) / lp.shape[0], "words": res}
    return run


ALIGNERS = {"w2v": w2v, "mms": mms, "ic": ic}


def stage(argv):
    """Stage-contract mode, so a benchmark aligner can run as a SageMaker job through pipeline/run.py (stage
    `align_bench`, params `-- --aligner ic`): hi_words from that aligner's raw word spans, its result in `_w5bench`."""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--aligner", required=True, choices=sorted(ALIGNERS))
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("records.jsonl", "rejects.jsonl"):
        p = a.out / f
        if p.exists() and (b := p.read_bytes()) and not b.endswith(b"\n"):
            p.write_bytes(b[: b.rfind(b"\n") + 1])
    done = {json.loads(l)["id"] for f in ("records.jsonl", "rejects.jsonl") if (a.out / f).exists() for l in open(a.out / f)}
    todo = [r for r in (json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()) if r["id"] not in done]
    fn = ALIGNERS[a.aligner]() if todo else None
    with open(a.out / "records.jsonl", "a", encoding="utf-8") as ok, open(a.out / "rejects.jsonl", "a", encoding="utf-8") as bad:
        for r in todo:
            x = read24(a.audio_dir / r["audio"])
            t = time.time()
            res = fn(x, [w for ws in A.words_of(r) for w in ws])
            if res is None:
                bad.write(json.dumps({"id": r["id"], "stage": "align_bench", "rule": "align_infeasible", "detail": a.aligner}) + "\n")
                continue
            A.set_times(r, [(s, e) for s, e, _ in res["words"]])
            r["_w5bench"] = {"aligner": a.aligner, "secs": round(time.time() - t, 3), "audio_s": round(len(x) / 24000, 3), **res}
            ok.write(json.dumps(r, ensure_ascii=False) + "\n")
            ok.flush()


# ---------------------------------------------------------------- conditions

def perturb(x, cond, rec, clean=None):
    """-> (audio, edit) for one condition. cut/dup edit the word span [s_i, s_{i+1}) taken from the clean run."""
    g = torch.Generator().manual_seed(int(rec["id"][4:]))
    if cond == "clean":
        return x, None
    if cond == "tel":
        y = AF.highpass_biquad(AF.resample(x, 24000, 8000), 8000, 300.0)
        return AF.resample(AF.mu_law_decoding(AF.mu_law_encoding(y.clamp(-1, 1), 256), 256), 8000, 24000), None
    if cond == "noise":
        return x + torch.randn(len(x), generator=g) * (x.pow(2).mean() / 10).sqrt(), None
    if cond == "shift":
        return torch.cat([torch.zeros(round(SHIFT * 24000)), x]), None
    # word spans of the clean run after align.fill_gaps, so a cut takes the whole word (a filler's CTC frame sits
    # mid-vowel) and the pauses around it: cut = [end of word i-1, start of word i+1); dup repeats [start_i, end_i)
    ws = A.fill_gaps(A.word_spans([tuple(s) for s in clean["spans"]], "|"), A.frame_db(AF.resample(x, 24000, A.SR)))
    words = [w for ws_ in A.words_of(rec) for w in ws_]
    cand = [i for i in range(1, len(words) - 1) if cond[-1] != "f" or words[i] in A.FILLERS]
    if not cand:
        return None, None
    i = cand[int(torch.randint(len(cand), (1,), generator=g))]
    sec = lambda f: round(f * A.FRAME * 24000)
    a, b = (sec(ws[i - 1][1]), sec(ws[i + 1][0])) if cond.startswith("cut") else (sec(ws[i][0]), sec(ws[i][1]))
    edit = {"word": i, "w": words[i], "secs": (b - a) / 24000}
    return (torch.cat([x[:a], x[b:]]) if cond.startswith("cut") else torch.cat([x[:b], x[a:b], x[b:]])), edit


def run(aligner, set_name, *conds):
    """COND@N: only the first N records (perturbed conditions on the 642-clip real set)."""
    fn = ALIGNERS[aligner]()
    adir = SETS[set_name][1]
    for cond in conds or ("clean",):
        cond, _, n = cond.partition("@")
        rs = recs(set_name)[:int(n) if n else None]
        out = W5 / "bench" / aligner / f"{set_name}.{cond}.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
        clean = {}
        if cond.startswith(("cut", "dup")):  # edit spans come from the production aligner's clean run
            clean = {json.loads(l)["id"]: json.loads(l) for l in open(W5 / "bench/w2v" / f"{set_name}.clean.jsonl")}
        with open(out, "a", encoding="utf-8") as f:
            for r in rs:
                if r["id"] in done or (clean and r["id"] not in clean):
                    continue
                x, edit = perturb(read24(adir / r["audio"]), cond, r, clean.get(r["id"]))
                if x is None:
                    continue
                t = time.time()
                res = fn(x, [w for ws in A.words_of(r) for w in ws])
                f.write(json.dumps({"id": r["id"], "secs": round(time.time() - t, 3), "audio_s": round(len(x) / 24000, 3),
                                    "edit": edit, **(res or {"fail": True})}, ensure_ascii=False) + "\n")
                f.flush()
        print(aligner, set_name, cond, "done", file=sys.stderr)


# ---------------------------------------------------------------- scoring

@functools.cache
def results(aligner, set_name, cond):
    p = W5 / "bench" / aligner / f"{set_name}.{cond}.jsonl"
    return {r["id"]: r for r in map(json.loads, open(p))} if p.exists() else {}


TRIM_DB = 15  # control truth: a word edge inside VITS's token allocation but below the clip floor + 15 dB is pause


def truth_intervals(rec, ctrl_by_id):
    """-> per word ((start_lo, start_hi), (end_lo, end_hi)) on the control set. VITS gives each character token a
    duration; a word's truth is its first..last character token, widened by the blank token next to it (an onset or
    offset may sit in either). VITS also hands pause frames to the token beside the pause (a phrase-final है "lasts"
    300 ms into silence), so edge frames that are silent in the audio are trimmed off (SPEC §3: pauses belong to
    no word). Long clips rebuild it from their parts."""
    parts = [ctrl_by_id[p] for p in rec["_w5"]["src"].split("+")] if rec["_w5"].get("long") else [rec]
    ivs, off = [], 0.0
    for r in parts:
        db, hop = fine_level(r)
        loud = db > np.percentile(db, 10) + TRIM_DB
        eb, tr = r["_w5"]["edge_blank"], []
        for k, (s, e) in enumerate(r["_w5"]["truth"]):
            a, b = int(s / hop), max(int(s / hop) + 1, int(e / hop))
            idx = np.flatnonzero(loud[a:b])
            tr.append((s, e) if not len(idx) else (max(s, (a + idx[0]) * hop), min(e, (a + idx[-1] + 1) * hop)))
        for k, ((s, e), (ts, te)) in enumerate(zip(r["_w5"]["truth"], tr)):
            lo_s = s - eb[k][0] * 0.016 if ts == s else ts
            hi_e = e + eb[k][1] * 0.016 if te == e else te
            # fluent junction (no silent frame between the words): the boundary may sit anywhere in the gap
            if k and loud[int(tr[k - 1][1] / hop):int(ts / hop)].all():
                lo_s = min(lo_s, tr[k - 1][1])
            if k + 1 < len(tr) and loud[int(te / hop):int(tr[k + 1][0] / hop)].all():
                hi_e = max(hi_e, tr[k + 1][0])
            ivs.append(((lo_s + off, ts + off), (te + off, hi_e + off)))
        off += r["duration"] + 0.4
    return ivs


_FINE = {}


def fine_level(rec):
    """dB per 5 ms hop (10 ms window) of a control clip, for trimming its truth."""
    if rec["id"] not in _FINE:
        x = read24(W5 / "ctrl" / rec["audio"]).numpy()
        n = (len(x) - 240) // 120 + 1
        fr = x[np.arange(n)[:, None] * 120 + np.arange(240)]
        _FINE[rec["id"]] = (10 * np.log10((fr ** 2).mean(1) + 1e-10), 0.005)
    return _FINE[rec["id"]]


def err(x, lo, hi):
    return 0.0 if lo <= x <= hi else min(abs(x - lo), abs(x - hi))


def pct(xs, t):
    return 100 * sum(x <= t for x in xs) / max(len(xs), 1)


def ctrl_errors(aligner, cond, timing=None):
    """-> {category: [abs boundary errors, s]} on the control set (starts and ends pooled), plus 'start'/'end' signed."""
    rs = recs("ctrl")
    by = {r["id"]: r for r in rs}
    res = results(aligner, "ctrl", cond)
    sh = SHIFT if cond == "shift" else 0.0
    out = {"all": [], "filler": [], "english": [], "hindi": [], "long": [], "start": [], "end": [], "fail": 0}
    for r in rs:
        x = res.get(r["id"])
        if x is None:
            continue
        if x.get("fail"):
            out["fail"] += 1
            continue
        ws = timing(x, r, cond) if timing else x["words"]
        for (s, e, *_), (si, ei), cat in zip(ws, truth_intervals(r, by), r["_w5"]["cat"]):
            a, b = err(s - sh, *si), err(e - sh, *ei)
            out["start"].append(s - sh - si[1])
            out["end"].append(e - sh - ei[0])
            for k in ("all", cat) + (("long",) if r["_w5"].get("long") else ()):
                out[k] += [a, b]
    return out


_DB = {}


def level(set_name, rec, cond):
    """Frame levels (align.frame_db) of the audio the aligner saw under `cond`."""
    k = (set_name, rec["id"], cond)
    if k not in _DB:
        clean = results("w2v", set_name, "clean").get(rec["id"]) if cond.startswith(("cut", "dup")) else None
        x, _ = perturb(read24(SETS[set_name][1] / rec["audio"]), cond, rec, clean)
        _DB[k] = A.frame_db(AF.resample(x, 24000, A.SR))
    return _DB[k]


def filled(rise=A.VAD_RISE, cap=A.VAD_CAP, set_name="ctrl"):
    """Timing variant: align.fill_gaps over the stored token spans."""
    def t(x, rec, cond):
        ws = A.word_spans([tuple(s) for s in x["spans"]], "|")
        return [[s * A.FRAME, e * A.FRAME] for s, e in A.fill_gaps(ws, level(set_name, rec, cond), rise, cap)]
    return t


def row(name, xs):
    if not xs:
        return f"| {name} | – | – | – | – |"
    return f"| {name} | {1000 * sum(xs) / len(xs):.1f} | {pct(xs, .05):.1f} | {pct(xs, .1):.1f} | {100 - pct(xs, .2):.2f} |"


def iv_segments(rec):
    """-> per IndicVoices segment (start, end, n_words), or None if the words can't be mapped (W4's converter, run per
    segment, must reproduce the transcript's words exactly)."""
    sys.path.insert(0, str(ROOT / "w4"))
    from convert import GLOSS, MARKS, STOP, clean, nasal, spell_acronyms
    glossed = {d: g.replace(".", "").replace(" ", "") for seg in rec["_w1"]["segments"]
               for d, g in GLOSS.findall(unicodedata.normalize("NFC", seg["normalized"]))}
    segs = [spell_acronyms([nasal(w) for w in clean(s["verbatim"], [], 0) if w != STOP and w not in MARKS], glossed)
            for s in rec["_w1"]["segments"]]
    if [w for x in segs for w in x] != [w for ws in A.words_of(rec) for w in ws]:
        return None
    return [(s["start"], s["end"], len(x)) for s, x in zip(rec["_w1"]["segments"], segs) if x]


def seg_edges(aligner, cond="clean", timing=None):
    """-> outside-segment overshoot (s) of each segment's first start and last end: >0 = word outside the human
    segment. Plus words inside the gaps between segments."""
    res, out, inside_gap, n = results(aligner, "real", cond), [], 0, 0
    for r in recs("real"):
        x, segs = res.get(r["id"]), iv_segments(r)
        if not x or x.get("fail") or not segs:
            continue
        ws = timing(x, r, cond) if timing else x["words"]
        k = 0
        for s, e, c in segs:
            out += [s - ws[k][0], ws[k + c - 1][1] - e]
            k += c
        for (s0, e0, _), (s1, e1, _) in zip(segs, segs[1:]):
            inside_gap += sum(1 for w in ws if w[0] >= e0 + 0.1 and w[1] <= s1 - 0.1)
        n += len(ws)
    return out, inside_gap, n


def consistency(aligner, set_name, cond, timing=None):
    """-> |boundary under cond - boundary on clean| (s); `shift` subtracts the known 0.37 s."""
    a, b = results(aligner, set_name, "clean"), results(aligner, set_name, cond)
    rs = {r["id"]: r for r in recs(set_name)}
    sh, out = (SHIFT if cond == "shift" else 0.0), []
    for k, y in b.items():
        x = a.get(k)
        if not x or x.get("fail") or y.get("fail"):
            continue
        wx = timing(x, rs[k], "clean") if timing else x["words"]
        wy = timing(y, rs[k], cond) if timing else y["words"]
        out += [abs(q - sh - p) for u, v in zip(wx, wy) for p, q in ((u[0], v[0]), (u[1], v[1]))]
    return out


def agreement(a1, a2, set_name, t1=None, t2=None):
    x1, x2 = results(a1, set_name, "clean"), results(a2, set_name, "clean")
    rs = {r["id"]: r for r in recs(set_name)}
    out = []
    for k in x1.keys() & x2.keys():
        if x1[k].get("fail") or x2[k].get("fail"):
            continue
        w1 = t1(x1[k], rs[k], "clean") if t1 else x1[k]["words"]
        w2 = t2(x2[k], rs[k], "clean") if t2 else x2[k]["words"]
        out += [abs(p - q) for u, v in zip(w1, w2) for p, q in ((u[0], v[0]), (u[1], v[1]))]
    return out


def qa_of(x, rec, set_name, cond):
    """The stage's QA (align.word_flags + verdict) recomputed from a stored w2v result."""
    ws = A.word_spans([tuple(t) for t in x["spans"]], "|")
    times = [(round(s * A.FRAME, 3), round(e * A.FRAME, 3)) for s, e in A.fill_gaps(ws, level(set_name, rec, cond))]
    flags = A.word_flags([w for ws_ in A.words_of(rec) for w in ws_], ws, times)
    return A.verdict({"regret": x["regret"], "flags": flags}), flags, times


def score():
    q = lambda xs: " / ".join(f"{1000 * v:+.0f}" for v in np.percentile(xs, [10, 50, 90]))
    V = [("w2v+fill (stage)", "w2v", filled), ("w2v raw CTC", "w2v", None), ("MMS-FA (NC)", "mms", None), ("IndicConformer CTC", "ic", None)]
    print("### Control set (known truth, junction-tolerant): boundary error\n")
    print("| aligner / condition / words | MAE ms | ≤50 ms % | ≤100 ms % | >200 ms % |\n|---|---|---|---|---|")
    for name, al, t in V:
        for cond in ("clean", "tel", "noise", "shift"):
            e = ctrl_errors(al, cond, t() if t else None)
            for k in (("all", "hindi", "english", "filler", "long") if cond == "clean" else ("all",)) if e["all"] else ():
                print(row(f"{name} / {cond} / {k}", e[k]))
    print("\nSigned error vs the strict (character-token) truth, p10 / p50 / p90 ms:")
    for name, al, t in V:
        e = ctrl_errors(al, "clean", t() if t else None)
        if e["start"]:
            print(f"- {name}: start {q(e['start'])}; end {q(e['end'])}")

    print("\n### Real speech: IndicVoices human segment edges (642 clips)\n")
    print("| aligner | edges | outside segment >100 ms % | >250 ms % | words inside inter-segment gaps |\n|---|---|---|---|---|")
    for name, al, t in V:
        o, g, n = seg_edges(al, "clean", t(set_name="real") if t else None)
        if o:
            o = np.array(o)
            print(f"| {name} | {len(o)} | {100 * (o > .1).mean():.2f} | {100 * (o > .25).mean():.2f} | {g} of {n} |")

    print("\n### Consistency under perturbation (|boundary - clean boundary|)\n")
    print("| aligner / set / condition | boundaries | median ms | ≤20 ms % | ≤50 ms % | >200 ms % |\n|---|---|---|---|---|---|")
    for name, al, t in V[:2]:
        for st, conds in (("real", ("shift", "tel", "noise")), ("ctrl", ("shift", "tel", "noise"))):
            for cond in conds:
                d = consistency(al, st, cond, t(set_name=st) if t else None)
                if d:
                    d = np.array(d)
                    print(f"| {name} / {st} / {cond} | {len(d)} | {1000 * np.median(d):.0f} | {100 * (d <= .02).mean():.1f} | "
                          f"{100 * (d <= .05).mean():.1f} | {100 * (d > .2).mean():.2f} |")

    print("\n### Agreement between independent aligners (clean)\n")
    print("| pair / set | boundaries | median ms | ≤50 ms % | ≤100 ms % | >200 ms % |\n|---|---|---|---|---|---|")
    for st in ("real", "syn"):
        for (n1, a1, t1), (n2, a2, t2) in ((V[0], V[2]), (V[0], V[3]), (V[2], V[3]), (V[1], V[2])):
            d = agreement(a1, a2, st, t1(set_name=st) if t1 else None, t2(set_name=st) if t2 else None)
            if d:
                d = np.array(d)
                print(f"| {n1} vs {n2} / {st} | {len(d)} | {1000 * np.median(d):.0f} | {100 * (d <= .05).mean():.1f} | "
                      f"{100 * (d <= .1).mean():.1f} | {100 * (d > .2).mean():.2f} |")

    print("\n### QA rules on clean and corrupted audio (stage thresholds)\n")
    print("| set / condition | records | rejected % | align_regret | unspoken | filler_short | edited word flagged % |\n|---|---|---|---|---|---|---|")
    for st in ("real", "syn", "ctrl"):
        rs = {r["id"]: r for r in recs(st)}
        for cond in ("clean", "cut", "cutf", "dup"):
            res = results("w2v", st, cond)
            if not res:
                continue
            v = [(x, *qa_of(x, rs[k], st, cond)) for k, x in res.items() if not x.get("fail")]
            rule = lambda r: sum(1 for _, x, _, _ in v if x == r)
            hit = [any(i == x["edit"]["word"] for i, _ in f) for x, _, f, _ in v if x.get("edit")]
            print(f"| {st} / {cond} | {len(v)} | {100 * sum(1 for _, x, _, _ in v if x) / len(v):.1f} | {rule('align_regret')} | "
                  f"{rule('unspoken')} | {rule('filler_short')} | {f'{100 * np.mean(hit):.1f}' if hit else '–'} |")

    print("\n### Filler durations after gap filling (ms, p10 / p50 / p90; share < 60 ms)\n")
    for st, cond in (("real", "clean"), ("syn", "clean"), ("ctrl", "clean"), ("real", "cutf"), ("ctrl", "cutf"), ("syn", "cutf")):
        rs, d = {r["id"]: r for r in recs(st)}, {}
        for k, x in results("w2v", st, cond).items():
            words = [w for ws in A.words_of(rs[k]) for w in ws]
            times = qa_of(x, rs[k], st, cond)[2]
            for i in ([x["edit"]["word"]] if x.get("edit") else [i for i, w in enumerate(words) if w in A.FILLERS]):
                d.setdefault(words[i], []).append(times[i][1] - times[i][0])
        print(f"- {st} / {cond}: " + "; ".join(f"{w} n={len(v)}: {' / '.join(f'{1000 * u:.0f}' for u in np.percentile(v, [10, 50, 90]))}, "
                                                 f"<60 ms {100 * np.mean(np.array(v) < .06):.0f}%" for w, v in sorted(d.items(), key=lambda z: -len(z[1]))))

    print("\n### CPU time (s per audio hour; machine shared with other jobs)\n")
    for al in ("w2v", "mms", "ic"):
        for st in ("real", "ctrl", "syn"):
            res = [x for x in results(al, st, "clean").values() if "secs" in x]
            if res:
                print(f"- {al} / {st}: {3600 * sum(x['secs'] for x in res) / sum(x['audio_s'] for x in res):.0f} s per audio hour over {len(res)} clips")


# ---------------------------------------------------------------- optional human check (one list, ~20 min)

HUMAN = W5 / "human"
QUESTIONS = {"bound": "Play the word alone. ok = whole word, nothing else | cut = part of it missing | extra = part of a neighbour audible",
             "spoken": "Play the context. Is the bold word actually spoken? y / n",
             "filler": "Play the context. Is there a filler sound (uh / umm / aa / hmm) where the bold word sits? y / n"}


def human(*stage_dirs):
    """Snippets + index.html + answers.tsv under data/w5/human/ from align stage outputs (DIR = a run's out/ folder
    with records.jsonl, rejects.jsonl, raw/; the audio dir is resolved from SETS by record id range)."""
    rnd = random.Random(14)
    bound, spoken, filler = [], [], []
    for d in map(Path, stage_dirs):
        for l in open(d / "records.jsonl", encoding="utf-8"):
            r = json.loads(l)
            for i, w in enumerate(r["hi_words"]):
                (filler if w["w"] in A.FILLERS else bound).append((r, i, None))
        for l in open(d / "rejects.jsonl", encoding="utf-8"):
            x = json.loads(l)
            if x.get("detail") and "flags" in x["detail"]:
                raw = json.loads((d / "raw" / f"{x['id']}.json").read_text())
                for i, f in x["detail"]["flags"]:
                    (spoken if f == "unspoken" else filler).append((x["id"], i, raw))
    tel = lambda r: r.get("_qa", {}).get("audio_qc", {}).get("telephone")
    pick = (rnd.sample([b for b in bound if tel(b[0])], 12) + rnd.sample([b for b in bound if tel(b[0]) is False], 12)
            + rnd.sample([b for b in bound if b[0]["source"]["synthetic"]], 6))
    items = [("bound", *b) for b in pick] + [("spoken", *s) for s in rnd.sample(spoken, min(15, len(spoken)))]
    short = [f for f in filler if f[2] is not None]  # filler_short rejects, then kept fillers up to 10
    short = rnd.sample(short, min(5, len(short)))
    items += [("filler", *f) for f in short + rnd.sample([f for f in filler if f[2] is None], 10 - len(short))]
    rnd.shuffle(items)
    allrecs = {r["id"]: r for s in ("real", "syn") for r in recs(s)}
    (HUMAN / "clips").mkdir(parents=True, exist_ok=True)
    rows, html = [], []
    for n, (kind, r, i, raw) in enumerate(items, 1):
        rec = allrecs[r["id"] if isinstance(r, dict) else r]
        words = [w for ws in A.words_of(rec) for w in ws]
        times = [(w["start"], w["end"]) for w in r["hi_words"]] if isinstance(r, dict) else \
            [(s * A.FRAME, e * A.FRAME) for s, e in raw["words"]]
        x = read24(SETS["syn" if rec["source"]["synthetic"] else "real"][1] / rec["audio"])
        a, b = (times[i][0], times[i][1]) if kind == "bound" else (max(0, times[max(0, i - 2)][0]), times[min(len(times) - 1, i + 2)][1])
        f = f"clips/{n:02d}.wav"
        write24(HUMAN / f, x[round(a * 24000):round(b * 24000)])
        ctx = " ".join(f"<b>{w}</b>" if j == i else w for j, w in enumerate(words) if abs(j - i) <= 3)
        rows.append([f"{n:02d}", kind, rec["id"], str(i), words[i], f"{times[i][0]:.3f}-{times[i][1]:.3f}", ""])
        html.append(f"<tr><td>{n:02d}</td><td>{kind}</td><td><audio controls preload=none src='{f}'></audio></td><td>{ctx}</td></tr>")
    (HUMAN / "answers.tsv").write_text("item\tkind\tid\tword_idx\tword\ttimes\tanswer\n" + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8")
    (HUMAN / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>W5 alignment check</title><style>body{font:16px sans-serif;margin:16px}"
        "td{padding:4px 8px;border-bottom:1px solid #ccc}</style><h1>W5 alignment check (optional, about 20 min)</h1><ul>"
        + "".join(f"<li><b>{k}</b>: {q}</li>" for k, q in QUESTIONS.items())
        + "</ul><p>Write each answer in the <code>answer</code> column of answers.tsv.</p><table>" + "".join(html) + "</table>",
        encoding="utf-8")
    print(f"{len(rows)} items -> {HUMAN}/index.html, answers.tsv")


if __name__ == "__main__":
    if sys.argv[1].startswith("--"):
        stage(sys.argv[1:])
    else:
        {"ctrl": ctrl, "run": run, "score": score, "human": human}[sys.argv[1]](*sys.argv[2:])
