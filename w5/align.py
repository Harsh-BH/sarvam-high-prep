# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "torchaudio", "transformers", "numpy", "huggingface_hub"]
# ///
"""`align` pipeline stage (W5): start/end of every transcript word by CTC forced alignment, on CPU.

    uv run w5/align.py --in RECORDS.jsonl --audio-dir DIR --out OUT_DIR

Model: vakyansh wav2vec2 Hindi (MIT; Devanagari characters, 20 ms frames), pinned revision, fixed thread count.
Emissions come from 10 s windows (8 s kept, 1 s context per side; the model degrades on long inputs), stitched
frame-exactly; one Viterbi pass (torchaudio `forced_align`) then covers the whole clip. The target is the transcript's
characters with the model's word delimiter `|` between words, so every word owns >= 1 frame and words never overlap.

Writes per record `hi_words`, `sentences[].hi_start/hi_end`, `_qa.align` and `_archive.align` (= `raw/<id>.json`
in this run folder: the Viterbi token spans with their frame posteriors, and the greedy CTC hypothesis). The rebuild
copies these timings; it never re-aligns. Records that fail a QA rule (RULES) go to rejects.jsonl (design §3.3:
drop, never review).
"""
import argparse, json, wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as AF

MODEL, REV = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200", "e2568c3f7868d8aa3aaabcf28fa100d10d54c170"
SR, HOP, RF, FRAME = 16000, 320, 400, 0.02  # wav2vec2 conv stack: frame k reads samples [320k, 320k + 400)
WIN, CTX = 400, 50  # frames kept per window, context frames on each side
THREADS = 8  # fixed: float reduction order, hence the emissions, can depend on the thread count
FINAL = "।?!"
FILLERS = {"अ", "अं", "आ", "ए", "उम्म", "हम्म"}  # SPEC R9
VAD_RISE, VAD_CAP = 10, 5  # fill_gaps: dB above the clip floor; frames (100 ms). Picked on the W5 control set
UNSPOKEN_POST, FILLER_MIN, REGRET_MAX = 0.01, 0.06, 0.5  # QA (word_flags, verdict); see tasks/W5-alignment.md
REJECT_FLAGS = {"unspoken", "filler_short"}


def words_of(rec):
    """-> per sentence, its words (SPEC R4: final mark stripped, split on single spaces)."""
    return [(t[:-1] if t[-1] in FINAL else t).split(" ") for t in (s["hi_text"] for s in rec["sentences"])]


def load(path):
    """WAV -> float32 tensor at 16 kHz."""
    with wave.open(str(path)) as w:
        sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2")
    x = torch.from_numpy(x.astype(np.float32) / 32768)
    return AF.resample(x, sr, SR) if sr != SR else x


class W2V:
    """Emissions: (T, V) log-probs, column 0 = blank. The checkpoint spreads blank mass over its four special tokens
    (<s> <pad> </s> <unk>), so they are summed into one blank column."""

    def __init__(self):
        from huggingface_hub import hf_hub_download
        from transformers import Wav2Vec2ForCTC
        torch.set_num_threads(THREADS)
        torch.use_deterministic_algorithms(True)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL, revision=REV).eval()
        vocab = json.loads(Path(hf_hub_download(MODEL, "vocab.json", revision=REV)).read_text())
        self.tokens = ["<b>"] + sorted(vocab, key=vocab.get)[4:]
        assert sorted(vocab.values()) == list(range(len(vocab))) and self.tokens[1] == "|"
        self.index = {t: i for i, t in enumerate(self.tokens)}

    def __call__(self, x):
        n = (len(x) - RF) // HOP + 1
        out = []
        for f in range(0, n, WIN):
            a, b = max(0, f - CTX), min(n, f + WIN + CTX)
            seg = x[a * HOP:(b - 1) * HOP + RF]  # exactly frames a..b-1
            seg = (seg - seg.mean()) / torch.sqrt(seg.var(unbiased=False) + 1e-7)  # the processor's do_normalize
            with torch.inference_mode():
                lp = self.model(seg[None]).logits[0].float().log_softmax(-1)
            out.append(lp[f - a:f - a + min(WIN, n - f)])
        lp = torch.cat(out)
        return torch.cat([lp[:, :4].logsumexp(-1, keepdim=True), lp[:, 4:]], 1)


def viterbi(lp, words, index, sep="|"):
    """-> (token spans [(token, start_frame, end_frame_excl, mean_posterior)] incl. one `sep` span between words,
    log-prob of the whole path), or None if the transcript can't fit the audio (CTC needs a frame per token plus one
    per repeated pair)."""
    ids = []
    for i, w in enumerate(words):
        ids += ([index[sep]] if i else []) + [index[c] for c in w]
    if len(ids) + sum(a == b for a, b in zip(ids, ids[1:])) > lp.shape[0]:
        return None
    path, scores = AF.forced_align(lp[None], torch.tensor([ids], dtype=torch.int32), blank=0)
    spans = AF.merge_tokens(path[0], scores[0].exp())
    assert [s.token for s in spans] == ids
    return [(int(s.token), s.start, s.end, float(s.score)) for s in spans], scores.sum().item()


def word_spans(spans, sep_id):
    """Token spans -> per word (start_frame, end_frame_excl, mean char posterior, max char posterior, n chars)."""
    out, cur = [], []
    for s in list(spans) + [(sep_id, 0, 0, 0.0)]:
        if s[0] == sep_id:
            out.append((cur[0][1], cur[-1][2], sum(c[3] for c in cur) / len(cur), max(c[3] for c in cur), len(cur)))
            cur = []
        else:
            cur.append(s)
    return out


def frame_db(x):
    """Level (dB) of the 16 kHz samples each CTC frame reads."""
    n = (len(x) - RF) // HOP + 1
    return 10 * torch.log10(x[torch.arange(n)[:, None] * HOP + torch.arange(RF)].pow(2).mean(1) + 1e-10)


def fill_gaps(ws, db, rise=VAD_RISE, cap=VAD_CAP):
    """CTC puts each character on one peaky frame, so a long vowel (phrase-final है, vowel-initial और, the filler अ)
    leaves the frames around it to blank. Hand the frames between two words back to them unless the audio is silent
    there: a gap with no silent frame is split in the middle; otherwise the left word's end and the right word's
    start move through non-silent frames up to the silence, at most `cap` frames each (also before the first and
    after the last word). Silent = below the clip's 10th-percentile level + `rise` dB. Pauses stay outside every
    word (SPEC §3). ws: [(start_frame, end_frame_excl, ...)] -> [[start, end]]."""
    speech = (db > torch.quantile(db, 0.1) + rise).tolist()
    out = [[w[0], w[1]] for w in ws]
    for i in range(len(out) + 1):
        lo, hi = out[i - 1][1] if i else 0, out[i][0] if i < len(out) else len(speech)
        if 0 < i < len(out) and all(speech[lo:hi]):
            out[i - 1][1] = out[i][0] = (lo + hi) // 2
            continue
        a, b = lo, hi
        while i and a < hi and speech[a] and a - lo < cap:
            a += 1
        while i < len(out) and b > a and speech[b - 1] and hi - b < cap:
            b -= 1
        if i:
            out[i - 1][1] = a
        if i < len(out):
            out[i][0] = b
    return out


def greedy(lp, tokens):
    ids = lp.argmax(-1).tolist()
    text = "".join(tokens[k] for k, p in zip(ids, [None] + ids[:-1]) if k != p and k != 0)
    return " ".join(text.replace("|", " ").split())


def finalize(times, dur):
    """Seconds, 3 decimals, 0 <= s_i < e_i <= s_{i+1}, last e <= dur. Inputs on the 20 ms grid already satisfy it;
    this only clamps the clip end and guards the invariant."""
    out = []
    for s, e in times:
        s, e = round(max(0.0, s), 3), round(min(dur, e), 3)
        if out and s < out[-1][1]:
            s = out[-1][1]
        if e <= s:
            raise ValueError(f"empty word span {s}-{e}")
        out.append((s, e))
    return out


def set_times(rec, times):
    """Word times (s) in transcript order -> rec's hi_words and sentences[].hi_start/hi_end (SPEC §3)."""
    sents = words_of(rec)
    times = finalize(times, rec["duration"])
    rec["hi_words"] = [{"w": w, "start": s, "end": e} for w, (s, e) in zip((w for ws in sents for w in ws), times)]
    j = 0
    for k, (sen, sw) in enumerate(zip(rec["sentences"], sents)):
        rest = {x: v for x, v in sen.items() if x not in ("hi_text", "hi_start", "hi_end")}
        rec["sentences"][k] = {"hi_text": sen["hi_text"], "hi_start": times[j][0], "hi_end": times[j + len(sw) - 1][1], **rest}
        j += len(sw)


def word_flags(words, ws, times):
    """QA flags per word: `unspoken` = a non-filler word whose every character has posterior < UNSPOKEN_POST and
    that Viterbi squeezed to at most one spare frame (a transcript word the audio lacks); `filler_short` = a filler
    shorter than FILLER_MIN s after gap filling. Fillers get no posterior test: the model never emits them."""
    flags = []
    for i, (w, (s, e, _, top, n), (t0, t1)) in enumerate(zip(words, ws, times)):
        if w in FILLERS and t1 - t0 < FILLER_MIN:
            flags.append([i, "filler_short"])
        elif w not in FILLERS and top < UNSPOKEN_POST and e - s <= n + 1:
            flags.append([i, "unspoken"])
    return flags


def align(rec, lp, db, tokens, index):
    """-> (record with timings, raw dict, qa dict) or (None, reject dict)."""
    words = [w for ws in words_of(rec) for w in ws]
    res = viterbi(lp, words, index)
    if res is None:
        return None, {"rule": "align_infeasible", "detail": {"tokens": sum(map(len, words)) + len(words) - 1, "frames": lp.shape[0]}}
    spans, forced = res
    ws = word_spans(spans, index["|"])
    filled = fill_gaps(ws, db)
    set_times(rec, [(s * FRAME, e * FRAME) for s, e in filled])
    times = [(x["start"], x["end"]) for x in rec["hi_words"]]
    best = lp.max(-1).values.sum().item()  # greedy path: forced == best iff the transcript is what the model hears
    qa = {"regret": round((best - forced) / lp.shape[0], 4), "flags": word_flags(words, ws, times)}
    raw = {"model": MODEL, "rev": REV, "frame": FRAME, "frames": lp.shape[0], "fill": [VAD_RISE, VAD_CAP],
           "greedy": greedy(lp, tokens), "spans": [[tokens[t], s, e, round(c, 4)] for t, s, e, c in spans],
           "words": filled}
    return rec, raw, qa


def verdict(qa):
    """-> reject rule or None (design §3.3: drop, never review)."""
    if qa["regret"] > REGRET_MAX:
        return "align_regret"
    return next((f for _, f in qa["flags"] if f in REJECT_FLAGS), None)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (a.out / "raw").mkdir(parents=True, exist_ok=True)
    for f in ("records.jsonl", "rejects.jsonl"):
        p = a.out / f
        if p.exists() and (b := p.read_bytes()) and not b.endswith(b"\n"):
            p.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: drop it, that id is redone
    done = {json.loads(l)["id"] for f in ("records.jsonl", "rejects.jsonl") if (a.out / f).exists() for l in open(a.out / f)}
    todo = [r for r in (json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()) if r["id"] not in done]
    if not todo:
        return
    m = W2V()
    with open(a.out / "records.jsonl", "a", encoding="utf-8") as ok, open(a.out / "rejects.jsonl", "a", encoding="utf-8") as bad:
        for r in todo:
            x = load(a.audio_dir / r["audio"])
            lp, db = m(x), frame_db(x)
            assert len(db) == lp.shape[0]
            res = align(r, lp, db, m.tokens, m.index)
            if res[0] is None:
                bad.write(json.dumps({"id": r["id"], "stage": "align", **res[1]}, ensure_ascii=False) + "\n")
                continue
            rec, raw, qa = res
            (a.out / "raw" / f"{r['id']}.json").write_text(json.dumps(raw, ensure_ascii=False))
            if rule := verdict(qa):
                bad.write(json.dumps({"id": r["id"], "stage": "align", "rule": rule, "detail": qa}, ensure_ascii=False) + "\n")
                continue
            rec.setdefault("_qa", {})["align"] = qa
            rec.setdefault("_archive", {})["align"] = f"raw/{r['id']}.json"
            ok.write(json.dumps(rec, ensure_ascii=False) + "\n")
            ok.flush()


if __name__ == "__main__":
    main()
