"""CTC-ASR QA over a W1 stage manifest: (1) speech outside the speaker's transcribed segments, (2) CER vs corpus text.

    uv run --no-project --with numpy --with torch --with transformers \
        --index https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match \
        python -I w1/asr_check.py data/w1/pool/manifest.jsonl data/w1/pool/asr_check.tsv

A greedy-CTC character emitted more than TOL s outside every `_w1.segments` interval is untranscribed speech:
either a second voice (PS §3A, SPEC R11) or speech the corpus left out. CTC emits nothing on silence/noise, so
unlike Whisper it does not hallucinate there. A run of >= MIN_CHARS such characters flags the clip.
"""
import json, re, subprocess, sys, unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL = "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"  # MIT, wav2vec2-base fine-tuned on 4,200 h Hindi
TOL, MIN_CHARS = 0.15, 4
TAG = re.compile(r"\[[^\]]*\]|<[^>]*>")


def norm(t):
    t = unicodedata.normalize("NFC", TAG.sub(" ", t))
    return " ".join(re.sub(r"[^\u0900-\u0963\u0971-\u097f ]", " ", t).split())


def cer(ref, hyp):
    r, h = ref.replace(" ", ""), hyp.replace(" ", "")
    d = np.arange(len(h) + 1)
    for i, c in enumerate(r, 1):
        prev, d[0] = d.copy(), i
        for j, k in enumerate(h, 1):
            d[j] = min(prev[j] + 1, d[j - 1] + 1, prev[j - 1] + (c != k))
    return d[-1] / max(1, len(r))


def load16k(path):
    pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768


def main():
    manifest, out = Path(sys.argv[1]), Path(sys.argv[2])
    proc, model = Wav2Vec2Processor.from_pretrained(MODEL), Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    vocab = {v: k for k, v in proc.tokenizer.get_vocab().items()}
    # this checkpoint (fairseq-converted) emits <s> as its CTC blank although config.pad_token_id says <pad>:
    # treat every special token as blank
    blank, delim = {proc.tokenizer.convert_tokens_to_ids(t) for t in ("<s>", "<pad>", "</s>", "<unk>")}, proc.tokenizer.word_delimiter_token
    torch.set_num_threads(8)
    rows = []
    for line in open(manifest):
        r = json.loads(line)
        x = load16k(str(manifest.parent / r["audio"]))
        with torch.inference_mode():
            ids = model(proc(x, sampling_rate=16000, return_tensors="pt").input_values).logits[0].argmax(-1).tolist()
        step = len(x) / 16000 / len(ids)
        chars, prev = [], None  # (time, char) after CTC collapse
        for i, k in enumerate(ids):
            if k != prev and k not in blank:
                chars.append((i * step, vocab[k]))
            prev = k
        hyp = norm("".join(" " if c == delim else c for _, c in chars))
        segs = [(s["start"], s["end"]) for s in r["_w1"]["segments"]]
        runs, cur = [], []
        for t, c in chars:
            if c != delim and not any(s - TOL <= t <= e + TOL for s, e in segs):
                if cur and t - cur[-1][0] > 0.5:
                    runs.append(cur)
                    cur = []
                cur.append((t, c))
        runs += [cur] if cur else []
        flagged = [f"{u[0][0]:.1f}-{u[-1][0]:.1f}s:{''.join(c for _, c in u)}" for u in runs if len(u) >= MIN_CHARS]
        rows.append((r["id"], r["_w1"]["scenario"], round(cer(norm(r["_orig_text"]), hyp), 3), len(flagged), " | ".join(flagged), hyp))
    with open(out, "w") as f:
        f.write("id\tscenario\tcer\tn_outside_runs\toutside_runs\thyp\n")
        f.writelines("\t".join(map(str, row)) + "\n" for row in rows)
    for k in sorted({r[1] for r in rows}):
        g = [r for r in rows if r[1] == k]
        c = np.array([r[2] for r in g])
        print(f"{k}: {len(g)} clips | CER p50/p90 {np.percentile(c, 50):.3f}/{np.percentile(c, 90):.3f} | "
              f"clips with untranscribed speech run >= {MIN_CHARS} chars: {sum(r[3] > 0 for r in g)}", file=sys.stderr)


if __name__ == "__main__":
    assert cer("abc", "abd") == 1 / 3 and norm("हाँ [uhh] <noise> जी, OK") == "हाँ जी"
    main()
