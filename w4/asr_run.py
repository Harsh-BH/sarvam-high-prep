"""Run one ASR model over the first N clips of a W1 stage manifest; write raw hypotheses (no normalisation).

    flock data/.gpu.lock uv run --no-project --with torch --with transformers --with accelerate \
        python -I w4/asr_run.py MODEL [N] [SET]   # MODEL: whisper-v3 | whisper-turbo | mms | ic-ctc | ic-rnnt | vakyansh-w2v2
    (ic-*: add --with onnxruntime --with boto3; weights via w4/ic_weights.py)

The `asr_check` pipeline stage (w4/asr_check.py) calls `hypotheses()` directly.

SET: pool (W1's pinned-ffmpeg pool, data/w1/pool_pinned/records.jsonl, default) or ivvalid (data/w4/ivvalid).
Output: data/w4/asr/[<SET>/]<MODEL>.jsonl, one {"id", "hyp", "secs"} per clip, appended, so a killed run resumes.
Scoring is w4/asr_score.py. Weights go to data/w4/hf (W4's disk budget), except models already in the shared cache.
"""
import gc, json, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / "data/w1/pool_pinned"
OUT = ROOT / "data/w4/asr"
CACHE = str(ROOT / "data/w4/hf")
HF = {"whisper-v3": "openai/whisper-large-v3",  # Apache-2.0
      "whisper-turbo": "openai/whisper-large-v3-turbo",  # MIT; already in the shared cache (W1)
      "mms": "facebook/mms-1b-all",  # CC BY-NC 4.0: benchmark only, never a dataset tool (research T3)
      "ic-ctc": "ai4bharat/indic-conformer-600m-multilingual",  # MIT; gated
      "ic-rnnt": "ai4bharat/indic-conformer-600m-multilingual",
      "vakyansh-w2v2": "Harveenchadha/vakyansh-wav2vec2-hindi-him-4200"}  # MIT; W1's QA model
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HALF = torch.float16 if DEVICE == "cuda" else torch.float32


def ffmpeg():
    """PATH ffmpeg, else the pipeline's pinned one (SageMaker images may have none). ASR input only: the gate's
    hypotheses are archived with each stage run, so the resampler's exact bytes never matter for a rebuild."""
    if exe := shutil.which("ffmpeg"):
        return exe
    sys.path.insert(0, str(ROOT / "pipeline"))
    import ffmpeg as pinned
    return str(pinned.binary())


def load16k(path):  # same as w1/asr_check.py
    pcm = subprocess.run([ffmpeg(), "-nostdin", "-v", "error", "-i", path, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768


def whisper(name):
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    cache = None if name == "whisper-turbo" else CACHE
    proc = WhisperProcessor.from_pretrained(HF[name], cache_dir=cache)
    model = WhisperForConditionalGeneration.from_pretrained(HF[name], cache_dir=cache, torch_dtype=HALF,
                                                            device_map=DEVICE).eval()

    def run(x):
        # > 30 s clips need the sequential long-form path: no truncation, timestamps on
        f = proc(x, sampling_rate=16000, return_tensors="pt", truncation=False, padding="longest",
                 return_attention_mask=True)
        if f.input_features.shape[-1] < 3000:  # short-form needs exactly 30 s of features
            f = proc(x, sampling_rate=16000, return_tensors="pt", return_attention_mask=True)
        ids = model.generate(f.input_features.to(DEVICE, HALF), attention_mask=f.attention_mask.to(DEVICE),
                             language="hi", task="transcribe", num_beams=1, return_timestamps=True,
                             condition_on_prev_tokens=False)
        return proc.batch_decode(ids, skip_special_tokens=True)[0]
    return run


def mms(name):
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    proc = AutoProcessor.from_pretrained(HF[name], cache_dir=CACHE, target_lang="hin")
    model = Wav2Vec2ForCTC.from_pretrained(HF[name], cache_dir=CACHE, target_lang="hin", ignore_mismatched_sizes=True,
                                           torch_dtype=HALF).to(DEVICE).eval()

    def run(x):
        v = proc(x, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE, HALF)
        return proc.decode(model(v).logits[0].argmax(-1))
    return run


def indicconformer(name):
    # HF wrapper: ONNX encoder (CPU onnxruntime here; it asks for CUDA and falls back), TorchScript preprocessor.
    # The ".nemo" repo is the same multisoftmax model and needs the AI4Bharat NeMo fork, so it isn't used.
    import importlib.util
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # `python -I` leaves the script's folder off sys.path
    import ic_weights  # local copy, else HF at the pinned revision (token), else the write-once S3 mirror (jobs)
    path = ic_weights.LOCAL  # real files: onnxruntime rejects external data reached through cache symlinks
    print("ic weights:", ic_weights.ensure(path), file=sys.stderr)
    spec = importlib.util.spec_from_file_location("model_onnx", f"{path}/model_onnx.py")  # = trust_remote_code
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    cfg = {k: v for k, v in json.load(open(f"{path}/config.json")).items() if k != "auto_map"}
    model = mod.IndicASRModel(mod.IndicASRConfig(ts_folder=path, **cfg))  # its from_pretrained ignores cache_dir
    torch.set_num_threads(8)

    def run(x):
        return model(torch.from_numpy(x)[None], "hi", name.split("-")[1])
    return run


def vakyansh(name):
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    proc = Wav2Vec2Processor.from_pretrained(HF[name], cache_dir=CACHE)
    model = Wav2Vec2ForCTC.from_pretrained(HF[name], cache_dir=CACHE).to(DEVICE).eval()
    vocab = {v: k for k, v in proc.tokenizer.get_vocab().items()}
    # as w1/asr_check.py: this fairseq-converted checkpoint emits <s> as its CTC blank, so every special token is blank
    blank = {proc.tokenizer.convert_tokens_to_ids(t) for t in ("<s>", "<pad>", "</s>", "<unk>")}

    def run(x):
        ids = model(proc(x, sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)).logits[0].argmax(-1).tolist()
        out = [vocab[k] for k, prev in zip(ids, [None] + ids[:-1]) if k != prev and k not in blank]
        return " ".join("".join(out).replace(proc.tokenizer.word_delimiter_token, " ").split())
    return run


LOADERS = {"whisper-v3": whisper, "whisper-turbo": whisper, "mms": mms, "ic-ctc": indicconformer, "ic-rnnt": indicconformer,
           "vakyansh-w2v2": vakyansh}


def hypotheses(name, recs, audio_dir, out):
    """Append {"id", "hyp", "secs"} to `out` for every record not in it yet; a killed run resumes. The model is
    loaded only if something is missing, and freed afterwards (one model on the GPU at a time)."""
    out = Path(out)
    if out.exists() and (b := out.read_bytes()) and not b.endswith(b"\n"):
        out.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: redo that id
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [r for r in recs if r["id"] not in done]
    if not todo:
        return
    run = LOADERS[name](name)
    with open(out, "a") as f, torch.inference_mode():
        for r in todo:
            t = time.time()
            hyp = run(load16k(str(Path(audio_dir) / r["audio"])))
            f.write(json.dumps({"id": r["id"], "hyp": hyp, "secs": round(time.time() - t, 2)}, ensure_ascii=False) + "\n")
            f.flush()
    del run
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


def main():
    name, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 174
    s = sys.argv[3] if len(sys.argv) > 3 else "pool"
    pool, out = (POOL, OUT / f"{name}.jsonl") if s == "pool" else (ROOT / f"data/w4/{s}", OUT / s / f"{name}.jsonl")
    out.parent.mkdir(exist_ok=True)
    man = pool / ("records.jsonl" if s == "pool" else "manifest.jsonl")
    hypotheses(name, [json.loads(l) for l in list(open(man))[:n]], pool, out)
    print(name, "done", out)


if __name__ == "__main__":
    main()
