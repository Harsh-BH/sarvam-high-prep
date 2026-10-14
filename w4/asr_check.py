# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "transformers", "accelerate", "onnxruntime", "numpy", "huggingface_hub", "boto3"]
# ///
"""`asr_check` pipeline stage (design C7): ASR drafts, then the P1/P2 gate of w4/review_list.check().

    uv run w4/asr_check.py --in RECORDS.jsonl --audio-dir DIR --out OUT_DIR

Input: the `transcribe` stage's records. For every record not done yet, OUT_DIR/asr/<model>.jsonl gets one
{"id","hyp","secs"} per model (vakyansh-w2v2, whisper-turbo, ic-ctc; w4/asr_run.py), appended and resumable: these
files are uploaded with the run, so the gate's input is archived and a re-run never re-decodes. Then each record goes
to OUT_DIR/records.jsonl (+ `_qa.asr_check`) or OUT_DIR/rejects.jsonl (rule P0/P1/P2). Models load only for ids that
lack a hypothesis, one at a time. GPU if present (Whisper, vakyansh); IndicConformer runs on CPU onnxruntime.
Locally: `flock data/.gpu.lock uv run w4/asr_check.py ...`. ic-ctc is gated on HF; its weights come from
data/w4/ic600m/, else HF at the pinned revision if a token is available, else the write-once S3 mirror
(w4/ic_weights.py) - the path a SageMaker job takes, since a job never gets an HF token.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from review_list import MODELS, check  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    (a.out / "asr").mkdir(parents=True, exist_ok=True)
    for f in ("records.jsonl", "rejects.jsonl"):
        p = a.out / f
        if p.exists() and (b := p.read_bytes()) and not b.endswith(b"\n"):
            p.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: drop it, that id is redone
    done = {json.loads(l)["id"] for f in ("records.jsonl", "rejects.jsonl") if (a.out / f).exists() for l in open(a.out / f)}
    todo = [r for r in (json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()) if r["id"] not in done]

    def hyps(m):
        p = a.out / "asr" / f"{m}.jsonl"
        return {json.loads(l)["id"]: json.loads(l)["hyp"] for l in open(p) if l.endswith("\n")} if p.exists() else {}
    for m in MODELS:
        have = hyps(m)
        if any(r["id"] not in have for r in todo):
            from asr_run import hypotheses  # torch etc. only when something must be decoded
            hypotheses(m, todo, a.audio_dir, a.out / "asr" / f"{m}.jsonl")
    h = {m: hyps(m) for m in MODELS}
    with open(a.out / "records.jsonl", "a", encoding="utf-8") as ok, open(a.out / "rejects.jsonl", "a", encoding="utf-8") as bad:
        for r in todo:
            rule, detail = check(r, {m: h[m][r["id"]] for m in MODELS})
            if rule:
                bad.write(json.dumps({"id": r["id"], "stage": "asr_check", "rule": rule, "detail": detail}, ensure_ascii=False) + "\n")
            else:
                r.setdefault("_qa", {})["asr_check"] = detail
                ok.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
