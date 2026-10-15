# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Run SeamlessStreaming (base or base + a fine-tuned decoder) on the eval sets and score it, as one SageMaker job.

    uv run w8c/sm_eval.py --sets stages/w8c_sets/<run> [--only dev,test,fleurs] [--ckpt s3://.../decoder.pt]
                          [--name base] [--limit N] [--instance ml.g5.xlarge] [--spot]

Outputs: stages/w8c_eval/<run>/<set>/{<id>.json,<id>.wav,asr.jsonl,per_record.jsonl,scores.json} + run.json.
Needs a committed, clean worktree (the job runs the code at HEAD, like pipeline/sm.py).
Image: PyTorch 2.2.0 / py3.10 (fairseq2 0.2.1 has no wheel for the repo's py3.12 / torch 2.9 image); the fairseq2n
build for that torch comes from Meta's index (env verified by w8d). Inside the job this same file is the entry point.
"""
import argparse, json, os, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "pipeline"))
IMAGE = "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker"
SEAMLESS = "https://github.com/facebookresearch/seamless_communication/archive/9a081e935c29c6b1e0e05bcbf3fcb623ad306524.tar.gz"
FAIRSEQ2 = ["--extra-index-url", "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.2.0/cu121",
            "fairseq2n==0.2.1+cu121", "fairseq2==0.2.1", "simuleval==1.1.4", "soundfile"]
SCORING = ["sacrebleu==2.4.3", "unbabel-comet==2.2.4", "transformers==4.46.3", "boto3"]
CKPT_DIR = Path("/opt/ml/checkpoints")  # synced to S3: a spot restart resumes per record


def args(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", required=True)
    ap.add_argument("--only", default="dev,test,fleurs")
    ap.add_argument("--ckpt")
    ap.add_argument("--name", default="base")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--instance", default="ml.g5.xlarge")
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--run")
    return ap.parse_args(argv)


def submit(argv):
    import run, sm
    a = args(argv)
    rid = a.run or run.run_id(run.git_sha(), datetime.now(timezone.utc))
    params = [x for x in argv if x not in ("--spot",)]
    base = sm.job_request

    def request(*x, **k):
        req = base(*x, **k)
        req["AlgorithmSpecification"]["TrainingImage"] = IMAGE
        req["HyperParameters"]["sagemaker_program"] = "w8c/sm_eval.py"
        return req
    sm.job_request = request  # sm.submit does the dirty check, code tarball, submit and wait
    rc = sm.submit("w8c_eval", a.sets, a.sets, params, run=rid, instance=a.instance, spot=a.spot)
    print(f"stages/w8c_eval/{rid}: exit {rc}")
    return rc


def sh(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def entry():
    import s3 as store
    hp = json.loads(Path("/opt/ml/input/config/hyperparameters.json").read_text())
    a = args(json.loads(hp["params"]))
    py = sys.executable
    import torch, torchaudio
    pins = CKPT_DIR.parent / "pins.txt"
    pins.write_text(f"torch=={torch.__version__}\ntorchaudio=={torchaudio.__version__}\nnumpy<2\n")
    sh(py, "-m", "pip", "install", "-q", "-c", pins, *FAIRSEQ2, *SCORING)
    sh(py, "-m", "pip", "install", "-q", "--no-deps", SEAMLESS)
    sh("bash", "-c", "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; pip list 2>/dev/null | grep -i -E 'torch|fairseq2|simuleval|comet|transformers|numpy'")
    sets = Path("/opt/ml/input/data/sets")
    store.download(a.sets.rstrip("/") + "/", sets)
    ckpt = None
    if a.ckpt:
        ckpt = Path("/opt/ml/input/data/ckpt") / Path(a.ckpt).name
        if a.ckpt.startswith("s3://"):
            key = a.ckpt.split("/", 3)[3]
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            ckpt.write_bytes(store.get(key))
        else:
            ckpt = Path(a.ckpt)
    prefix = f"stages/w8c_eval/{hp['run']}"
    summary = {}
    for s in a.only.split(","):
        out = CKPT_DIR / s
        cmd = [py, HERE / "run_seamless.py", "--set", sets / f"{s}.jsonl", "--audio-dir", sets, "--out", out]
        sh(*cmd, *(["--ckpt", ckpt] if ckpt else []), *(["--limit", a.limit] if a.limit else []))
    for s in a.only.split(","):  # scoring after all inference: one model family on the GPU at a time
        out = CKPT_DIR / s
        sh(py, HERE / "eval.py", "score", "--set", sets / f"{s}.jsonl", "--sys", out)
        summary[s] = json.loads((out / "scores.json").read_text())
        store.upload_dir(out, f"{prefix}/{s}")
    meta = {"stage": "w8c_eval", "run": hp["run"], "sets": a.sets, "only": a.only, "ckpt": a.ckpt, "name": a.name,
            "limit": a.limit, "instance": os.environ.get("SM_CURRENT_INSTANCE_TYPE"), "image": IMAGE, "seamless": SEAMLESS,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scores": summary}
    store.put(f"{prefix}/run.json", json.dumps(meta, indent=1).encode())
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    if Path("/opt/ml/input/config/hyperparameters.json").exists():
        entry()
    else:
        sys.exit(submit(sys.argv[1:]))
