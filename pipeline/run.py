# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Run one pipeline stage locally or on SageMaker (design §4.1). LATEST moves only on a clean, validated run.

    uv run pipeline/run.py <stage> --in <stage>|stages/<stage>/<run> [--audio-from ...] [--local | --sm INSTANCE [--spot]] [-- params]
"""
import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import s3 as store

ROOT = Path(__file__).resolve().parent.parent


def _main_checkout():
    """The repo's main checkout (shared by every worktree), or ROOT when there's no git (a SageMaker job)."""
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True)
    return Path(r.stdout.strip()).parent if r.returncode == 0 else ROOT


CACHE = _main_checkout() / "data/cache"  # one cache for all worktrees, so a session never re-downloads what's here
WHERE = "sagemaker" if Path("/opt/ml/input/config").exists() else "local"
VALIDATE = ["uv", "run", str(ROOT / "validate.py")]
STAGES = {
    "audio_qc": {"cmd": ["uv", "run", str(ROOT / "pipeline/audio_qc.py")], "makes_audio": False, "gpu": False},
    "dedupe": {"cmd": ["uv", "run", str(ROOT / "pipeline/dedupe.py")], "makes_audio": False, "gpu": False},
    "transcribe": {"cmd": ["uv", "run", str(ROOT / "w4/convert.py")], "makes_audio": False, "gpu": False},
    "asr_check": {"cmd": ["uv", "run", str(ROOT / "w4/asr_check.py")], "makes_audio": False, "gpu": True},
    "align": {"cmd": ["uv", "run", str(ROOT / "w5/align.py")], "makes_audio": False, "gpu": False},
    "align_bench": {"cmd": ["uv", "run", str(ROOT / "w5/bench.py")], "makes_audio": False, "gpu": False},
    "translate": {"cmd": ["uv", "run", str(ROOT / "w6/translate.py")], "makes_audio": False, "gpu": False},
    "rebuild": {"cmd": ["uv", "run", str(ROOT / "w7/rebuild_stage.py")], "makes_audio": False, "gpu": False},
    "tts": {"cmd": ["uv", "run", str(ROOT / "w3/tts.py")], "makes_audio": True, "gpu": True},
    "tts_bench": {"cmd": ["uv", "run", str(ROOT / "w3/tts_bench.py")], "makes_audio": True, "gpu": True},
}


def run_id(git_sha, now):
    return f"{now:%Y%m%dT%H%M%SZ}-{git_sha}"


def git_sha():
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def latest(stage, s3=None):
    b = store.get(f"stages/{stage}/LATEST", s3)
    return b.decode().strip() if b else None


def resolve(ref, s3=None):
    """'<stage>' or 'stages/<stage>/<run>' -> 'stages/<stage>/<run>'."""
    if ref.startswith("stages/"):
        return ref.rstrip("/")
    r = latest(ref, s3)
    if r is None:
        raise SystemExit(f"no LATEST run for stage {ref!r}")
    return f"stages/{ref}/{r}"


def audio_of(src, s3=None):
    """Default --audio-from: `src` itself if its stage makes audio, else the audio `src` read (its run.json), else cut's LATEST."""
    if STAGES.get(src.split("/")[1], {}).get("makes_audio"):
        return src
    meta = store.get(f"{src}/run.json", s3)
    return json.loads(meta)["audio_from"].rstrip("/").removesuffix("/audio") if meta else resolve("cut", s3)


def _count(p):
    return sum(1 for l in open(p) if l.strip()) if p.exists() else 0


def run_local(stage, spec, in_records, audio_prefix, params, *, run, cache=CACHE, audio_cache=None, s3=None, validate_cmd=VALIDATE):
    work = Path(cache) / stage / run
    inp, out = work / "in" / "records.jsonl", work / "out"  # out/ survives a crash: the stage resumes into it
    inp.parent.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    args, pinned = {"input": in_records, "audio_from": audio_prefix, "params": params}, work / "args.json"
    if pinned.exists() and json.loads(pinned.read_text()) != args:
        raise SystemExit(f"stages/{stage}/{run} was started with {pinned.read_text()}; resume it with those, or start a new run")
    pinned.write_text(json.dumps(args))
    for p in (out / "records.jsonl", out / "rejects.jsonl"):
        if p.exists() and not (b := p.read_bytes()).endswith(b"\n"):
            p.write_bytes(b[:b.rfind(b"\n") + 1])  # a crash tore the last line: drop it, the stage redoes that id
    inp.write_bytes(store.get(in_records, s3))
    # records say audio="audio/<id>.wav" (SPEC §5) and stages + validate.py resolve it against --audio-dir
    audio = Path(audio_cache or Path(cache) / "audio") / audio_prefix.strip("/").replace("/", "_")
    store.download(audio_prefix, audio / "audio", s3)  # shared cache across runs; skips files already present
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rc = subprocess.run(spec["cmd"] + ["--in", str(inp), "--audio-dir", str(audio), "--out", str(out)] + params).returncode
    violations = None
    if rc == 0:
        v = subprocess.run(validate_cmd + [str(out / "records.jsonl"), "--audio-dir", str(out if spec["makes_audio"] else audio), "--partial"],
                           capture_output=True, text=True)
        # a validator that dies before printing (traceback on stderr) still blocks LATEST
        violations = max(1, sum(1 for l in v.stdout.splitlines() if l.strip())) if v.returncode else 0
        (out / "validate.txt").write_text(v.stdout + v.stderr)
    prefix = f"stages/{stage}/{run}"
    store.upload_dir(out, prefix, s3)
    meta = {"stage": stage, "run": run, "input": in_records, "audio_from": audio_prefix, "params": params, "where": WHERE,
            "started": started, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "exit": rc,
            "violations": violations, "records": _count(out / "records.jsonl"), "rejects": _count(out / "rejects.jsonl")}
    store.put(f"{prefix}/run.json", json.dumps(meta, indent=1).encode(), s3)
    if rc != 0:
        return 1
    if violations:
        return 2
    # ponytail: one LATEST per stage, last writer wins; a pilot/experiment run moves it too. Point experiments at
    # explicit stages/<stage>/<run> paths and restore LATEST after, or add per-dataset pointers if this bites.
    store.put(f"stages/{stage}/LATEST", run.encode(), s3)
    return 0


def main():
    argv = sys.argv[1:]
    params = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=sorted(STAGES))
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--audio-from")
    ap.add_argument("--run", help="resume this run id instead of starting a new one")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--local", action="store_true")
    g.add_argument("--sm", metavar="INSTANCE")
    ap.add_argument("--spot", action="store_true")
    a = ap.parse_args(argv)
    spec = STAGES[a.stage]
    src = resolve(a.inp)
    audio_src = resolve(a.audio_from) if a.audio_from else audio_of(src)
    run = a.run or run_id(git_sha(), datetime.now(timezone.utc))
    if a.sm:
        import sm  # Task 4
        sys.exit(sm.submit(a.stage, src, audio_src, params, run=run, instance=a.sm, spot=a.spot))
    rc = run_local(a.stage, spec, f"{src}/records.jsonl", f"{audio_src}/audio/", params, run=run)
    print(f"stages/{a.stage}/{run}: " + {0: "ok, LATEST moved", 1: "stage failed", 2: "validation failed"}[rc], file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
