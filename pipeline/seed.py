# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Seed a stage run in S3 from a local stage-layout folder, so data made outside run.py can enter the pipeline.

    uv run pipeline/seed.py <stage> <dir> [--audio-from stages/<stage>/<run> --audio-dir LOCAL] [--note TEXT]

<dir> holds records.jsonl (+ rejects.jsonl, + audio/<id>.wav if this data owns its audio). Validates with
validate.py --partial first; uploads, writes run.json (audio_from = its own audio/, or --audio-from), moves LATEST.
"""
import argparse, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import run, s3 as store


def seed(stage, d, *, audio_from=None, audio_dir=None, note="", s3=None, validate_cmd=run.VALIDATE, git_sha=None):
    d = Path(d)
    own = (d / "audio").is_dir()
    if not own and not (audio_from and audio_dir):
        raise SystemExit("no audio/ in the folder: pass --audio-from (S3 run) and --audio-dir (local copy to validate against)")
    v = subprocess.run(validate_cmd + [str(d / "records.jsonl"), "--audio-dir", str(d if own else audio_dir), "--partial"],
                       capture_output=True, text=True)
    if v.returncode:
        print(v.stdout[-2000:] + v.stderr[-500:], file=sys.stderr)
        raise SystemExit("validate.py failed; nothing seeded")
    rid = run.run_id(git_sha or run.git_sha(), datetime.now(timezone.utc))
    prefix = f"stages/{stage}/{rid}"
    store.upload_dir(d, prefix, s3)
    count = lambda f: sum(1 for l in open(d / f) if l.strip()) if (d / f).exists() else 0
    meta = {"stage": stage, "run": rid, "input": f"seed: {d}" + (f" ({note})" if note else ""), "params": [], "where": "local",
            "audio_from": f"{prefix}/audio/" if own else f"{audio_from.rstrip('/').removesuffix('/audio')}/audio/",
            "records": count("records.jsonl"), "rejects": count("rejects.jsonl"), "violations": 0, "exit": 0}
    store.put(f"{prefix}/run.json", json.dumps(meta, indent=1).encode(), s3)
    store.put(f"stages/{stage}/LATEST", rid.encode(), s3)
    return rid


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage")
    ap.add_argument("dir", type=Path)
    ap.add_argument("--audio-from")
    ap.add_argument("--audio-dir", type=Path)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    rid = seed(a.stage, a.dir, audio_from=a.audio_from, audio_dir=a.audio_dir, note=a.note)
    print(f"stages/{a.stage}/{rid}: seeded, LATEST moved")


if __name__ == "__main__":
    main()
