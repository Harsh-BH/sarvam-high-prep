# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""`rebuild` as a pipeline stage, so a SageMaker CPU job can be the clean machine (§3A, §10 R7).

    uv run pipeline/run.py rebuild --in stages/<s>/<run> --audio-from stages/<s>/<run> --sm ml.m5.large -- <tag>

Its input is ignored: point --in and --audio-from at any run without audio/, so nothing is downloaded. Rebuilds
release/<tag> from the bucket alone, in a scratch dir beside the audio cache (not --out: on SageMaker that is the
checkpoint dir, synced to S3). Writes OUT/rebuild.json and an empty records.jsonl; exits 1 on any difference.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import rebuild  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="inp", type=Path, required=True)
ap.add_argument("--audio-dir", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("tag")
a = ap.parse_args()
rep = rebuild.rebuild(a.tag, work=a.audio_dir.parent / "_rebuild")
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "rebuild.json").write_text(json.dumps(rep, indent=1))
(a.out / "records.jsonl").touch()
print(json.dumps(rep), file=sys.stderr)
sys.exit(0 if not rep["audio_diff"] and rep["manifest_identical"] else 1)
