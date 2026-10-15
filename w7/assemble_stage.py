# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""`assemble` as a pipeline stage, so a big release is built on a SageMaker job's disk, not the laptop's.

    uv run pipeline/run.py assemble --in stages/<s>/<run> --audio-from stages/<s>/<run> --sm ml.c5.2xlarge --spot \
        -- <tag> --runs stages/<stage>/<run> [...]

Its input is ignored (point --in/--audio-from at any run without audio/, as for rebuild). Works in a scratch dir
beside the audio cache (on SageMaker: the large ML volume, not the checkpoint dir), then w7/assemble.py does the rest.
Writes OUT/assemble.json and an empty records.jsonl.
"""
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--in", dest="inp", type=Path, required=True)
ap.add_argument("--audio-dir", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
ap.add_argument("tag")
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--partial", action="store_true")
a = ap.parse_args()
scratch = a.audio_dir.parent
n = assemble.assemble(a.tag, a.runs, partial=a.partial, work=scratch / "_assemble", cache=scratch / "_assemble_cache")
a.out.mkdir(parents=True, exist_ok=True)
(a.out / "assemble.json").write_text(json.dumps({"tag": a.tag, "records": n, "runs": a.runs}, indent=1))
(a.out / "records.jsonl").touch()
print(f"release/{a.tag}: {n} records", file=sys.stderr)
