# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "boto3", "torch", "transformers", "onnxruntime", "huggingface_hub", "soundfile"]
# ///
"""`tts_bench` stage = `w3/tts.py --bench`: one take per script, no records. Its own stage, so a benchmark never moves
the `tts` LATEST (sm_entry runs a stage's script without extra argv, so the flag can't live in run.py STAGES)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts  # noqa: E402

tts.main(sys.argv[1:] + ["--bench"])
