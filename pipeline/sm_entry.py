"""Entry point inside the SageMaker container: install the stage's deps, then run it locally into the checkpoint dir."""
import json, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
hp = json.loads(Path("/opt/ml/input/config/hyperparameters.json").read_text())
sys.path.insert(0, str(HERE))
import run, sm  # noqa: E402

spec = run.STAGES[hp["stage"]]
script = next(Path(x) for x in spec["cmd"] if x.endswith(".py"))
if deps := sm.stage_deps(script):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *deps], check=True)  # into the image's python
spec = {**spec, "cmd": [sys.executable, str(script)]}  # run with the image's python, not `uv run`
validator = [sys.executable, str(HERE.parent / "validate.py")]
# /opt/ml/checkpoints is synced to S3 (spot restarts resume from it); input audio stays out of it
rc = run.run_local(hp["stage"], spec, f"{hp['src']}/records.jsonl", f"{hp['audio_src']}/audio/", json.loads(hp["params"]),
                   run=hp["run"], cache=Path("/opt/ml/checkpoints"), audio_cache=Path("/opt/ml/input/data/_audio"),
                   validate_cmd=validator)
sys.exit(rc)
