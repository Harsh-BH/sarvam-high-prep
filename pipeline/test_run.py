# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
import json, sys, tempfile
from pathlib import Path
import run
from test_s3 import FakeS3

# a stand-in stage: copies records, rejects ids listed in REJECT, exits 3 if FAIL is set.
# Like the real stages it opens each record's audio at --audio-dir / rec["audio"] (SPEC §5: "audio/<id>.wav").
STAGE = r'''
import argparse, json, os, pathlib
ap = argparse.ArgumentParser(); ap.add_argument("--in", dest="inp"); ap.add_argument("--audio-dir"); ap.add_argument("--out"); a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["id"] for f in ("records.jsonl", "rejects.jsonl") if (out / f).exists() for l in open(out / f)}
rej = set(os.environ.get("REJECT", "").split())
with open(out / "records.jsonl", "a") as r, open(out / "rejects.jsonl", "a") as x:
    for l in open(a.inp):
        rec = json.loads(l)
        if rec["id"] in done: continue
        if "audio" in rec: open(pathlib.Path(a.audio_dir) / rec["audio"], "rb").close()
        if rec["id"] in rej: x.write(json.dumps({"id": rec["id"], "stage": "t", "rule": "r", "detail": ""}) + "\n")
        else: r.write(json.dumps(rec) + "\n")
if os.environ.get("FAIL"): raise SystemExit(3)
'''


def setup():
    f, d = FakeS3(), Path(tempfile.mkdtemp())
    (d / "stage.py").write_text(STAGE)
    recs = [{"id": f"rec-00000{i}", "audio": f"audio/rec-00000{i}.wav", "translation_method": "x"} for i in (1, 2, 3)]
    f.objects["stages/prev/R0/records.jsonl"] = "".join(json.dumps(r) + "\n" for r in recs).encode()
    for r in recs:
        f.objects[f"stages/cut/R0/{r['audio']}"] = b"RIFF"
    spec = {"cmd": [sys.executable, str(d / "stage.py")], "makes_audio": False, "gpu": False}
    # passes only if the audio dir it is handed resolves "audio/<id>.wav", as validate.py does
    ok_validator = [sys.executable, "-c", "import sys, pathlib; a = sys.argv[sys.argv.index('--audio-dir') + 1]; "
                    "sys.exit(0 if (pathlib.Path(a) / 'audio/rec-000001.wav').exists() else 1)"]
    return f, d, spec, ok_validator


def go(f, d, spec, validator, run_name, env=None):
    import os
    old = dict(os.environ); os.environ.update(env or {})
    try:
        return run.run_local("t", spec, "stages/prev/R0/records.jsonl", "stages/cut/R0/audio/", [], run=run_name,
                             cache=d / "cache", audio_cache=d / "audio", s3=f, validate_cmd=validator)
    finally:
        os.environ.clear(); os.environ.update(old)


def main():
    f, d, spec, okv = setup()
    assert go(f, d, spec, okv, "R1", {"REJECT": "rec-000002"}) == 0
    assert run.latest("t", f) == "R1"
    out = f.objects["stages/t/R1/records.jsonl"].decode().splitlines()
    assert [json.loads(l)["id"] for l in out] == ["rec-000001", "rec-000003"]
    assert json.loads(f.objects["stages/t/R1/rejects.jsonl"])["id"] == "rec-000002"
    meta = json.loads(f.objects["stages/t/R1/run.json"])
    assert meta["records"] == 2 and meta["rejects"] == 1 and meta["input"] == "stages/prev/R0/records.jsonl"

    # latest_guard: stage exit != 0 -> run uploaded, LATEST unchanged
    assert go(f, d, spec, okv, "R2", {"FAIL": "1"}) == 1 and run.latest("t", f) == "R1"
    assert "stages/t/R2/run.json" in f.objects
    # latest_guard: validator violations -> 2, LATEST unchanged
    badv = [sys.executable, "-c", "import sys; print('rec-000001 hi_final: x'); sys.exit(1)"]
    assert go(f, d, spec, badv, "R3") == 2
    assert run.latest("t", f) == "R1"
    # latest_guard: a validator that dies without printing a violation (traceback on stderr) is a failure too
    crashv = [sys.executable, "-c", "raise SystemExit(1)"]
    assert go(f, d, spec, crashv, "R3b") == 2 and run.latest("t", f) == "R1"

    # resume: a crashed run's partial output is reused; only missing ids are processed, none duplicated.
    # The crash tore the last line mid-write (buffered writes are flushed in chunks, not lines).
    part = d / "cache" / "t" / "R4" / "out"
    part.mkdir(parents=True)
    (part / "records.jsonl").write_text(json.dumps({"id": "rec-000001", "translation_method": "x"}) + "\n" + '{"id": "rec-000002", "tra')
    assert go(f, d, spec, okv, "R4") == 0
    ids = [json.loads(l)["id"] for l in f.objects["stages/t/R4/records.jsonl"].decode().splitlines()]
    assert sorted(ids) == ["rec-000001", "rec-000002", "rec-000003"] and len(ids) == 3

    # input audio is cached outside `cache` (on SageMaker `cache` is the checkpoint dir, synced to S3)
    assert not any((d / "cache").rglob("*.wav")) and (d / "audio").exists()

    from datetime import datetime, timezone
    assert run.run_id("abc1234", datetime(2026, 10, 14, 7, 5, 9, tzinfo=timezone.utc)) == "20261014T070509Z-abc1234"
    print("ok: run")


if __name__ == "__main__":
    main()
