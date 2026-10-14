# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
import json, sys, tempfile
from pathlib import Path
import run, seed
from test_s3 import FakeS3

OK = [sys.executable, "-c", "import sys; sys.exit(0)"]
BAD = [sys.executable, "-c", "import sys; print('rec-000001 hi_final: x'); sys.exit(1)"]


def main():
    f, d = FakeS3(), Path(tempfile.mkdtemp())
    (d / "audio").mkdir()
    (d / "audio/rec-400001.wav").write_bytes(b"RIFF")
    (d / "records.jsonl").write_text(json.dumps({"id": "rec-400001"}) + "\n")
    rid = seed.seed("tts", d, s3=f, validate_cmd=OK, git_sha="abc1234")
    meta = json.loads(f.objects[f"stages/tts/{rid}/run.json"])
    assert run.latest("tts", f) == rid and meta["audio_from"] == f"stages/tts/{rid}/audio/" and meta["records"] == 1
    assert f"stages/tts/{rid}/audio/rec-400001.wav" in f.objects
    assert run.audio_of(f"stages/tts/{rid}", f) == f"stages/tts/{rid}"  # the launcher finds the audio downstream

    t = Path(tempfile.mkdtemp())  # text-only data points at another run's audio
    (t / "records.jsonl").write_text(json.dumps({"id": "rec-400001"}) + "\n")
    rid2 = seed.seed("scripts", t, audio_from=f"stages/tts/{rid}/audio", audio_dir=d, s3=f, validate_cmd=OK, git_sha="abc1234")
    assert json.loads(f.objects[f"stages/scripts/{rid2}/run.json"])["audio_from"] == f"stages/tts/{rid}/audio/"

    n = len(f.objects)  # a failed validation seeds nothing and leaves LATEST alone
    try:
        seed.seed("tts", d, s3=f, validate_cmd=BAD, git_sha="abc1234")
        raise AssertionError("seeded despite validator failure")
    except SystemExit:
        pass
    assert len(f.objects) == n and run.latest("tts", f) == rid
    try:
        seed.seed("x", t, s3=f, validate_cmd=OK, git_sha="abc1234")
        raise AssertionError("text-only seed without --audio-from accepted")
    except SystemExit:
        pass
    print("ok: seed")


if __name__ == "__main__":
    main()
