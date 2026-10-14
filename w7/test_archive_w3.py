# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""uv run w7/test_archive_w3.py: renders archived verbatim, ids bound write-once, spend counted, re-run is a no-op."""
import json, sys, tempfile
from datetime import date
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parent.parent / "pipeline")]
import archive, archive_w3  # noqa: E402
from test_s3 import FakeS3  # noqa: E402


def main():
    f, root = FakeS3(), Path(tempfile.mkdtemp())
    w3 = root / "data/w3"
    (w3 / "archive/bulbul").mkdir(parents=True)
    (w3 / "archive/bulbul/k1.wav").write_bytes(b"RIFF-take-1")
    (w3 / "archive/bulbul/k1.json").write_text(json.dumps({"request": {"text": "x"}, "billed_chars": 100}))
    (w3 / "renders.jsonl").write_text(json.dumps({"key": "k1", "engine": "bulbul", "wav": "data/w3/archive/bulbul/k1.wav"}) + "\n")
    (w3 / "qa.tsv").write_text("key\tverdict\nk1\tkeep\n")
    ids = root / "ids.tsv"
    ids.write_text("id\tkey\tengine\tvoice\tpid\nrec-400000\tk1\tbulbul\tritu\tp001\n")
    kw = dict(s3=f, w3=w3, ids_tsv=ids, today=date(2026, 10, 14))
    archive_w3.main(**kw)
    archive_w3.main(**kw)  # idempotent
    assert f.objects["archive/tts/rec-400000.wav"] == b"RIFF-take-1" == f.objects["archive/w3/bulbul/k1.wav"]
    meta = json.loads(f.objects["archive/tts/rec-400000.wav.meta.json"])
    assert meta["take"] == "archive/w3/bulbul/k1.wav" and len(meta["sha256"]) == 64
    archive._spent.clear()
    assert abs(archive.spent("w3", f) - 0.3) < 1e-9, archive.spent("w3", f)
    (w3 / "archive/bulbul/k1.wav").write_bytes(b"RIFF-other-audio")
    try:  # an id never gets other audio
        archive_w3.main(**kw)
        raise AssertionError("rebinding not refused")
    except SystemExit:
        pass
    assert f.objects["archive/tts/rec-400000.wav"] == b"RIFF-take-1"
    print("ok: archive_w3")


if __name__ == "__main__":
    main()
