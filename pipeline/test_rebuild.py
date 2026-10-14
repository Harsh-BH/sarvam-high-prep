# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""uv run pipeline/test_rebuild.py: snapshot a one-record release into FakeS3, rebuild it, catch tampering.
ffmpeg.cut fetches the pinned binary from the real bucket the first time: load .env first."""
import hashlib, json, math, struct, tempfile, wave
from pathlib import Path
import ffmpeg, rebuild, release
from test_s3 import FakeS3


def tone(path, sr, secs):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(2 * math.pi * 300 * i / sr))) for i in range(int(sr * secs))))


def main():
    f, d = FakeS3(), Path(tempfile.mkdtemp())
    tone(d / "long.wav", 16000, 30.0)
    src = d / "long.wav"
    sf = "indicvoices/Hindi/v4/train/x_0.wav"
    f.objects[f"raw/{sf}"] = src.read_bytes()
    f.objects[f"raw/{sf}.meta.json"] = json.dumps({"sha256": hashlib.sha256(src.read_bytes()).hexdigest()}).encode()
    (d / "audio").mkdir()
    ffmpeg.cut(src, 2.0, 14.5, d / "audio/rec-000001.wav")
    a = {"id": "rec-000001", "audio": "audio/rec-000001.wav", "duration": 12.5, "speaker_id": "spk-x",
         "source": {"corpus": "IndicVoices", "source_file": sf, "start": 2.0, "end": 14.5, "synthetic": False},
         "_rev": "u etag=E"}
    b = {"id": "rec-000001", "sentences": [{"hi_text": "नमस्ते।"}], "_orig_text": "नमस्ते"}
    (d / "a.jsonl").write_text(json.dumps(a, ensure_ascii=False) + "\n")
    (d / "b.jsonl").write_text(json.dumps(b, ensure_ascii=False) + "\n")
    line = release.serialise(release.merge([d / "a.jsonl", d / "b.jsonl"])[0])
    assert line.startswith('{"id": "rec-000001", "audio"') and "_rev" not in line and "_orig_text" not in line
    assert release.snapshot("t1", [d / "a.jsonl", d / "b.jsonl"], d, partial=True, s3=f) == 1
    try:
        release.snapshot("t1", [d / "a.jsonl"], d, partial=True, s3=f)
        raise AssertionError("release overwrite not refused")
    except SystemExit:
        pass
    rep = rebuild.rebuild("t1", s3=f, work=d / "rb")
    assert rep == {"records": 1, "audio_identical": 1, "audio_diff": [], "manifest_identical": True}, rep
    f.objects["release/t1/audio/rec-000001.wav"] = b"RIFF-tampered"
    assert rebuild.rebuild("t1", s3=f, work=d / "rb2")["audio_diff"] == ["rec-000001"]
    (d / "b.jsonl").write_text(json.dumps(dict(b, sentences=[{"hi_text": "नमस्कार।"}]), ensure_ascii=False) + "\n")
    assert rebuild.rebuild("t1", s3=f, work=d / "rb2")["manifest_identical"] is False  # stage file drifted
    (d / "c.jsonl").write_text(json.dumps(dict(b, sentences=[{"hi_text": "नमस्ते"}]), ensure_ascii=False) + "\n")
    try:  # no final mark: validate.py fails, so nothing at all lands under release/t2/
        release.snapshot("t2", [d / "a.jsonl", d / "c.jsonl"], d, partial=True, s3=f)
        raise AssertionError("invalid release not refused")
    except SystemExit:
        pass
    assert not [k for k in f.objects if k.startswith("release/t2/")]
    f.objects["release/t3/audio/rec-000001.wav"] = b"RIFF-from-a-crashed-run"
    try:  # a re-run after a crash must not keep bytes it never validated
        release.snapshot("t3", [d / "a.jsonl", d / "b.jsonl"], d, partial=True, s3=f)
        raise AssertionError("stale audio from an interrupted release kept")
    except SystemExit:
        pass
    assert "release/t3/manifest.jsonl" not in f.objects
    print("ok: rebuild")


if __name__ == "__main__":
    main()
