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
    # synthetic: a 22,050 Hz take bound under archive/tts/, release audio = tts_audio(take) via the pinned ffmpeg
    tone(d / "take.wav", 22050, 3.0)
    take = (d / "take.wav").read_bytes()
    f.objects["archive/tts/rec-400000.wav"] = take
    f.objects["archive/tts/rec-400000.wav.meta.json"] = json.dumps({"sha256": hashlib.sha256(take).hexdigest()}).encode()
    (d / "audio/rec-400000.wav").write_bytes(rebuild.tts_audio(take))
    with wave.open(str(d / "audio/rec-400000.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (24000, 1, 2, 72000)
    s = {"id": "rec-400000", "audio": "audio/rec-400000.wav", "duration": 3.0, "speaker_id": "spk-tts-x-y",
         "source": {"corpus": "w2-scripts-v1", "source_file": "tts/rec-400000.wav", "start": 0, "end": 3.0, "synthetic": True,
                    "model": "x", "voice": "y"}, "sentences": [{"hi_text": "नमस्ते।"}]}
    (d / "a.jsonl").write_text(json.dumps(a, ensure_ascii=False) + "\n")
    (d / "b.jsonl").write_text(json.dumps(b, ensure_ascii=False) + "\n")
    (d / "s.jsonl").write_text(json.dumps(s, ensure_ascii=False) + "\n")
    line = release.serialise(release.merge([d / "a.jsonl", d / "b.jsonl"])[0])
    assert line.startswith('{"id": "rec-000001", "audio"') and "_rev" not in line and "_orig_text" not in line
    assert release.snapshot("t1", [d / "a.jsonl", d / "b.jsonl", d / "s.jsonl"], d, partial=True, origins=["x", "y", "z"], s3=f) == 2
    assert f.objects["release/t1/inputs/2.jsonl"] == (d / "s.jsonl").read_bytes()
    try:
        release.snapshot("t1", [d / "a.jsonl"], d, partial=True, s3=f)
        raise AssertionError("release overwrite not refused")
    except SystemExit:
        pass
    (d / "b.jsonl").write_text(json.dumps(dict(b, sentences=[{"hi_text": "नमस्कार।"}]), ensure_ascii=False) + "\n")
    rep = rebuild.rebuild("t1", s3=f, work=d / "rb")  # local stage files may drift: the release froze its own copy
    assert rep == {"records": 2, "audio_identical": 2, "audio_diff": [], "manifest_identical": True}, rep
    f.objects["release/t1/audio/rec-000001.wav"] = b"RIFF-tampered"
    assert rebuild.rebuild("t1", s3=f, work=d / "rb2")["audio_diff"] == ["rec-000001"]
    f.objects["release/t1/inputs/1.jsonl"] = (d / "b.jsonl").read_bytes()
    assert rebuild.rebuild("t1", s3=f, work=d / "rb3")["manifest_identical"] is False  # frozen input tampered
    f.objects["archive/tts/rec-400000.wav"] = take[:-2] + b"\x00\x01"
    try:  # an archived take that no longer matches its sidecar is never used
        rebuild.rebuild("t1", s3=f, work=d / "rb4")
        raise AssertionError("tampered take used")
    except SystemExit:
        pass
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
    (d / "typo.jsonl").write_text(json.dumps(dict(b, speakerid="spk-y"), ensure_ascii=False) + "\n")
    (d / "dup.jsonl").write_text((json.dumps(b, ensure_ascii=False) + "\n") * 2)
    (d / "textonly.jsonl").write_text(json.dumps(dict(b, id="rec-000002"), ensure_ascii=False) + "\n")
    for tag, extra in (("t4", "typo"), ("t5", "dup"), ("t6", "textonly")):
        try:  # unknown key / duplicate id in one stage file / a record with no audio: refused before any write
            release.snapshot(tag, [d / "a.jsonl", d / f"{extra}.jsonl"], d, partial=True, s3=f)
            raise AssertionError(f"{extra} released")
        except SystemExit:
            pass
        assert not [k for k in f.objects if k.startswith(f"release/{tag}/")], extra
    f.objects["release/t7/audio/rec-000099.wav"] = b"RIFF-left-by-a-crashed-run"
    try:  # an object the new manifest wouldn't list must not sit in the frozen snapshot
        release.snapshot("t7", [d / "a.jsonl", d / "b.jsonl"], d, partial=True, s3=f)
        raise AssertionError("leftover object under release/t7/ not refused")
    except SystemExit:
        pass
    assert "release/t7/manifest.jsonl" not in f.objects
    print("ok: rebuild")


if __name__ == "__main__":
    main()
