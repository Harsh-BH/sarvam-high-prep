# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""uv run pipeline/test_ffmpeg.py [--pilot N]: synthetic cut check; --pilot compares N pilot clips byte-for-byte."""
import json, math, struct, sys, tempfile, wave
from pathlib import Path
import ffmpeg

DATA = Path("/home/harsh1/github-repos/sarvam-high-prep/data")


def tone(path, sr, secs):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / sr))) for i in range(int(sr * secs))))


def main():
    d = Path(tempfile.mkdtemp())
    tone(d / "src.wav", 8000, 3.0)
    ffmpeg.cut(d / "src.wav", 0.5, 2.25, d / "a.wav")
    ffmpeg.cut(d / "src.wav", 0.5, 2.25, d / "b.wav")
    with wave.open(str(d / "a.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (24000, 1, 2, 42000)
    assert (d / "a.wav").read_bytes() == (d / "b.wav").read_bytes()  # deterministic
    print("ok: ffmpeg synthetic")
    if "--pilot" in sys.argv:
        n = int(sys.argv[sys.argv.index("--pilot") + 1])
        same = 0
        recs = [json.loads(l) for l in open(DATA / "w1/pilot/manifest.jsonl")][:n]
        for r in recs:
            # iv_train mirrors the archive layout: Hindi/<v>/<split>/<name>.wav (w1/cut_pilot.py `rel`)
            src = DATA / "w1/iv_train" / Path(r["source"]["source_file"]).relative_to("indicvoices")
            out = d / f"{r['id']}.wav"
            ffmpeg.cut(src, r["source"]["start"], r["source"]["end"], out)
            same += out.read_bytes() == (DATA / "w1/pilot" / r["audio"]).read_bytes()
        print(f"pilot: {same}/{len(recs)} byte-identical to the system-ffmpeg cuts")


if __name__ == "__main__":
    main()
