# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
import json, subprocess, sys, tempfile, wave
from pathlib import Path
import numpy as np
import audio_qc

SR = 24000


def write(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.clip(x, -32768, 32767).astype("<i2").tobytes())


def speechlike(secs, seed, band=None):
    rng = np.random.default_rng(seed)
    n = int(SR * secs)
    x = rng.normal(0, 3000, n) * (0.2 + 0.8 * (np.sin(np.arange(n) * 2 * np.pi * 3 / SR) > 0))  # bursts + pauses at 3 Hz
    if band:
        X = np.fft.rfft(x); X[np.fft.rfftfreq(n, 1 / SR) > band] = 0; x = np.fft.irfft(X, n)
    return x


def main():
    d = Path(tempfile.mkdtemp()); (d / "audio").mkdir()
    write(d / "audio/rec-000001.wav", speechlike(15, 1))                    # wideband, fine
    write(d / "audio/rec-000002.wav", speechlike(15, 2, band=3400))         # telephone band, fine
    write(d / "audio/rec-000003.wav", speechlike(15, 3) * 40)               # heavy clipping
    write(d / "audio/rec-000004.wav", np.zeros(SR * 15))                    # silence
    write(d / "audio/rec-000005.wav", speechlike(5, 5))                     # too short
    m1, m2 = audio_qc.measure(d / "audio/rec-000001.wav"), audio_qc.measure(d / "audio/rec-000002.wav")
    assert not m1["telephone"] and m2["telephone"] and abs(m1["duration"] - 15) < 1e-6
    assert audio_qc.verdict(m1, audio_qc.DEFAULTS) == [] and audio_qc.verdict(m2, audio_qc.DEFAULTS) == []
    x = speechlike(15, 6); X = np.fft.rfft(x); X[np.fft.rfftfreq(len(x), 1 / SR) > 4000] *= 0.05
    write(d / "dull.wav", np.fft.irfft(X, len(x)))  # wideband, but HF share ~0.005 like the W1 pool's 44.1 kHz median
    assert not audio_qc.measure(d / "dull.wav")["telephone"]
    write(d / "quiet.wav", 32768 * 10 ** (-60 / 20) * np.sin(np.arange(SR * 15) * 2 * np.pi * 440 / SR))  # frames at -63 dBFS
    assert audio_qc.measure(d / "quiet.wav")["silence_ratio"] == 1.0
    recs = "".join(json.dumps({"id": f"rec-00000{i}", "audio": f"audio/rec-00000{i}.wav"}) + "\n" for i in range(1, 6))
    (d / "in.jsonl").write_text(recs)
    cmd = [sys.executable, str(Path(audio_qc.__file__)), "--in", str(d / "in.jsonl"), "--audio-dir", str(d), "--out", str(d / "out")]
    subprocess.run(cmd, check=True)
    kept = [json.loads(l) for l in open(d / "out/records.jsonl")]
    rej = {json.loads(l)["id"]: json.loads(l)["rule"] for l in open(d / "out/rejects.jsonl")}
    assert [r["id"] for r in kept] == ["rec-000001", "rec-000002"] and kept[1]["_qa"]["audio_qc"]["telephone"] is True
    assert rej == {"rec-000003": "clipping", "rec-000004": "silence", "rec-000005": "duration"}, rej
    failed = {json.loads(l)["id"]: json.loads(l)["detail"]["failed"] for l in open(d / "out/rejects.jsonl")}
    assert failed["rec-000004"] == ["silence", "level", "snr"], failed
    subprocess.run(cmd, check=True)  # resume: nothing duplicated
    assert sum(1 for _ in open(d / "out/records.jsonl")) == 2 and sum(1 for _ in open(d / "out/rejects.jsonl")) == 3
    lines = (d / "out/rejects.jsonl").read_text().splitlines(keepends=True)
    (d / "out/rejects.jsonl").write_text("".join(lines[:-1]) + lines[-1][:15])  # crash tore the last line mid-write
    subprocess.run(cmd, check=True)
    assert sorted(json.loads(l)["id"] for l in open(d / "out/rejects.jsonl")) == sorted(rej)
    print("ok: audio_qc")


if __name__ == "__main__":
    main()
