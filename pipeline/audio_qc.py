# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
"""C4 audio QC stage (design §3.1): measure every clip, reject clear outliers, label telephone audio.

    uv run pipeline/audio_qc.py --in RECORDS.jsonl --audio-dir DIR --out OUT_DIR [--max-clip-ratio X ...]
"""
import argparse, json, wave
from pathlib import Path
import numpy as np

DEFAULTS = {"max_clip_ratio": 0.001, "max_silence_ratio": 0.5, "min_rms_dbfs": -40.0, "min_snr_db": 10.0, "min_dur": 10.0, "max_dur": 60.0}
ORDER = ["clipping", "silence", "level", "snr", "duration"]
# Telephone = HF share (>4 kHz) below this. W1 pool: 8 kHz sources max 1.1e-4 (clipped clips leak HF), 44.1 kHz min 1.7e-4.
# ponytail: one global cutoff with a thin margin; use the source's orig_sample_rate where the corpus gives one.
TEL_HF = 1.4e-4


def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-9) / 32768)


def measure(path):
    with wave.open(str(path)) as w:
        sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64)
    hop = sr // 50
    frames = x[: len(x) // hop * hop].reshape(-1, hop)
    fdb = _db(np.sqrt((frames ** 2).mean(axis=1))) if len(frames) else np.array([-180.0])
    spec = np.abs(np.fft.rfft(x)) ** 2
    hf = spec[np.fft.rfftfreq(len(x), 1 / sr) > 4000].sum() / max(spec.sum(), 1e-9)
    return {"duration": len(x) / sr, "rms_dbfs": float(_db(np.sqrt((x ** 2).mean()) if len(x) else 0)),
            "clip_ratio": float((np.abs(x) >= 32767).mean()) if len(x) else 0.0,
            "silence_ratio": float((fdb < -50).mean()), "snr_db": float(np.percentile(fdb, 90) - np.percentile(fdb, 10)),
            "hf_ratio": float(hf), "telephone": bool(hf < TEL_HF)}


def verdict(m, t):
    fail = {"clipping": m["clip_ratio"] > t["max_clip_ratio"], "silence": m["silence_ratio"] > t["max_silence_ratio"],
            "level": m["rms_dbfs"] < t["min_rms_dbfs"], "snr": m["snr_db"] < t["min_snr_db"],
            "duration": not t["min_dur"] <= m["duration"] <= t["max_dur"]}
    return [r for r in ORDER if fail[r]]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), type=float, default=v)
    a = ap.parse_args()
    t = {k: getattr(a, k) for k in DEFAULTS}
    a.out.mkdir(parents=True, exist_ok=True)
    for f in ("records.jsonl", "rejects.jsonl"):
        p = a.out / f
        if p.exists() and (b := p.read_bytes()) and not b.endswith(b"\n"):
            p.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: drop it, that id is redone
    done ={json.loads(l)["id"] for f in ("records.jsonl", "rejects.jsonl") if (a.out / f).exists() for l in open(a.out / f)}
    with open(a.out / "records.jsonl", "a", encoding="utf-8") as ok, open(a.out / "rejects.jsonl", "a", encoding="utf-8") as bad:
        for l in open(a.inp, encoding="utf-8"):
            rec = json.loads(l)
            if rec["id"] in done:
                continue
            m = measure(a.audio_dir / rec["audio"])
            fails = verdict(m, t)
            if fails:
                bad.write(json.dumps({"id": rec["id"], "stage": "audio_qc", "rule": fails[0], "detail": {"failed": fails, **m}}) + "\n")
            else:
                rec.setdefault("_qa", {})["audio_qc"] = m
                ok.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
