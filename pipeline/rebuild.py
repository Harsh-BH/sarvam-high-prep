# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Rebuild a release from raw/ + archive/ + its frozen stage files and compare byte-for-byte (§3A, design §4.2).

    uv run pipeline/rebuild.py <tag>

Real record: fetch raw/<source_file>, check its sidecar sha256, re-cut with the pinned ffmpeg.
Synthetic record: archive/<source_file> (the take bound to this id, w7/archive_w3.py), sidecar sha256, then
tts_audio(); never re-synthesised. Needs nothing but the bucket. Exits 1 on any difference.
"""
import hashlib, io, json, subprocess, sys, wave
from pathlib import Path
import ffmpeg, release, s3 as store

ROOT = Path(__file__).resolve().parent.parent


def tts_audio(take):
    """Engine take (WAV bytes) -> release audio: 24 kHz mono s16, canonical 44-byte header.
    Same as w3/pilot.py to24k, but with the pinned ffmpeg (the system one differs by ±1 LSB after soxr)."""
    with wave.open(io.BytesIO(take)) as w:
        sr, ch, pcm = w.getframerate(), w.getnchannels(), w.readframes(w.getnframes())
    if (sr, ch) != (ffmpeg.RATE, 1):
        pcm = subprocess.run([str(ffmpeg.binary()), "-nostdin", "-v", "error", "-i", "-", "-af", "aresample=resampler=soxr",
                              "-ar", str(ffmpeg.RATE), "-ac", "1", "-f", "s16le", "-"], input=take, capture_output=True, check=True).stdout
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(ffmpeg.RATE), w.writeframes(pcm)
    return out.getvalue()


def source_audio(r, work, s3=None):
    """A manifest record's audio, made from the bucket alone: raw/ re-cut (real) or archive/ take (synthetic)."""
    key = f"{'archive' if r['source']['synthetic'] else 'raw'}/{r['source']['source_file']}"
    src = Path(work) / key
    if not src.exists():
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_bytes(store.get(key, s3))
    if hashlib.sha256(src.read_bytes()).hexdigest() != json.loads(store.get(key + ".meta.json", s3))["sha256"]:
        raise SystemExit(f"{key}: sha256 differs from its sidecar")
    if r["source"]["synthetic"]:
        return tts_audio(src.read_bytes())
    out = Path(work) / f"{r['id']}.wav"
    ffmpeg.cut(src, r["source"]["start"], r["source"]["end"], out)
    return out.read_bytes()


def _stage_files(prov, work, s3=None):
    """The stage files the manifest was merged from: frozen under release/<tag>/inputs/ (pilot-v0 predates that and
    names local paths). -> local paths, or None if one is missing."""
    keys = next(iter(prov.values()))["stage_files"] if prov else []
    out = []
    for k in keys:
        p = Path(k)
        if k.startswith("release/"):
            p = Path(work) / k
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(store.get(k, s3))
        out.append(p)
    return out if out and all(p.exists() for p in out) else None


def rebuild(tag, *, s3=None, work=ROOT / "data/rebuild"):
    """-> {"records", "audio_identical", "audio_diff": [ids], "manifest_identical"}"""
    work = Path(work) / tag
    work.mkdir(parents=True, exist_ok=True)
    manifest = store.get(f"release/{tag}/manifest.jsonl", s3).decode().splitlines()
    prov = {p["id"]: p for p in map(json.loads, store.get(f"release/{tag}/provenance.jsonl", s3).decode().splitlines())}
    same, diff = 0, []
    for line in manifest:
        r = json.loads(line)
        if source_audio(r, work, s3) == store.get(f"release/{tag}/audio/{r['id']}.wav", s3):
            same += 1
        else:
            diff.append(r["id"])
    files = _stage_files(prov, work, s3)
    remade = [release.serialise(x) for x in release.merge(files)] if files else None
    return {"records": len(manifest), "audio_identical": same, "audio_diff": diff, "manifest_identical": remade == manifest}


def main():
    rep = rebuild(sys.argv[1])
    print(json.dumps(rep, indent=1))
    sys.exit(0 if not rep["audio_diff"] and rep["manifest_identical"] else 1)


if __name__ == "__main__":
    main()
