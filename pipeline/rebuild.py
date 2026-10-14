# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Rebuild a release from raw/ + archive/ + stage files and compare byte-for-byte (§3A, design §4.2).

    uv run pipeline/rebuild.py <tag>

Real record: fetch raw/<source_file>, check its sidecar sha256, re-cut with the pinned ffmpeg.
Synthetic record: the archived take named in provenance `_archive["tts"]`; never re-synthesised.
Exits 1 on any difference.
"""
import hashlib, json, sys
from pathlib import Path
import ffmpeg, release, s3 as store

ROOT = Path(__file__).resolve().parent.parent


def rebuild(tag, *, s3=None, work=ROOT / "data/rebuild"):
    """-> {"records", "audio_identical", "audio_diff": [ids], "manifest_identical"}"""
    work = Path(work) / tag
    work.mkdir(parents=True, exist_ok=True)
    manifest = store.get(f"release/{tag}/manifest.jsonl", s3).decode().splitlines()
    prov = {p["id"]: p for p in map(json.loads, store.get(f"release/{tag}/provenance.jsonl", s3).decode().splitlines())}
    same, diff = 0, []
    for line in manifest:
        r = json.loads(line)
        want = store.get(f"release/{tag}/audio/{r['id']}.wav", s3)
        if r["source"]["synthetic"]:
            got = store.get(prov[r["id"]]["_archive"]["tts"], s3)  # the archived take; never re-synthesised
        else:
            key = f"raw/{r['source']['source_file']}"
            src = work / "raw" / r["source"]["source_file"]
            if not src.exists():
                src.parent.mkdir(parents=True, exist_ok=True)
                src.write_bytes(store.get(key, s3))
            if hashlib.sha256(src.read_bytes()).hexdigest() != json.loads(store.get(key + ".meta.json", s3))["sha256"]:
                raise SystemExit(f"{key}: sha256 differs from its sidecar")
            out = work / f"{r['id']}.wav"
            ffmpeg.cut(src, r["source"]["start"], r["source"]["end"], out)
            got = out.read_bytes()
        if got == want:
            same += 1
        else:
            diff.append(r["id"])
    files = next(iter(prov.values()))["stage_files"] if prov else []
    remade = [release.serialise(x) for x in release.merge(files)] if files and all(Path(p).exists() for p in files) else None
    return {"records": len(manifest), "audio_identical": same, "audio_diff": diff, "manifest_identical": remade == manifest}


def main():
    rep = rebuild(sys.argv[1])
    print(json.dumps(rep, indent=1))
    sys.exit(0 if not rep["audio_diff"] and rep["manifest_identical"] else 1)


if __name__ == "__main__":
    main()
