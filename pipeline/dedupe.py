# /// script
# requires-python = ">=3.12"
# ///
"""C5 dedupe stage (design §3.1): drop exact re-cuts and byte-identical audio; keep the lowest id.

    uv run pipeline/dedupe.py --in RECORDS.jsonl --audio-dir DIR --out OUT_DIR
"""
import argparse, hashlib, json
from pathlib import Path


def run(records, audio_dir):
    kept, rejects, seen_iv, seen_sha = [], [], {}, {}
    for r in sorted(records, key=lambda r: r["id"]):
        s = r["source"]
        iv = (s["source_file"], round(s["start"], 3), round(s["end"], 3))
        sha = hashlib.sha256((Path(audio_dir) / r["audio"]).read_bytes()).hexdigest()
        if iv in seen_iv:
            rejects.append({"id": r["id"], "stage": "dedupe", "rule": "same_interval", "detail": f"same as {seen_iv[iv]}"})
        elif sha in seen_sha:
            rejects.append({"id": r["id"], "stage": "dedupe", "rule": "same_audio", "detail": f"same bytes as {seen_sha[sha]}"})
        else:
            seen_iv[iv], seen_sha[sha] = r["id"], r["id"]
            kept.append(r)
    return kept, rejects


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    # dedupe needs every record at once, so a re-run recomputes from scratch instead of resuming
    kept, rejects = run([json.loads(l) for l in open(a.inp, encoding="utf-8") if l.strip()], a.audio_dir)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    (a.out / "rejects.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rejects), encoding="utf-8")


if __name__ == "__main__":
    main()
