# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Freeze a release snapshot in S3 (design §1 release/, SPEC §6 serialisation).

    uv run pipeline/release.py <tag> --records A.jsonl [B.jsonl ...] --audio-dir DIR [--partial]

Records are joined on `id` (later files override earlier fields), `_` keys dropped, validated,
then written once to release/<tag>/: audio/<id>.wav, provenance.jsonl, manifest.jsonl (last).
"""
import argparse, json, sys
from pathlib import Path
import s3 as store

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import validate  # noqa: E402

KEYS = ["id", "audio", "duration", "speaker_id", "source", "sentences", "hi_words", "translation_method"]


def _strip(x):
    if isinstance(x, dict):
        return {k: _strip(v) for k, v in x.items() if not k.startswith("_")}
    if isinstance(x, list):
        return [_strip(v) for v in x]
    return round(x, 3) if isinstance(x, float) else x


def serialise(rec):
    """SPEC §6: `_` keys dropped at every level, §5 key order, times rounded to 3 decimals."""
    r = _strip(rec)
    return json.dumps({k: r[k] for k in KEYS if k in r}, ensure_ascii=False)


def merge(stage_files):
    by = {}
    for p in stage_files:
        for l in open(p, encoding="utf-8"):
            if l.strip():
                rec = json.loads(l)
                by.setdefault(rec["id"], {}).update(rec)
    return [by[k] for k in sorted(by)]


def _once(key, body, s3):
    if not store.put_once(key, body, s3) and store.get(key, s3) != body:
        raise SystemExit(f"{key} exists with other bytes (an interrupted release?); pick a new tag")


def snapshot(tag, stage_files, audio_dir, *, partial, s3=None):
    if store.exists(f"release/{tag}/manifest.jsonl", s3):
        raise SystemExit(f"release/{tag} exists; releases are frozen, pick a new tag")
    recs = merge(stage_files)
    body = "".join(serialise(r) + "\n" for r in recs)
    out, _ = validate.validate(body.split("\n"), audio_dir, partial)
    for n, rid, rule, msg in out:
        print(f"{rid} {rule}: {msg}", file=sys.stderr)
    if out:
        raise SystemExit(f"validate.py: {len(out)} violations; nothing released")
    for r in recs:
        _once(f"release/{tag}/audio/{r['id']}.wav", (Path(audio_dir) / r["audio"]).read_bytes(), s3)
    # ponytail: local stage-file paths until stages run through run.py (Task 10)
    files = [str(Path(p).resolve()) for p in stage_files]
    prov = "".join(json.dumps({"id": r["id"], "_rev": r.get("_rev"), "_archive": r.get("_archive"), "stage_files": files},
                              ensure_ascii=False) + "\n" for r in recs)
    _once(f"release/{tag}/provenance.jsonl", prov.encode(), s3)
    _once(f"release/{tag}/manifest.jsonl", body.encode(), s3)  # last: its presence marks a complete release
    return len(recs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--records", nargs="+", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    print(f"release/{a.tag}: {snapshot(a.tag, a.records, a.audio_dir, partial=a.partial)} records")


if __name__ == "__main__":
    main()
