# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""One-time: W3's pre-archive_call renders -> S3, write-once (design §1 archive/, §2.2; SPEC §4 source_file).

    uv run w7/archive_w3.py [--dry-run]

- every render, kept or rejected: data/w3/archive/<engine>/<key>.wav|.json -> archive/w3/<engine>/<key>.wav|.json, verbatim
- W3's index and QA verdicts, as of today: archive/w3/renders-<date>.jsonl, archive/w3/qa-<date>.tsv
- Bulbul spend markers (archive/spend/w3/), so archive.spent("w3") counts the ₹ already paid
- each id allocated in w3/ids.tsv is bound to its take: archive/tts/<id>.wav (the take, byte for byte = archive/<source_file>)
  + archive/tts/<id>.wav.meta.json {sha256, take, render}. Write-once: an id can never be rebound to other audio (G3).
Re-running is a no-op; an existing object with different bytes stops the run.
"""
import hashlib, json, sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import archive, s3 as store  # noqa: E402
from run import _main_checkout  # noqa: E402

W3 = _main_checkout() / "data/w3"
RUPEES_PER_CHAR = 30 / 10_000  # w3/render.py: Bulbul v3 list price


def once(key, body, s3=None):
    if not store.put_once(key, body, s3) and store.get(key, s3) != body:
        raise SystemExit(f"{key} exists with other bytes; refusing")


def bind(rid, take, render, s3=None):
    """id -> take, forever. -> the sidecar written."""
    meta = {"sha256": hashlib.sha256(take).hexdigest(), "take": f"archive/w3/{render['engine']}/{render['key']}.wav", "render": render}
    once(f"archive/tts/{rid}.wav", take, s3)
    once(f"archive/tts/{rid}.wav.meta.json", json.dumps(meta, ensure_ascii=False, indent=1).encode(), s3)
    return meta


def main(dry=False, s3=None, w3=W3, ids_tsv=ROOT / "w3/ids.tsv", today=None):
    renders = {r["key"]: r for r in map(json.loads, open(w3 / "renders.jsonl", encoding="utf-8"))}
    files = sorted(p for p in (w3 / "archive").glob("*/*") if p.suffix in (".wav", ".json"))
    ids = [l.rstrip("\n").split("\t") for l in list(open(ids_tsv))[1:] if l.strip()]
    spend = [(p.stem, json.loads(p.read_text())["billed_chars"] * RUPEES_PER_CHAR) for p in files if p.parent.name == "bulbul" and p.suffix == ".json"]
    print(f"{len(files)} render files, {len(spend)} Bulbul calls ₹{sum(i for _, i in spend):.2f}, {len(ids)} ids to bind", file=sys.stderr)
    if dry:
        return
    for p in files:
        once(f"archive/w3/{p.parent.name}/{p.name}", p.read_bytes(), s3)
    d = (today or date.today()).strftime("%Y%m%d")
    once(f"archive/w3/renders-{d}.jsonl", (w3 / "renders.jsonl").read_bytes(), s3)
    once(f"archive/w3/qa-{d}.tsv", (w3 / "qa.tsv").read_bytes(), s3)
    for key, inr in spend:
        archive._ledger("w3", "bulbul", key, round(inr, 4), s3)
    for rid, key, engine, *_ in ids:
        r = renders[key]
        assert r["engine"] == engine, (rid, key)
        bind(rid, (w3.parent.parent / r["wav"]).read_bytes(), r, s3)  # r["wav"] is relative to the main checkout
    print(f"archived; spent(w3) = ₹{archive.spent('w3', s3):.2f}", file=sys.stderr)


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
