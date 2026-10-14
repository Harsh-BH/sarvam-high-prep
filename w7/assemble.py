# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Assemble the release: final stage runs (real + synthetic) -> one validated, frozen snapshot (SPEC §6, §8).

    uv run w7/assemble.py <tag> --runs stages/<stage>/<run> [...] [--partial]

Each run contributes whole records. An id in two runs is refused: release.merge() would union their fields and
could bring back an id a later gate dropped. Real audio: the run's audio_from (the shared cache, symlinked).
Synthetic audio: rebuild.source_audio(), the archived take through the pinned ffmpeg, so the release is by
construction what rebuild makes on any machine. Then release.snapshot() writes release/<tag>/.
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import rebuild, release, run, s3 as store  # noqa: E402

WORK = run._main_checkout() / "data/w7/assemble"


def bind_take(x, s3=None):
    """Bind a released synthetic id to its kept take, write-once: archive/<source_file> is what rebuild reads.
    -> True when archive/<source_file> holds this record's take (bound now if it was unbound); False when the id is
    already bound to other audio (left by an abandoned run): that id is burned and the record leaves the release."""
    key = f"archive/{x['source']['source_file']}"
    tk = (x.get("_archive") or {}).get("tts")
    take = store.get(tk, s3) if tk else None
    bound = store.get(key, s3)
    if bound is not None:
        return take is None or bound == take  # no recorded take (round 1): the binding is the take
    if take is None:
        raise SystemExit(f"{x['id']}: synthetic, no {key} and no _archive.tts take")
    meta = {"sha256": hashlib.sha256(take).hexdigest(), "take": tk, "bound_by": "w7/assemble.py"}
    store.put_once(key, take, s3)
    store.put_once(key + ".meta.json", json.dumps(meta, indent=1).encode(), s3)
    return store.get(key, s3) == take  # a concurrent binder may have won with other bytes


def assemble(tag, runs, *, partial, s3=None, work=WORK, cache=run.CACHE):
    d = Path(work) / tag
    (d / "audio").mkdir(parents=True, exist_ok=True)
    runs = [r.rstrip("/") for r in runs]
    files, seen = [], {}
    for i, r in enumerate(runs):
        f = d / f"{i}.jsonl"
        f.write_bytes(store.get(f"{r}/records.jsonl", s3))
        recs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        burned = [x["id"] for x in recs if x["source"]["synthetic"] and not bind_take(x, s3)]
        if burned:
            print(f"{r}: {len(burned)} ids bound to other audio, burned and left out: {burned}", file=sys.stderr)
            recs = [x for x in recs if x["id"] not in burned]
            f.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in recs), encoding="utf-8")
        if dup := [x["id"] for x in recs if x["id"] in seen]:
            raise SystemExit(f"{r}: {len(dup)} ids also in {seen[dup[0]]}: {dup[:5]}")
        seen.update((x["id"], r) for x in recs)
        audio_from = json.loads(store.get(f"{r}/run.json", s3))["audio_from"]
        local = Path(cache) / "audio" / audio_from.strip("/").replace("/", "_")  # run.run_local's cache layout
        if any(not x["source"]["synthetic"] for x in recs):
            store.download(audio_from, local / "audio", s3)
        for x in recs:
            dst = d / "audio" / f"{x['id']}.wav"
            dst.unlink(missing_ok=True)
            if x["source"]["synthetic"]:
                dst.write_bytes(rebuild.source_audio(x, d / "src", s3))
            else:
                dst.symlink_to(local / x["audio"])
        files.append(f)
    return release.snapshot(tag, files, d, partial=partial, origins=[f"{r}/records.jsonl" for r in runs], s3=s3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag")
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    print(f"release/{a.tag}: {assemble(a.tag, a.runs, partial=a.partial)} records")


if __name__ == "__main__":
    main()
