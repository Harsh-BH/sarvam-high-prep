# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Build the frozen eval sets and put them in S3 (tasks/W8c-eval.md "Eval sets").

    uv run w8c/sets.py fleurs                       # -> data/w8c/sets/fleurs.jsonl + audio/
    uv run w8c/sets.py heldout stages/<stage>/<run> # dev/test: final real run + pilot-v2 held-out synthetic
    uv run w8c/sets.py upload                       # -> stages/w8c_sets/<run>/ (no LATEST: pass the path)

Records keep the SPEC §5 fields (`_` keys dropped, release.serialise); audio is 24 kHz mono s16 at audio/<id>.wav.
FLEURS hi_in test (CC BY 4.0, HF google/fleurs @ FLEURS_REV): one record per recording, en_text = the en_us
raw transcription of the same FLoRes sentence id, no hi_words (start offset and sentence lag are skipped for it).
"""
import csv, io, json, subprocess, sys, tarfile, urllib.request, wave
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pipeline"))
import ffmpeg, release, s3 as store  # noqa: E402

DATA = Path("/home/harsh1/github-repos/sarvam-high-prep/data/w8c")
SETS = DATA / "sets"
FLEURS_REV = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
HF = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REV}/data"


def _get(url):
    import os
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {})
    return urllib.request.urlopen(req).read()


def _tsv(b):
    return list(csv.reader(io.StringIO(b.decode()), delimiter="\t", quoting=csv.QUOTE_NONE))


def fleurs():
    en = {r[0]: r[2] for r in _tsv(_get(f"{HF}/en_us/test.tsv"))}
    hi = _tsv(_get(f"{HF}/hi_in/test.tsv"))
    (SETS / "audio").mkdir(parents=True, exist_ok=True)
    tgz = SETS / "fleurs_hi_test.tar.gz"
    if not tgz.exists():
        tgz.write_bytes(_get(f"{HF}/hi_in/audio/test.tar.gz"))
    want = {r[1]: r for r in hi if r[0] in en}
    lines = []
    with tarfile.open(tgz) as t:
        for m in t:
            name = Path(m.name).name
            if name not in want:
                continue
            fid, rid = want[name][0], f"fleurs-{Path(name).stem}"
            out = SETS / "audio" / f"{rid}.wav"
            pcm = subprocess.run([str(ffmpeg.binary()), "-nostdin", "-v", "error", "-i", "-", "-af", "aresample=resampler=soxr",
                                  "-ar", "24000", "-ac", "1", "-f", "s16le", "-"], input=t.extractfile(m).read(),
                                 capture_output=True, check=True).stdout
            with wave.open(str(out), "wb") as w:
                w.setnchannels(1), w.setsampwidth(2), w.setframerate(24000), w.writeframes(pcm)
            lines.append({"id": rid, "audio": f"audio/{rid}.wav", "duration": round(len(pcm) / 2 / 24000, 3),
                          "source": {"corpus": "FLEURS", "source_file": f"fleurs/hi_in/test/{name}", "start": 0,
                                     "end": round(len(pcm) / 2 / 24000, 3), "synthetic": False},
                          "sentences": [{"hi_text": want[name][2], "en_text": en[fid]}]})
    lines.sort(key=lambda r: r["id"])
    (SETS / "fleurs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in lines))
    tgz.unlink()
    print(f"fleurs: {len(lines)} of {len(hi)} recordings ({len(hi) - len(want)} without an en_us sentence), "
          f"{sum(r['duration'] for r in lines) / 3600:.3f} h")


def heldout(real_run):
    split = {l.strip(): sp for sp in ("dev", "test") for l in open(HERE / f"splits/{sp}.txt") if l.strip()}
    real = [json.loads(l) for l in store.get(f"{real_run}/records.jsonl").decode().splitlines() if l.strip()]
    pv2 = [json.loads(l) for l in store.get("release/pilot-v2/manifest.jsonl").decode().splitlines()]
    (SETS / "audio").mkdir(parents=True, exist_ok=True)
    out = {"dev": [], "test": []}
    for r in real + [r for r in pv2 if r["id"] in split]:
        assert r["id"] in split, f"{r['id']} is not a dev/test id"
        p = SETS / "audio" / f"{r['id']}.wav"
        if not p.exists():
            src = DATA / "cut" / r["audio"] if not r["source"]["synthetic"] else None
            p.write_bytes(src.read_bytes() if src and src.exists() else store.get(f"release/pilot-v2/audio/{r['id']}.wav"))
        out[split[r["id"]]].append(r)
    for sp, rs in out.items():
        rs.sort(key=lambda r: r["id"])
        (SETS / f"{sp}.jsonl").write_text("".join(release.serialise(r) + "\n" for r in rs))
        syn = [r for r in rs if r["source"]["synthetic"]]
        print(f"{sp}: {len(rs)} records, {sum(r['duration'] for r in rs) / 3600:.3f} h "
              f"(real {len(rs) - len(syn)}, synthetic {len(syn)} / {sum(r['duration'] for r in syn) / 3600:.3f} h)")
    (SETS / "heldout_from.txt").write_text(real_run + "\n")


def upload():
    sha = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    prefix = f"stages/w8c_sets/{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{sha}"
    store.upload_dir(SETS, prefix)
    print(prefix)


if __name__ == "__main__":
    {"fleurs": fleurs, "heldout": heldout, "upload": upload}[sys.argv[1]](*sys.argv[2:])
