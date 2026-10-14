# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Ingest the IndicVoices source files we cut from into raw/, pinned and verifiable (design §2.1).

    uv run pipeline/ingest_iv.py --records data/w1/pool/manifest.jsonl [--local data/w1/iv_train]
    uv run pipeline/ingest_iv.py --verify

--local reads members already extracted under DIR/<member path> (DIR/Hindi/v4/train/<name>.wav)
instead of streaming the archive; the sidecar still records the archive URL and pinned ETag.
"""
import argparse, collections, hashlib, json, sys, tarfile, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import s3 as store

CORPORA = json.loads((Path(__file__).resolve().parent / "corpora.json").read_text())


def gate(corpus):
    st = CORPORA.get(corpus, {}).get("status")
    if st != "OK":
        raise SystemExit(f"{corpus}: catalog status {st!r}, ingestion refused (design §2.1)")


def _put(member, data, url, etag, s3):
    key = f"raw/indicvoices/{member}"
    if store.exists(key + ".meta.json", s3):
        return 0
    store.put_once(key, data, s3)  # False after a crash between the two writes; verify() catches a mismatch
    side = {"corpus": "indicvoices", "url": url, "etag": etag, "member": member, "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data), "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    store.put_once(key + ".meta.json", json.dumps(side).encode(), s3)  # last: its presence marks the member done
    return 2


def ingest(wanted, *, rev, s3=None, opener=urllib.request.urlopen, local=None):
    """wanted: archive URL -> member names (+ their sibling .json). -> objects written (members + sidecars)."""
    gate("indicvoices")
    n = 0
    for url, members in wanted.items():
        want = set(members) | {m.rsplit(".", 1)[0] + ".json" for m in members}
        if local:
            missing = sorted(m for m in want if not (Path(local) / m).exists())
            if missing:
                raise SystemExit(f"{len(missing)} members missing under {local}, nothing written for {url}: {missing[:5]}")
            for m in sorted(want):
                n += _put(m, (Path(local) / m).read_bytes(), url, rev[url], s3)
            continue
        with opener(url) as r:
            etag = r.headers.get("ETag", "").strip('"')
            if etag != rev[url]:
                raise SystemExit(f"{url}: ETag {etag!r} != pinned {rev[url]!r}; upstream changed, refusing")
            with tarfile.open(fileobj=r, mode="r|gz") as t:
                for ti in t:
                    if ti.name in want:
                        n += _put(ti.name, t.extractfile(ti).read(), url, etag, s3)
                        want.discard(ti.name)
                        if not want:
                            break
        if want:
            raise SystemExit(f"{url}: {len(want)} wanted members not in the archive: {sorted(want)[:5]}")
    return n


def verify(prefix="raw/indicvoices/", s3=None):
    """-> keys whose bytes don't match their sidecar's sha256 (or have no sidecar)."""
    bad = []
    for k in store.keys(prefix, s3):
        if k.endswith(".meta.json"):
            continue
        side = store.get(k + ".meta.json", s3)
        if side is None or hashlib.sha256(store.get(k, s3)).hexdigest() != json.loads(side)["sha256"]:
            bad.append(k)
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--local", type=Path)
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.verify:
        bad = verify()
        print("\n".join(bad) or "ok: every raw/indicvoices object matches its sidecar")
        sys.exit(1 if bad else 0)
    if not a.records:
        ap.error("--records or --verify")
    wanted, rev = collections.defaultdict(set), {}
    for l in open(a.records):
        r = json.loads(l)
        url, etag = r["_rev"].split(" etag=")  # W1: '<url> etag="<etag>"'
        wanted[url].add(r["source"]["source_file"].removeprefix("indicvoices/"))
        rev[url] = etag.strip('"')
    print(f"{ingest(wanted, rev=rev, local=a.local)} objects written")


if __name__ == "__main__":
    main()
