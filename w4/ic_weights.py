# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""IndicConformer-600M weights, mirrored write-once in S3: the HF repo is gated, and a job never gets an HF token.

    uv run w4/ic_weights.py upload [DIR]   # DIR (default data/w4/ic600m) -> raw/_models/indic-conformer-600m-multilingual/<REV>/
    uv run w4/ic_weights.py check [DIR]    # every local file has a stored sidecar with the same sha256 (no download)
    uv run w4/ic_weights.py fetch DIR      # S3 -> DIR, each file checked against its sidecar's sha256 and size

Each object F has a sidecar F.meta.json = {repo, revision, path, sha256, size, licence, licence_url}. Licence: MIT
(HF card `license: mit`, checked 2026-10-14). asr_run.indicconformer() calls ensure(): a local copy wins; else HF at
the pinned revision when a token is available (env or `hf auth login`); else this mirror.
"""
import hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import s3 as store  # noqa: E402

REPO = "ai4bharat/indic-conformer-600m-multilingual"
REV = "2a77d305038dfc502fd9464e7d028b90381976cb"  # HF commit; all 404 files matched HF's LFS sha256 / git blob ids
PREFIX = f"raw/_models/indic-conformer-600m-multilingual/{REV}/"
LOCAL = ROOT / "data/w4/ic600m"
SKIP = {".cache", "__pycache__"}  # hub download metadata and python bytecode: not repo files


def files(d):
    return sorted(p for p in Path(d).rglob("*") if p.is_file() and not SKIP & set(p.relative_to(d).parts))


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def upload(d=LOCAL, s3=None):
    """Write-once, data before sidecar (a sidecar means its data is stored). A re-run skips files whose sidecar
    matches and refuses ones that differ. -> number of files written."""
    n = 0
    for p in files(d):
        rel, b = p.relative_to(d).as_posix(), p.read_bytes()
        meta = json.dumps({"repo": REPO, "revision": REV, "path": rel, "sha256": sha256(b), "size": len(b),
                           "licence": "MIT", "licence_url": f"https://huggingface.co/{REPO}"}, indent=1).encode()
        old = store.get(PREFIX + rel + ".meta.json", s3)
        if old is not None:
            if json.loads(old)["sha256"] != sha256(b):
                raise SystemExit(f"{PREFIX + rel}: stored sha256 differs from {p}")
            continue
        if not store.put_once(PREFIX + rel, b, s3) and sha256(store.get(PREFIX + rel, s3)) != sha256(b):
            raise SystemExit(f"{PREFIX + rel} already holds different bytes")  # crashed between data and sidecar
        store.put_once(PREFIX + rel + ".meta.json", meta, s3)
        n += 1
    return n


def check(d=LOCAL, s3=None):
    """Every local repo file has a stored sidecar with its sha256 and size. -> problems (empty = ok)."""
    bad = []
    for p in files(d):
        rel, b = p.relative_to(d).as_posix(), p.read_bytes()
        m = store.get(PREFIX + rel + ".meta.json", s3)
        if m is None or json.loads(m)["sha256"] != sha256(b) or json.loads(m)["size"] != len(b):
            bad.append(rel)
    return bad


def fetch(dest, s3=None):
    """S3 mirror -> dest, through dest.part/ (resumable; only fully verified copies are renamed into place)."""
    dest, tmp = Path(dest), Path(str(dest) + ".part")
    keys = [k for k in store.keys(PREFIX, s3) if not k.endswith(".meta.json")]
    if not keys:
        raise SystemExit(f"no S3 mirror at {PREFIX}")
    for k in keys:
        m = store.get(k + ".meta.json", s3)
        if m is None:
            raise SystemExit(f"{k}: no sidecar")
        m, p = json.loads(m), tmp / k[len(PREFIX):]
        if p.exists() and sha256(p.read_bytes()) == m["sha256"]:
            continue
        b = store.get(k, s3)
        if sha256(b) != m["sha256"] or len(b) != m["size"]:
            raise SystemExit(f"{k}: sha256/size mismatch with its sidecar, refusing")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b)
    tmp.rename(dest)


def ensure(dest=LOCAL, s3=None):
    """Make sure the weights are at `dest`. -> where they came from: 'local', 'hf' or 's3'."""
    if (Path(dest) / "config.json").exists():
        return "local"
    try:
        from huggingface_hub import get_token
    except ImportError:
        get_token = lambda: os.environ.get("HF_TOKEN")  # noqa: E731
    if get_token():
        from huggingface_hub import snapshot_download
        snapshot_download(REPO, revision=REV, local_dir=dest)
        return "hf"
    fetch(dest, s3)
    return "s3"


if __name__ == "__main__":
    cmd, d = sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else LOCAL
    if cmd == "upload":
        print(f"{upload(d)} files written under s3://{store.BUCKET}/{PREFIX}")
    elif cmd == "check":
        bad = check(d)
        print(f"{len(files(d))} files checked against stored sidecars:", "ok" if not bad else f"{len(bad)} bad, e.g. {bad[:3]}")
        sys.exit(1 if bad else 0)
    elif cmd == "fetch":
        fetch(d)
        print(f"fetched -> {d}")
