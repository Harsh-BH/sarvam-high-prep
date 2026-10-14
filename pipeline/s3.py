"""S3 access for the pipeline: one bucket, write-once where the design needs it (design §1).

Every function takes an optional `s3` client so tests can pass `FakeS3` (test_s3.py).
Credentials come from the environment: AWS_PROFILE=harsh-admin locally, the job role on SageMaker.
"""
from pathlib import Path

BUCKET = "sarvam-parta-114872709824"
REGION = "ap-south-1"
_client = None


def client():
    global _client
    if _client is None:
        import boto3
        _client = boto3.client("s3", region_name=REGION)
    return _client


def _code(e):
    return getattr(e, "response", {}).get("Error", {}).get("Code", "")


def put_once(key, body, s3=None):
    """Write `body` (bytes) only if `key` doesn't exist. -> True if written, False if it already existed.
    The bucket policy rejects any archive/ PUT without If-None-Match, so this is the only way to write there."""
    try:
        (s3 or client()).put_object(Bucket=BUCKET, Key=key, Body=body, IfNoneMatch="*")
        return True
    except Exception as e:
        if _code(e) in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
            return False
        raise


def put(key, body, s3=None):
    """Plain write, for mutable areas only (stages/, code/). Overwrites keep old versions (versioning)."""
    (s3 or client()).put_object(Bucket=BUCKET, Key=key, Body=body)


def get(key, s3=None):
    """-> bytes, or None if the key doesn't exist."""
    try:
        return (s3 or client()).get_object(Bucket=BUCKET, Key=key)["Body"].read()
    except Exception as e:
        if _code(e) in ("NoSuchKey", "404"):
            return None
        raise


def exists(key, s3=None):
    try:
        (s3 or client()).head_object(Bucket=BUCKET, Key=key)
        return True
    except Exception as e:
        if _code(e) in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def keys(prefix, s3=None):
    """Every key under `prefix`, in S3's (lexicographic) order."""
    for page in (s3 or client()).get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            yield o["Key"]


def download(prefix, dest, s3=None):
    """Copy every key under `prefix` to dest/<key minus prefix>, skipping files already present. -> local paths."""
    out = []
    for k in keys(prefix, s3):
        p = Path(dest) / k[len(prefix):].lstrip("/")
        if not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            # ponytail: whole object in memory; fine for clips and JSONL, use download_file for multi-GB sources
            p.write_bytes(get(k, s3))
        out.append(p)
    return out


def upload_dir(src, prefix, s3=None):
    """Plain-write every file under src to prefix/<relative path>. -> keys written."""
    out = []
    for p in sorted(Path(src).rglob("*")):
        if p.is_file():
            k = f"{prefix.rstrip('/')}/{p.relative_to(src).as_posix()}"
            put(k, p.read_bytes(), s3)
            out.append(k)
    return out
