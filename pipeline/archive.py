"""archive_call: every paid API call goes through here (design §2.2). Never pays twice, never loses a paid output."""
import hashlib, json, time
from datetime import datetime, timezone
from pathlib import Path
import s3 as store

SPOOL = Path(__file__).resolve().parent.parent / "data/archive_spool"  # local write-ahead copy, survives a crash
_spent = {}  # stream -> INR, filled lazily from the archive


class SpendCapError(Exception):
    pass


def request_key(request):
    return hashlib.sha256(json.dumps(request, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _load(service, key, s3=None):
    meta = store.get(f"archive/api/{service}/{key}.json", s3)
    if meta is None:
        return None
    meta = json.loads(meta)
    payload = store.get(f"archive/api/{service}/{meta['payload']}", s3) if meta.get("payload") else None
    return meta["response"], payload


def spent(stream, s3=None):
    if stream not in _spent:
        # ponytail: one GET per archived call, fine for a few thousand calls; keep a spend index if it gets slow
        total = 0.0
        for k in store.keys("archive/api/", s3):
            if k.endswith(".json"):
                m = json.loads(store.get(k, s3))
                total += m.get("inr", 0) if m.get("stream") == stream else 0
        _spent[stream] = total
    return _spent[stream]


def _upload(service, key, meta_bytes, payload, s3=None):
    """-> True if our copy is now the archived one, False if another writer got there first (to either file).
    The .json is written even when the .bin was lost, so a crashed .bin winner can't leave the key without metadata."""
    bin_won = store.put_once(f"archive/api/{service}/{key}.bin", payload, s3) if payload is not None else True
    return store.put_once(f"archive/api/{service}/{key}.json", meta_bytes, s3) and bin_won


def _flush_one(p, s3=None):
    """Upload one spool entry (<spool>/<service>/<key>.json [+ .bin]), then drop it. -> _upload's result."""
    b = p.with_suffix(".bin")
    won = _upload(p.parent.name, p.stem, p.read_bytes(), b.read_bytes() if b.exists() else None, s3)
    p.unlink(missing_ok=True)  # another process may flush the same entry
    b.unlink(missing_ok=True)
    return won


def flush_spool(s3=None, spool=SPOOL):
    paths = sorted(Path(spool).glob("*/*.json"))
    for p in paths:
        _flush_one(p, s3)
    return len(paths)


def archive_call(service, request, call, *, stream, inr, cap_inr, s3=None, spool=SPOOL, retries=4, sleep=time.sleep):
    key = request_key(request)
    hit = _load(service, key, s3)
    if hit is not None:
        return hit
    d = Path(spool) / service
    if (d / f"{key}.json").exists():  # paid for before a crash, never reached S3: archive it, don't pay again
        _flush_one(d / f"{key}.json", s3)
        return _load(service, key, s3)
    for attempt in range(retries + 1):
        if spent(stream, s3) + inr > cap_inr:
            raise SpendCapError(f"{stream}: {spent(stream, s3):.2f} + {inr:.2f} INR > cap {cap_inr:.2f}")
        _spent[stream] += inr  # charged per attempt: a call can bill and still raise (timeout after the server finished)
        try:
            response, payload = call(request)
            break
        except Exception:
            if attempt == retries:
                raise
            sleep(2 ** attempt)
    meta = json.dumps({"service": service, "request": request, "response": response, "stream": stream, "inr": inr,
                       "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "payload": f"{key}.bin" if payload is not None else None}, ensure_ascii=False).encode()
    d.mkdir(parents=True, exist_ok=True)
    if payload is not None:
        (d / f"{key}.bin").write_bytes(payload)
    (d / f"{key}.json").write_bytes(meta)  # written last: its presence means the spool entry is complete
    won = _flush_one(d / f"{key}.json", s3)
    return (response, payload) if won else _load(service, key, s3)
