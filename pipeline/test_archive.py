# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
import json, tempfile
from pathlib import Path
import archive
from test_s3 import FakeS3


def caller(log, fail_first=0, payload=b"WAV"):
    def call(req):
        log.append(req)
        if len(log) <= fail_first:
            raise ConnectionError("flaky")
        return {"ok": True, "n": len(log)}, payload
    return call


def main():
    f, spool = FakeS3(), Path(tempfile.mkdtemp())
    kw = dict(stream="w3", inr=1.5, cap_inr=10, s3=f, spool=spool, sleep=lambda s: None)
    req = {"model": "bulbul:v3", "text": "पाँच लाख", "take": 1}
    assert archive.request_key(req) == archive.request_key(dict(reversed(list(req.items()))))  # key order irrelevant

    log = []
    r1 = archive.archive_call("bulbul", req, caller(log), **kw)
    r2 = archive.archive_call("bulbul", req, caller(log), **kw)
    assert len(log) == 1 and r1 == r2 == ({"ok": True, "n": 1}, b"WAV")  # cache hit, one paid call
    k = archive.request_key(req)
    meta = json.loads(f.objects[f"archive/api/bulbul/{k}.json"])
    assert meta["inr"] == 1.5 and meta["stream"] == "w3" and meta["payload"] == f"{k}.bin"

    archive.archive_call("bulbul", {**req, "take": 2}, caller(log), **kw)  # a new take is a new request
    assert len(log) == 2 and archive.spent("w3", f) == 3.0

    log3 = []  # retries: two failures, then success; one archive write
    archive.archive_call("bulbul", {**req, "take": 3}, caller(log3, fail_first=2), **kw)
    assert len(log3) == 3

    try:  # cap: 4.5 spent, next 1.5 would exceed 5.0
        archive.archive_call("bulbul", {**req, "take": 4}, caller([]), **{**kw, "cap_inr": 5.0})
        raise AssertionError("cap not enforced")
    except archive.SpendCapError:
        pass

    # race: another process archived first; we must return ITS response
    req5 = {**req, "take": 5}
    k5 = archive.request_key(req5)
    real_put_once = archive.store.put_once
    def loser(key, body, s3=None):
        if key.endswith(".json"):
            f.objects[key] = json.dumps({"response": {"winner": True}, "payload": None, "stream": "w3", "inr": 1.5}).encode()
            return False
        return real_put_once(key, body, s3)
    archive.store.put_once = loser
    try:
        assert archive.archive_call("bulbul", req5, caller([]), **{**kw, "cap_inr": 99}) == ({"winner": True}, None)
    finally:
        archive.store.put_once = real_put_once

    # race on the payload: another writer stored <key>.bin first; the caller must get the archived bytes, not its own
    req6 = {**req, "take": 6}
    f.objects[f"archive/api/bulbul/{archive.request_key(req6)}.bin"] = b"OTHER"
    assert archive.archive_call("bulbul", req6, caller([]), **{**kw, "cap_inr": 99}) == ({"ok": True, "n": 1}, b"OTHER")

    # spool: a response that reached disk but not S3 is uploaded by flush_spool
    (spool / "bulbul").mkdir(exist_ok=True)
    (spool / "bulbul" / "abc.json").write_text(json.dumps({"response": {}, "payload": None, "stream": "w3", "inr": 0}))
    assert archive.flush_spool(f, spool) == 1 and "archive/api/bulbul/abc.json" in f.objects
    assert not any(spool.rglob("*.json"))
    print("ok: archive")


if __name__ == "__main__":
    main()
