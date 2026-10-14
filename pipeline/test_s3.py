# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Self-check for s3.py against an in-memory fake with the real bucket's write-once semantics.

    uv run pipeline/test_s3.py            # fake only
    uv run pipeline/test_s3.py --live     # also round-trips stages/_selftest/ in the real bucket
"""
import io
import sys
import tempfile
from pathlib import Path

from botocore.exceptions import ClientError

import s3


class FakeS3:
    """The subset of the boto3 S3 client that s3.py uses. Other pipeline tests import it."""

    def __init__(self):
        self.objects = {}

    def _err(self, code):
        return ClientError({"Error": {"Code": code}}, "op")

    def put_object(self, Bucket, Key, Body, IfNoneMatch=None):
        if IfNoneMatch == "*" and Key in self.objects:
            raise self._err("PreconditionFailed")
        if Key.startswith("archive/") and IfNoneMatch != "*":
            raise self._err("AccessDenied")  # mirrors the bucket policy
        self.objects[Key] = Body if isinstance(Body, bytes) else Body.encode()

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._err("NoSuchKey")
        return {"Body": io.BytesIO(self.objects[Key])}

    def head_object(self, Bucket, Key):
        if Key not in self.objects:
            raise self._err("404")
        return {}

    def get_paginator(self, name):
        fake = self

        class P:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k} for k in sorted(fake.objects) if k.startswith(Prefix)]}
        return P()


def check(c, prefix):
    k = f"{prefix}a.txt"
    assert s3.get(k, c) is None and not s3.exists(k, c)
    s3.put(k, b"one", c)
    s3.put(k, b"two", c)  # plain put overwrites
    assert s3.get(k, c) == b"two" and s3.exists(k, c)
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "sub").mkdir()
        (Path(d) / "sub/b.txt").write_bytes(b"bee")
        assert s3.upload_dir(d, prefix + "up", c) == [f"{prefix}up/sub/b.txt"]
        got = s3.download(prefix + "up/", Path(d) / "back", c)
        assert [p.read_bytes() for p in got] == [b"bee"]
    assert list(s3.keys(prefix, c)) == [f"{prefix}a.txt", f"{prefix}up/sub/b.txt"]


def main():
    f = FakeS3()
    check(f, "stages/_t/")
    assert s3.put_once("archive/_t/x", b"1", f) is True
    assert s3.put_once("archive/_t/x", b"2", f) is False and f.objects["archive/_t/x"] == b"1"
    try:
        s3.put("archive/_t/y", b"z", f)
        raise AssertionError("plain put to archive/ must be refused")
    except ClientError:
        pass
    # never fall back to the machine's `default` profile (a different AWS account)
    assert s3.profile({}, on_sagemaker=False) == "harsh-admin"
    assert s3.profile({}, on_sagemaker=True) is None
    assert s3.profile({"AWS_PROFILE": "x"}, on_sagemaker=False) == "x"
    print("ok: fake")
    if "--live" in sys.argv:
        c = s3.client()
        check(c, "stages/_selftest/")
        for k in list(s3.keys("stages/_selftest/", c)):
            c.delete_object(Bucket=s3.BUCKET, Key=k)
        print("ok: live bucket round-trip (stages/_selftest/ removed; old versions expire in 30 days)")


if __name__ == "__main__":
    main()
