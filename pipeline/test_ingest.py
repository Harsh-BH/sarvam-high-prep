# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""uv run pipeline/test_ingest.py: ingest from a fake archive into FakeS3, verify, ETag and licence guards."""
import hashlib, io, json, tarfile, tempfile
from pathlib import Path
import ingest_iv
from test_s3 import FakeS3


def make_tgz():
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in (("Hindi/v4/train/a_0.wav", b"RIFFaaaa"), ("Hindi/v4/train/a_0.json", b"{}"),
                           ("Hindi/v4/train/b_0.wav", b"RIFFbbbb"), ("Hindi/v4/train/b_0.json", b"{}")):
            ti = tarfile.TarInfo(name); ti.size = len(data); t.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


class Resp(io.BytesIO):
    def __init__(self, b, etag):
        super().__init__(b); self.headers = {"ETag": f'"{etag}"'}
    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def main():
    tgz, url, f = make_tgz(), "https://x/v4_Hindi_train.tgz", FakeS3()
    opener = lambda u: Resp(tgz, "E1")
    n = ingest_iv.ingest({url: {"Hindi/v4/train/a_0.wav"}}, rev={url: "E1"}, s3=f, opener=opener)
    assert n == 4  # wav + json, each with a sidecar
    meta = json.loads(f.objects["raw/indicvoices/Hindi/v4/train/a_0.wav.meta.json"])
    assert meta["sha256"] == hashlib.sha256(b"RIFFaaaa").hexdigest() and meta["etag"] == "E1" and meta["member"] == "Hindi/v4/train/a_0.wav"
    assert "raw/indicvoices/Hindi/v4/train/b_0.wav" not in f.objects  # only wanted members
    assert ingest_iv.ingest({url: {"Hindi/v4/train/a_0.wav"}}, rev={url: "E1"}, s3=f, opener=opener) == 0  # idempotent
    assert ingest_iv.verify(s3=f) == []
    f.objects["raw/indicvoices/Hindi/v4/train/a_0.wav"] = b"tampered"
    assert ingest_iv.verify(s3=f) == ["raw/indicvoices/Hindi/v4/train/a_0.wav"]
    try:  # etag_guard
        ingest_iv.ingest({url: {"Hindi/v4/train/b_0.wav"}}, rev={url: "OLD"}, s3=f, opener=opener)
        raise AssertionError("ETag change not refused")
    except SystemExit:
        pass
    try:  # a wanted member the archive doesn't hold is an error, not a silent gap in raw/
        ingest_iv.ingest({url: {"Hindi/v4/train/c_0.wav"}}, rev={url: "E1"}, s3=f, opener=opener)
        raise AssertionError("member missing from the archive not reported")
    except SystemExit:
        pass
    for bad in ("kathbath", "gram-vaani", "nope"):
        try:
            ingest_iv.gate(bad); raise AssertionError(f"{bad} passed the licence gate")
        except SystemExit:
            pass
    ingest_iv.gate("indicvoices")
    with tempfile.TemporaryDirectory() as d:  # --local: members sit at <local>/<member path>, as extracted
        (Path(d) / "Hindi/v4/train").mkdir(parents=True)
        (Path(d) / "Hindi/v4/train/b_0.wav").write_bytes(b"RIFFbbbb")
        try:
            ingest_iv.ingest({url: {"Hindi/v4/train/b_0.wav"}}, rev={url: "E1"}, s3=f, local=Path(d))
            raise AssertionError("missing local member (b_0.json) not refused")
        except SystemExit:
            pass
        (Path(d) / "Hindi/v4/train/b_0.json").write_bytes(b"{}")
        assert ingest_iv.ingest({url: {"Hindi/v4/train/b_0.wav"}}, rev={url: "E1"}, s3=f, local=Path(d)) == 4  # refusal wrote nothing
        assert json.loads(f.objects["raw/indicvoices/Hindi/v4/train/b_0.json.meta.json"])["url"] == url
    print("ok: ingest")


if __name__ == "__main__":
    main()
