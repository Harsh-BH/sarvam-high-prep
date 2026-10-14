# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Self-check for w4/ic_weights.py (the S3 mirror of the gated IndicConformer weights) against FakeS3.

    uv run w4/test_ic_weights.py           # fake bucket only
    uv run w4/test_ic_weights.py --live    # also: every local file in data/w4/ic600m has its sidecar in the real bucket
"""
import json, os, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pipeline")]
import ic_weights as ic  # noqa: E402
from test_s3 import FakeS3  # noqa: E402


def tree(d, files):
    for rel, b in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_bytes(b)


def main():
    d, s3 = Path(tempfile.mkdtemp()), FakeS3()
    repo = {"config.json": b'{"BLANK_ID": 256}', "model_onnx.py": b"# code", "assets/encoder.onnx": b"\x00onnx" * 1000,
            "assets/layers.0.weight": os.urandom(4096)}
    tree(d / "m", {**repo, "__pycache__/model_onnx.cpython-313.pyc": b"pyc", ".cache/huggingface/download/x.metadata": b"rev"})
    assert ic.upload(d / "m", s3) == 4
    stored = {k[len(ic.PREFIX):] for k in s3.objects}
    assert stored == set(repo) | {r + ".meta.json" for r in repo}, stored  # no .cache/, no __pycache__/
    meta = json.loads(s3.objects[ic.PREFIX + "assets/encoder.onnx.meta.json"])
    assert meta == {"repo": ic.REPO, "revision": ic.REV, "path": "assets/encoder.onnx", "sha256": ic.sha256(repo["assets/encoder.onnx"]),
                    "size": 5000, "licence": "MIT", "licence_url": f"https://huggingface.co/{ic.REPO}"}
    assert ic.upload(d / "m", s3) == 0 and ic.check(d / "m", s3) == []  # re-run: nothing written
    del s3.objects[ic.PREFIX + "model_onnx.py.meta.json"]  # crash between data and sidecar
    assert ic.upload(d / "m", s3) == 1 and ic.check(d / "m", s3) == []
    (d / "m/model_onnx.py").write_bytes(b"# changed")
    assert ic.check(d / "m", s3) == ["model_onnx.py"]
    try:
        ic.upload(d / "m", s3)
        raise AssertionError("upload over different stored bytes must refuse")
    except SystemExit:
        pass
    # fetch: byte-identical copy, sidecars and repo-external files left out, nothing left behind
    ic.fetch(d / "f", s3)
    assert {p.relative_to(d / "f").as_posix(): p.read_bytes() for p in (d / "f").rglob("*") if p.is_file()} == repo
    assert not (d / "f.part").exists()
    # ensure(): a local copy wins; without any HF token it fetches the mirror
    os.environ.pop("HF_TOKEN", None)
    os.environ["HF_HOME"] = str(d / "no-hf-login")
    assert ic.ensure(d / "f", s3) == "local" and ic.ensure(d / "e", s3) == "s3" and (d / "e/config.json").exists()
    # a tampered object is refused and never renamed into place
    s3.objects[ic.PREFIX + "assets/encoder.onnx"] = b"evil"
    try:
        ic.fetch(d / "t", s3)
        raise AssertionError("tampered object must be refused")
    except SystemExit as e:
        assert "mismatch" in str(e)
    assert not (d / "t").exists()
    if "--live" in sys.argv:
        bad = ic.check()
        assert not bad, bad[:5]
        print(f"live: {len(ic.files(ic.LOCAL))} local files match their sidecars in s3://{ic.store.BUCKET}/{ic.PREFIX}")
    print("ok: ic_weights")


if __name__ == "__main__":
    main()
