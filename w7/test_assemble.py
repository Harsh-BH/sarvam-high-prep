# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Self-check for assemble.bind_take: bind once, accept the same take, burn an id bound to other audio."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import assemble  # noqa: E402
from test_s3 import FakeS3  # noqa: E402


def rec(i, take_key):
    return {"id": f"rec-40000{i}", "source": {"source_file": f"tts/rec-40000{i}.wav", "synthetic": True},
            "_archive": {"tts": take_key} if take_key else {}}


def main():
    f = FakeS3()
    f.objects["archive/tts/rec-400001-aaaa.wav"] = b"TAKE-A"
    assert assemble.bind_take(rec(1, "archive/tts/rec-400001-aaaa.wav"), f) is True      # unbound -> bound now
    assert f.objects["archive/tts/rec-400001.wav"] == b"TAKE-A" and "archive/tts/rec-400001.wav.meta.json" in f.objects
    assert assemble.bind_take(rec(1, "archive/tts/rec-400001-aaaa.wav"), f) is True      # same take again -> fine
    f.objects["archive/tts/rec-400001-bbbb.wav"] = b"TAKE-B"
    assert assemble.bind_take(rec(1, "archive/tts/rec-400001-bbbb.wav"), f) is False     # bound to other audio -> burned
    assert f.objects["archive/tts/rec-400001.wav"] == b"TAKE-A"                          # never rebound
    f.objects["archive/tts/rec-400002.wav"] = b"ROUND1"
    assert assemble.bind_take(rec(2, None), f) is True                                   # round-1: binding is the take
    try:
        assemble.bind_take(rec(3, None), f)
        raise AssertionError("unbound synthetic with no take accepted")
    except SystemExit:
        pass
    print("ok: assemble bind")


if __name__ == "__main__":
    main()
