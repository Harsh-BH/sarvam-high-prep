# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Check w8d/submit.py pairs(): only train ids become training pairs, a held-out speaker/voice/id stops everything,
dev keeps only ids that already have English.    uv run w8d/test_submit.py"""
import json, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import submit


def rec(i, spk=None, voice=None, en="Hi."):
    src = {"synthetic": voice is not None, **({"model": "k2-fsa/OmniVoice", "voice": voice} if voice else {})}
    return {"id": i, "speaker_id": spk, "source": src, "sentences": [{"en_text": en}, {"en_text": "Ok."}] if en else [{}]}


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "train.txt").write_text("rec-1\nrec-2\nrec-3\n")
        (d / "dev.txt").write_text("rec-4\nrec-5\n")
        (d / "test.txt").write_text("rec-6\n")
        (d / "heldout.json").write_text(json.dumps({"speakers": ["S9"], "voices": [["k2-fsa/OmniVoice", "iv-S7"]], "scripts": []}))
        R = {i: (r, "release/x/audio/") for i, r in [("rec-1", rec("rec-1", "spk-indicvoices-S1")), ("rec-2", rec("rec-2", voice="iv-S2")),
                                                       ("rec-4", rec("rec-4", "spk-indicvoices-S9")), ("rec-5", rec("rec-5", en=None)),
                                                       ("rec-9", rec("rec-9"))]}
        train, dev = submit.pairs(R, d)
        assert [p["id"] for p in train] == ["rec-1", "rec-2"], train  # rec-3 absent from sources, rec-9 not in train.txt
        assert train[0] == {"id": "rec-1", "audio_key": "release/x/audio/rec-1.wav", "en": "Hi. Ok."}, train[0]
        assert [p["id"] for p in dev] == ["rec-4"], dev  # rec-5 has no English yet
        for bad in (rec("rec-1", "spk-indicvoices-S9"), rec("rec-1", voice="iv-S9"), rec("rec-1", voice="iv-S7")):
            try:
                submit.pairs({**R, "rec-1": (bad, "a/")}, d)
                raise AssertionError(f"held-out record accepted: {bad}")
            except SystemExit:
                pass
        (d / "train.txt").write_text("rec-1\nrec-6\n")
        R["rec-6"] = (rec("rec-6"), "a/")
        try:
            submit.pairs(R, d)
            raise AssertionError("test id accepted for training")
        except SystemExit:
            pass
    print("ok")


if __name__ == "__main__":
    main()
