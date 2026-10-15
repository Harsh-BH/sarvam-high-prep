"""S2 voice bank: w3/voices.tsv minus every voice w8c's is_heldout() flags for OmniVoice (tasks/S2-scale-synth.md).

    python3 scale/synth/voices.py    # -> scale/synth/voices-train.tsv; render with `tts.py --voices` it
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "w8c"))
from split import is_heldout  # noqa: E402

MODEL = "k2-fsa/OmniVoice"


def keep(line):
    cols = line.split("\t")
    if line.startswith("#") or cols[0] == "voice" or cols[1] != "*":
        return True  # comments, header, an engine's built-in voices
    return not is_heldout({"source": {"synthetic": True, "model": MODEL, "voice": cols[0]}})


if __name__ == "__main__":
    lines = (ROOT / "w3/voices.tsv").read_text(encoding="utf-8").splitlines(keepends=True)
    out = [l for l in lines if keep(l)]
    assert not keep("iv-S4257045200388748\t*\tm\t") and keep("iv-S4257790700335697\t*\tm\t")  # held out / train (w8c 2026-10-15)
    (HERE / "voices-train.tsv").write_text("".join(out), encoding="utf-8")
    print(f"{len(lines) - len(out)} held-out voices dropped, {sum(l.split(chr(9))[1:2] == ['*'] for l in out)} cloned voices kept")
