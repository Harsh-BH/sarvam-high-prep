# /// script
# requires-python = ">=3.12"
# dependencies = ["torch", "torchaudio", "numpy"]
# ///
"""Self-check for w5/align.py on hand-made emissions (no model).  uv run w5/test_align.py"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import align as A  # noqa: E402
import validate  # noqa: E402

TOK = ["<b>", "|", "क", "ख", "ग"]
IDX = {t: i for i, t in enumerate(TOK)}


def emissions(path):
    """One frame per symbol of `path` ('-' = blank), 0.9 on that token."""
    lp = torch.full((len(path), len(TOK)), 0.1 / (len(TOK) - 1)).log()
    for f, c in enumerate(path):
        lp[f, IDX["<b>" if c == "-" else c]] = torch.tensor(0.9).log()
    return lp


def main():
    lp = emissions("--क-|ख--ग|ग---")  # words: क, खग, ग
    spans, _ = A.viterbi(lp, ["क", "खग", "ग"], IDX)
    assert [s[:3] for s in spans] == [(2, 2, 3), (1, 4, 5), (3, 5, 6), (4, 8, 9), (1, 9, 10), (4, 10, 11)], spans
    ws = A.word_spans(spans, IDX["|"])
    assert [w[:2] for w in ws] == [(2, 3), (5, 9), (10, 11)]
    assert A.viterbi(emissions("-क-"), ["कक", "ख"], IDX) is None  # 3 tokens + 1 repeat need 5 frames: infeasible

    # gap filling: silence 0-4, speech 5-24, silence 25-29, speech 30-39, silence 40-44
    db = torch.tensor([-60.0] * 5 + [-20.0] * 20 + [-60.0] * 5 + [-20.0] * 10 + [-60.0] * 5)
    assert A.fill_gaps([(6, 8), (12, 15), (31, 33)], db, 10, 3) == [[5, 10], [10, 18], [30, 36]]  # mid split, caps
    assert A.fill_gaps([(6, 8), (12, 15), (31, 33)], db, 10, 50) == [[5, 10], [10, 25], [30, 40]]  # stops at silence
    assert A.fill_gaps([(6, 7), (20, 22)], db, 10, 3) == [[5, 13], [13, 25]]  # a voiced gap splits uncapped

    # QA flags: an unspoken word (all characters ~0, squeezed) and a 40 ms filler; a spoken low-posterior filler passes
    ws = [(0, 2, 0.001, 0.002, 2), (2, 9, 0.001, 0.002, 2), (9, 10, 0.0, 0.0, 1), (10, 20, 0.0, 0.0, 1)]
    times = [(0.0, 0.04), (0.04, 0.18), (0.18, 0.22), (0.22, 0.4)]
    assert A.word_flags(["कल", "कल", "अ", "अ"], ws, times) == [[0, "unspoken"], [2, "filler_short"]]
    assert A.verdict({"regret": 0.1, "flags": [[2, "filler_short"]]}) == "filler_short"
    assert A.verdict({"regret": A.REGRET_MAX + 0.1, "flags": []}) == "align_regret"

    assert A.finalize([(0.0, 0.2), (0.2, 0.62)], 0.6) == [(0.0, 0.2), (0.2, 0.6)]  # clamped to the clip
    try:
        A.finalize([(0.6, 0.7)], 0.6)
        raise AssertionError("empty span accepted")
    except ValueError:
        pass

    rec = {"id": "rec-000001", "duration": 2.0, "sentences": [{"hi_text": "क खग।"}, {"hi_text": "ग?", "en_text": "x"}]}
    A.set_times(rec, [(0.04, 0.06), (0.1, 0.18), (0.2, 0.2224)])
    assert rec["hi_words"] == [{"w": "क", "start": 0.04, "end": 0.06}, {"w": "खग", "start": 0.1, "end": 0.18},
                               {"w": "ग", "start": 0.2, "end": 0.222}]
    assert list(rec["sentences"][1]) == ["hi_text", "hi_start", "hi_end", "en_text"]  # SPEC §5 key order
    assert validate.check_record(rec, ".", partial=True) == []
    print("ok: align")


if __name__ == "__main__":
    main()
