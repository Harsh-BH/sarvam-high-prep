# /// script
# requires-python = ">=3.12"
# ///
"""Self-check for the `translate` stage (w6/translate.py) with a fake LLM: no network, no spend.
    uv run w6/test_translate.py"""
import json, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "w6")]
import archive, translate as tr  # noqa: E402

HI = ["मेरे खाते से साढ़े तीन लाख रुपये कट गए नहीं नहीं पौने चार लाख कटे हैं।", "मैंने नहीं बताया।"]
REC = {"id": "rec-400000", "audio": "audio/rec-400000.wav", "duration": 5.0,
       "source": {"corpus": "t", "source_file": "tts/rec-400000.wav", "start": 0, "end": 5.0, "synthetic": True,
                  "model": "m", "voice": "v"}, "sentences": [{"hi_text": h} for h in HI]}
calls = []


def fake_llm(name, system, user, take, max_tokens):
    calls.append((name, "judge" if system == tr.JUDGE_SYS else "repair" if "previous translation" in user else "translate"))
    kind = calls[-1][1]
    if kind == "translate":  # take 0: wrong value in en, an alt pulling a number from nowhere
        segs = [{"en": "Uh, four lakh rupees were debited from my account.",
                 "alt": "Three and a half lakh rupees, no, no, three and three-quarter lakh, five rupees."},
                {"en": "I didn’t tell them."}]
    elif kind == "repair":
        segs = [{"en": "Three and three-quarter lakh rupees were debited from my account.",
                 "alt": "Three and a half lakh rupees were debited, no, no, three and three-quarter lakh."},
                {"en": "I didn't tell them."}]
    else:
        segs = json.loads(user)["segments"]
        bad = "four lakh" in segs[0]["en"].lower()
        return json.dumps({"segments": [
            {"i": 0, "errors": [{"code": "number", "field": "en", "en_span": "four lakh", "hi_span": "पौने चार लाख",
                                 "fix": "three and three-quarter lakh"}] if bad else []},
            {"i": 1, "errors": [{"code": "number", "field": "en", "en_span": "them", "hi_span": "बताया", "fix": "them"}]}]}), "k"
    return json.dumps({"segments": segs}), "k"


tr.llm = fake_llm
archive.spent = lambda *a, **k: 0.0
args = type("A", (), {"engine": "mistral3", "judge": "gptoss", "no_repair": False})()

# 1. translate -> judge flags the number -> repair -> re-judge clean: repaired, filler and curly quote cleaned
rec, rej = tr.process(json.loads(json.dumps(REC)), args)
assert rej is None
w = rec["_qa"]["w6"]
assert w["repaired"] and rec["translation_method"] == "w6-style-v1/mistral.mistral-large-3-675b-instruct+repair", rec["translation_method"]
assert rec["sentences"][0]["en_text"] == "Three and three-quarter lakh rupees were debited from my account."
assert rec["sentences"][0]["en_alt"] == ["Three and a half lakh rupees were debited, no, no, three and three-quarter lakh."]
assert rec["sentences"][1]["en_text"] == "I didn't tell them."
assert w["verdict"][1]["errors"] == [] and w["verdict"][1]["dropped"] == 1  # fix == span: hallucinated, dropped
assert [c[1] for c in calls] == ["translate", "judge", "repair", "judge"], calls

# 2. no repair: take 0 kept; its alt has a number not in the Hindi (num_extra), so the alt is dropped
calls.clear()
args.no_repair = True
rec, _ = tr.process(json.loads(json.dumps(REC)), args)
assert rec["sentences"][0]["en_text"] == "Four lakh rupees were debited from my account."  # "Uh," stripped (rule 3)
assert "en_alt" not in rec["sentences"][0] and rec["_qa"]["w6"]["alt_dropped"] == [0]
assert not rec["translation_method"].endswith("+repair") and rec["_qa"]["w6"]["critical"] >= 1

# 3. parse: wrong segment count is a hard failure
try:
    tr.parse('{"segments": [{"en": "a"}]}', 2)
    raise AssertionError("count mismatch accepted")
except ValueError:
    pass

# 4. patches: applied when the auto text is unchanged, refused (stale) otherwise
p = {rec["id"]: [{"id": rec["id"], "i": 1, "en_text": "I didn't tell anyone.", "was": "I didn't tell them."},
                 {"id": rec["id"], "i": 0, "en_text": "x", "was": "something else"}]}
rec = tr.apply_patches(rec, p)
assert rec["sentences"][1]["en_text"] == "I didn't tell anyone." and rec["translation_method"].endswith("+human")
assert rec["sentences"][0]["en_text"] != "x" and rec["_qa"]["w6"]["stale_patch"] == [0]

# 5. the stage CLI resumes: ids already written are skipped
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    (d / "in.jsonl").write_text(json.dumps(REC, ensure_ascii=False) + "\n" + json.dumps({**REC, "id": "rec-400001"}, ensure_ascii=False) + "\n")
    (d / "out").mkdir()
    (d / "out/records.jsonl").write_text(json.dumps({**REC, "id": "rec-400000"}, ensure_ascii=False) + "\n")
    calls.clear()
    sys.argv = ["translate.py", "--in", str(d / "in.jsonl"), "--audio-dir", str(d), "--out", str(d / "out"), "--workers", "1",
                "--engine", "mistral3", "--judge", "gptoss"]
    tr.main()
    ids = [json.loads(l)["id"] for l in open(d / "out/records.jsonl")]
    assert ids == ["rec-400000", "rec-400001"], ids
    assert calls[0] == ("mistral3", "translate") and len([c for c in calls if c[1] == "translate"]) == 1
print("ok: translate")
