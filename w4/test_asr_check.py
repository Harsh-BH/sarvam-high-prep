# /// script
# requires-python = ">=3.12"
# ///
"""Self-check for the `asr_check` stage (w4/asr_check.py) with archived hypotheses: no model is loaded (this script's
environment has no torch, so a decode attempt would fail the test).  uv run w4/test_asr_check.py"""
import json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def rec(n, text, normalized=None, **w4):
    return {"id": f"rec-90000{n}", "sentences": [{"hi_text": text}], "_w4": {"flags": [], **w4},
            "_w1": {"segments": [{"verbatim": text[:-1], "normalized": normalized or text[:-1]}], "scenario": "Extempore",
                    "orig_sample_rate": 44100}}


def main():
    d = Path(tempfile.mkdtemp())
    good = "मेरा नाम राम है और मैं दिल्ली में रहता हूँ।"
    recs = [rec(1, good),
            rec(2, good),                                                     # both routers hear something else -> P1
            rec(3, "मुझे पचास रुपये दो।"),                                   # draft hears a different number -> P2
            rec(4, good, review={"action": "ok", "reviewer": "t", "date": "d"}),  # a human listened: overrides P1
            rec(5, good, flags=["stammers@1"]),                               # guard -> P0
            rec(6, "मैंने कलर और साइज देखा।", "मैंने कलर [colour] और साइज़ [size] देखा")]  # 2/2 English words unmatched -> P2
    other = "क ख ग घ च छ ज झ ट ठ ड ढ त थ द ध न प फ ब भ म"
    hyp = {"vakyansh-w2v2": {2: other, 4: other}, "whisper-turbo": {2: other, 4: other},
           "ic-ctc": {3: "मुझे साठ रुपये दो", 6: "मैंने तोता और मैना देखा"}}
    (d / "in.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    (d / "out/asr").mkdir(parents=True)
    for m, special in hyp.items():
        (d / f"out/asr/{m}.jsonl").write_text("".join(json.dumps({"id": r["id"], "hyp": special.get(int(r["id"][-1]), r["sentences"][0]["hi_text"][:-1])},
                                                                 ensure_ascii=False) + "\n" for r in recs))
    cmd = [sys.executable, str(HERE / "asr_check.py"), "--in", str(d / "in.jsonl"), "--audio-dir", str(d), "--out", str(d / "out")]
    subprocess.run(cmd, check=True)
    ok = {json.loads(l)["id"]: json.loads(l) for l in open(d / "out/records.jsonl")}
    bad = {json.loads(l)["id"]: json.loads(l) for l in open(d / "out/rejects.jsonl")}
    assert sorted(ok) == ["rec-900001", "rec-900004"], sorted(ok)
    assert {i: x["rule"] for i, x in bad.items()} == {"rec-900002": "P1", "rec-900003": "P2", "rec-900005": "P0", "rec-900006": "P2"}, bad
    assert ok["rec-900001"]["_qa"]["asr_check"]["cer"] == {"vakyansh-w2v2": 0.0, "whisper-turbo": 0.0}
    assert bad["rec-900003"]["detail"]["number_conflicts"] == 1 and bad["rec-900006"]["detail"]["entity_misses"] == 2
    assert bad["rec-900002"]["stage"] == "asr_check" and bad["rec-900002"]["detail"]["cer"]["whisper-turbo"] > 0.35
    before = [(d / f"out/{f}.jsonl").read_bytes() for f in ("records", "rejects")]
    subprocess.run(cmd, check=True)  # resume: nothing duplicated
    assert [(d / f"out/{f}.jsonl").read_bytes() for f in ("records", "rejects")] == before
    lines = before[1].decode().splitlines(keepends=True)
    (d / "out/rejects.jsonl").write_text("".join(lines[:-1]) + lines[-1][:30])  # crash tore the last line
    subprocess.run(cmd, check=True)
    assert (d / "out/rejects.jsonl").read_bytes() == before[1]
    print("ok: asr_check")


if __name__ == "__main__":
    main()
