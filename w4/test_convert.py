# /// script
# requires-python = ">=3.12"
# ///
"""Self-check for the `transcribe` stage (w4/convert.py). No model: --no-sbd.  uv run w4/test_convert.py"""
import hashlib, json, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ZWJ = chr(0x200D)


def rec(n, *segs):
    """W1-style record: per segment (L1 verbatim, L2 normalized)."""
    return {"id": f"rec-90000{n}", "_orig_text": " ".join(v for v, _ in segs),
            "_w1": {"segments": [{"start": 0.0, "end": 1.0, "verbatim": v, "normalized": m} for v, m in segs]}}


def sha(t):
    return hashlib.sha1(t.encode()).hexdigest()[:12]


def run(d, *extra):
    cmd = [sys.executable, str(HERE / "convert.py"), "--in", str(d / "in.jsonl"), "--audio-dir", str(d), "--out", str(d / "out"),
           "--no-sbd", "--patches", str(d / "patches.jsonl"), *extra]
    subprocess.run(cmd, check=True)
    return ({json.loads(l)["id"]: json.loads(l) for l in open(d / "out/records.jsonl")},
            {json.loads(l)["id"]: json.loads(l) for l in open(d / "out/rejects.jsonl")})


def main():
    d = Path(tempfile.mkdtemp())
    recs = [rec(1, (f"<noise> हिन्दी में [uhh] बात कर-कर के अप{ZWJ}लीकेसन भेजो", "हिन्दी में बात"),
                ("मेरा पी.आर.ए.एन. नंबर छ: है।", "मेरा पी.आर.ए.एन. नंबर छ: है।")),
            rec(2, ("मतलब [stammers] ज्यादा", "मतलब [stammers] ज्यादा")),
            rec(3, ("हम <talking> चलते हैं", "हम <talking> चलते हैं")),
            rec(4, ("मेरा आईडी है", "मेरा आईडी [ID] है")),
            rec(5, ("ठीक है", "ठीक है")), rec(6, ("ठीक है", "ठीक है")), rec(7, ("ठीक है", "ठीक है"))]
    (d / "in.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    (d / "patches.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in [
        {"id": "rec-900005", "base": sha("ठीक है।"), "action": "fix", "reviewer": "t", "date": "d", "sentences": ["ठीक है?"],
         "archive": "archive/human/transcribe/d-harsh.jsonl"},
        {"id": "rec-900006", "base": "000000000000", "action": "ok", "reviewer": "t", "date": "d"},
        {"id": "rec-900007", "base": sha("ठीक है।"), "action": "drop", "reviewer": "t", "date": "d"}]))
    ok, bad = run(d)
    assert ok["rec-900001"]["sentences"] == [{"hi_text": "हिंदी में अ बात कर कर के अपलीकेसन भेजो मेरा पी आर ए एन नंबर छः है।"}], ok["rec-900001"]
    assert ok["rec-900004"]["sentences"] == [{"hi_text": "मेरा आई डी है।"}]
    assert ok["rec-900005"]["sentences"] == [{"hi_text": "ठीक है?"}] and ok["rec-900005"]["_w4"]["review"]["action"] == "fix"
    assert ok["rec-900005"]["_archive"] == {"transcribe": "archive/human/transcribe/d-harsh.jsonl"}
    assert {i: x["rule"] for i, x in bad.items()} == {"rec-900002": "ear_tag", "rec-900003": "second_voice",
                                                      "rec-900006": "stale_patch", "rec-900007": "audit_drop"}, bad
    assert bad["rec-900002"]["detail"]["flags"] == ["stammers@1"] and bad["rec-900002"]["stage"] == "transcribe"
    assert list(ok["rec-900001"])[:2] == ["id", "sentences"] and "_w1" in ok["rec-900001"]  # input fields carried through
    v = subprocess.run(["uv", "run", str(HERE.parent / "validate.py"), str(d / "out/records.jsonl"), "--partial"], capture_output=True, text=True)
    assert v.returncode == 0 and "0 violations" in v.stderr, v.stdout + v.stderr  # summary goes to stderr
    before = (d / "out/records.jsonl").read_bytes()
    run(d)  # resume: nothing duplicated
    assert (d / "out/records.jsonl").read_bytes() == before and sum(1 for _ in open(d / "out/rejects.jsonl")) == 4
    lines = before.decode().splitlines(keepends=True)
    (d / "out/records.jsonl").write_text("".join(lines[:-1]) + lines[-1][:20])  # crash tore the last line
    run(d)
    assert (d / "out/records.jsonl").read_bytes() == before
    print("ok: transcribe")


if __name__ == "__main__":
    main()
