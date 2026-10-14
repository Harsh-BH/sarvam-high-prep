# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""Self-check for `review_list.py import`: audit rows -> write-once archive/human/transcribe/ (FakeS3) + the git copy,
then the `transcribe` stage applies the patch.  uv run w4/test_review_list.py"""
import csv, io, json, subprocess, sys, tempfile
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pipeline")]
import review_list  # noqa: E402
from test_s3 import FakeS3  # noqa: E402


def transcribe(d, out, patches):
    subprocess.run([sys.executable, str(HERE / "convert.py"), "--in", str(d / "in.jsonl"), "--audio-dir", str(d), "--out", str(d / out),
                    "--no-sbd", "--patches", str(patches)], check=True)
    return {json.loads(l)["id"]: json.loads(l) for l in open(d / out / "records.jsonl")}


def tsv(path, actions):
    rows = [{c: "" for c in review_list.COLS} | {"id": i, **a} for i, a in actions.items()]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, review_list.COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main():
    d, s3 = Path(tempfile.mkdtemp()), FakeS3()
    seg = lambda t: {"start": 0.0, "end": 1.0, "verbatim": t, "normalized": t}
    recs = [{"id": f"rec-90000{n}", "_orig_text": t, "_w1": {"segments": [seg(t)]}}
            for n, t in ((1, "मैं घर जा रहा हूँ"), (2, "ठीक है"), (3, "हाँ जी"))]
    (d / "in.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
    auto = transcribe(d, "t1", d / "none.jsonl")
    patches, day = d / "patches.jsonl", "2026-10-15"
    tsv(d / "review.tsv", {"rec-900001": {"action": "fix", "corrected": "मैं घर जा रहा हूँ ना?", "minutes": "2"},
                           "rec-900002": {"action": "ok"}, "rec-900003": {}})
    with redirect_stdout(io.StringIO()):
        k1 = review_list.import_(d / "review.tsv", d / "t1/records.jsonl", patches, s3=s3, day=day)
    assert k1 == f"archive/human/transcribe/{day}-harsh.jsonl" and s3.objects[k1] == patches.read_bytes()
    lines = [json.loads(l) for l in open(patches)]
    assert [p["id"] for p in lines] == ["rec-900001", "rec-900002"] and all(p["archive"] == k1 for p in lines)
    assert lines[0]["sentences"] == ["मैं घर जा रहा हूँ ना?"] and lines[0]["base"] == review_list.sha(review_list.text_of(auto["rec-900001"]))
    # a second batch the same day: the first object stays as it was, the new one gets -2
    first = s3.objects[k1]
    tsv(d / "review.tsv", {"rec-900001": {"action": "fix", "corrected": "x।"}, "rec-900003": {"action": "drop", "note": "noise"}})
    with redirect_stdout(io.StringIO()):
        k2 = review_list.import_(d / "review.tsv", d / "t1/records.jsonl", patches, s3=s3, day=day)
        assert review_list.import_(d / "review.tsv", d / "t1/records.jsonl", patches, s3=s3, day=day) is None  # nothing new
    assert k2.endswith(f"{day}-harsh-2.jsonl") and s3.objects[k1] == first and sum(1 for _ in open(patches)) == 3
    # crash after the archive write, before the git copy: the re-run reuses the archived batch, no duplicate key
    lost = d / "lost.jsonl"
    tsv(d / "review.tsv", {"rec-900001": {"action": "fix", "corrected": "मैं घर जा रहा हूँ ना?", "minutes": "2"}, "rec-900002": {"action": "ok"}})
    with redirect_stdout(io.StringIO()):
        assert review_list.import_(d / "review.tsv", d / "t1/records.jsonl", lost, s3=s3, day=day) == k1
    assert lost.read_bytes() == first and len([k for k in s3.objects if k.startswith("archive/human/")]) == 2
    # the transcribe stage applies the git copy: fix -> new text + archive key, ok -> kept, drop -> rejected
    ok = transcribe(d, "t2", patches)
    assert ok["rec-900001"]["sentences"] == [{"hi_text": "मैं घर जा रहा हूँ ना?"}] and ok["rec-900001"]["_archive"]["transcribe"] == k1
    assert ok["rec-900002"]["_w4"]["review"]["action"] == "ok" and "rec-900003" not in ok
    assert json.loads(open(d / "t2/rejects.jsonl").read())["rule"] == "audit_drop"
    print("ok: review_list import")


if __name__ == "__main__":
    main()
