# /// script
# requires-python = ">=3.12"
# ///
import json, subprocess, sys, tempfile
from pathlib import Path
import dedupe


def rec(i, sf, s, e):
    return {"id": f"rec-00000{i}", "audio": f"audio/rec-00000{i}.wav", "source": {"source_file": sf, "start": s, "end": e}}


def main():
    d = Path(tempfile.mkdtemp()); (d / "audio").mkdir()
    for i, b in ((1, b"A"), (2, b"B"), (3, b"C"), (4, b"A"), (5, b"E")):
        (d / f"audio/rec-00000{i}.wav").write_bytes(b)
    recs = [rec(2, "f1", 0.0, 12.0), rec(1, "f1", 5.0, 20.0), rec(3, "f1", 0.0, 12.0004), rec(4, "f2", 0.0, 11.0), rec(5, "f1", 0.0, 12.5)]
    kept, rej = dedupe.run(recs, d)
    assert [r["id"] for r in kept] == ["rec-000001", "rec-000002", "rec-000005"], kept
    assert {(x["id"], x["rule"]) for x in rej} == {("rec-000003", "same_interval"), ("rec-000004", "same_audio")}, rej
    (d / "in.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    subprocess.run([sys.executable, str(Path(dedupe.__file__)), "--in", str(d / "in.jsonl"), "--audio-dir", str(d), "--out", str(d / "out")], check=True)
    assert sum(1 for _ in open(d / "out/records.jsonl")) == 3 and sum(1 for _ in open(d / "out/rejects.jsonl")) == 2
    print("ok: dedupe")


if __name__ == "__main__":
    main()
