# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "boto3"]
# ///
"""uv run scale/real/test_cut_iv.py: the cut stage on a fake archive (reconnects, quota, held-out, ids, resume)."""
import io, json, sys, tarfile, tempfile, wave
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent.parent)]
import cut_iv, validate  # noqa: E402


def wav(dur, sr, segs):
    x = np.zeros(int(dur * sr), dtype=np.int16)
    for s, e in segs:
        x[int(s * sr):int(e * sr)] = (3000 * np.sin(np.arange(int((e - s) * sr)) * 0.3)).astype(np.int16)
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(x.tobytes())
    return b.getvalue()


def js(spk, scenario, dur, segs):
    seg = [{"start": s, "end": e, "text": "नमस्ते जी", "speaker_id": 0} for s, e in segs]
    return json.dumps({"speaker_id": spk, "scenario": scenario, "task_name": scenario, "gender": "Female", "age_group": "18-30",
                       "state": "Bihar", "district": "Patna", "audio_raw_duration": dur, "verbatim": seg,
                       "normalized": [dict(x, text="नमस्ते जी") for x in seg],
                       "verification_report": {"decision": "excellent"}}, ensure_ascii=False).encode()


SEGS, PARTNER = [(1, 6), (6.5, 12), (12.5, 20)], [(21, 23)]
MEMBERS = [  # sorted, json before wav, as in the real archives
    ("Hindi/v1/train/aa_0.json", js("SA", "Conversation", 30, SEGS)), ("Hindi/v1/train/aa_0.wav", wav(30, 8000, SEGS)),
    ("Hindi/v1/train/aa_1.json", js("SB", "Conversation", 30, PARTNER)), ("Hindi/v1/train/aa_1.wav", wav(30, 8000, PARTNER)),
    ("Hindi/v1/train/bb.json", js("SHELD", "Extempore", 30, SEGS)), ("Hindi/v1/train/bb.wav", wav(30, 16000, SEGS)),
    ("Hindi/v1/train/cc.json", js("SC", "Extempore", 30, SEGS)), ("Hindi/v1/train/cc.wav", wav(30, 16000, SEGS)),
    ("Hindi/v1/train/dd.json", js("SD", "Extempore", 30, SEGS)), ("Hindi/v1/train/dd.wav", wav(30, 16000, SEGS)),
]


def tgz():
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        for n, d in MEMBERS:
            ti = tarfile.TarInfo(n)
            ti.size = len(d)
            t.addfile(ti, io.BytesIO(d))
    return b.getvalue()


class Resp:
    def __init__(self, body, etag, cut):
        self.f, self.left, self.headers = io.BytesIO(body), cut, {"ETag": f'"{etag}"', "Content-Length": str(len(body))}

    def readinto(self, b):  # serves `cut` bytes, then an early close (0 bytes) like http.client does
        n = self.f.readinto(memoryview(b)[: max(0, min(len(b), self.left))])
        self.left -= n
        return n


def opener(data, etags, cut=50_000):
    calls = []

    def op(req):
        start = int(req.get_header("Range", "bytes=0-")[6:-1])
        calls.append(start)
        return Resp(data[start:], etags[min(len(calls), len(etags)) - 1], cut)
    return op, calls


def main():
    data, etag = tgz(), cut_iv.W1.IV_REV["v1"].strip('"')
    op, calls = opener(data, [etag])  # every connection closes early: reconnects must stitch the bytes
    s = cut_iv.Stream("https://x/v1_Hindi_train.tgz", etag, op, sleep=lambda _: None)
    assert io.BufferedReader(s).read() == data and len(calls) > 2 and calls[1] == 50_000, calls
    op, _ = opener(data, [etag, "OTHER"])
    try:
        io.BufferedReader(cut_iv.Stream("https://x/v1_Hindi_train.tgz", etag, op, sleep=lambda _: None)).read()
        raise AssertionError("an ETag change mid-stream was not refused")
    except SystemExit:
        pass

    with tempfile.TemporaryDirectory() as d:
        cut_iv.W1.IDS = Path(d) / "ids.tsv"  # an empty register: the real one grows, and its spans aren't in the fake archive
        cut_iv.W1.IDS.write_text("id\tsource_file\tstart_ms\tend_ms\n")
        out, puts = Path(d) / "out", []
        (out / "audio").mkdir(parents=True)
        put = lambda m, b, url, et, s3: puts.append(m)

        def shard():
            return cut_iv.Shard(out, {"Conversation": 1.0, "Extempore": 0.004}, 0, 1, 1001, 240.0, frozenset({"SHELD"}))
        sh = shard()
        cut_iv.run(sh, "v1", Path(d) / "tmp", opener(data, [etag])[0], put)
        recs = [json.loads(l) for l in open(out / "records.jsonl")]
        got = {(r["id"], r["source"]["source_file"].rsplit("/", 1)[1]) for r in recs}
        # aa_0: one clip; aa_1: its segments are < 10 s; bb: held out; cc: fills the 0.004 h (14 s) quota; dd: quota full
        assert got == {("rec-001001", "aa_0.wav"), ("rec-001002", "cc.wav")}, got
        assert puts == ["Hindi/v1/train/aa_0.wav", "Hindi/v1/train/aa_0.json", "Hindi/v1/train/cc.wav", "Hindi/v1/train/cc.json"], puts
        assert recs[0]["_rev"].endswith(f'etag="{etag}"') and recs[0]["source"]["source_file"] == "indicvoices/Hindi/v1/train/aa_0.wav"
        st = json.loads((out / "cut_stats.json").read_text())
        assert st["file skipped: held-out speaker (IV valid / W3 voice)"] == 1 and st["file skipped: kind quota full"] == 1, st
        assert validate.validate(open(out / "records.jsonl").read().splitlines(), out, True)[0] == []
        assert [l.split("\t")[0] for l in open(out / "ids_new.tsv")] == ["rec-001001", "rec-001002"]
        before = (out / "records.jsonl").read_bytes()
        cut_iv.run(shard(), "v1", Path(d) / "tmp", opener(data, [etag])[0], put)  # a restart redoes nothing
        assert (out / "records.jsonl").read_bytes() == before and len(puts) == 4
        (out / "seen.txt").unlink()  # crash before seen.txt / records: the same spans get the same ids back
        (out / "records.jsonl").write_bytes(before.splitlines(keepends=True)[0])
        cut_iv.run(shard(), "v1", Path(d) / "tmp", opener(data, [etag])[0], put)
        assert (out / "records.jsonl").read_bytes() == before and len(open(out / "ids_new.tsv").readlines()) == 2
        # --continue: the first run's files are done and its spans used; dd is new, with a fresh quota and block
        prev = tuple(open(out / f).read().splitlines() for f in ("records.jsonl", "seen.txt", "ids_new.tsv"))
        out2 = Path(d) / "out2"
        (out2 / "audio").mkdir(parents=True)
        sh2 = cut_iv.Shard(out2, {"Extempore": 1.0}, 0, 1, 2001, 240.0, frozenset({"SHELD"}), prev)
        cut_iv.run(sh2, "v1", Path(d) / "tmp", opener(data, [etag])[0], put)
        got = [(json.loads(l)["id"], json.loads(l)["source"]["source_file"].rsplit("/", 1)[1]) for l in open(out2 / "records.jsonl")]
        assert got == [("rec-002001", "dd.wav")], got
    held = cut_iv.held_out()
    assert len(held) > 483 and "S4259465500359288" in held  # IV valid + the W3 voice cloned from a train speaker
    print("ok: cut_iv")


if __name__ == "__main__":
    main()
