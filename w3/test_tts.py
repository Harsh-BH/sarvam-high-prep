# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "boto3"]
# ///
"""Self-check for the tts stage: retakes, write-once archive, crash resume without re-rendering, bench mode.

    uv run w3/test_tts.py
"""
import json, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "pipeline"), str(HERE.parent)]
import tts  # noqa: E402
from test_s3 import FakeS3  # noqa: E402
from validate import validate  # noqa: E402


def script(pid, words, ents, gender):
    hi = " ".join(words) + "।"
    return {"pid": pid, "tts_dev": ", ".join(words) + "।", "hi_text": hi, "sentences": [hi], "entities": ents,
            "en_words": [], "chars": len(hi.replace(" ", "")), "gender": gender}


S = [script("w2-00001", ["मेरा", "नाम", "सुनीता", "है", "और", "मैं", "चेन्नई", "से", "बोल", "रही", "हूँ"], ["सुनीता", "चेन्नई"], "f"),
     script("w2-00002", ["कल", "पौने", "चार", "लाख", "कटे", "हैं"], ["पौने चार लाख"], "f"),
     script("w2-00003", ["मैंने", "ऑर्डर", "दिया", "था"], [], "m")]
GOOD = {s["pid"]: s["hi_text"][:-1] for s in S}
BANK = [{"voice": v, "engine": "fake", "gender": g, "ref": "", "ref_text": "", "source": "", "licence": ""}
        for v, g in (("A", "f"), ("B", "f"), ("C", "m"))]


def run(out, scripts, answers, f, spool, synth=None, engine="fake", **kw):
    """answers: hyps in call order (both ASRs hear the same); synth: optional wrapper around the fake engine."""
    it = iter(answers)
    cur = {}

    def asr_w(x):
        cur["h"] = next(it)
        return cur["h"]
    ids = {tts.text_key(s): (f"rec-4{i:05d}", s["pid"]) for i, s in enumerate(S, 100)}
    tts.stage(scripts, out, engine, s3=f, synth=synth or tts.load_fake(), asr={"w2v": asr_w, "ic": lambda x: cur["h"]},
              load16k=lambda p: None, ids=ids, bank=BANK, spool=spool, **kw)


def main():
    f, spool = FakeS3(), Path(tempfile.mkdtemp())
    out = Path(tempfile.mkdtemp())
    calls = []

    def counting(text, v, seed):
        calls.append(seed)
        if len(calls) == 2 and crash:
            raise KeyboardInterrupt("crash after take 1 was archived and judged")
        return tts.load_fake()(text, v, seed)

    # script 1: take 1 drops चेन्नई (entity miss + skip), take 2 is right -> crash before take 2 renders
    crash = True
    try:
        run(out, S[:1], [GOOD["w2-00001"].replace(" चेन्नई", "")], f, spool, synth=counting)
        raise AssertionError("no crash")
    except KeyboardInterrupt:
        pass
    assert len(calls) == 2 and not (out / "records.jsonl").read_text()
    crash = False
    run(out, S, [GOOD["w2-00001"], "कुछ और", "बिल्कुल अलग", "तीसरा गलत", GOOD["w2-00003"]], f, spool, synth=counting)
    assert calls == [1, 2, 2, 1, 2, 3, 1], calls  # take 1 of script 1 was never rendered again
    recs = [json.loads(l) for l in open(out / "records.jsonl")]
    rej = [json.loads(l) for l in open(out / "rejects.jsonl")]
    assert [r["id"] for r in recs] == ["rec-400100", "rec-400102"] and [r["id"] for r in rej] == ["rec-400101"]
    assert recs[0]["_w3"]["take"] == 2 and len(recs[0]["_w3"]["rejected_takes"]) == 1 and recs[1]["_w3"]["take"] == 1
    assert rej[0]["rule"] == "qa" and rej[0]["detail"].count("take") == 3
    assert recs[0]["source"]["voice"] in ("A", "B") and recs[1]["source"]["voice"] == "C"  # gender-matched
    assert recs[0]["sentences"] == [{"hi_text": S[0]["hi_text"]}] and recs[0]["source"]["source_file"] == "tts/rec-400100.wav"
    bins = [k for k in f.objects if k.startswith("archive/api/tts-fake/") and k.endswith(".bin")]
    assert len(bins) == 6, bins  # every take archived, kept or not
    kept = f.objects["archive/tts/rec-400100.wav"]
    assert kept == (out / "audio/rec-400100.wav").read_bytes() and "archive/tts/rec-400101.wav" not in f.objects
    errs, hours = validate(open(out / "records.jsonl").read().splitlines(), out, partial=True)
    assert not errs, errs
    takes = [json.loads(l) for l in open(out / "takes.jsonl")]
    assert [t["verdict"] for t in takes] == ["reject", "keep", "reject", "reject", "reject", "keep"]
    assert "ent1" in takes[0]["why"] and "skip" in takes[0]["why"]

    n = len(calls)  # a finished run is a no-op: nothing rendered, nothing heard
    run(out, S, [], f, spool, synth=counting)
    assert len(calls) == n and len(open(out / "records.jsonl").readlines()) == 2

    bench = Path(tempfile.mkdtemp())  # bench: one take per script, no ids, no records
    run(bench, S, ["x", "y", GOOD["w2-00003"]], f, spool, bench=True)
    assert not (bench / "records.jsonl").read_text() and len(open(bench / "takes.jsonl").readlines()) == 3
    assert len(calls) == n  # take 1 of every script was already archived: bench re-used it

    tts.ENGINES["fake2"] = {**tts.ENGINES["fake"], "model": "w3-selfcheck-tone-2"}  # two engines, split by pid
    mix = Path(tempfile.mkdtemp())
    S2 = [script(f"w2-1{i:04d}", ["नमस्ते", "जी"] + ["हाँ"] * i, [], "f") for i in range(12)]
    GOOD.update({x["pid"]: x["hi_text"][:-1] for x in S2})
    ids2 = {tts.text_key(x): (f"rec-49{i:04d}", x["pid"]) for i, x in enumerate(S2)}
    it = iter([GOOD[x["pid"]] for x in S2])
    tts.stage(S2, mix, "fake,fake2", s3=f, asr={"w2v": lambda x: next(it)}, load16k=lambda p: None, ids=ids2,
              bank=BANK, spool=spool)
    models = {json.loads(l)["source"]["model"] for l in open(mix / "records.jsonl")}
    assert models == {"w3-selfcheck-tone", "w3-selfcheck-tone-2"}, models
    rot = Path(tempfile.mkdtemp())  # a rejected take is retaken on the other engine
    it = iter(["गलत", GOOD[S2[0]["pid"]]])
    tts.stage(S2[:1], rot, "fake,fake2", s3=f, asr={"w2v": lambda x: next(it)}, load16k=lambda p: None, ids=ids2,
              bank=BANK, spool=spool)
    t = [json.loads(l) for l in open(rot / "takes.jsonl")]
    assert [x["verdict"] for x in t] == ["reject", "keep"] and t[0]["engine"] != t[1]["engine"], t

    n, mins = tts.listen(out, Path(tempfile.mkdtemp()), per_gender=1, fillers=1)
    assert n == 2 and mins > 0  # one voice per gender; no filler in these scripts

    assert tts.to24k(tts.wav_bytes(tts.np.zeros(22050, "float32"), 22050)) == tts.wav_bytes(tts.np.zeros(24000, "<i2"), 24000)
    print("ok: tts stage")


if __name__ == "__main__":
    main()
