# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "sacrebleu"]
# ///
"""Score a streaming Hi->En system's outputs on an eval set (tasks/W8c-eval.md "Harness").

    python w8c/eval.py score --set SET.jsonl --sys OUT [--asr openai/whisper-large-v3-turbo] [--comet Unbabel/wmt22-comet-da]
    uv run w8c/eval.py selftest        # toy outputs, no models

System output (w8c/run_seamless.py writes it; any system can): OUT/<id>.json
    {"id", "frames": [{"t", "t0"?, "n"}], "text", "text_events": [{"t", "t0"?, "text"}], "failed"}
plus OUT/<id>.wav: the frames' audio concatenated as emitted (24 kHz mono s16). t = arrival in seconds, 0 = Hindi
frame 0 sent (§5.1); t0 (optional) = the same with zero compute. Each timing key present is scored.

Speech track: output ASR (Whisper large-v3-turbo, word timestamps) on OUT/<id>.wav, cached in OUT/asr.jsonl; word
times map onto the §5.1 playback timeline (frame k starts at max(t_k, end of frame k-1)).
  ASR-BLEU / ASR-chrF++ (sacrebleu, both sides lowercased, punctuation stripped), ASR-COMET (wmt22-comet-da, src = Hindi).
  Entities: numbers via W6's parser (w6/qa.en_numbers: lakh/crore, digit strings, 25k = 25,000) and names
  (capitalised, non-initial reference words); score = found / gold, 1 when a record has none.
  Latency (s): start_offset = first output word start - first Hindi word start; end_offset = last output word end -
  last Hindi word end; end_offset_audio = last output sample - source end (SimulEval's); laal = LAAL over output
  words (delay = word end, source = whole clip); sent_lag = mean over output words of (word end - end of its source
  sentence), output word i mapped to the reference word at the same relative position.
Text track (S2TT): BLEU / chrF++ / COMET on "text", numbers + names, and the same latencies with delay = emission time.
A failed session (or no output) gets the worst value: empty hypothesis, delays = source end + 30 s (§5.1 R3, R7).
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "w6"))
import qa  # noqa: E402  (W6's number parser)

RATE, DEADLINE = 24000, 30.0


def playback(frames, key):
    """§5.1 R1: frame k starts at max(arrival_k, end of frame k-1). -> start times (s)."""
    out, end = [], None
    for f in frames:
        s = f[key] if end is None else max(f[key], end)
        out.append(s)
        end = s + f["n"] / RATE
    return out


def to_timeline(words, frames, key):
    """Word times measured in the concatenated output audio -> playback-timeline times."""
    starts, cum = playback(frames, key), np.cumsum([0] + [f["n"] for f in frames])

    def at(sec, end=False):
        s = sec * RATE - end  # an end time belongs to the frame of its last sample
        k = max(0, min(len(frames) - 1, int(np.searchsorted(cum, s, side="right")) - 1))
        return starts[k] + (sec * RATE - cum[k]) / RATE
    return [(w, at(a), at(b, True)) for w, a, b in words]


def norm(s):
    return " ".join(re.sub(r"[^a-z0-9' ]+", " ", s.lower().replace("’", "'")).split())


def names(ref):
    """Capitalised reference words that don't start a sentence (and aren't "I")."""
    out = []
    for sent in re.split(r"(?<=[.?!])\s+", ref):
        toks = re.findall(r"[A-Za-z][A-Za-z'-]*", sent)
        out += [t for t in toks[1:] if t[0].isupper() and t not in ("I", "I'm", "I'll", "I've", "I'd")]
    return out


def entity_hits(ref, hyp):
    """-> (gold numbers, found numbers, gold names, found names); multiset matching."""
    from collections import Counter
    gn, hn = Counter(map(qa._key, qa.en_numbers(ref))), Counter(map(qa._key, qa.en_numbers(hyp)))
    gw, hw = Counter(w.lower() for w in names(ref)), Counter(re.findall(r"[a-z][a-z'-]*", hyp.lower()))
    return sum(gn.values()), sum((gn & hn).values()), sum(gw.values()), sum((gw & hw).values())


def laal(delays, src_len, n_ref):
    if not delays:
        return src_len + DEADLINE
    gamma = src_len / max(len(delays), n_ref)
    tau = next((i + 1 for i, d in enumerate(delays) if d >= src_len), len(delays))
    return float(np.mean([d - i * gamma for i, d in enumerate(delays[:tau])]))


def latency(rec, words, failed):
    """words: [(w, start, end)] on the playback timeline (or emission times). -> {metric: seconds}"""
    hw, dur = rec.get("hi_words") or [], rec["duration"]
    src0, src1 = (hw[0]["start"], hw[-1]["end"]) if hw else (None, dur)
    if failed or not words:
        words = [("", dur + DEADLINE, dur + DEADLINE)]
    ref_words = [len(s["en_text"].split()) for s in rec["sentences"]]
    n_ref = sum(ref_words)
    out = {"end_offset": words[-1][2] - src1, "laal": laal([e for _, _, e in words], dur, n_ref)}
    if src0 is not None:
        out["start_offset"] = words[0][1] - src0
    if all("hi_end" in s for s in rec["sentences"]):
        bounds, lag = np.cumsum(ref_words), []
        for i, (_, _, e) in enumerate(words):
            j = min(int(np.searchsorted(bounds, (i + 0.5) * n_ref / len(words))), len(bounds) - 1)
            lag.append(e - rec["sentences"][j]["hi_end"])
        out["sent_lag"] = float(np.mean(lag))
    return {k: round(float(v), 3) for k, v in out.items()}


def asr_words(wav_paths, model):
    """-> [(text, [(word, start, end)])] with Whisper word timestamps; times in the given audio."""
    import torch, torchaudio
    from transformers import pipeline
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    pipe = pipeline("automatic-speech-recognition", model=model, device=dev,
                    torch_dtype=torch.float16 if dev != "cpu" else torch.float32)
    out = []
    for p in wav_paths:
        x, sr = torchaudio.load(str(p))
        if x.shape[1] < sr // 10:
            out.append(("", []))
            continue
        x = torchaudio.functional.resample(x[0], sr, 16000).numpy()
        r = pipe({"raw": x, "sampling_rate": 16000}, return_timestamps="word", chunk_length_s=30, batch_size=8,
                 generate_kwargs={"language": "english", "task": "transcribe"})
        ws = [(c["text"].strip(), c["timestamp"][0], c["timestamp"][1] if c["timestamp"][1] is not None else c["timestamp"][0])
              for c in r["chunks"] if c["text"].strip()]
        out.append((r["text"].strip(), ws))
    return out


def comet_scores(triples, model):
    if model is None:
        return None
    import torch
    from comet import download_model, load_from_checkpoint
    m = load_from_checkpoint(download_model(model))
    return m.predict([{"src": s, "mt": h, "ref": r} for s, h, r in triples], batch_size=16,
                     gpus=1 if torch.cuda.is_available() else 0, progress_bar=False).scores


def score(recs, sys_dir, asr=None, comet=None, asr_model="openai/whisper-large-v3-turbo", comet_model="Unbabel/wmt22-comet-da"):
    """asr / comet: injectable for tests (asr: paths -> [(text, words)]; comet: triples -> scores or None)."""
    import sacrebleu
    sys_dir = Path(sys_dir)
    outs = {r["id"]: json.loads((sys_dir / f"{r['id']}.json").read_text()) for r in recs}
    cache = sys_dir / "asr.jsonl"
    got = {d["id"]: d for d in map(json.loads, open(cache))} if cache.exists() else {}
    todo = [r["id"] for r in recs if r["id"] not in got]
    if todo:
        res = (asr or (lambda ps: asr_words(ps, asr_model)))([sys_dir / f"{i}.wav" for i in todo])
        with open(cache, "a") as f:
            for i, (text, ws) in zip(todo, res):
                got[i] = {"id": i, "text": text, "words": ws}
                f.write(json.dumps(got[i], ensure_ascii=False) + "\n")
    keys = [k for k in ("t", "t0") if all(k in f for o in outs.values() for f in o["frames"])]
    per = []
    for r in recs:
        o, a = outs[r["id"]], got[r["id"]]
        ref, src = " ".join(s["en_text"] for s in r["sentences"]), " ".join(s["hi_text"] for s in r["sentences"])
        p = {"id": r["id"], "kind": "fleurs" if r["source"]["corpus"] == "FLEURS" else
             "synthetic" if r["source"]["synthetic"] else "real", "failed": bool(o.get("failed")),
             "src": src, "ref": ref, "asr": "" if o.get("failed") else a["text"], "text": "" if o.get("failed") else o["text"]}
        for track in ("asr", "text"):
            p[f"{track}_ent"] = entity_hits(ref, p[track])
        for k in keys:
            p[f"lat_{k}"] = latency(r, to_timeline(a["words"], o["frames"], k), p["failed"])
            tw = [(w, e[k], e[k]) for e in o.get("text_events", []) for w in e["text"].split()]
            p[f"text_lat_{k}"] = latency(r, tw, p["failed"])
        per.append(p)
    comet_fn = comet or (lambda t: comet_scores(t, comet_model))
    for track in ("asr", "text"):
        cs = comet_fn([(p["src"], p[track], p["ref"]) for p in per])
        for p, c in zip(per, cs or [None] * len(per)):
            p[f"{track}_comet"] = None if c is None else round(float(c), 4)

    def summary(ps):
        s = {"n": len(ps), "failed": sum(p["failed"] for p in ps), "src_h": None}
        for track, nrm in (("asr", norm), ("text", lambda x: x)):
            hyp, ref = [nrm(p[track]) for p in ps], [[nrm(p["ref"]) for p in ps]]
            s[f"{track}_bleu"] = round(sacrebleu.corpus_bleu(hyp, ref).score, 2)
            s[f"{track}_chrf"] = round(sacrebleu.corpus_chrf(hyp, ref, word_order=2).score, 2)
            cs = [p[f"{track}_comet"] for p in ps if p[f"{track}_comet"] is not None]
            s[f"{track}_comet"] = round(float(np.mean(cs)), 4) if cs else None
            e = np.array([p[f"{track}_ent"] for p in ps]).reshape(-1, 4).sum(0)
            s[f"{track}_num_acc"] = round(e[1] / e[0], 4) if e[0] else None
            s[f"{track}_name_acc"] = round(e[3] / e[2], 4) if e[2] else None
            s[f"{track}_ent_acc"] = round(float(np.mean([(x[1] + x[3]) / (x[0] + x[2]) if x[0] + x[2] else 1.0
                                                          for x in (p[f"{track}_ent"] for p in ps)])), 4)
        for k in keys:
            for pre in ("lat", "text_lat"):
                ms = {m for p in ps for m in p[f"{pre}_{k}"]}
                s[f"{pre}_{k}"] = {m: round(float(np.mean([p[f"{pre}_{k}"][m] for p in ps if m in p[f"{pre}_{k}"]])), 3) for m in sorted(ms)}
        return s

    rep = {"all": summary(per)}
    for kind in sorted({p["kind"] for p in per}):
        rep[kind] = summary([p for p in per if p["kind"] == kind])
    for kind, s in rep.items():
        s["src_h"] = round(sum(r["duration"] for r in recs if kind == "all" or
                               next(p["kind"] for p in per if p["id"] == r["id"]) == kind) / 3600, 3)
    return rep, per


def selftest():
    import tempfile, wave
    d = Path(tempfile.mkdtemp())
    rec = lambda i, syn: {"id": i, "duration": 6.0, "source": {"corpus": "x", "synthetic": syn},
                          "hi_words": [{"w": "क", "start": 0.5, "end": 1.0}, {"w": "ख", "start": 4.0, "end": 5.0}],
                          "sentences": [{"hi_text": "क।", "hi_start": 0.5, "hi_end": 1.0, "en_text": "Pay Ravi 25,000 rupees."},
                                        {"hi_text": "ख।", "hi_start": 4.0, "hi_end": 5.0, "en_text": "Call me at nine."}]}
    recs = [rec("rec-1", False), rec("rec-2", True), rec("rec-3", False)]
    # rec-1: two 1 s frames; the second arrives at 2.5 s, before the first ends -> plays at 3.0 s
    out = {"rec-1": {"frames": [{"t": 2.0, "t0": 1.5, "n": RATE}, {"t": 2.5, "t0": 2.0, "n": RATE}], "failed": False,
                     "text": "Pay Ravi 25k rupees. Call me at nine.",
                     "text_events": [{"t": 1.5, "t0": 1.0, "text": "Pay Ravi 25k rupees."}, {"t": 5.5, "t0": 5.0, "text": "Call me at nine."}]},
           "rec-2": {"frames": [{"t": 1.0, "t0": 1.0, "n": RATE}], "failed": False, "text": "Pay someone.",
                     "text_events": [{"t": 1.0, "t0": 1.0, "text": "Pay someone."}]},
           "rec-3": {"frames": [], "failed": True, "text": "", "text_events": []}}
    for i, o in out.items():
        (d / f"{i}.json").write_text(json.dumps({"id": i, **o}))
        with wave.open(str(d / f"{i}.wav"), "wb") as w:
            w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE), w.writeframes(b"\0\0" * sum(f["n"] for f in o["frames"]))
    words = {"rec-1": ("Pay Ravi 25,000 rupees. Call me at nine.",
                       [("Pay", 0.1, 0.3), ("Ravi", 0.3, 0.6), ("25,000", 0.6, 0.9), ("rupees.", 0.9, 1.0),
                        ("Call", 1.2, 1.4), ("me", 1.4, 1.5), ("at", 1.5, 1.6), ("nine.", 1.6, 1.9)]),
             "rec-2": ("Pay someone.", [("Pay", 0.0, 0.2), ("someone.", 0.2, 0.5)]), "rec-3": ("", [])}
    assert playback(out["rec-1"]["frames"], "t") == [2.0, 3.0]
    tl = to_timeline(words["rec-1"][1], out["rec-1"]["frames"], "t")
    assert [round(x, 3) for x in tl[0][1:]] == [2.1, 2.3] and [round(x, 3) for x in tl[-1][1:]] == [3.6, 3.9], tl
    assert round(to_timeline([("x", 0.5, 1.0)], out["rec-1"]["frames"], "t")[0][2], 3) == 3.0  # end on a frame edge
    assert entity_hits("Pay Ravi 25,000 rupees.", "pay ravi twenty five thousand rupees") == (1, 1, 1, 1)
    assert entity_hits("Pay Ravi 3.5 lakh.", "Pay Ravi three lakh.") == (1, 0, 1, 1)
    assert laal([1.0, 2.0], 2.0, 2) == 1.0  # gamma 1: delays 1 and 2-1
    rep, per = score(recs, d, asr=lambda ps: [words[p.stem] for p in ps], comet=lambda t: [1.0 if h else 0.0 for _, h, _ in t])
    p1, p3 = per[0], per[2]
    assert p1["lat_t"] == {"start_offset": 1.6, "end_offset": -1.1, "laal": 0.525, "sent_lag": 0.15}, p1["lat_t"]
    assert p1["lat_t0"]["start_offset"] == 1.1 and p1["text_lat_t"]["end_offset"] == 0.5, (p1["lat_t0"], p1["text_lat_t"])
    assert p3["lat_t"]["end_offset"] == 31.0 and p3["lat_t"]["laal"] == 36.0 and p3["asr"] == ""  # failed: worst values
    a = rep["all"]
    assert rep["real"]["n"] == 2 and rep["synthetic"]["asr_ent_acc"] == 0.0 and rep["real"]["asr_ent_acc"] == 0.5, rep  # rec-2 finds nothing
    assert a["asr_num_acc"] == round(1 / 3, 4) and a["text_num_acc"] == round(1 / 3, 4), a  # 25k = 25,000; nine x3 gold
    assert 0 < a["asr_bleu"] < 100 and a["asr_comet"] == round(2 / 3, 4) and a["failed"] == 1, a
    assert (d / "asr.jsonl").exists() and score(recs, d, asr=lambda ps: 1 / 0, comet=lambda t: None)[0]["all"]["asr_comet"] is None
    print("ok: eval selftest")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["score", "selftest"])
    ap.add_argument("--set", type=Path)
    ap.add_argument("--sys", type=Path)
    ap.add_argument("--asr", default="openai/whisper-large-v3-turbo")
    ap.add_argument("--comet", default="Unbabel/wmt22-comet-da", help="'' to skip")
    a = ap.parse_args()
    if a.cmd == "selftest":
        return selftest()
    recs = [json.loads(l) for l in open(a.set) if l.strip()]
    recs = [r for r in recs if (a.sys / f"{r['id']}.json").exists()]
    rep, per = score(recs, a.sys, asr_model=a.asr, comet_model=a.comet or None,
                     comet=None if a.comet else (lambda t: None))
    (a.sys / "per_record.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in per))
    (a.sys / "scores.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
