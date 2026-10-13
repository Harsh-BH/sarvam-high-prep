"""Per-corpus inventory numbers for tasks/W1-real-audio.md, measured on whatever sample is on disk.

    uv run --no-project --with numpy python -I w1/inventory.py iv data/w1/iv_full
    uv run --no-project --with numpy python -I w1/inventory.py spring data/w1/spring
    uv run --no-project --with numpy python -I w1/inventory.py mucs data/w1/mucs/test

Every loader yields one dict per source recording:
    corpus, file (wav path), raw_dur, sr, speaker, gender, region, kind, segs=[(start, end, text)], en_words
"""
import collections as C, glob, json, re, statistics as st, sys, unicodedata, wave
from pathlib import Path

import numpy as np

TAG = re.compile(r"\[[^\]]*\]|<[^>]*>")
NUM_WORDS = set("""शून्य सून्य एक दो तीन चार पाँच पांच छह छः छे छेह सात आठ नौ दस ग्यारह बारह तेरह चौदह पंद्रह पन्द्रह सोलह सत्रह अठारह
उन्नीस बीस पच्चीस तीस चालीस पचास साठ सत्तर अस्सी नब्बे सौ हज़ार हजार लाख करोड़ करोड डेढ़ ढाई सवा साढ़े पौने पहला पहली दूसरा दूसरी तीसरा
रुपये रुपए रूपये रुपया""".split())
DIGIT = re.compile(r"[0-9०-९]")


def wav_meta(path):
    with wave.open(str(path)) as w:
        return w.getframerate(), w.getnchannels(), w.getnframes() / w.getframerate()


def load_iv(root):
    for jf in sorted(glob.glob(f"{root}/Hindi/*/*/*.json")):
        d = json.load(open(jf))
        wav = jf[:-5] + ".wav"
        # ponytail: WAVs without a usable window were deleted to save disk; fall back to the JSON's duration
        sr, ch, dur = wav_meta(wav) if Path(wav).exists() else (None, None, d["audio_raw_duration"])
        segs = [(s["start"], s["end"], s["text"]) for s in d["verbatim"]]
        en = sum(len(re.findall(r"\[[A-Za-z][^\]]*\]", s["text"])) for s in d["normalized"])
        yield dict(corpus="iv", file=wav, raw_dur=dur, sr=sr, ch=ch, speaker=d["speaker_id"], gender=d["gender"],
                   region=d["state"], kind=d["scenario"], task=d["task_name"], segs=segs, en_words=en,
                   n_spk=len({s.get("speaker_id") for s in d["verbatim"]}), decision=d["verification_report"].get("decision"), raw=d)


def load_mucs(root):
    text = dict(l.rstrip("\n").split(" ", 1) for l in open(f"{root}/transcription.txt"))
    for uid, t in text.items():
        sr, ch, dur = wav_meta(f"{root}/audio/{uid}.wav")
        yield dict(corpus="mucs", file=f"{root}/audio/{uid}.wav", raw_dur=dur, sr=sr, ch=ch, speaker=uid.split("_")[0],
                   gender="?", region="?", kind="Read", task="story", segs=[(0, dur, t)], en_words=None, n_spk=1)


def load_spring(root):
    """Kaldi dirs train/dev/eval under one release; durations of every recording from tar_scan's members.tsv."""
    rel = next(Path(root).glob("SPRING_INX_Hindi_R*"))
    dur = {Path(l.split("\t")[0]).stem: float(l.split("\t")[5]) for l in list(open(f"{root}/members.tsv"))[1:]
           if l.split("\t")[0].endswith(".wav")}
    segs = C.defaultdict(list)
    for split in ("train", "dev", "eval"):
        text = dict(l.rstrip("\n").split("\t", 1) if "\t" in l else l.rstrip("\n").split(" ", 1) for l in open(rel / split / "text"))
        for l in open(rel / split / "segments"):
            u, r, s, e = l.split()
            segs[r].append((float(s), float(e), text.get(u, "")))
    for r, ss in segs.items():
        wav = rel / "Audio" / f"{r}.wav"
        kind = {"M": "Monologue"}.get(r.split("_")[2], "Conversation(" + r.split("_")[2] + ")")
        en = sum(len(re.findall(r"[A-Za-z]+", t)) for *_, t in ss)
        yield dict(corpus="spring", file=str(wav), raw_dur=dur.get(r, max(e for _, e, _ in ss)), sr=16000 if r in dur else None,
                   ch=1, speaker=r, gender="?", region="?", kind=kind, task="", segs=ss, en_words=en, n_spk=None)


def windows(segs, raw_dur, max_gap=2.0, lo=10.0, hi=60.0):
    """Greedy non-overlapping clips of consecutive segments: gap <= max_gap, lo <= len <= hi."""
    out, i = [], 0
    segs = sorted(segs)
    while i < len(segs):
        j = i
        while j + 1 < len(segs) and segs[j + 1][0] - segs[j][1] <= max_gap and segs[j + 1][1] - segs[i][0] <= hi:
            j += 1
        s, e = segs[i][0], segs[j][1]
        if lo <= e - s <= hi:
            out.append((s, e, j - i + 1))
        i = j + 1
    return out


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)


def gap_energy(rec, n_max=200):
    """dB of each inter-segment gap (>=0.5 s) relative to the file's speech-segment level."""
    if not Path(rec["file"]).exists():
        return []
    with wave.open(rec["file"]) as w:
        sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    sp = [x[int(s * sr):int(e * sr)] for s, e, _ in rec["segs"] if e - s > 0.3]
    if not sp:
        return []
    ref = rms_db(np.concatenate(sp))
    segs = sorted(rec["segs"])
    gaps = [(a[1], b[0]) for a, b in zip(segs, segs[1:]) if b[0] - a[1] >= 0.5]
    # 95th-percentile 50 ms frame energy in the gap: catches untranscribed speech, ignores clicks
    out = []
    for s, e in gaps[:n_max]:
        g = x[int(s * sr) + int(0.1 * sr):int(e * sr) - int(0.1 * sr)]
        f = sr // 20
        if len(g) < f:
            continue
        frames = g[: len(g) // f * f].reshape(-1, f).astype(np.float64)
        p95 = np.percentile(20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9), 95)
        out.append(p95 - ref)
    return out


def pct(xs, ps=(10, 50, 90)):
    xs = sorted(xs)
    return "/".join(f"{xs[min(len(xs) - 1, int(p / 100 * len(xs)))]:.1f}" for p in ps) if xs else "-"


def report(recs, by="kind"):
    groups = C.defaultdict(list)
    for r in recs:
        groups[r[by]].append(r)
    groups["ALL"] = recs
    for k, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        segs = [(e - s) for r in rs for s, e, _ in r["segs"]]
        gaps = [b[0] - a[1] for r in rs for a, b in zip(sorted(r["segs"]), sorted(r["segs"])[1:])]
        words = [w for r in rs for *_, t in r["segs"] for w in unicodedata.normalize("NFC", TAG.sub(" ", t)).split()]
        tags = C.Counter(m for r in rs for *_, t in r["segs"] for m in TAG.findall(t))
        speech_min = sum(segs) / 60
        nums = sum(w in NUM_WORDS or bool(DIGIT.search(w)) for w in words)
        win = [w for r in rs for w in windows(r["segs"], r["raw_dur"])]
        en = [r["en_words"] for r in rs if r["en_words"] is not None]
        print(f"\n## {k}: {len(rs)} files")
        print(f"  raw h {sum(r['raw_dur'] for r in rs) / 3600:.2f} | transcribed-seg h {sum(segs) / 3600:.2f}"
              f" | speakers {len({r['speaker'] for r in rs})} | gender {dict(C.Counter(r['gender'] for r in rs))}")
        print(f"  sr {dict(C.Counter(r['sr'] for r in rs))} | ch {dict(C.Counter(r['ch'] for r in rs))}"
              f" | multi-speaker-id files {sum((r['n_spk'] or 1) > 1 for r in rs)}")
        print(f"  regions(top) {C.Counter(r['region'] for r in rs).most_common(8)}")
        print(f"  file dur p10/50/90 {pct([r['raw_dur'] for r in rs])} s | seg dur p10/50/90 {pct(segs)} s"
              f" | gap p10/50/90 {pct(gaps)} s")
        print(f"  10-60 s windows (gap<=2 s): {len(win)} clips, {sum(e - s for s, e, _ in win) / 3600:.2f} h,"
              f" len p10/50/90 {pct([e - s for s, e, _ in win])} s, segs/clip median {st.median([n for *_, n in win]) if win else 0}")
        if en:
            print(f"  code-mix (normalized [Latin] glosses / words) {sum(en) / max(1, len(words)):.3f}")
        print(f"  number/amount words per speech-minute {nums / max(speech_min, 1e-9):.2f}"
              f" | words/min {len(words) / max(speech_min, 1e-9):.0f} | tags/100 words {100 * sum(tags.values()) / max(1, len(words)):.1f}")
        print(f"  tags {tags.most_common(12)}")


if __name__ == "__main__":
    segs = [(0, 4, ""), (5, 9, ""), (9.5, 14, ""), (30, 31, ""), (40, 70, ""), (70.5, 75, "")]
    assert windows(segs, 80) == [(0, 14, 3), (40, 75, 2)], windows(segs, 80)
    kind, root = sys.argv[1], sys.argv[2]
    recs = list({"iv": load_iv, "mucs": load_mucs, "spring": load_spring}[kind](root))
    report(recs)
    if kind == "spring":
        for kd in sorted({r["kind"] for r in recs}):
            g = [v for r in recs if r["kind"] == kd and Path(r["file"]).exists() for v in gap_energy(r)]
            if g:
                print(f"## gap p95-frame energy rel. to speech, {kd}: n={len(g)} p10/50/90 {pct(g)} dB")
    if kind == "iv":
        report(recs, by="sr")
        print("\n## verification decisions", C.Counter(r["decision"] for r in recs))
        for kd in ("Conversation", "Extempore"):
            g = [v for r in recs if r["kind"] == kd for v in gap_energy(r)]
            print(f"## gap p95-frame energy rel. to speech, {kd}: n={len(g)} p10/50/90 {pct(g)} dB,"
                  f" share > -10 dB {np.mean(np.array(g) > -10):.2f}, > -20 dB {np.mean(np.array(g) > -20):.2f}")
