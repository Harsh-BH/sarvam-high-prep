"""Held-out control set for the ASR benchmark: IndicVoices Hindi *valid* segments (W1 keeps valid out of the pool).

    python3 -I w4/ivvalid_set.py        # -> data/w4/ivvalid/{manifest.jsonl,audio/}

Why: the pool is IV *train*. A model trained on IV train (IndicConformer, I) scores optimistically on the pool and
can't act as an independent check of IV references. Comparing its pool-vs-valid gap with other models' gap measures
that. One 5-20 s segment per long-form file (speaker spread), first qualifying one, 60 per scenario. Records use the
W1 stage shape so w4/convert.py, w4/asr_run.py and w4/asr_score.py run on them unchanged.
"""
import json, re, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC, OUT = ROOT / "data/w1/iv_full/Hindi", ROOT / "data/w4/ivvalid"
SKIP = re.compile(r"stammers|uh-huh|ugh|unintelligible|talking|whisper|yelling|singing", re.I)
PER = {"Conversation": 60, "Extempore": 60}


def main():
    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    got, recs = dict.fromkeys(PER, 0), []
    for js in sorted(SRC.glob("*/valid/*.json")):
        d = json.load(open(js))
        sc = d["scenario"]
        if got.get(sc, 99) >= PER.get(sc, 0):
            continue
        for v, n in zip(d["verbatim"], d["normalized"]):
            if v.get("speaker_id", 0) == 0 and 5 <= v["end"] - v["start"] <= 20 and not SKIP.search(v["text"]):
                break
        else:
            continue
        rid = f"ivv-{js.stem}"
        wav = js.with_suffix(".wav")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", str(v["start"]), "-to", str(v["end"]), "-i", str(wav),
                        "-ar", "16000", "-ac", "1", str(OUT / f"audio/{rid}.wav")], check=True)
        sr = int(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=sample_rate", "-of", "csv=p=0", str(wav)],
                                capture_output=True, text=True, check=True).stdout.split()[0])
        dur = round(v["end"] - v["start"], 3)
        recs.append({"id": rid, "audio": f"audio/{rid}.wav", "duration": dur, "_orig_text": v["text"],
                     "_w1": {"segments": [{"start": 0.0, "end": dur, "verbatim": v["text"], "normalized": n["text"]}],
                             "orig_sample_rate": sr, "scenario": sc, "source": str(js.relative_to(ROOT))}})
        got[sc] += 1
    with open(OUT / "manifest.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
    print(got, f"{sum(r['duration'] for r in recs) / 3600:.3f} h ->", OUT)


if __name__ == "__main__":
    main()
