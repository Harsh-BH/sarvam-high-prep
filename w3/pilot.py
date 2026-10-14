"""QA-kept renders -> synthetic stage records (SPEC §4, §7, §8) for W5/W6.

    uv run --no-project python w3/pilot.py [N]       # default 50 records

Reads data/w3/renders.jsonl + data/w3/qa.tsv. Picks `keep` renders: one per probe text, Bulbul then Magpie,
Devanagari input first, at most N/10 per voice.
Ids come from w3/ids.tsv (rec-400000 up, allocated once per archive key, never reused). Writes data/w3/pilot/:
  tts/<id>.wav    the archived take, byte for byte as the engine returned it (source.source_file)
  audio/<id>.wav  24 kHz mono 16-bit (soxr resample if the engine rate differs; canonical 44-byte header)
  manifest.jsonl  id, audio, duration, speaker_id, source, sentences[].hi_text, _w3/_qa/_archive
hi_text is the input transcript; only renders where every ASR agrees nothing was skipped, added or misread are kept.
"""
import json
import re
import subprocess
import sys
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/w3"
OUT = DATA / "pilot"
IDS = ROOT / "w3/ids.tsv"
FIRST_ID, RATE = 400000, 24000
SENT_END = re.compile(r"(?<=[।?!]) ")
CORPUS = "w3-probe-v0"  # provisional bake-off text (w3/probe.tsv), not W2 scripts


def to24k(src, dst):
    with wave.open(str(src)) as w:
        sr, pcm = w.getframerate(), w.readframes(w.getnframes())
    if sr != RATE:
        pcm = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(src), "-af", "aresample=resampler=soxr",
                              "-ar", str(RATE), "-ac", "1", "-f", "s16le", "-"], check=True, capture_output=True).stdout
    with wave.open(str(dst), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE), w.writeframes(pcm)
    return round(len(pcm) / 2 / RATE, 3)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    qa = {}
    for i, l in enumerate(open(DATA / "qa.tsv", encoding="utf-8")):
        f = l.rstrip("\n").split("\t")
        if i == 0:
            cols = f
            continue
        qa[f[0]] = dict(zip(cols, f))
    ids = {}  # archive key -> id
    if IDS.exists():
        for l in list(open(IDS))[1:]:
            i, k = l.split("\t")[:2]
            ids[k] = i
    rend = [json.loads(l) for l in open(DATA / "renders.jsonl", encoding="utf-8")]
    # QA keeps with no word every ASR heard differently (those go to human review), Parler excluded (40% QA reject)
    kept = [r for r in rend if qa.get(r["key"], {}).get("verdict") == "keep" and qa[r["key"]]["subs"] == "[]"
            and r["variant"] in ("dev", "lat") and "+" not in r["pid"]]
    order = {"bulbul": 0, "magpie": 1}
    kept = [r for r in kept if r["engine"] in order]
    # renders that already hold an id come first, so a re-run never churns ids
    kept.sort(key=lambda r: (r["key"] not in ids, order[r["engine"]], r["variant"] != "dev", r["pid"], r["voice"]))
    pick, seen, voices = [], set(), {}
    for r in kept:  # one render per text; spread voices
        if r["pid"] in seen:
            continue
        if voices.get((r["engine"], r["voice"]), 0) >= max(3, n // 10):
            continue
        pick.append(r), seen.add(r["pid"])
        voices[(r["engine"], r["voice"])] = voices.get((r["engine"], r["voice"]), 0) + 1
        if len(pick) == n:
            break

    nxt = max([int(i[4:]) for i in ids.values()] + [FIRST_ID - 1]) + 1
    with open(IDS, "a") as f:
        if not ids:
            f.write("id\tkey\tengine\tvoice\tpid\n")
        for r in pick:
            if r["key"] not in ids:
                ids[r["key"]] = f"rec-{nxt:06d}"
                f.write(f"{ids[r['key']]}\t{r['key']}\t{r['engine']}\t{r['voice']}\t{r['pid']}\n")
                nxt += 1

    (OUT / "audio").mkdir(parents=True, exist_ok=True)
    (OUT / "tts").mkdir(exist_ok=True)
    recs = []
    for r in pick:
        rid, q = ids[r["key"]], qa[r["key"]]
        (OUT / "tts" / f"{rid}.wav").write_bytes((ROOT / r["wav"]).read_bytes())
        dur = to24k(ROOT / r["wav"], OUT / "audio" / f"{rid}.wav")
        model = r["model"]
        recs.append({
            "id": rid, "audio": f"audio/{rid}.wav", "duration": dur,
            "speaker_id": f"spk-tts-{slug(model.split('/')[-1])}-{slug(r['voice'])}",
            "source": {"corpus": CORPUS, "source_file": f"tts/{rid}.wav", "start": 0, "end": dur, "synthetic": True,
                       "model": model, "voice": r["voice"]},
            "sentences": [{"hi_text": s} for s in SENT_END.split(r["hi_text"])],
            "_w3": {"pid": r["pid"], "key": r["key"], "variant": r["variant"], "params": r["params"],
                    "text_in": r["text_in"], "entities": r["entities"], "src_rate": r["sr"]},
            "_qa": {"w3": {"best_cer": float(q["best_cer"]), "cer": json.loads(q["cer"]), "ent_n": int(q["ent_n"]),
                           "spc_rel": float(q["spc_rel"]), "pause": float(q["pause"])}},
            "_archive": {"w3": r["wav"]},
        })
    recs.sort(key=lambda x: x["id"])
    with open(OUT / "manifest.jsonl", "w", encoding="utf-8") as f:
        for x in recs:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    by = {}
    for r in pick:
        by[r["engine"]] = by.get(r["engine"], 0) + 1
    print(f"{len(recs)} records -> {OUT / 'manifest.jsonl'}; {by}; {sum(x['duration'] for x in recs) / 60:.1f} min")


if __name__ == "__main__":
    assert slug("bulbul:v3") == "bulbul-v3" and slug("magpie_tts_multilingual_357m") == "magpie-tts-multilingual-357m"
    main()
