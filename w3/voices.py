# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "boto3"]
# ///
"""Reference clips for voice-cloning TTS engines -> raw/_voices/<voice>.wav (+ .meta.json, write-once) + w3/voices.tsv.

    uv run w3/voices.py [N_PER_GENDER]     # default 20

Source: IndicVoices Hindi (CC BY 4.0), the files W1 already holds (data/w1/iv_train, iv_full). Per speaker, one
segment of 6-11 s from a non-conversation (wideband) recording whose verification report is excellent/good with no
noise/echo/chatter/volume flags and whose verbatim text has no tags; the speaker's cleanest such segment (SNR, as
pipeline/audio_qc.py measures it) is cut to 24 kHz with the pinned ffmpeg. The N cleanest speakers per gender are kept.
Consent caveat (tasks/W3-hindi-tts.md): these speakers agreed to ASR data collection, not to having their voice cloned.
"""
import ast, glob, hashlib, json, re, shutil, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent / "pipeline")]
import audio_qc, ffmpeg, s3 as store  # noqa: E402
from run import _main_checkout  # noqa: E402

MAIN = _main_checkout()
ffmpeg.TOOLS = MAIN / "data/tools"
BAD = ("noise_persistent", "noise_intermittent", "echo_present", "unclear_audio", "low_volume", "chatter_persistent",
       "chatter_intermittent", "stretching", "long_pauses")
LIC = ("CC BY 4.0", "https://huggingface.co/datasets/ai4bharat/IndicVoices")


def candidates():
    """-> {(speaker, gender): [(wav, start, end, text, meta)]} for every clean segment."""
    out = {}
    for f in sorted(glob.glob(str(MAIN / "data/w1/iv_*/Hindi/*/*/*.json"))):
        d, wav = json.load(open(f)), Path(f[:-5] + ".wav")
        vr = d.get("verification_report") or {}
        if (not wav.exists() or d.get("scenario") == "Conversation" or vr.get("decision") not in ("excellent", "good")
                or any(vr.get(k) for k in BAD) or d.get("gender") not in ("Male", "Female")):
            continue
        v = ast.literal_eval(d["verbatim"]) if isinstance(d["verbatim"], str) else d["verbatim"]
        for s in v:
            t = " ".join(s["text"].split())
            if 6 <= s["end"] - s["start"] <= 11 and t and not re.search(r"[<>\[\]A-Za-z0-9०-९]", t):
                out.setdefault((d["speaker_id"], d["gender"][0].lower()), []).append((wav, s["start"], s["end"], t, d))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    tmp, best = Path(tempfile.mkdtemp(dir=MAIN / "data/w3")), []
    for (spk, g), segs in candidates().items():
        scored = []
        for i, (wav, a, b, t, d) in enumerate(segs[:6]):  # ponytail: first 6 per speaker bound the cutting time
            p = tmp / f"{spk}-{i}.wav"
            try:
                ffmpeg.cut(wav, a, b, p)
            except AssertionError:  # the segment runs past the end of the file's audio
                continue
            scored.append((audio_qc.measure(p)["snr_db"], p, wav, a, b, t, d))
        if scored:
            best.append((spk, g, max(scored, key=lambda x: x[0])))
    rows = []
    for g in ("f", "m"):
        for spk, _, (snr, p, wav, a, b, t, d) in sorted((x for x in best if x[1] == g), key=lambda x: -x[2][0])[:n]:
            voice, body = f"iv-{spk}", p.read_bytes()
            src = "indicvoices/" + wav.relative_to(next(q for q in wav.parents if q.name in ("iv_train", "iv_full"))).as_posix()
            meta = {"voice": voice, "corpus": "IndicVoices", "source_file": src, "start": a, "end": b, "text": t,
                    "speaker_id": spk, "gender": g, "state": d.get("state"), "district": d.get("district"),
                    "age_group": d.get("age_group"), "scenario": d.get("scenario"), "snr_db": round(snr, 1),
                    "sha256": hashlib.sha256(body).hexdigest(), "licence": LIC[0], "licence_url": LIC[1],
                    "attribution": "AI4Bharat, IndicVoices (arXiv 2403.01926)",
                    "consent": "collected for ASR; cloning not explicitly consented (W3 caveat)"}
            key = f"raw/_voices/{voice}.wav"
            if not store.put_once(key, body) and hashlib.sha256(store.get(key)).hexdigest() != meta["sha256"]:
                raise SystemExit(f"{key} already holds a different clip")
            store.put_once(key + ".meta.json", json.dumps(meta, ensure_ascii=False, indent=1).encode())
            (MAIN / "data/w3/voices").mkdir(parents=True, exist_ok=True)
            (MAIN / "data/w3/voices" / f"{voice}.wav").write_bytes(body)
            rows.append(f"{voice}\t*\t{g}\t{key}\t{t}\t{src}@{a}-{b}\t{LIC[0]}\n")
    head = [l for l in open(HERE / "voices.tsv", encoding="utf-8") if not l.startswith("iv-")]
    (HERE / "voices.tsv").write_text("".join(head + rows), encoding="utf-8")
    shutil.rmtree(tmp)
    print(f"{len(rows)} reference voices -> raw/_voices/ and w3/voices.tsv")


if __name__ == "__main__":
    main()
