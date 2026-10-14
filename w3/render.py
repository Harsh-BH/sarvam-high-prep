"""Render the W3 probe set with one TTS engine. Archive-first: every output is kept, nothing is ever re-requested.

    uv run --no-project python w3/render.py bulbul [--limit N]
    flock data/.gpu.lock uv run --no-project python w3/render.py magpie [--limit N]  # NeMo-Speech.cpp CUDA build
    flock data/.gpu.lock uv run --no-project --python 3.13 --with torch \
        --with "parler-tts @ git+https://github.com/huggingface/parler-tts@d108732cd57788ec86bc857d99a6cabd66663d68" python w3/render.py parler

A job is (engine, voice, params, input text). Its key is a hash of exactly that, so the same request maps to the same
archive file: data/w3/archive/<engine>/<key>.wav (audio exactly as the engine returned it) + <key>.json (request
without secrets, response metadata, timing). data/w3/renders.jsonl indexes the archive with the reference transcript
(hi_text = what the input asked the TTS to say; QA decides whether it did).
"""
import argparse
import base64
import hashlib
import io
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data/w3"
ARCH = DATA / "archive"
INDEX = DATA / "renders.jsonl"

# Bulbul v3 hi-IN: 6 + 6 voices (gender from the docs' names, INFERRED; check F0 later). Each probe gets one of each.
BULBUL_F = ["ritu", "priya", "simran", "kavya", "ishita", "roopa"]
BULBUL_M = ["aditya", "rahul", "amit", "manan", "rohan", "kabir"]
MAGPIE = ["Aria", "Jason", "John", "Leo", "Sofia"]  # all five fixed voices; no cloning in this release
PARLER = ["Rohit", "Divya", "Aman", "Rani"]  # the four Hindi speakers on the model card
PARLER_DESC = "{}'s voice is clear and natural, speaking at a moderate pace with a very close recording that has no background noise."
MAGPIE_BIN = DATA / "tools/nemo-speech/nemo-speech-0.2.0-linux-x86_64-cuda/bin/nemo-speech"  # CPU build: 4-12x slower than real time
RUPEES_PER_CHAR = 30 / 10_000  # Bulbul v3 list price (docs, 2026-10-14); confirm against the dashboard


def load_probe():
    return [json.loads(l) for l in open(DATA / "probe.jsonl", encoding="utf-8")]


def long_items(probe, n=10):
    """Same-domain triples joined into ~35-45 s inputs: long-form stability and the 60 s cap."""
    by = {}
    for p in probe:
        if p["domain"] != "stress":
            by.setdefault(p["domain"], []).append(p)
    out = []
    for dom in sorted(by):
        g = by[dom]
        for i in range(0, len(g) - 2, 3):
            t = g[i:i + 3]
            out.append({"pid": "+".join(x["pid"] for x in t), "tts_dev": " ".join(x["tts_dev"] for x in t),
                        "tts_lat": " ".join(x["tts_lat"] for x in t), "hi_text": " ".join(x["hi_text"] for x in t),
                        "entities": sum((x["entities"] for x in t), []), "en_words": sum((x["en_words"] for x in t), []),
                        "chars": sum(x["chars"] for x in t)})
    return out[:n]


def lat_pids(probe, n=40):
    """The n probes with the most English spans: the Devanagari-vs-Latin input test."""
    return {p["pid"] for p in sorted(probe, key=lambda p: (-len(p["en_words"]), p["pid"]))[:n]}


def jobs(engine, probe):
    lat = lat_pids(probe)
    out = []  # (item, variant, voice, params)
    for i, p in enumerate(probe):
        if engine == "bulbul":
            v1, v2 = BULBUL_F[i % 6], BULBUL_M[i % 6]
            out += [(p, "dev", v1, {}), (p, "dev", v2, {})]
            if p["pid"] in lat:
                out.append((p, "lat", v1, {}))
            if i % 3 == 0:  # 34 probes: same voice at temperature 0.3, paired with the default 0.6 render
                out.append((p, "dev", v2, {"temperature": 0.3}))
        else:
            v = (MAGPIE if engine == "magpie" else PARLER)[i % (5 if engine == "magpie" else 4)]
            out.append((p, "dev", v, {}))
            if p["pid"] in lat:
                out.append((p, "lat", v, {}))
    if engine == "bulbul":
        out += [(p, "dev", (BULBUL_F + BULBUL_M)[2 * i % 12], {}) for i, p in enumerate(long_items(probe))]
    return out


def key_of(engine, voice, params, text):
    blob = json.dumps([engine, voice, params, text], ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha1(blob).hexdigest()[:16]


def wav_info(b):
    with wave.open(io.BytesIO(b)) as w:
        return w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes() / w.getframerate()


# --- engines: each takes (text, voice, params) and returns (wav bytes, response metadata) ---

def bulbul(text, voice, params):
    key = os.environ.get("SARVAM_API_KEY")
    if not key:
        sys.exit("SARVAM_API_KEY missing from .env")
    req = {"text": text, "language_code": "hi-IN", "model": "bulbul:v3", "speaker": voice,
           "speech_sample_rate": 24000, "output_audio_codec": "wav", **params}
    body = json.dumps(req, ensure_ascii=False).encode()
    for attempt in range(4):
        r = urllib.request.Request("https://api.sarvam.ai/text-to-speech", data=body, method="POST",
                                   headers={"api-subscription-key": key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(r, timeout=120) as resp:
                d = json.load(resp)
            break
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            if e.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(5 * 2 ** attempt)
                continue
            sys.exit(f"Bulbul HTTP {e.code}: {msg}")
    assert len(d["audios"]) == 1, d.keys()
    return base64.b64decode(d["audios"][0]), {"request": req, "request_id": d.get("request_id"),
                                               "billed_chars": len(text)}


def magpie(text, voice, params):
    out = DATA / "scratch/magpie_tmp.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    seed = params.get("seed", 1)
    cmd = [str(MAGPIE_BIN), "synthesize", text, "--language", "hi", "--voice", voice, "--seed", str(seed),
           "--device", "cuda", "--no-warmup", "-o", str(out), "--force"]
    env = {**os.environ, "NEMO_SPEECH_MODEL_DIR": str(DATA / "models")}
    log = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stderr
    return out.read_bytes(), {"cmd": [c if c != text else "<text_in>" for c in cmd[1:-3]],
                              "model_rev": "v2607 GGUF, nemo-speech 0.2.0", "log": log[-400:]}


_parler = None


def parler(text, voice, params):
    global _parler
    import numpy as np
    import torch
    if _parler is None:
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        name = "ai4bharat/indic-parler-tts"
        m = ParlerTTSForConditionalGeneration.from_pretrained(name, cache_dir=DATA / "hf", torch_dtype=torch.float16).to("cuda").eval()
        _parler = (m, AutoTokenizer.from_pretrained(name, cache_dir=DATA / "hf"),
                   AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path, cache_dir=DATA / "hf"))
    m, tok, dtok = _parler
    seed = params.get("seed", 1)
    torch.manual_seed(seed)
    desc = PARLER_DESC.format(voice)
    d, p = dtok(desc, return_tensors="pt").to("cuda"), tok(text, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        a = m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids,
                       prompt_attention_mask=p.attention_mask, do_sample=True).float().cpu().numpy().squeeze()
    pcm = (np.clip(a, -1, 1) * 32767).round().astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(m.config.sampling_rate), w.writeframes(pcm)
    return buf.getvalue(), {"description": desc, "seed": seed, "dtype": "float16", "do_sample": True}


ENGINES = {"bulbul": (bulbul, "bulbul:v3"), "magpie": (magpie, "nvidia/magpie_tts_multilingual_357m"),
           "parler": (parler, "ai4bharat/indic-parler-tts")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine", choices=ENGINES)
    ap.add_argument("--limit", type=int, default=0, help="render at most N new jobs")
    a = ap.parse_args()
    for line in open(ROOT / ".env"):
        k, _, v = line.strip().partition("=")
        if k and v and not k.startswith("#"):
            os.environ.setdefault(k, v)
    fn, model = ENGINES[a.engine]
    (ARCH / a.engine).mkdir(parents=True, exist_ok=True)
    indexed = {json.loads(l)["key"] for l in open(INDEX)} if INDEX.exists() else set()
    new = spent = 0
    for item, variant, voice, params in jobs(a.engine, load_probe()):
        text = item["tts_lat" if variant == "lat" else "tts_dev"]
        k = key_of(a.engine, voice, params, text)
        wav, meta = ARCH / a.engine / f"{k}.wav", ARCH / a.engine / f"{k}.json"
        if not wav.exists():
            if a.limit and new >= a.limit:
                continue
            t = time.time()
            audio, resp = fn(text, voice, params)
            wav.write_bytes(audio)  # archive first, then index
            resp["secs"] = round(time.time() - t, 2)
            meta.write_text(json.dumps(resp, ensure_ascii=False, indent=1))
            new += 1
            spent += resp.get("billed_chars", 0)
        if k not in indexed:
            sr, ch, sw, dur = wav_info(wav.read_bytes())
            assert ch == 1 and sw == 2, (k, ch, sw)
            rec = {"key": k, "pid": item["pid"], "variant": variant, "engine": a.engine, "model": model, "voice": voice,
                   "params": params, "text_in": text, "wav": str(wav.relative_to(ROOT)), "sr": sr, "dur": round(dur, 3),
                   **{f: item[f] for f in ("hi_text", "entities", "en_words", "chars")},
                   "secs": json.loads(meta.read_text()).get("secs")}
            with open(INDEX, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            indexed.add(k)
            print(k, item["pid"], variant, voice, params or "", f"{dur:.1f}s", flush=True)
    total = sum(json.loads(p.read_text()).get("billed_chars", 0) for p in (ARCH / "bulbul").glob("*.json")) \
        if (ARCH / "bulbul").exists() else 0
    print(f"{a.engine}: {new} new renders; Bulbul chars this run {spent} (₹{spent * RUPEES_PER_CHAR:.0f}), "
          f"all-time {total} (₹{total * RUPEES_PER_CHAR:.0f})")


if __name__ == "__main__":
    assert key_of("e", "v", {}, "x") == key_of("e", "v", {}, "x") != key_of("e", "v", {"temperature": 0.3}, "x")
    main()
