# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "boto3", "torch", "transformers", "onnxruntime", "huggingface_hub", "soundfile"]
# ///
"""`tts` stage (design S2 + S3): W2 scripts -> QA-passed synthetic Hindi records; a failed take is regenerated.

    uv run w3/tts.py --in SCRIPTS.jsonl --audio-dir DIR --out OUT_DIR [--engine E] [--max-takes N] [--corpus C] [--bench]
    uv run w3/tts.py alloc SCRIPTS.jsonl          # record ids for new scripts -> w3/ids.tsv; commit before any run
    uv run w3/tts.py upload SCRIPTS.jsonl STAGE   # -> stages/STAGE/<run>/records.jsonl + LATEST, for run.py --in STAGE
    uv run w3/tts.py report SCRIPTS.jsonl DIR...  # bench tables (first takes) from DIR/takes.jsonl
    uv run w3/tts.py listen RUN_DIR OUT_DIR [N [S3_RUN]]  # the <= 10 min listening list for Harsh

Input: W2 scripts (tasks/W2-scripts.md: pid, tts_dev, hi_text, sentences, entities, en_words, chars, gender).
--audio-dir is unused: scripts have no audio. Per script: a voice of the script's gender (stable hash of pid, from
w3/voices.tsv), then take 1, 2, ... (seed = take) through archive.archive_call, so every take, kept or rejected, is
archived write-once under archive/api/tts-<engine>/<key>.bin|.json and no take is ever rendered twice. Each take is
heard by two ASRs from other families than the TTS (vakyansh CTC, IndicConformer RNNT) and judged by w3/qa.py's rules
plus "no agreed substitution" (design §3.3: drop, don't review). The first passing take becomes the record:
OUT/audio/<id>.wav (24 kHz s16) and the same bytes write-once at archive/tts/<id>-<take key[:16]>.wav (+ .meta.json),
which the rebuild copies (`_archive.tts`). hi_text = the script's transcript, which QA says the take actually said.
OUT/takes.jsonl: one QA row per take. OUT/asr/<model>.jsonl: hypotheses by take key, so a re-run never re-decodes.
--bench: no ids, no records, take 1 of every --engine per script: takes.jsonl is the benchmark (stage `tts_bench`).
"""
import argparse, hashlib, io, json, os, re, subprocess, sys, tempfile, time, wave
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "pipeline"), str(ROOT / "w4")]
import archive, qa, s3 as store  # noqa: E402
from run import _main_checkout  # noqa: E402

MAIN = _main_checkout()  # the laptop's data lives in the main checkout; a SageMaker job has only ROOT
IDS, VOICES = HERE / "ids.tsv", HERE / "voices.tsv"
FIRST_ID, RATE = 400000, 24000
MAX_SEC = 55.0  # OmniVoice length cap (the clip limit is 60 s, §3A)
ON_SM = Path("/opt/ml/input/config").exists()
NO_SUBS = "subs1"  # one word every ASR heard differently: review in round 1, a retake here (one human, §3.3)


def text_key(s):
    return "txt:" + hashlib.sha256(s["tts_dev"].encode()).hexdigest()[:16]


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def jsonl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if Path(p).exists() else []


def ids_table(path=IDS):
    """w3/ids.tsv -> {key: (id, pid)}. Keys: a render's archive key (round-1 pilot) or txt:<sha256(tts_dev)[:16]>."""
    rows = [l.rstrip("\n").split("\t") for l in open(path)][1:] if Path(path).exists() else []
    return {r[1]: (r[0], r[4]) for r in rows}


def alloc(scripts, path=IDS):
    """Give every new script text an id, append-only (SPEC §4: assigned once, never reused). -> ids added."""
    have = ids_table(path)
    nxt = max([int(i[4:]) for i, _ in have.values()] + [FIRST_ID - 1]) + 1
    new = []
    for s in scripts:
        k = text_key(s)
        if k in have:
            if have[k][1] != s["pid"]:
                print(f"{s['pid']}: same text as {have[k][1]} ({have[k][0]}), not allocated", file=sys.stderr)
            continue
        have[k] = (f"rec-{nxt:06d}", s["pid"])
        new.append(f"{have[k][0]}\t{k}\t-\t-\t{s['pid']}\n")
        nxt += 1
    with open(path, "a") as f:
        f.writelines(new)
    return len(new)


def upload(path, stage, s3=None):
    """Put a scripts file in S3 as a stage run (no audio), so run.py can feed it to `tts`. -> run id."""
    import run
    rid = run.run_id(run.git_sha(), datetime.now(timezone.utc))
    prefix, body = f"stages/{stage}/{rid}", Path(path).read_bytes()
    store.put(f"{prefix}/records.jsonl", body, s3)
    meta = {"stage": stage, "run": rid, "input": f"upload: {path}", "params": [], "where": "local",
            "audio_from": f"{prefix}/audio/", "records": body.count(b"\n"), "rejects": 0, "violations": 0, "exit": 0}
    store.put(f"{prefix}/run.json", json.dumps(meta, indent=1).encode(), s3)
    store.put(f"stages/{stage}/LATEST", rid.encode(), s3)
    return rid


def voices(engine, clone, path=VOICES):
    """Voices an engine may use: its own built-in ones, plus the shared reference clips (engine '*') if it clones."""
    rows = [dict(zip(["voice", "engine", "gender", "ref", "ref_text", "source", "licence"], l.rstrip("\n").split("\t")))
            for l in open(path, encoding="utf-8") if l.strip() and not l.startswith("#")][1:]
    return [v for v in rows if v["engine"] == engine or (clone and v["engine"] == "*")]


def pick_voice(s, bank):
    """Same script -> same voice, spread evenly; gender-matched when the script fixes it (Hindi verbs agree, W2)."""
    pool = sorted((v for v in bank if s.get("gender") in (None, v["gender"])), key=lambda v: v["voice"])
    if not pool:
        raise SystemExit(f"{s['pid']}: no {s.get('gender')} voice in the bank")
    return pool[int(hashlib.sha256(s["pid"].encode()).hexdigest()[:8], 16) % len(pool)]


def wav_bytes(x, sr):
    """float [-1, 1] or int16 -> canonical 16-bit mono WAV (44-byte header)."""
    x = np.asarray(x)
    pcm = x.astype("<i2") if x.dtype == np.int16 else (np.clip(x, -1, 1) * 32767).round().astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(sr), w.writeframes(pcm.tobytes())
    return buf.getvalue()


def to24k(b):
    """A take's WAV -> the record's 24 kHz s16 WAV: the same bytes if already 24 kHz, else soxr with the pinned ffmpeg."""
    with wave.open(io.BytesIO(b)) as w:
        sr, ch, sw, pcm = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.readframes(w.getnframes())
    assert ch == 1 and sw == 2, (ch, sw)
    if sr == RATE:
        return wav_bytes(np.frombuffer(pcm, "<i2"), RATE)
    import ffmpeg
    ffmpeg.TOOLS = MAIN / "data/tools"  # the pinned build is already unpacked there on the laptop
    pcm = subprocess.run([str(ffmpeg.binary()), "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0", "-af",
                          "aresample=resampler=soxr", "-ar", str(RATE), "-ac", "1", "-f", "s16le", "-fflags", "+bitexact",
                          "-"], input=b, check=True, capture_output=True).stdout
    return wav_bytes(np.frombuffer(pcm, "<i2"), RATE)


def seconds(b):
    with wave.open(io.BytesIO(b)) as w:
        return w.getnframes() / w.getframerate(), w.getframerate()


# --- engines: load() -> synth(text, voice_row, seed) -> (float array or int16, sample rate) ---

def need(*pkgs):
    """Install an engine's packages into the job's python. Locally they come from `uv run --with ...` instead."""
    if ON_SM:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)


def seed_all(seed):
    import random
    import torch
    random.seed(seed), np.random.seed(seed), torch.manual_seed(seed)


def ref_path(v):
    """A reference clip (raw/_voices/<voice>.wav in S3) as a local file."""
    p = MAIN / "data/w3/voices" / Path(v["ref"]).name
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(store.get(v["ref"]))
    return str(p)


def load_parler():
    """Indic Parler-TTS (Apache-2.0, gated): named speakers via a text description. Round 1: 40% QA reject."""
    need("parler-tts @ git+https://github.com/huggingface/parler-tts@d108732cd57788ec86bc857d99a6cabd66663d68")
    import torch
    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    name, rev = ENGINES["parler"]["model"], ENGINES["parler"]["rev"]
    m = ParlerTTSForConditionalGeneration.from_pretrained(name, revision=rev, torch_dtype=torch.float16).to("cuda").eval()
    tok, dtok = AutoTokenizer.from_pretrained(name, revision=rev), AutoTokenizer.from_pretrained(m.config.text_encoder._name_or_path)
    desc = "{}'s voice is clear and natural, speaking at a moderate pace with a very close recording that has no background noise."

    def synth(text, v, seed):
        seed_all(seed)
        d, p = dtok(desc.format(v["voice"]), return_tensors="pt").to("cuda"), tok(text, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            a = m.generate(input_ids=d.input_ids, attention_mask=d.attention_mask, prompt_input_ids=p.input_ids,
                           prompt_attention_mask=p.attention_mask, do_sample=True)
        return a.float().cpu().numpy().squeeze(), m.config.sampling_rate
    return synth


def ref_audio(v):
    """A reference clip as (float32 mono array, sample rate)."""
    import soundfile as sf
    x, sr = sf.read(ref_path(v), dtype="float32")
    return x, sr


def snapshot(engine):
    """The engine's HF weights at the pinned revision (ungated: a job has no HF token)."""
    from huggingface_hub import snapshot_download
    return snapshot_download(ENGINES[engine]["model"], revision=ENGINES[engine]["rev"])


def load_omnivoice():
    """OmniVoice (k2-fsa; code Apache-2.0, weights CC BY-NC): zero-shot cloning from a 3-10 s reference + transcript.
    It speaks at the reference's pace (the bank keeps 8-14 chars/s references, w3/voices.py). Forcing a conversational
    duration made it drop words (agreed skips 12/44 -> 0/44 without it), so a duration is set only when the
    reference's pace would run past 55 s (the clip limit is 60)."""
    need("omnivoice==0.2.1")
    import torch
    from omnivoice import OmniVoice
    m = OmniVoice.from_pretrained(snapshot("omnivoice"), device_map="cuda:0", dtype=torch.float16)
    prompts, rates = {}, {}

    def synth(text, v, seed):
        if v["voice"] not in prompts:
            x, sr = ref_audio(v)
            prompts[v["voice"]] = m.create_voice_clone_prompt(ref_audio=(torch.from_numpy(x), sr), ref_text=v["ref_text"])
            rates[v["voice"]] = len(v["ref_text"].replace(" ", "")) / qa.audio_stats(ref_path(v))["speech"]
        seed_all(seed)
        est = len(re.sub(r"[\s,]", "", text)) / rates[v["voice"]]
        return m.generate(text=text, language="hi", voice_clone_prompt=prompts[v["voice"]],
                          duration=MAX_SEC if est > MAX_SEC else None)[0], 24000
    return synth


def load_voxcpm2():
    """VoxCPM2 (OpenBMB, Apache-2.0): cloning with the reference's transcript ("ultimate cloning"), 48 kHz out."""
    need("voxcpm==2.0.3")
    from voxcpm import VoxCPM
    m = VoxCPM.from_pretrained(snapshot("voxcpm2"), load_denoiser=False)

    def synth(text, v, seed):
        seed_all(seed)
        ref = ref_path(v)
        wav = m.generate(text=text, prompt_wav_path=ref, prompt_text=v["ref_text"], reference_wav_path=ref,
                         cfg_value=2.0, inference_timesteps=10)
        return wav, m.tts_model.sample_rate
    return synth


def load_pocket():
    """pocket-tts-hindi (Saryps Labs, CC BY 4.0; runtime MIT): 110M, cloning from a reference clip, CPU-fast.
    Its config points at the repo's main branch, so the pinned revision is checked against main instead."""
    need("pocket-tts==3.3.0")
    from huggingface_hub import model_info
    from pocket_tts import TTSModel
    E = ENGINES["pocket"]
    if model_info(E["model"]).sha != E["rev"]:
        raise SystemExit(f"{E['model']} main moved off the pinned {E['rev']}: re-benchmark before using it")
    m = TTSModel.load_model(config=f"hf://{E['model']}/config.yaml", eos_threshold=-1.0)  # the card's eval setting
    states = {}

    def synth(text, v, seed):
        if v["voice"] not in states:
            states[v["voice"]] = m.get_state_for_audio_prompt(ref_path(v))
        seed_all(seed)
        plain = " ".join(re.sub(r"[,।?!]", " ", text).split())  # trained without punctuation (card)
        return m.generate_audio(states[v["voice"]], plain).float().numpy(), m.sample_rate
    return synth


def load_fake():
    """Self-check engine: a tone whose length follows the text. Never a record outside the test."""
    return lambda text, v, seed: (0.3 * np.sin(np.arange(int(0.09 * len(text) * RATE)) * 0.05), RATE)


ENGINES = {
    "omnivoice": {"model": "k2-fsa/OmniVoice", "rev": "c5fdb5ccb189668d56333f77ba2629f4cd7535f4", "clone": True,
                  "load": load_omnivoice, "opts": "own pace; duration 55 s if the reference pace runs past it"},
    "voxcpm2": {"model": "openbmb/VoxCPM2", "rev": "32279effe8c19989596f05d353d1447f51d9e915", "clone": True,
                "load": load_voxcpm2},
    "pocket": {"model": "saryps-labs/pocket-tts-hindi", "rev": "6bff9c1e776bb90c68dd05c2467c62d1ecdf1b51", "clone": True,
               "load": load_pocket},
    "parler": {"model": "ai4bharat/indic-parler-tts", "rev": "660bc53c", "clone": False, "load": load_parler},
    "fake": {"model": "w3-selfcheck-tone", "rev": "0", "clone": False, "load": load_fake},
}


def load_asr():
    """{name: fn(16 kHz float32) -> hyp}: W4's loaders (GPU when present), vakyansh decoded in quiet chunks."""
    import asr_run, ic_weights, torch
    asr_run.CACHE = None  # the default HF cache: vakyansh is already there on the laptop; a job downloads it
    ic_weights.LOCAL = MAIN / "data/w4/ic600m"  # laptop: W4's copy; a job fetches the write-once S3 mirror here
    v, ic = asr_run.vakyansh("vakyansh-w2v2"), asr_run.indicconformer("ic-rnnt")

    def w2v(x):
        with torch.inference_mode():
            return " ".join(v(x[i:j]) for i, j in qa.quiet_chunks(x))

    def icr(x):
        with torch.inference_mode():
            return ic(x)
    return {"w2v": w2v, "ic": icr}, asr_run.load16k


def spc_ref(rows, voice, engine, n=8):
    """Seconds-per-char reference: median over this voice's earlier takes, else the engine's, else none yet."""
    for sel in ([r["spc"] for r in rows if r["voice"] == voice], [r["spc"] for r in rows if r["engine"] == engine]):
        if len(sel) >= n:
            return float(np.median(sel))
    return None


def stage(scripts, out, engine, *, max_takes=3, bench=False, corpus="w2-scripts-v1", s3=None, synth=None, asr=None,
          load16k=None, ids=None, bank=None, spool=archive.SPOOL, archive_only=False):
    out = Path(out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    (out / "asr").mkdir(exist_ok=True)
    for f in ("records.jsonl", "rejects.jsonl", "takes.jsonl"):
        p = out / f
        p.touch()
        if (b := p.read_bytes()) and not b.endswith(b"\n"):
            p.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: drop it, that script is redone
    engines = engine.split(",")  # several: a script starts on one (stable hash of its pid), retakes rotate
    banks = {e: bank or voices(e, ENGINES[e]["clone"]) for e in engines}
    ids = ids_table() if ids is None else ids
    takes = {r["key"]: r for r in jsonl(out / "takes.jsonl")}
    done = {r["id"] for f in ("records.jsonl", "rejects.jsonl") for r in jsonl(out / f)}
    have = {}
    for r in takes.values():
        have.setdefault(r["pid"], set()).add(r["engine"])
    done_pids = {p for p, e in have.items() if e >= set(engine.split(","))} if bench else set()
    todo, seen = [], set()
    for s in scripts:
        k = text_key(s)
        if bench:
            if s["pid"] not in done_pids:
                todo.append((None, s))
            continue
        if k not in ids:
            raise SystemExit(f"{s['pid']} has no id: run `uv run w3/tts.py alloc` and commit w3/ids.tsv first")
        rid, pid = ids[k]
        if pid != s["pid"] or k in seen:
            print(f"{s['pid']}: duplicate text of {pid} ({rid}), skipped", file=sys.stderr)
            continue
        seen.add(k)
        if rid not in done:
            todo.append((rid, s))
    print(f"tts {engine}: {len(todo)} to do ({len(scripts)} in, {len(done) or len(done_pids)} done)", file=sys.stderr)
    if not todo:
        return
    hyps = {m.stem: {r["key"]: r["hyp"] for r in jsonl(m)} for m in (out / "asr").glob("*.jsonl")}
    synths = {}
    names = list(asr) if asr else ["w2v", "ic"]
    for m in names:
        hyps.setdefault(m, {})
    ears = {"asr": asr, "load16k": load16k}  # the ASR models load only when a hypothesis is missing

    def hear(m, path):
        if ears["asr"] is None:
            ears["asr"], ears["load16k"] = load_asr()
        return ears["asr"][m](ears["load16k"](path))

    def call(req):
        if archive_only:
            raise SystemExit(f"--archive-only: take {archive.request_key(req)} is not archived")
        t = time.time()
        if req["engine"] not in synths:
            synths[req["engine"]] = synth or ENGINES[req["engine"]]["load"]()
        x, sr = synths[req["engine"]](req["text"], voice_row[req["voice"]], req["seed"])
        b = wav_bytes(x, sr)
        return {"sr": sr, "secs": round(time.time() - t, 2), "dur": round(seconds(b)[0], 3)}, b

    voice_row = {v["voice"]: v for b in banks.values() for v in b}
    for rid, s in todo:
        first = int(hashlib.sha256(f"engine {s['pid']}".encode()).hexdigest()[:8], 16)
        tried = []
        # bench: take 1 of every engine; else the first take by hash, each retake on the next engine
        plan = [(e, 1) for e in engines] if bench else [(engines[(first + t - 1) % len(engines)], t)
                                                         for t in range(1, max_takes + 1)]
        for eng, take in plan:
            E, service = ENGINES[eng], f"tts-{eng}"
            v = pick_voice(s, banks[eng])
            req = {"engine": eng, "model": E["model"], "rev": E["rev"], "voice": v["voice"], "ref": v["ref"],
                   "text": s["tts_dev"], "seed": take, "take": take, **({"opts": E["opts"]} if "opts" in E else {})}
            # own GPU, no API bill: inr 0, no cap
            meta, b = archive.archive_call(service, req, call, stream="w3", inr=0.0, cap_inr=float("inf"), s3=s3, spool=spool)
            key = archive.request_key(req)
            row = takes.get(key)
            if row is None:
                t_asr = time.time()
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    f.write(b), f.flush()
                    for m in names:
                        if key not in hyps[m]:
                            hyps[m][key] = hear(m, f.name)
                            with open(out / "asr" / f"{m}.jsonl", "a", encoding="utf-8") as h:
                                h.write(json.dumps({"key": key, "hyp": hyps[m][key]}, ensure_ascii=False) + "\n")
                    q = qa.measure(s, {m: hyps[m][key] for m in names}, f.name)
                q["spc_rel"] = None if (ref := spc_ref(list(takes.values()), v["voice"], eng)) is None else q["spc"] / ref
                dur = seconds(b)[0]
                why = qa.rules(q, dur) + ([NO_SUBS] if len(q["subs"]) == 1 else [])
                row = {"pid": s["pid"], "id": rid, "take": take, "key": key, "engine": eng, "voice": v["voice"],
                       "gender": v["gender"], "dur": round(dur, 3), "sr": meta["sr"], "secs": meta["secs"],
                       "asr_secs": round(time.time() - t_asr, 2),
                       "chars": s["chars"], "tags": s.get("tags", []), **q, "why": why,
                       "verdict": "reject" if why else "keep"}
                takes[key] = row
                with open(out / "takes.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if bench:
                continue
            if row["verdict"] == "keep":
                b24 = to24k(b)
                dur = round(seconds(b24)[0], 3)
                take_key = f"archive/api/{service}/{key}.bin"
                # named by id and take: another run (other engine, voice or rules) may keep a different take for this id
                kept = f"archive/tts/{rid}-{key[:16]}.wav"
                if not store.put_once(kept, b24, s3) and store.get(kept, s3) != b24:
                    raise SystemExit(f"{kept} already holds different audio")
                store.put_once(f"{kept}.meta.json", json.dumps(
                    {"sha256": hashlib.sha256(b24).hexdigest(), "take": take_key, "request": req, "render": meta},
                    ensure_ascii=False, indent=1).encode(), s3)
                (out / "audio" / f"{rid}.wav").write_bytes(b24)
                rec = {"id": rid, "audio": f"audio/{rid}.wav", "duration": dur,
                       "speaker_id": f"spk-tts-{slug(E['model'].split('/')[-1])}-{slug(v['voice'])}",
                       "source": {"corpus": corpus, "source_file": f"tts/{rid}.wav", "start": 0, "end": dur,
                                  "synthetic": True, "model": E["model"], "voice": v["voice"]},
                       "sentences": [{"hi_text": t} for t in s["sentences"]],
                       "_w2": {k: x for k, x in s.items() if k not in ("_gen", "sentences", "hi_text")},
                       "_w3": {"text_key": text_key(s), "engine": eng, "rev": E["rev"], "take": take,
                               "take_key": take_key, "rejected_takes": tried, "src_rate": meta["sr"]},
                       "_qa": {"tts": {k: row[k] for k in ("best_cer", "cer", "ent_n", "en_n", "spc_rel", "pause", "lead",
                                                           "trail", "asrs")}},
                       "_archive": {"tts": kept}}
                with open(out / "records.jsonl", "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                break
            tried.append(f"archive/api/{service}/{key}.bin")
        else:
            if not bench:
                with open(out / "rejects.jsonl", "a", encoding="utf-8") as f:
                    detail = "; ".join(f"take{r['take']}: {','.join(r['why'])}" for r in takes.values()
                                       if r["pid"] == s["pid"])
                    f.write(json.dumps({"id": rid, "stage": "tts", "rule": "qa", "detail": detail}, ensure_ascii=False) + "\n")


def report(scripts, dirs):
    """Markdown tables from takes.jsonl files (first takes only, so engines compare like round 1), one group per
    run directory (named after the engine), then by voice."""
    ents = {x["pid"]: [qa.canon(e) for e in x["entities"]] for x in scripts}
    rows = [{**r, "engine": f"{Path(d).name}/{r['engine']}"} for d in dirs for r in jsonl(Path(d) / "takes.jsonl")
            if r["take"] == 1]

    def pct(a, b):
        return f"{100 * a / b:.1f}%" if b else "–"

    def table(groups):
        print("| group | takes | audio h | CER ic p50/p90 | entity hit | number-entity hit | English-span hit | agreed skip "
              "| agreed insert | QA reject (r1 rules) | reject incl. 1 subst. | s/char p50 | wall s per audio s |")
        print("|" + "---|" * 13)
        for name, g in groups:
            ic = [r["cer"].get("ic", np.nan) for r in g]
            ne = sum(1 for r in g for e in ents[r["pid"]] if qa.num_runs(e))
            nm = sum(1 for r in g for e in r["ent_miss"] if qa.num_runs(e))
            en_n, en_m = sum(r["ent_n"] for r in g), sum(len(r["ent_miss"]) for r in g)
            sp_n, sp_m = sum(r["en_n"] for r in g), sum(len(r["en_miss"]) for r in g)
            r1 = sum(1 for r in g if set(r["why"]) - {NO_SUBS})
            dur = sum(r["dur"] for r in g)
            print(f"| {name} | {len(g)} | {dur / 3600:.2f} | {np.nanmedian(ic):.3f}/{np.nanpercentile(ic, 90):.3f} | "
                  f"{pct(en_n - en_m, en_n)} | {pct(ne - nm, ne)} | {pct(sp_n - sp_m, sp_n)} | "
                  f"{pct(sum(bool(r['skips']) for r in g), len(g))} | {pct(sum(bool(r['adds']) for r in g), len(g))} | "
                  f"{pct(r1, len(g))} | {pct(sum(r['verdict'] == 'reject' for r in g), len(g))} | "
                  f"{np.median([r['spc'] for r in g]):.3f} | {sum(r['secs'] for r in g) / dur:.2f} |")
    engines = sorted({r["engine"] for r in rows})
    print("\n**By engine**\n")
    table([(e, [r for r in rows if r["engine"] == e]) for e in engines])
    print("\n**By voice**\n")
    table([(f"{e} {v} ({g})", [r for r in rows if r["engine"] == e and r["voice"] == v])
           for e in engines for v, g in sorted({(r["voice"], r["gender"]) for r in rows if r["engine"] == e})])
    why = {}
    for r in rows:
        for w in r["why"]:
            k = (r["engine"], re.sub(r"[\d.]+$", "", w))
            why[k] = why.get(k, 0) + 1
    print("\n**Reject reasons** (takes can have several): " + "; ".join(
        f"{e}: " + ", ".join(f"{w} {n}" for (e2, w), n in sorted(why.items(), key=lambda x: -x[1]) if e2 == e) for e in engines))


def listen(run, out, per_gender=6, fillers=4, s3_run=None):
    """The one list for Harsh (<= 10 min): one clip per voice for `per_gender` voices of each gender (Indian accent?
    gender right?), plus clips whose transcript has a filler (no ASR keeps fillers, W4). Shortest records first.
    s3_run (stages/tts/<run>): fetch only the picked clips from there when RUN has no audio/ locally."""
    run, out = Path(run), Path(out)
    recs = sorted(jsonl(run / "records.jsonl"), key=lambda r: (r["duration"], r["id"]))
    pick, voices = [], {}
    for r in recs:
        g = r["_w2"]["gender"]
        if r["source"]["voice"] not in voices and sum(v == g for v in voices.values()) < per_gender:
            voices[r["source"]["voice"]] = g
            pick.append((r, f"native Indian Hindi accent? a {'female' if g == 'f' else 'male'} voice?"))
    for r in recs:
        f = [w for s in r["sentences"] for w in s["hi_text"][:-1].split() if w in qa.FILLERS]
        if f and r not in [x for x, _ in pick] and len(pick) < 2 * per_gender + fillers:
            pick.append((r, f"is the filler '{' '.join(f)}' audible where the transcript has it?"))
    (out / "audio").mkdir(parents=True, exist_ok=True)
    with open(out / "listen.tsv", "w", encoding="utf-8") as f:
        f.write("n\tfile\tid\tvoice\tquestion\ttranscript\tanswer\n")
        for i, (r, q) in enumerate(pick, 1):
            src = run / r["audio"]
            b = src.read_bytes() if src.exists() else store.get(f"{s3_run}/{r['audio']}")
            (out / "audio" / f"{i:02d}-{r['id']}.wav").write_bytes(b)
            text = " ".join(s["hi_text"] for s in r["sentences"])
            f.write(f"{i}\taudio/{i:02d}-{r['id']}.wav\t{r['id']}\t{r['source']['voice']}\t{q}\t{text}\t\n")
    return len(pick), sum(r["duration"] for r, _ in pick) / 60


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "alloc":
        print(f"{alloc(jsonl(argv[1]))} ids added to {IDS}")
        return
    if argv and argv[0] == "report":
        return report(jsonl(argv[1]), argv[2:])
    if argv and argv[0] == "listen":
        n, mins = listen(argv[1], argv[2], per_gender=int(argv[3]) if len(argv) > 3 else 6,
                         s3_run=argv[4] if len(argv) > 4 else None)
        print(f"{n} clips, {mins:.1f} min of audio -> {argv[2]}/listen.tsv")
        return
    if argv and argv[0] == "upload":
        print(f"stages/{argv[2]}/{upload(argv[1], argv[2])}: uploaded, LATEST moved")
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine", default="omnivoice", help=f"one of {sorted(ENGINES)}, or several comma-separated")
    ap.add_argument("--max-takes", type=int, default=3)
    ap.add_argument("--corpus", default="w2-scripts-v1")
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N scripts only (smoke runs)")
    ap.add_argument("--archive-only", action="store_true", help="re-judge archived takes; fail if one would be rendered")
    a = ap.parse_args(argv)
    if bad := set(a.engine.split(",")) - set(ENGINES):
        ap.error(f"unknown engine(s) {sorted(bad)}")
    scripts = jsonl(a.inp)
    stage(scripts[: a.limit] if a.limit else scripts, a.out, a.engine, max_takes=a.max_takes, bench=a.bench, corpus=a.corpus,
          archive_only=a.archive_only)


if __name__ == "__main__":
    assert text_key({"tts_dev": "क"}) == text_key({"tts_dev": "क", "pid": "x"}) != text_key({"tts_dev": "ख"})
    assert slug("ai4bharat/indic-parler-tts".split("/")[-1]) == "indic-parler-tts"
    main()
