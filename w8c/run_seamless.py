"""SeamlessStreaming on an eval set, simulated in real time on the §5.1 timeline (CC BY-NC weights: private eval only).

    python w8c/run_seamless.py --set SET.jsonl --audio-dir DIR --out OUT [--ckpt decoder.pt|s3://...] [--limit N]

Runs in the fairseq2 0.2.1 env (w8c/sm_eval.py installs it). Per record: OUT/<id>.json + OUT/<id>.wav (the
system-output contract eval.py scores, tasks/W8c-eval.md "Harness").

Simulation: our 24 kHz clip is sent as 80 ms frames, frame k at t = k * 0.08 (§5.1). The agent reads 320 ms blocks
(4 frames, streaming_evaluate's source_segment_size), resampled to 16 kHz float in [-1, 1]. A block is available
when its last frame is sent. Every output gets two arrival times:
  t  = max(block available, previous compute end) + compute seconds   (measured on this GPU: lag piles up if slow)
  t0 = block available                                                  (zero compute: hardware-independent)
Vocoder audio (16 kHz) is resampled to 24 kHz and cut into <= 1 s frames (§5.1). failed = no finish within 30 s of
input_end (§5.1 R3, on t). The model sees the true end of input (input_end) when the last block is sent.
Agent: SeamlessStreaming's s2st pipeline without VAD, plus the detokenizer branch for the text track; settings
agreed with w8d (max_len_a=1, max_len_b=200 = the text decoder's own defaults; streaming_evaluate's 0/100 truncates
our 60 s records). --ckpt: a w8d fine-tuned monotonic decoder, {"model": state_dict}, loaded strict over the base.
"""
import argparse, json, re, time, wave
from pathlib import Path

import numpy as np
import torch, torchaudio
from simuleval import options
from simuleval.data.segments import EmptySegment, SpeechSegment, TextSegment
from seamless_communication.streaming.agents.detokenizer import UnitYDetokenizerAgent
from seamless_communication.streaming.agents.offline_w2v_bert_encoder import OfflineWav2VecBertEncoderAgent
from seamless_communication.streaming.agents.online_feature_extractor import OnlineFeatureExtractorAgent
from seamless_communication.streaming.agents.online_text_decoder import UnitYMMATextDecoderAgent
from seamless_communication.streaming.agents.online_unit_decoder import NARUnitYUnitDecoderAgent
from seamless_communication.streaming.agents.online_vocoder import VocoderAgent
from seamless_communication.streaming.agents.unity_pipeline import UnitYAgentTreePipeline

RATE, FRAME, BLOCK_FRAMES, DEADLINE, MAX_TAIL_STEPS = 24000, 1920, 4, 30.0, 400
SETTINGS = dict(task="s2st", tgt_lang="eng", device="cuda:0", dtype="fp16", min_starting_wait_w2vbert=192,
                decision_threshold=0.5, no_early_stop=True, max_len_a=1, max_len_b=200, min_unit_chunk_size=50)


class Agent(UnitYAgentTreePipeline):
    pipeline = {
        OnlineFeatureExtractorAgent: [OfflineWav2VecBertEncoderAgent],
        OfflineWav2VecBertEncoderAgent: [UnitYMMATextDecoderAgent],
        UnitYMMATextDecoderAgent: [UnitYDetokenizerAgent, NARUnitYUnitDecoderAgent],
        UnitYDetokenizerAgent: [],
        NARUnitYUnitDecoderAgent: [VocoderAgent],
        VocoderAgent: [],
    }
    ckpt = None

    @classmethod
    def load_model(cls, args):
        m = super().load_model(args)
        if cls.ckpt:
            m["monotonic_decoder_model"].load_state_dict(torch.load(cls.ckpt, map_location=args.device)["model"], strict=True)
        return m


def build(ckpt=None, settings=SETTINGS):
    Agent.ckpt = ckpt
    parser = options.general_parser()
    Agent.add_args(parser)
    argv = []
    for k, v in settings.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    args, _ = parser.parse_known_args(argv)
    return Agent.from_args(args)


def blocks(n24):
    """-> [(start_16k, end_16k, available_s)] for a clip of n24 samples at 24 kHz, and input_end time."""
    frames = -(-n24 // FRAME)
    n16 = round(n24 * 16000 / RATE)
    out = []
    for f0 in range(0, frames, BLOCK_FRAMES):
        f1 = min(f0 + BLOCK_FRAMES, frames)
        out.append((round(f0 * FRAME * 16000 / RATE), min(n16, round(f1 * FRAME * 16000 / RATE)), (f1 - 1) * FRAME / RATE))
    return out, (frames - 1) * FRAME / RATE


def run_one(agent, x24, sync=torch.cuda.synchronize):
    """x24: int16 samples at 24 kHz -> (frames [{t, t0, n}], pcm24 int16, text_events [{t, t0, text}], failed)."""
    x16 = torchaudio.functional.resample(torch.from_numpy(x24.astype(np.float32) / 32768), RATE, 16000).numpy()
    bl, input_end = blocks(len(x24))
    agent.reset()
    frames, pcm, events, busy = [], [], [], 0.0
    steps = [(a, b, avail, i == len(bl) - 1) for i, (a, b, avail) in enumerate(bl)] + [None] * MAX_TAIL_STEPS
    for st in steps:
        if st is None:
            seg, avail = EmptySegment(finished=True), input_end
        else:
            a, b, avail, last = st
            seg = SpeechSegment(content=x16[a:b].tolist(), sample_rate=16000, finished=last, tgt_lang="eng")
        start = max(avail, busy)
        tic = time.perf_counter()
        outs = agent.pushpop(seg)
        sync()
        busy = start + time.perf_counter() - tic
        for o in outs if isinstance(outs, list) else [outs]:
            if isinstance(o, TextSegment) and isinstance(o.content, str) and o.content.strip():
                events.append({"t": round(busy, 3), "t0": round(avail, 3), "text": o.content.strip()})
            elif isinstance(o, SpeechSegment) and len(o.content):
                y = torchaudio.functional.resample(torch.tensor(o.content, dtype=torch.float32), o.sample_rate, RATE)
                y = (y.clamp(-1, 1) * 32767).round().to(torch.int16).numpy()
                for i in range(0, len(y), RATE):  # §5.1: at most 1 s per frame
                    frames.append({"t": round(busy, 3), "t0": round(avail, 3), "n": len(y[i:i + RATE])})
                    pcm.append(y[i:i + RATE])
        done = all(o.finished for o in (outs if isinstance(outs, list) else [outs]))
        if st is None or st[3]:
            if done or busy - input_end > DEADLINE:
                break
    failed = not done or busy - input_end > DEADLINE
    return frames, np.concatenate(pcm) if pcm else np.zeros(0, np.int16), events, failed


def read_wav(p):
    with wave.open(str(p)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (RATE, 1, 2), p
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def write_wav(p, x):
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1), w.setsampwidth(2), w.setframerate(RATE), w.writeframes(x.tobytes())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", type=Path, required=True)
    ap.add_argument("--audio-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in open(a.set) if l.strip()][:a.limit]
    agent = build(a.ckpt)
    for r in recs:
        if (a.out / f"{r['id']}.json").exists():
            continue  # resumable: a spot restart redoes only the missing records
        x = read_wav(a.audio_dir / r["audio"])
        tic = time.perf_counter()
        frames, pcm, events, failed = run_one(agent, x)
        write_wav(a.out / f"{r['id']}.wav", pcm)
        text = re.sub(r" ([,.?!;:])", r"\1", " ".join(e["text"] for e in events))  # the detokenizer emits "," alone
        (a.out / f"{r['id']}.json").write_text(json.dumps({
            "id": r["id"], "frames": frames, "text": text, "text_events": events, "failed": failed,
            "compute_s": round(time.perf_counter() - tic, 3), "src_s": round(len(x) / RATE, 3)}, ensure_ascii=False))
        print(f"{r['id']}: {len(x) / RATE:.1f} s in {time.perf_counter() - tic:.1f} s, {len(pcm) / RATE:.1f} s out, "
              f"failed={failed}: {text[:120]}", flush=True)


if __name__ == "__main__":
    main()
