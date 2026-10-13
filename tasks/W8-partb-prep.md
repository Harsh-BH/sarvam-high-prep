# W8: Part B prep (independent of Part A; start now)

**Goal:** be ready on the first day after the mid-term. The base checkpoint, its inference code and the reference client arrive only then (§11 FAQ). Training code is ours to write (§3B).

## W8a: Streaming server + local harness

PS rules (§5.1, §6):
- WebSocket server on port 8000, path `/ws`; one connection = one session.
- **Input:**
  - `session_start` JSON, then 80 ms binary frames (3,840 B of s16le at 24 kHz) in real time.
  - Then `input_end` with `valid_samples`.
  - Then silent frames until we send `session_complete`.
- **Output:**
  - PCM frames of at most 1 s each.
  - Exactly one `session_complete`, within 30 s of `input_end`, and nothing after it.
- **A session fails** (worst value on every metric) if any of these happen:
  - we stop reading for more than 1 s;
  - we take more than 10 s to accept the connection;
  - we send an extra text frame;
  - we close early.
- `/health` must return 200 within 300 s of start-up. No network; one H100.

Build:
- A harness that mimics the organiser exactly:
  - a real-time sender;
  - playback-timeline accounting (each frame plays at max(arrival, previous end));
  - checks for every failure rule.
- A server skeleton: an I/O loop that never blocks; the inference worker in a separate thread or process; a dummy model behind a clean interface; an offline Docker image.
- Fault injection: a slow model, exceptions, runaway generation after `input_end`.

Research:
- How websockets, uvicorn and aiohttp buffer incoming data and apply back-pressure.
- CUDA-graph and compile warm-up before `/health` returns 200.
- Offline flags such as `HF_HUB_OFFLINE`.

## W8b: English target pipeline

PS rules (§3B):
- English target audio = Bulbul `bulbul:v3`, speaker `shubh`, `en-IN`, **24 kHz**. Pace and temperature are free but must be declared per training pair.
- We must align the English audio ourselves; its timings must be ordered, non-negative and inside the audio.

Build: a Bulbul client (always `speech_sample_rate=24000`; batching; archiving), an English forced aligner (MFA, wav2vec2 or NeMo), and the checks.

Measure:
- How Bulbul pronounces lakh/crore, digit strings and Indian names. Does a pronunciation dictionary (`dict_id`) help?
- How pace affects duration.
- The English/Hindi duration ratio on the pilot (lag accumulates if the English runs longer).
- Character cost per hour of English.

## W8c: Eval harness and dev set

G2 requires scores before and after fine-tuning (§8), and Sarvam gives no KPI feedback (§6). Build our own proxies:
- Output ASR (Whisper or Parakeet).
- Translation quality: chrF, BLEU, COMET against references.
- **Entity accuracy:** normalised number and name matching (25,000 = 25k; §11 FAQ).
- **Intent accuracy:** an LLM judge with a rubric (asked, refused, confirmed, corrected, negation, condition, roles).
- **Latency:** start offset, end offset and average lag, from word alignment of our output on the playback timeline against the source word times.
- **Audio quality:** UTMOS or DNSMOS.

Dev set (dev only, not Part A):
- FLEURS-hi, which has parallel English.
- A small team-recorded role-play set over a phone line, with references and entity/intent labels.

## W8d: Model literature and adaptation playbooks

- [Inference] 24 kHz with 80 ms frames matches Kyutai's Mimi codec (Moshi, Hibiki). Read up on:
  - Hibiki: contextual alignment, silence insertion, its text stream, CFG;
  - Moshi;
  - StreamSpeech, Seamless Streaming and SimulS2S.
- Fine-tuning recipes: LoRA on multistream decoders, delay conditioning, training-pair construction from word alignments.
- Compute: estimate H100 hours for 1–7B models against our Modal budget (~250 H100-hours at about $3.95/h; verify the price).
- Write playbooks for the 2–3 most likely architecture families: what we fine-tune, how we build pairs, how we serve it in real time.

## Done when

- 8a passes our harness under fault injection.
- 8b produces aligned English targets for the pilot.
- 8c computes every KPI proxy on toy outputs.
- 8d has its playbooks written.
