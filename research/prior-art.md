# Prior art: streaming Hindi → English speech-to-speech translation (S2ST)

**What this is.** Open-source repos, papers, model cards and shared tasks we can borrow from, organised by workstream. It feeds W2–W8, mostly W8d. The prior-art scout owns this file. Everything was checked on **2026-10-14**.

**How it was checked.** Every item was fetched; nothing comes from memory.
- 136 arXiv IDs checked against the arXiv API (title and date).
- 105 GitHub repos checked with `gh api` (licence, last push, stars, archived).
- 84 Hugging Face (HF) models and datasets checked against the HF API (licence, gating).
- Where an HF metadata tag and the card text disagree, the card text wins and the conflict is flagged ⚠.
- No models or datasets were downloaded (about 10 MB of pages in total). Raw scout notes are in the session scratchpad and are not committed.

**Labels.** **S** = stated by the source. **I** = our inference. **A** = ambiguous; both readings are given.

**Licence flags.** §10 R5 says Sarvam owns all code we submit. That includes the Part A data-pipeline commit (§9.2), not only the container. §10 R1 governs data.

| Flag | Meaning |
|---|---|
| OK | Permissive (MIT, BSD, Apache, CC BY, CC0) |
| ATTR | Permissive, but publications must acknowledge it |
| NC | Non-commercial |
| CL | Copyleft or share-alike |
| NOLIC | No licence, so all rights are reserved |
| GATED | You must accept terms on HF first |
| BASE | The upstream model's terms bind outputs or derivatives |
| ⚠ | Sources disagree |

---

## 0. Headlines (read these if nothing else)

1. **Our 80 ms frame is exactly one frame of Kyutai's Mimi codec (S).**
   - Mimi works at 24 kHz with 1920 samples per frame, i.e. 12.5 Hz or 80 ms, and 1920 × 2 B = 3840 B. That is the §5.1 frame (Moshi §3.3; `moshi/models/loaders.py`).
   - That makes a Hibiki/Moshi-style base the most likely guess, but it is **not proof**: Qwen3-Omni's speech generator also emits one 12.5 Hz token per 80 ms (S, arXiv 2509.17765). See §8d.1.
2. **Hibiki-Zero (ICML 2026 oral) is the closest prior art to Part B.**
   - Its last fine-tune used **under 200 h, 1K steps, batch 16, lr 1e-6** (S). That is our scale.
   - It needs no word-level alignment: it trains on sentence-level "coarse alignment", then cuts latency with GRPO, a reinforcement-learning method.
   - Kyutai has not released its training or RL code. Its weights are **CC BY-NC-SA 4.0**: the card text says so while the HF metadata licence field is empty.
3. **Sarvam has published the rubric it uses to score intent and entities**: [sarvamai/llm_intent_entity](https://github.com/sarvamai/llm_intent_entity). Intent is a strict 0/1; entity is the fraction of entities preserved. It is very close to the §7 KPIs. It has **no licence**, so we reimplement it rather than copy it.
4. **Sarvam's Bulbul docs now recommend `bulbul:v4-flash`, and on that model "The v3 default `shubh` is not valid"** (S, [docs](https://docs.sarvam.ai/api/getting-started/models/bulbul.md)). Always pass `model="bulbul:v3"` explicitly. The other default is a trap too: leaving out the sample rate gives 22,050 Hz.
5. **There is no open-weight Hindi→English S2ST model with a usable licence.**
   - SeamlessStreaming and IndicSeamless are NC.
   - IWSLT 2026 ran a Hindi→English speech-to-speech track, and its findings say "none of the teams submitted a system that achieved significant performance" (S).
6. **A licence trap sits inside the Kyutai fine-tuning stack.** `moshi-finetune/annotate.py` imports **whisper-timestamped (AGPL-3.0)**, the tool Kyutai used for word timestamps in every paper. Use MFA, WhisperX or Qwen3-ForcedAligner instead.
7. **Two serving bugs would each fail a session (§5.1 R4/R7):**
   - uvicorn stops reading the socket until the app calls `receive()` again.
   - Starlette's `receive_bytes()` raises `KeyError` when the `input_end` text frame arrives (S, code; see §8a).
8. **No Inter IIT edition from 9.0 to 14.0 had a Sarvam or speech problem statement**, so there are no past team repos to reuse (S, ten GitHub searches plus a compilation of past problem statements).

---

## W1: Real-audio sourcing (additions to W1's corpus table only)

| Item | Link | What it gives us | Licence | Maturity |
|---|---|---|---|---|
| LAHAJA | [HF](https://huggingface.co/datasets/ai4bharat/Lahaja) · arXiv 2408.11440 | 12.5 h of Hindi, read and extempore; 132 speakers from 83 districts; entity-heavy (S) | Card says CC BY 4.0, HF tag says MIT ⚠. GATED | 2025-03 |
| IndicVoices-R | [HF](https://huggingface.co/datasets/ai4bharat/indicvoices_r) · arXiv 2409.05356 | 1,704 h (22 languages) of **enhanced** IndicVoices audio (S). I: same source recordings as IndicVoices, so hours count once (§3A), and it is cleaner than test-like audio (§4) | CC BY 4.0. GATED | 2025-03 |
| IndicVoices-ST | [HF](https://huggingface.co/datasets/ai4bharat/IndicVoices-ST) · arXiv 2411.04699 | IndicVoices Hindi paired with **synthetic** English translations (S). A format reference for W6, not ground truth | CC BY 4.0. GATED | 2025-01 |
| Indic DiarBench | [HF](https://huggingface.co/datasets/sarvamai/indic-diarbench) · arXiv 2607.23808 | 10.7 h of real multi-speaker Hindi at 16 kHz, transcribed in Devanagari and normalised (S). I: a good real-audio dev set for W8c; single-speaker spans would have to be cut for Part A | CC BY 4.0. OK | 2026-08 |
| FLEURS (hi) | [HF](https://huggingface.co/datasets/google/fleurs) | Read speech with parallel English. Dev/eval only | CC BY 4.0. OK | — |
| Svarah | [HF](https://huggingface.co/datasets/ai4bharat/Svarah) · arXiv 2305.15760 | 9.6 h of Indian-accented **English**. For choosing the W8b/W8c output ASR, not for Hindi records | CC BY 4.0 (card). GATED | 2025-03 |
| Silero VAD | [GH](https://github.com/snakers4/silero-vad) | Segmentation. IndicVoices used it for segments under 20 s, averaging 9 s (S, 2403.01926 §6) | MIT. OK | 2026-09, 10.4k★ |
| pyannote community-1 | [GH](https://github.com/pyannote/pyannote-audio) · [HF](https://huggingface.co/pyannote/speaker-diarization-community-1) | Diarisation, to enforce one speaker per clip | Code MIT; weights CC BY 4.0. GATED | 2026-10, 10.6k★ |
| MUCS 2021 Hi-En (SLR104) | [site](https://navana-tech.github.io/MUCS2021/data.html) · arXiv 2104.00235 | About 90 h of code-switched lecture audio; English in **Latin** script | **CL (CC BY-SA 4.0)**: W1 already excludes it | 2021 |

**Rejected:**
- Rural_Women_ASR_v2 (NOLIC).
- CS-FLEURS (NC).
- `dianavdavidson/MUCS-Hinglish`: a third party re-uploaded SLR104 and **re-tagged it CC BY**. Trap.
- InfoBayAI Hindi call-centre: tagged CC BY, but it is a vendor sample and consent can't be checked.

---

## W2: Script generation

| Item | Link | What it gives us | Licence | Maturity |
|---|---|---|---|---|
| **IndicCMix** | [HF](https://huggingface.co/datasets/ai4bharat/IndicCMix) | 104,800 Hindi rows. Each has an English source, a code-mixed Latin version and a **code-mixed version with English transliterated into Devanagari**, word-aligned (S). That last version already follows our transcript rule. I: written register; use it for few-shot seeds and as an English→Devanagari loanword lexicon | MIT. GATED. **Don't use its audio** (made with F5-TTS, NC) | 2026-04 |
| COMI-LINGUA | [HF](https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA) · arXiv 2503.21670 | 125K+ expert-annotated Hinglish instances, Devanagari and Roman, with an MT subset (S) | CC BY 4.0. OK | 2026-03 |
| PHINC | [HF](https://huggingface.co/datasets/LingoIITGN/PHINC) | Romanised Hinglish ↔ English parallel text | CC BY 4.0. OK | 2025-03 |
| samvaad-hi-v1 | [HF](https://huggingface.co/datasets/sarvamai/samvaad-hi-v1) | 100k English, Hindi and Hinglish conversations: style seeds | Apache-2.0. OK | 2024-07 |
| tts-robustness-benchmark | [HF](https://huggingface.co/datasets/sarvamai/tts-robustness-benchmark) | 959 stress prompts: 267 numerics, 215 Indian named entities, 70 code-mixed (S). Shows what Sarvam thinks is hard; use it to shape the coverage matrix | "other". **Eval-only; don't copy prompts into records** | 2026-02 |
| TTS-STT flywheel | arXiv 2605.03073 · [GH](https://github.com/praxelhq/stt-flywheel) | Recipe: entity dictionaries → LLM text → **five different TTS systems** so the ASR does not learn one voice; Entity-Hit-Rate scores "5 lakh" equal to "five hundred thousand" (S) | Paper CC BY; repo NOASSERTION, so treat as NOLIC | 2026-05 |
| indic-numtowords | [GH](https://github.com/AI4Bharat/indic-numtowords) (Sarvam fork: [sarvamai](https://github.com/sarvamai/indic-numtowords)) | Hindi number words with लाख/करोड़, **spoken variants** (डेढ़ सौ, ढाई सौ, पौने दो सौ) and a digit-by-digit mode for phone and account numbers (S). Integers ≥ 0 only | MIT. OK | 2025-09 |
| Script-generation LLMs, clean | [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) · [gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) · [sarvam-30b](https://huggingface.co/sarvamai/sarvam-30b) · [sarvam-105b](https://huggingface.co/sarvamai/sarvam-105b) | Open weights to run on Modal | All Apache-2.0. OK | 2026 |
| Script-generation LLMs, binding terms | Gemma 3 · Llama 3.x | Gemma 3: a model trained on Gemma "synthetic data Outputs" is a Model Derivative (S). Llama 3.3: a model trained on its outputs must be named "Llama…" (S) | **BASE**. Avoid for anything that trains the submitted model | — |

**Rejected:**
- Microsoft GCM toolkit: pins Python 3.7 and needs the Stanford Parser (**GPL**).
- L3Cube HingCorpus (**NC-SA**).
- CMU Hinglish DoG (**CL**).
- LinCE (NOLIC).

The CMI code-mixing index is a few lines of code; write it ourselves.

---

## W3: Hindi TTS and QA

| Item | Link | What it gives us | Licence | Maturity |
|---|---|---|---|---|
| Bulbul v3 hi-IN (API) | [docs](https://docs.sarvam.ai/api/getting-started/models/bulbul.md) | Facts in W8b. In the SpeechArena human-preference study (arXiv 2604.21481), Bulbul v3 beta scored Elo 1021.9 overall and 1031.3 on code-mixed text. IndicF5, the only open model tested, scored 805.8 (S) | §10 R2/R3 | — |
| Indic Parler-TTS | [HF](https://huggingface.co/ai4bharat/indic-parler-tts) | Named Hindi voices (Rohit, Divya, Aman, Rani), controlled by a text description; no cloning | Apache-2.0. GATED | 210k dl |
| Veena | [HF](https://huggingface.co/maya-research/Veena) | 4 fixed voices, Hindi and English including code-mixed, 24 kHz | Apache-2.0. OK | 2025-10 |
| Indic-Speak (Bodhan AI + AI4Bharat) | [HF](https://huggingface.co/bodhan-ai/indic-speak) | **95 voices from contracted artists who consented**; English words inside Hindi read in the same voice (S) | Indic Open Model License: **CL**, hosting needs sign-off, full text gated. BASE (Llama-3.2, I). GATED | 2026-09 |
| svara-TTS v1 | [HF](https://huggingface.co/kenpath/svara-tts-v1) | 19 languages; the card says code-mixing is "not deterministic" | Tagged Apache-2.0, but the base chain leads to Llama-3.2 (I): BASE | 83k dl |
| pocket-tts-hindi | [HF](https://huggingface.co/saryps-labs/pocket-tts-hindi) · [kyutai pocket-tts](https://github.com/kyutai-labs/pocket-tts) | 100M-parameter TTS that runs on CPU; a community Hindi model. Untested, 0 downloads | CC BY 4.0 / MIT. OK | 2026-09 |
| Kokoro-82M | [HF](https://huggingface.co/hexgrad/Kokoro-82M) | Hindi voices graded C | Apache-2.0, but its Hindi path runs **espeak-ng (GPL-3.0)** | 9.3k★ |
| IndicF5 | [HF](https://huggingface.co/ai4bharat/IndicF5) · [GH](https://github.com/AI4Bharat/IndicF5) | Voice cloning. Terms: "only clone voices for which you have explicit permission" (S) | Weights MIT; **code NOLIC**. GATED. Consent risk | 145★ |
| SpeechArenaBench | [HF](https://huggingface.co/datasets/ai4bharat/SpeechArenaBench) · arXiv 2604.21481 | Hindi code-mixed preference prompts for TTS QA | MIT. GATED | 2026-06 |
| kyutai tts_longeval | [GH](https://github.com/kyutai-labs/tts_longeval) | WER and speaker-similarity harness for TTS QA | MIT. "No support" | 29★ |

**Rejected:**
- **XTTS-v2**: the CPML licence makes "a model **and its outputs**" non-commercial.
- **F5-TTS base**: CC BY-NC.
- AI4Bharat Indic-TTS: MIT repo, but trained on the IndicTTS data W1 avoids, with a Coqui (MPL) dependency.

**Consent (I):** never clone W1 corpus speakers. They consented to ASR data collection, not to having their voices cloned.

---

## W4: Transcription of real audio

| Item | Link | What it gives us | Licence | Maturity |
|---|---|---|---|---|
| Indic-Transcribe-core | [HF](https://huggingface.co/bodhan-ai/indic-transcribe-core) | The card says it writes English words in Devanagari and keeps spoken numbers as words, which matches §3A. Hindi OIWER 3.5, against Saaras V3 3.8 and IndicConformer 6.5, on the vendor's own benchmark (S). **Refuses audio over 45 s; no timestamps** | Indic Open Model License, **CL**. Whether training on its outputs makes a derivative is unknown (full text gated). GATED | 2026-09 |
| IndicConformer 600M multilingual | [HF](https://huggingface.co/ai4bharat/indic-conformer-600m-multilingual) | Hybrid CTC/RNNT. The HF wrapper returns text only | MIT. GATED | 485k dl |
| IndicConformer Hindi `.nemo` | [HF](https://huggingface.co/ai4bharat/indicconformer_stt_hi_hybrid_ctc_rnnt_large) · [AI4Bharat NeMo fork](https://github.com/AI4Bharat/NeMo) | Its **CTC head feeds the NeMo Forced Aligner**, so ASR, alignment and confidence come from one stack (I). Needs the fork, branch `nemo-v2` | MIT / Apache-2.0. OK | — |
| IndicWhisper (Vistaar) | [GH](https://github.com/AI4Bharat/vistaar) · arXiv 2305.15386 | Hindi WER: Kathbath 10.3, FLEURS 11.4, GramVaani 26.8 (S) | MIT, models included. OK | 93★ |
| IndicWav2Vec-Hindi | [HF](https://huggingface.co/ai4bharat/indicwav2vec-hindi) | Character-level CTC, a source of alignment and confidence signals | Apache-2.0. GATED | — |
| Qwen3-ASR 1.7B | [HF](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) · arXiv 2601.21337 | Supports Hindi; a second-opinion ASR from a different model family (W3 QA needs that) | Apache-2.0. OK | 1.1M dl |
| Nemotron-3.5-ASR-Streaming-0.6B | [HF](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b) | Hindi FLEURS WER 8.13% at 80 ms latency, 6.81% at 1.12 s (S). Also a streaming design to study for Part B | OpenMDW-1.1, no restrictions on outputs. OK | 1.3M dl |
| Saaras v4 (API) | [docs](https://docs.sarvam.ai/api-reference/speech-to-text/transcribe.md) | Modes: transcribe, translate, verbatim, translit, codemix. "**Word-level timestamps are not supported.**" REST takes at most 30 s; raw PCM must be 16 kHz; ₹30/h (S) | §10 R2 | — |
| NeMo-text-processing | [GH](https://github.com/NVIDIA/NeMo-text-processing) | **Hindi text normalisation**: लाख/करोड़, phone numbers, dates, money. Also Hindi, Hindi-English code-mixed and English inverse normalisation | Apache-2.0. OK | 2026-10 |
| IndicXlit | [GH](https://github.com/AI4Bharat/IndicXlit) | Romanised → Devanagari. Hindi top-1 accuracy 55.6–60.6% (S). I: trained on romanised Hindi, not English spellings, so weak for loanwords | MIT; depends on fairseq (archived). Stale | 145★ |
| Cadence | [HF](https://huggingface.co/ai4bharat/Cadence) · arXiv 2506.03793 | Punctuation restoration, for the । ? ! the transcripts need | Tagged MIT, built on Gemma 3 (BASE). GATED | 2025-11 |
| sarvamai/llm_wer | [GH](https://github.com/sarvamai/llm_wer) | Sarvam's WER, where an LLM decides whether mismatched spans are equivalent (digits vs words, Amazon = अमेज़न) | **NOLIC**: read the idea, don't copy | 2026-04 |
| sft_saaras_flow | [HF](https://huggingface.co/datasets/sarvam/sft_saaras_flow) | 1,100 ASR-correction examples showing Sarvam's formatting prompts, including spoken self-corrections | Apache-2.0. The `sarvam` org is **not HF-verified** (I) | 2026-06 |

**IndicVoices transcription levels** (S, arXiv 2403.01926; W0 and W4 must handle both readings):
- Level 1 = "transcription is performed exactly as spoken". Level 2 = "standardizing words to their textbook form".
- The rule to add the English spelling in brackets is listed under Level 2. Yet Table 5's Level 1 examples already contain `[OTP]` and `[breathing]` ⚠.
- Reduplication gets "a hyphen(-) in between" (चाय-वाय). W0 must decide whether that is one timed word or two.

**Disfluencies:** no ASR card claims to keep fillers or disfluencies. The W4 pilot has to measure it.

**Scoring Hindi:** never use Whisper's normaliser. `BasicTextNormalizer` replaces combining marks with spaces, which strips Devanagari vowel signs. It turns FLEURS-hi WER of 38.46% into 21.45% (S).

---

## W5: Hindi forced alignment

| Item | Link | What it gives us | Licence | Maturity |
|---|---|---|---|---|
| Montreal Forced Aligner (MFA) | [GH](https://github.com/MontrealCorpusTools/Montreal-Forced-Aligner) · arXiv 2606.18466 | **No pretrained Hindi acoustic model.** There is a Hindi config and a Hindi Common Voice dictionary (CC0), so we train on our own audio (S). MFA 3.0 reports under 15 ms mean boundary error on its benchmarks (S) | Code MIT; models CC BY 4.0. OK | 2026-10, 1.9k★ |
| Code-mixed Hindi-English MFA study | arXiv 2607.25581 · [GH](https://github.com/Ayushi113/mfa-hindi-code-mixed) | An acoustic model trained on code-mixed speech: **4.15 ms** error, against 38.18 ms for a Hindi-only model and 37.58 ms for English-only (S; the gold set is tiny) | Repo NOLIC and uses eSpeak (GPL). **Use the method only** | 2026-07 |
| NeMo Forced Aligner | [GH](https://github.com/NVIDIA-NeMo/Speech/tree/main/tools/nemo_forced_aligner) | Word and token times from any CTC or hybrid model, including IndicConformer Hindi; CTC posteriors give confidence | Apache-2.0. OK | 2026-10 |
| ctc-forced-aligner | [GH](https://github.com/MahmoudAshraf97/ctc-forced-aligner) | Works with any HF CTC model. Pair it with indicwav2vec-hindi to avoid NC weights | Code BSD-2; **default model CC BY-NC** | 569★ |
| torchaudio `forced_align` + MMS_FA | [docs](https://docs.pytorch.org/audio/stable/tutorials/forced_alignment_for_multilingual_data_tutorial.html) · [uroman](https://github.com/isi-nlp/uroman) | The "MMS-FA baseline" in W5's plan | torchaudio BSD-2; **MMS_FA weights CC BY-NC (S, docs)**; uroman ATTR. See trap T3 | — |
| CTC-segmentation | [GH](https://github.com/lumaku/ctc-segmentation) | Utterance-level alignment with confidence; useful for long clips | Apache-2.0. OK | 346★ |
| Seamless UnitY2 aligner | [docs](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/unity2_aligner_README.md) | Text-to-unit aligner at 20 ms, 38 languages; I: Hindi is among them | Code MIT; checkpoint licence **not verified** | — |
| WhisperX | [GH](https://github.com/m-bain/whisperX) | ASR plus alignment | BSD-2; **its default Hindi alignment model has no licence** | 24.5k★ |
| Newer multilingual aligners | arXiv 2601.18220 (LLM-ForcedAligner), 2606.10675, 2609.33650 (FuseAlign) | Candidates if MFA and CTC both disappoint | Titles verified; code and licences not checked | 2026 |

**The only same-protocol comparison found** is FA-Bench (arXiv 2609.32396): English conversational speech (Buckeye), word boundaries, reference text given (S):

| Aligner | Mean error, clean (ms) | Mean error, noisy (ms) | F1 within 20 ms |
|---|---|---|---|
| **MFA 3.4** | **21.6** | **36.3** | **0.676** |
| Qwen3-ForcedAligner | 34.7 | 54.6 | 0.440 |
| MMS-FA | 35.4 | 40.6 | 0.225 |
| WhisperX | 40.3 | 51.3 | 0.162 |
| NeMo Forced Aligner (80 ms model) | 90.2 | 92.6 | 0.219 |

Whisper places words about 150 ms early (S). arXiv 2406.19363 also finds MFA beats WhisperX and MMS.

I: train MFA on our own code-mixed audio as the boundary-accuracy candidate. Keep NeMo Forced Aligner or ctc-forced-aligner as the cheap one, since their CTC confidences also flag TTS skips and repeats for W3. Decide on the W5 gold set.

---

## W6: Translation and target style

| Item | Link | What it gives us | Licence | Maturity |
|---|---|---|---|---|
| IndicTrans2 | [GH](https://github.com/AI4Bharat/IndicTrans2) · [toolkit](https://github.com/VarunGumma/IndicTransToolkit) · [HF](https://huggingface.co/ai4bharat/indictrans2-indic-en-1B) | Sentence-level Hindi → English, and an MT scorer for contextual alignment (§W8d.5) | Code and checkpoints MIT. GATED. OK | 2025-10, 481★ |
| IndicTrans3-beta | [HF](https://huggingface.co/ai4bharat/IndicTrans3-beta) | Document-level translation on Gemma 3 | CC BY 4.0 + Gemma 3 terms (BASE). GATED | 2025-05 |
| Sarvam-Translate | [HF](https://huggingface.co/sarvamai/sarvam-translate) · [API](https://docs.sarvam.ai/api-reference/text/translate-text.md) | Gemma-3-4B fine-tune. API: sarvam-translate:v1 takes at most 2,000 characters, formal style only; mayura:v1 at most 1,000; ₹20 per 10K characters; numerals default to **digits** (S) | **Weights GPL-3.0** + Gemma 3. Fine as a Part A teacher (§10 R1); never in the submission | 2026-07 |
| Indic-Translate (Bodhan) | [HF](https://huggingface.co/bodhan-ai/indic-translate) | Code-mixed translation on Gemma 4 | **CL**, GATED | 2026-09 |
| IN22-Conv | [HF](https://huggingface.co/datasets/ai4bharat/IN22-Conv) | Conversational translation test set | CC BY 4.0. OK | — |

**Target-style literature.** All of it can be done with Part A text, before the mid-term.
- **Models copy their teacher's style.** Hibiki scores ASR-BLEU 47.9 against its MT teacher's translations but 35.5 against the references (S, 2502.03382 §4.6.2). W6's choices on numbers, names and corrections become how the model speaks.
- **Chunk-wise monotonic translation for a verb-final source** (2406.08940, English↔Japanese simultaneous interpretation). Japanese is verb-final like Hindi, so this is the closest analogue for our sentence-boundary mapping.
- **EASiST** (2504.11809): an LLM writes **monotonic translations at several latencies** per semantic chunk. The code is NOLIC, so take the recipe only.
- **Action-aware simultaneous interpretation** (2509.21801): sentence-cut plus drop gave the best fluency–latency balance (S). **But drop and summarise cost entity and intent points and risk §10 R8.** Take sentence-cut only.
- **Prefix-to-prefix targets** from a teacher: CSSEL-P2P (2607.13158) and "Learning When to Commit" (2610.02612). Relevant only if we build prefix-level `en_alt`.
- **Disfluency paper** (2608.02138): argues for keeping self-repairs in the translation. **The PS overrides it**: references carry the corrected meaning (FAQ).

**Rejected:** NLLB-200, SeamlessM4T-v2 and indic-seamless are all **CC BY-NC**.

---

## W7: Assembly and reproducibility

- [kyutai-labs/sphn](https://github.com/kyutai-labs/sphn) (Apache-2.0): fast audio reads and durations, used by moshi-finetune.
- Bulbul has no `seed` parameter (S, docs), so archive every output. The rebuild must never re-synthesise (CLAUDE.md).
- Nothing else found that beats stdlib plus pinned ffmpeg. HF repo limits are still W7's own research item.

---

## W8a: Streaming server

| Item | Link | What it gives us | Licence | Maturity |
|---|---|---|---|---|
| Kyutai `moshi.server` / Hibiki-Zero `inference.py` | [moshi](https://github.com/kyutai-labs/moshi) · [hibiki-zero](https://github.com/kyutai-labs/hibiki-zero) | The per-frame loop to reuse: `mimi.encode` → `lm_gen.step` → `mimi.decode`, with **one PS frame (3,840 B) per model step**. It uses aiohttp WebSockets, one session at a time behind a lock, 4 warm-up frames and CUDA graphs on by default. Hibiki-Zero writes **input end-of-stream (EOS) as token 2048 at frame `ceil(duration·12.5)`**, keeps stepping until the text stream emits EOS, and **drops the first encoded frame** (S, code) | Apache-2.0 / MIT. OK | 11.2k★ / 274★ |
| Kyutai Rust `moshi-server` | [delayed-streams-modeling](https://github.com/kyutai-labs/delayed-streams-modeling) | msgpack audio in 1920-sample chunks. **The client sends an end-of-stream `Marker` and keeps sending silence; the server echoes the marker once it has processed everything** (S). This is our `input_end` → silence → `session_complete` pattern. Batch 64; 400 real-time speech-to-text streams per H100 | Apache-2.0 / MIT. OK | 3.0k★ |
| simulstream | [GH](https://github.com/hlt-mt/simulstream) · arXiv 2512.17648 | WebSocket server and client for streaming speech translation, with JSONL logs for scoring. The IWSLT 2026 toolkit and the closest open analogue to the organiser's harness. Text output only | Apache-2.0. OK | 2026-09 |
| SimulStreaming / WhisperLive / FastRTC | [ufal](https://github.com/ufal/SimulStreaming) · [collabora](https://github.com/collabora/WhisperLive) · [gradio](https://github.com/gradio-app/fastrtc) | Server and warm-up patterns; FastRTC for the live demo UI (§9.2) | MIT. OK | 687★ / 4.3k★ / 4.6k★ |
| vLLM-Omni | [GH](https://github.com/vllm-project/vllm-omni) | Full-duplex realtime endpoint for MiniCPM-o 4.5, Qwen3-Omni and Nemotron VoiceChat (S). Only relevant if the base is one of those; vLLM's own `/v1/realtime` is ASR only | Apache-2.0 | 7.2k★ |

**Facts to build against** (S unless marked):
1. **uvicorn** (both WebSocket backends) **stops reading the socket after each message until the app calls `receive()`**. One task must always be waiting on `receive()`, and nothing in that path may wait for the model.
2. **Starlette `receive_bytes()`** returns `message["bytes"]`, but a text frame arrives with only a `"text"` key. `input_end` therefore raises `KeyError`, the connection closes early, and the session fails. Call `receive()` and branch on the key. (Checked against `starlette/websockets.py` on main.)
3. **websockets 17.2 defaults:**
   - receive queue of 16 frames before reading pauses;
   - write limit 32 KiB;
   - ping 20 s; open and close timeouts 10 s;
   - maximum message size 1 MiB;
   - deflate compression on, which we should turn off since PCM barely compresses (I).
   - `/health` can be served on the same port via `process_request`.
4. The initial Linux TCP receive buffer is 131,072 B, so about 34 frames (~2.7 s) can pile up before the harness's send blocks (I, arithmetic). We don't know how the harness detects "not reading", so don't rely on this slack.
5. Kyutai's server runs the model inside the WebSocket handler. That only works while every step finishes in under 80 ms. It also **fetches its web UI from HF at start-up unless run with `--static none`**, which fails with no network (§6).
6. **Warm-up:** capture CUDA graphs on a side stream after a few eager iterations (PyTorch docs). Return 503 from `/health` until warm-up is done (I). Set `HF_HUB_OFFLINE=1`.
7. **Measured overheads:**
   - CMU's IWSLT 2025 system measured 0.5–0.6 s above its theoretical latency (S).
   - At IWSLT 2026, measuring real compute time added about 0.3 s on average, and the only end-to-end 1B system had the **largest** gap, over 0.4 s (S).
   - Profile; a small model is not automatically fast.

---

## W8b: English targets (Bulbul) and English alignment

**Bulbul v3 facts** (S, [model page](https://docs.sarvam.ai/api/getting-started/models/bulbul.md) and [API reference](https://docs.sarvam.ai/api-reference/text-to-speech/convert.md), fetched 2026-10-14):
- **Model and voice.** `bulbul:v3` is still the default when `model` is omitted. The docs recommend `bulbul:v4-flash` for new integrations, and **`shubh` is invalid on v4-flash**. The v3 default speaker is `shubh`.
- **Sample rate.** REST accepts 8–48 kHz; streaming accepts at most 24 kHz. **If omitted, output is 22,050 Hz.** The **WebSocket output codec defaults to mp3**.
- **Controls.**
  - `pace`: 0.5–2.0.
  - `temperature`: default 0.6.
  - **No seed. No SSML.**
  - `pitch` and `loudness`: the API reference says v2 only, while [sarvamai/skills](https://github.com/sarvamai/skills) says v3 accepts them ⚠. Avoid both; §3B frees only pace and temperature.
- **Limits.**
  - 2,500 characters per request; the docs advise writing numbers over 4 digits with commas.
  - Pronunciation dictionary (`dict_id`): at most 10 dictionaries of 100 words each.
- **Price.** ₹30 per 10K characters.
  - I: about ₹160 per English audio-hour, so roughly ₹32K per full 200 h synthesis pass.
  - Credits are in $ and prices in ₹; confirm the conversion in the dashboard.

| English aligner | Link | Evidence | Licence |
|---|---|---|---|
| **MFA `english_mfa` v3.1** | [mfa-models](https://github.com/MontrealCorpusTools/mfa-models) | Best word-boundary error in FA-Bench (W5 table) | CC BY 4.0. OK |
| Qwen3-ForcedAligner-0.6B | [HF](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) | English yes, **Hindi no**. Vendor-reported average shift 37.5 ms, against 107.5 ms for NeMo Forced Aligner and 92.1 ms for WhisperX (S) | Apache-2.0. OK |
| NeMo Forced Aligner + FastConformer / Parakeet CTC | [HF](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_pc) | CTC confidences | CC BY 4.0. OK |
| Kyutai DSM recipe | arXiv 2509.08753 §3.2 | **We know the exact text we sent to Bulbul**: get Whisper word times, then DTW them onto the known text (S). Kyutai `stt-2.6b-en` gives word timestamps (CC BY 4.0) | — |
| torchaudio `WAV2VEC2_ASR_BASE_960H` + `forced_align` | [docs](https://docs.pytorch.org/audio) | English CTC alignment with no NC weights | MIT. OK |

**Probe before the bulk synthesis run (I):**
- Run the numerics and named-entity prompts from Sarvam's tts-robustness-benchmark through v3 `shubh`.
- Fix one convention for "2.5 lakh" vs "two point five lakh" vs "250,000" in W6.

Pair construction is in §W8d.5.

---

## W8c: Evaluation harness

### KPI → proxy map (§7)

Run all metrics on the English speech **as played on the §5.1 timeline**: harness log → rendered wav → ASR → normalised text. A failed session gets the worst value.

| KPI (§9.1 points) | Primary proxy | Secondary | Tools |
|---|---|---|---|
| Translation quality (12) | ASR-COMET with `wmt22-comet-da`, one segment per 10–60 s session | ASR-chrF++ and ASR-BLEU (the method Hibiki and Seamless use), MetricX-24; paired bootstrap; read effect sizes against Kocmi et al. (2401.06760) | [COMET](https://github.com/Unbabel/COMET) + [wmt22-comet-da](https://huggingface.co/Unbabel/wmt22-comet-da) (Apache-2.0); [sacreBLEU](https://github.com/mjpost/sacrebleu); [MetricX](https://github.com/google-research/metricx) (Apache-2.0) |
| Intent (18) | LLM judge with a binary score per utterance, following **Sarvam's rubric**. Score 0 if who does what to whom changes, the action is reversed, a statement becomes a question, yes/no flips or a number changes (S, llm_intent_entity) | Our checklist on top: asked, refused, confirmed, corrected; negation; each condition; the final value after a self-correction | Open judge (Qwen3.5, Apache-2.0) so G2's before/after scores are reproducible. Stress-test the judge with ACES / DEMETR-style perturbations (2210.15615, 2210.13746) |
| Entity (18) | Deterministic: canonical gold values (rupees as integers, digit strings, ISO dates, names) matched in the normalised output. Score = correct / gold, and **1 when there are no entities** (Sarvam's rule) | Per-type breakdown; a **forbidden-value list** for abandoned self-corrections (FAQ); LLM entity score as a cross-check | Whisper normaliser **plus** lakh = 1e5, crore = 1e7, k = 1e3, ₹ / Rs (it has none of these, S). [NeMo-text-processing](https://github.com/NVIDIA/NeMo-text-processing) English ITN parses lakh/crore but not "lakh crore" (S). Gaido et al. 2109.07439 defines entity accuracy |
| Latency (16) | EndOffset (both definitions) and LAAL/YAAL on speech output | StartOffset; sentence-group lag (2606.15059); word-aligned lag | Our own small scorer, cross-checked with [OmniSTEval](https://github.com/pe-trik/OmniSTEval) (Apache-2.0) and [Speech-to-Speech-Latency](https://github.com/SakaiXue6666/Speech-to-Speech-Latency) (Apache-2.0) |
| Audio (6) | UTMOSv2 | DNSMOS P.835; Audiobox Aesthetics; ECAPA similarity to `shubh`; **gap count and total gap time on the playback timeline** (listeners hear gaps) | [UTMOSv2](https://github.com/sarulab-speech/UTMOSv2) MIT; [DNS-Challenge](https://github.com/microsoft/DNS-Challenge) CC BY 4.0; [audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics) CC BY 4.0; [ECAPA](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) Apache-2.0 |

**Supporting facts:**
- Sarvam's own ASR-evaluation blog makes intent the **primary** score for speech translation, with `wmt22-comet-da` and entity score secondary. It also says to pin the judge version and keep a 50-sample calibration set (S, [blog](https://www.sarvam.ai/blogs/evaluating-indian-language-asr), 2026-04-02).
- **Caveat on Sarvam's rubric:** it accepts a bare "Yes." for "Yes, I will clear the loan immediately" (S, scout read of the prompt). Our checklist must catch truncation.
- Audio-aware metrics did not beat ASR plus a text metric (S, 2605.28227, ACL 2026.iwslt-1.34). ASR → text metric is fine.
- **Fix one output ASR model** for both the base and fine-tuned runs, and measure its own error rate on `shubh` audio of known text. Its errors show up as our entity errors.

### Speech-output latency recipe

Sources: Hibiki-Zero §4.4, SimulEval `latency` code, ACL 2026.iwslt-1.3.

1. **Playback timeline.** For each output frame k: `p_1 = a_1` and `p_k = max(a_k, p_{k-1} + n_{k-1}/24000)`. Here `a_k` is the arrival time, with t = 0 when Hindi frame 0 is sent, and `n_k` is the frame's sample count.
   - This is §5.1 R1 and matches SimulEval's code.
   - I: because it uses arrival times, compute time is already included.
2. **Render** one 24 kHz wav with zeros before the first frame and in every gap.
3. **Output word times** from ASR plus alignment. Pick the aligner by measured error on `shubh` audio of known text.
4. **Metrics:**
   - **StartOffset:** first English word start minus first Hindi word start.
   - **EndOffset (Hibiki):** last English word end minus last Hindi word end. **Also report** the SimulEval/Seamless version: last audio sample minus source end, where source end is `valid_samples/24000`.
   - **LAAL** (2206.05807): `γ = Δsrc / max(n_gen, n_ref)`; average `d_i − (i−1)γ` over words up to the first one at or after the source end.
   - **YAAL** (2509.17349): the same, but only over words spoken before the source ends. Always pair it with EndOffset.
   - **Sentence-group lag** (2606.15059): align output sentences to source sentences, then measure lag against the real sentence spans. It finds **lag accumulates on long input**, so test full 60 s sessions.
5. **Failed session:** worst value. **Empty output:** cap the offsets (I).

### Dev sets

- **Fully usable:** [FLEURS-hi](https://huggingface.co/datasets/google/fleurs) (CC BY 4.0) and [Indic DiarBench](https://huggingface.co/datasets/sarvamai/indic-diarbench) (CC BY 4.0).
- **Eval only:**
  - [contextual_asr_benchmark](https://huggingface.co/datasets/sarvamai/contextual_asr_benchmark): 611 Hindi clips at 16 kHz, the set Sarvam scores with llm_intent_entity. **No licence**, so internal eval only.
  - [IWSLT 2026 Indic](https://iwslt.org/2026/indic): about 40 h, Hindi→English speech-to-speech. **No licence stated**, so not usable for Part A (§10 R1); internal eval at most.
  - [CS-FLEURS](https://huggingface.co/datasets/byan/cs-fleurs): Hindi code-switched, but **NC**; internal eval only.

---

## W8d: Model literature and adaptation playbooks

### 8d.1 What the base checkpoint might be (all INFERRED)

The PS says only "streaming Hindi → English speech-to-speech model trained on synthetic data, with inference code" (§3B). Architecture, size and framework come after the mid-term (FAQ).

| Hypothesis | Evidence for | Evidence against | What it means for pairs |
|---|---|---|---|
| **H1: Kyutai Mimi multistream** (Hibiki / Hibiki-Zero lineage) | 24 kHz with 80 ms frames is exactly Mimi (S). "Synthetic data plus inference code" is Hibiki's recipe. "Add heads" and "draft model verified by the checkpoint" (FAQ) fit a token decoder. Mimi was kept unchanged when Moshi was moved to Hindi (Human-1, 2604.23295, S) and to Japanese (J-Moshi, S); Kyutai calls Mimi "fairly resilient to using different languages" | No Sarvam page or job post mentions Moshi, Mimi or Hibiki | Stereo 24 kHz: channel 0 Bulbul English, channel 1 Part A Hindi, plus an English text stream word-aligned to the English audio |
| **H2: Sarvam's own stack.** Causal streaming encoder (as in Saaras V3 realtime) → adapter → Sarvam decoder emitting text and audio tokens | Every component is described in Sarvam's own blogs. Bulbul streaming tops out at exactly 24 kHz. Requiring `shubh` targets suggests the output head was trained on that voice | Saaras takes 8/16 kHz input only. 80 ms is also the native frame rate of common conformer encoders after 8× subsampling | Hindi audio + English text + Bulbul audio, possibly interleaved rather than parallel streams |
| **H3: text "inner monologue" inside the model** | Requiring English word timings implies a time-aligned text stream | Overlaps H1 and H2; not a separate architecture | English word times needed whichever it is |
| H4: unit-based model with a unit vocoder | IWSLT 2026 Indic baseline family | Unit vocoders are usually 16 kHz | Least likely |

**Size (I):** real time on one H100 points to roughly 1–4B parameters. Hibiki is 2.7B, Hibiki-Zero 3B.

**What to prepare whichever hypothesis holds:**
- English forced alignment of the Bulbul audio.
- Hindi↔English word mapping.
- A pair builder that can write both stereo and interleaved layouts.
- A test of how well Mimi reconstructs Bulbul `shubh` and our Hindi audio, measured with the W8c metrics.

**Day-1 checklist when the checkpoint lands:**
- end-of-input and EOS conventions;
- per-stream delays;
- text tokenizer, and whether a source text stream exists;
- conditioning labels;
- which streams the loss covers;
- number of codebooks;
- LoRA and quantisation support in the loader.

### 8d.2 Kyutai family

| Item | Link | What it gives us | Licence | Maturity |
|---|---|---|---|---|
| **Hibiki** | arXiv 2502.03382 · [GH](https://github.com/kyutai-labs/hibiki) · [HF 2B](https://huggingface.co/kyutai/hibiki-2b-pytorch-bf16) | The word-level contextual-alignment recipe (§8d.5); 2.7B; real time on an H100 for up to **320 parallel streams** (160 with classifier-free guidance, CFG) (S) | Code MIT + Apache; weights CC BY 4.0. OK | ICML 2025; 1.5k★ |
| **Hibiki-Zero** | arXiv 2602.11072 · [GH](https://github.com/kyutai-labs/hibiki-zero) · [HF 3B](https://huggingface.co/kyutai/hibiki-zero-3b-pytorch-bf16) | Sentence-level coarse alignment plus GRPO. A new source language (Italian) from under 1000 h. ~580-line inference server | Code MIT; **weights CC BY-NC-SA 4.0** (card text) ⚠ | ICML 2026 oral; 274★; training code not released (issue #8) |
| Moshi + Mimi | arXiv 2410.00037 · [GH](https://github.com/kyutai-labs/moshi) · [HF mimi](https://huggingface.co/kyutai/mimi) | Mimi: 2048-entry codebooks, up to 32 levels, first level distilled from WavLM. Acoustic delay 1–2 steps. Inner Monologue: text tokens placed at `round(start·12.5)` with padding between words, about 65% of text tokens are padding (S) | MIT / Apache; weights CC BY 4.0. OK | 11.2k★ |
| **moshi-finetune** | [GH](https://github.com/kyutai-labs/moshi-finetune) | LoRA or full fine-tuning. Data is a **stereo WAV (channel 0 = the model's output, channel 1 = input) plus JSON `alignments: [[word,[start,end],speaker]]`**. Kyutai says it can be adapted to Hibiki (S, hibiki#13/#15) | Apache-2.0; **`annotate.py` imports AGPL whisper-timestamped** | 496★ |
| Delayed Streams Modeling (DSM) | arXiv 2509.08753 · [GH](https://github.com/kyutai-labs/delayed-streams-modeling) | Pre-align streams, then delay one. **Delay conditioning**: one model hits a chosen delay to within about 300 ms (S). Per H100 step: backbone ≈ 5 ms, **depth transformer ≈ 25 ms at 32 codebooks**, so the depth transformer is the per-frame bottleneck (S) | MIT / Apache. OK | 3.0k★ |
| J-Moshi | arXiv 2506.02979 · [GH](https://github.com/nu-dialogue/moshi-finetune) · [HF](https://huggingface.co/nu-dialogue/j-moshi-ext) | Moshi moved to Japanese: swapped tokenizer, Mimi frozen, full fine-tune; stereo fine-tune on 320 h, 1,423 steps, lr 2e-6 / 4e-6 (S) | Code Apache-2.0; **weights CC BY-NC** | Interspeech 2025 |
| Human-1 (Josh Talks) | arXiv 2604.23295 | **Moshi moved to Hindi**: custom Hindi tokenizer, audio components kept, 26,000 h pre-training, then a 1,000 h fine-tune (S) | Paper only; no weights found on HF | 2026-04, submitted to ICASSP 2027 |
| CALM | arXiv 2509.06926 | Swaps the depth transformer for a consistency head about 12× faster, but **needs continuous latents** (S). Doesn't transfer to discrete Mimi | — | 2026-01 |

**Do not use:**
- Hindi Moshi attempts on HF: `tharwaniajay/moshi-hindi-merged` (no licence) and `Ranjit/moshiko-kame-hinglish-ft-exp` (NC, and it runs an outside LLM plus Deepgram ASR, i.e. a cascade).
- `kyutai/moshika-rl-seamless` (NC, gated).
- `kyutai/helium-1-2b` (**CL**; it is Hibiki-Zero's text backbone).

### 8d.3 Other streaming S2ST and simultaneous speech translation (ST)

| Item | Link | What it gives us | Licence (code / weights) |
|---|---|---|---|
| SeamlessStreaming | arXiv 2312.05187 · [GH](https://github.com/facebookresearch/seamless_communication) | EMMA policy learned **from ordinary, non-simultaneous pairs**. Freezes the encoder, then the speech-to-text part. X→English ASR-BLEU 22.1, Ending Offset 2.79 s. Supports Hindi (S) | MIT / **NC** |
| StreamSpeech | arXiv 2406.03049 · [GH](https://github.com/ictnlp/StreamSpeech) | Policy from two CTC heads: **write when source-CTC sees a new token and target-CTC count exceeds what was already emitted**; random chunk size covers every latency (S) | **MIT** / weights NOLIC |
| InfiniSST | arXiv 2503.02969 · [GH](https://github.com/LeiLiLab/InfiniSST) | Pair builder from word alignments, with targets inherited by source-word time (§8d.5). EOT token as the read signal; KV cache reused for unbounded speech | **MIT** |
| FAST-CAP | arXiv 2609.30416 · [GH](https://github.com/AmirHussein96/FAST-CAP) | Full causal chunk-pair pipeline (MonotonicAlign, silence padding) plus CAAL, an alignment-aware lag metric | **NC (custom). Read, don't copy** |
| HPO | arXiv 2604.21045 · [GH](https://github.com/owaski/HPO) | GRPO for simultaneous ST. **The latency reward drops to its worst value when MetricX < −5**, so latency can't be gamed. MetricX was the only reward model that held up in both automatic and human eval (S) | Apache-2.0 |
| NaturalFlow | arXiv 2606.13121 | **LoRA r=128 on Hibiki-2B, then DPO** (β 0.1, lr 2e-6, 400 steps, batch 32) to remove disruptive pauses. Preferences scored by Whisper BLEU and Silero-VAD silence ratio (S) | No code |
| Seed LiveInterpret 2.0 | arXiv 2507.17527 | SFT on human read/write data, then **PPO with adaptive KL**; staged rewards include time compliance and lag. Chinese→English: AL 5.18 s, first-letter lag 2.71 s (S) | Closed |
| SimulS2S-LLM | arXiv 2504.15509 | Trained offline, run with **wait-k at test time**; +3 ASR-BLEU (S) | No code |
| StreamUni | arXiv 2507.07803 · [GH](https://github.com/ictnlp/StreamUni) | Transcribe-then-translate chain; drops history at punctuation or when the transcript is stable (S). Legal only inside the checkpoint | Code NOLIC / Apache-2.0 |
| CMU IWSLT 2025 | arXiv 2506.13143 | Encoder and adapter first, then **LoRA r=32, α=16, all linear layers, lr 1e-4, 1 epoch**; one latency-multiplier knob (S) | Paper |
| Long-form S2ST eval | arXiv 2606.15059 · [GH](https://github.com/SakaiXue6666/Speech-to-Speech-Latency) | Qwen3-ASR → Qwen3-ForcedAligner → sentence alignment → YAAL / xCOMET / ending offset. Nearly our W8c harness | Apache-2.0 |
| Others, ideas only | NAST-S2X 2406.06937 (code NOLIC), EASiST 2504.11809 (NOLIC), SimulMEGA 2509.01200 (NOLIC), SimulU 2603.16924 (training-free long-form policy), Stream-Omni 2506.13642 (**GPL**), TransVIP 2405.17809 (duration control; NOLIC), DuraS2ST 2609.33742 (duration-margin RL reward; Apache-2.0) | — | — |

### 8d.4 Full-duplex speech LMs (other possible base families)

Codec and frame-rate details are as stated in each report (S).

| Model | Design | Licence code / weights |
|---|---|---|
| **Qwen3-Omni** (2509.17765) | Thinker–Talker; **12.5 Hz multi-codebook, one token = 80 ms**; causal ConvNet vocoder; first audio after 234 ms | Apache / Apache |
| FLM-Audio (2509.02521) | Moshi-style on Mimi with "contiguous monologues": needs **sentence-level** transcripts only | Apache / **NC** |
| PersonaPlex (2602.06053) | Full fine-tune of Moshi weights (compute in §8d.6) | MIT / NVIDIA Open Model License, gated |
| MiniCPM-o 4.5 (2604.27393) | Full duplex on a shared time axis; vLLM-Omni serving | Apache / Apache |
| Step-Audio 2 (2507.16632), Kimi-Audio (2504.18425) | 12.5 Hz audio tokens | Apache / Apache; MIT |
| GLM-4-Voice (2412.02612) | 12.5 Hz single codebook | Apache / custom (registration required; derivative names must start "glm-4") |
| LLaMA-Omni 2 (2505.02625), Qwen2.5-Omni-3B | — | **NC** weights |
| Freeze-Omni (2411.00774), VITA-Audio (2505.03739) | — | ⚠ NC on GitHub, Apache on HF |

### 8d.5 Building training pairs ("how you arrange the English against the Hindi", §3B)

Part A already gives Hindi word times (W5) and sentence-level English (W6). Bulbul English plus our English aligner (W8b) adds English word times. So every recipe below can be built from Part A alone (I).

1. **Hibiki contextual alignment** (S, 2502.03382 §3.2):
   - For English word *j*: `a_j = argmax_i [log p(T_j | S_≤i) − log p(T_j | S_≤i−1)]`. That is the Hindi word whose arrival gives the **largest jump** in the target word's log-probability, scored by an MT model (MADLAD-3B in the paper).
   - Spikes over 25% of the mean delay in a 5-word window are smoothed.
   - Then **silence is inserted** so each English word starts ≥ 2 s after Hindi word `a_j`.
   - Ablation (long-form): this gives ASR-BLEU 26.6 at LAAL 5.0 s, against 25.6 at 21.5 s for sentence alignment and 4.2 for no lag.
   - I: swap MADLAD for IndicTrans2 (MIT). Hindi's late verbs and negation then automatically hold the English back until the evidence arrives, which serves the intent KPI.
2. **Hibiki-Zero coarse alignment** (S, 2602.11072 §3.2.1):
   - English sentence *i* starts δ_i ~ U(0, 0.5·d_i) after Hindi sentence *i* starts, where d_i is the Hindi sentence's duration.
   - U(0, 2 s) silences are added at punctuation.
   - **Ablation B:** a base trained with full-sentence delay could not be taught by RL to start earlier. Supervised pairs need some within-sentence overlap.
3. **InfiniSST chunking** (S, 2503.02969):
   - Monotonic word alignment (SimAlign + LaBSE); each target token inherits its source word's time.
   - 960 ms chunks; chunks randomly merged by *m* to give several latencies from one alignment.
   - 28.8 s multi-sentence segments with random silences.
4. **FAST-CAP MonotonicAlign** (S, 2609.30416; read only, the code is NC):
   - Running maximum over source indices.
   - Chunk boundaries at alignment anchors.
   - **When the English chunk is shorter than the next Hindi segment, pad with Hamming-smoothed silence.**
5. **Sentence-level only** (S, FLM-Audio 2509.02521): text leads audio by about 2 tokens, then `<wait>`. The authors argue word-level alignment errors propagate.
6. **Bulbul-specific (I):**
   - Synthesise per sentence or clause so silences can be placed.
   - Use `pace` (declared per pair, §3B) where Hibiki used a padding-penalty speed-up.
   - Best-of-N by ASR WER is allowed for data, never at inference.
   - **Watch the English/Hindi duration ratio**: if English runs longer, lag piles up (Seed, DuraS2ST).

### 8d.6 Fine-tuning recipes and compute anchors

| Source | Data | Method | Steps / LR | Compute |
|---|---|---|---|---|
| **Hibiki-Zero final stage** (S) | **< 200 h** | Fine-tune, then distil into a lighter copy | **1K steps, batch 16, lr 1e-6** | 48 H100s across the whole project |
| Hibiki fine-tune (S) | ~900 h | Fine-tune | 8K steps, batch 8, lr 2e-6 | — |
| Hibiki-Zero Italian (S) | ~850 h | Fine-tune, then RL | ASR-BLEU 14.3 → 31.0 → 31.9; End Offset 3.9 → 3.7 → 3.0 s | — |
| **moshi-finetune default** (S) | 100 s windows | LoRA on every linear layer, rank 128, scaling 2; loss **on the model's own stream only**; first-codebook loss weight ×100 | 2,000 steps, batch 16, lr 2e-6 | **1×H100: ~12k tokens/s, 39.6 GB peak**. I: about 8 h per run |
| J-Moshi (S) | 320 h stereo | Full fine-tune | 1,423 steps | 2 h on 16 V100s |
| PersonaPlex (S) | ~2,250 h synthetic | Full fine-tune of Moshi | 24,576 steps × batch 32 | 6 h on 8×A100 |
| CMU IWSLT 2025 (S) | > 3,000 h | LoRA r=32, α=16 | 1 epoch, lr 1e-4 | — |

**Budget (I):** Modal $1,000 at about $3.95 per H100-hour (W8 still to verify) is roughly 250 H100-hours. At ~8 H100-hours per moshi-finetune-sized run, that is on the order of 30 runs.

**RL** (all S):
- **Hibiki-Zero GRPO:**
  - batch 32, group of 4, lr 2e-7, 2000 updates, 120 s rollouts;
  - reward = 0.6·BLEU(prefix) + 0.4·BLEU(full), every 8 source words;
  - clip 0.2, no KL term.
  - Effect: End Offset 3.5 → 2.3 s at about the same ASR-BLEU, so **RL mostly buys latency**.
- **HPO:** latency reward gated on quality.
- **Seed:** reward on lag alone gets gamed; use adaptive KL.
- The FAQ allows outside reward models and teachers during training. Whether on-policy rollouts are allowed when their audio isn't Bulbul is **A** (organiser question Q1).

**Forgetting** (S, 2505.17496): experience replay beats model merging and LoRA scaling. But training data must come only from Part A (§3B), so replaying the base model's own synthetic data is out (I). Use adapters plus low LR instead.

### 8d.7 Speculative decoding and multi-token prediction

FAQ: "A draft model may only propose tokens that the checkpoint verifies, as in speculative decoding."

| Method | Link | Speed-up | Exact? |
|---|---|---|---|
| Speculative sampling | arXiv 2211.17192, 2302.01318 | — | **Exact** (rejection sampling) |
| **RVQ position-aware speculative decoding** | arXiv 2609.37007 | 2–2.2× on the loop over 15 codebooks per **80 ms frame** (on-device TTS) | **Lossless under top-k** (S) |
| EAGLE-style for ASR | arXiv 2610.13506 | 1.9–2.4× | Lossless |
| VADUSA | arXiv 2410.21951 | 2.4× (2.9× with its "tolerance" rule) | **Not exact** |
| MTP + Viterbi (codec TTS) | arXiv 2410.13839 | 4–5× per token | Not exact |
| SSD / PCG | arXiv 2505.15380 / 2511.13732 | 1.4× / — | Not exact / exact only within groups of similar tokens |
| VocalNet(-M2), MTP-S2UT | arXiv 2504.04060, 2511.10232, 2510.10003 · [GH](https://github.com/SJTU-OmniAgent/VocalNet) (Apache-2.0) | Multi-token-prediction heads trained into speech LMs | — |

I: on a Mimi-style model the per-frame cost is in the **depth transformer** (DSM: ~25 ms of ~30 ms per step). That is where an exact, self-drafting head (FAQ: "add layers or heads") pays off.
- **No verified speculative-decoding work exists for Moshi/Mimi** (S, scout search).
- Using a *different* translation model as the drafter is **A**: it is NC-licensed anyway and arguably "alongside" the checkpoint (§3B).

---

## Past competitions and shared tasks

| Event | Link | What to take |
|---|---|---|
| IWSLT 2026 Indic S2S (Hindi↔English) | [iwslt.org/2026/indic](https://iwslt.org/2026/indic) | About 40 h per language, n-way parallel; baseline Hindi→English BLEU 26.08; scored by sacreBLEU, TER and BLASER, with no latency metric (S). Findings: no team achieved significant performance. **Data licence not stated → not for Part A** |
| IWSLT 2026 Simultaneous | ACL 2026.iwslt-1.39 · [OmniSTEval](https://github.com/pe-trik/OmniSTEval) | **LongYAAL** ranking, **XCOMET-XL** quality, Docker on **one H100 80 GB**: the closest analogue to our §6 setup |
| IWSLT 2025 Simultaneous / Indic | ACL 2025.iwslt-1.44 · arXiv 2506.17077 (CUNI) · 2507.19616 (HITSZ) | StreamLAAL ranking. CUNI's SimulStreaming won the simultaneous track. The Indic track (653.88 h of Hindi→English speech-to-text) was won mostly by cascades, which are forbidden for us |
| IWSLT 2023/2024 speech-to-speech subtracks | ACL 2023.iwslt-1.1 · arXiv 2411.05088 | ASR-BLEU (Whisper) plus BLASER; start and end offsets reported. **Different latency metrics ranked the systems differently**, so report several |
| Interspeech 2026 TOPI | [cs.utep.edu/topi](https://www.cs.utep.edu/topi/) | "Transfer of Pragmatic Intent" in S2ST (English↔Spanish), human-rated 1–5. Ideas for our intent rubric |
| MUCS 2021 | arXiv 2104.00235 | Code-switching ASR baselines (data is CL, see W1) |
| Inter IIT 9.0–14.0 | [compilation](https://criticic.github.io/InterIIT-TechMeet-Problem-Statements) | **No Sarvam or speech problem statement; no team repos** |

---

## Top 10 to steal from (ranked)

1. **Hibiki-Zero** (arXiv 2602.11072 + [hibiki-zero](https://github.com/kyutai-labs/hibiki-zero)). **W8d, W8a.**
   - Write Playbook A as "our fine-tune = Hibiki-Zero's last stage": under 200 h, ~1K steps, batch 16, lr ~1e-6, then optional GRPO.
   - Port its end-of-input handling: `valid_samples` → EOS at frame `ceil(valid_samples/1920)`, keep stepping until text EOS, then `session_complete`.
   - Port the first-frame drop into the W8a dummy model now.
2. **Hibiki contextual alignment** (2502.03382 §3.2). **W6, W8b.**
   - Before the mid-term, compute per-English-word earliest-time maps from our Hindi word times, scoring log-probability jumps with IndicTrans2 (MIT).
   - Store them as an internal (non-released) field.
   - This decides whether W6's sentence boundaries and word order are streamable, and gives Part B a ready pair builder.
3. **moshi-finetune data format and LoRA defaults** ([repo](https://github.com/kyutai-labs/moshi-finetune)). **W8b, W8d.**
   - Make the W8b pair builder emit stereo 24 kHz WAVs (channel 0 Bulbul, channel 1 Hindi) plus `alignments` JSON.
   - Replace `annotate.py`'s AGPL whisper-timestamped with our aligner.
   - Dry-run LoRA on public Moshi weights so the training loop exists before the checkpoint does.
4. **Sarvam's intent/entity rubric** ([llm_intent_entity](https://github.com/sarvamai/llm_intent_entity) + [eval blog](https://www.sarvam.ai/blogs/evaluating-indian-language-asr)). **W8c, W2.**
   - Reimplement it with our own prompts and an open judge: binary intent, entity fraction with 1 when there are none.
   - Add conditions, self-corrections, lakh/crore and a truncation check.
   - Calibrate on 50 hand-labelled items. W2 should weight its coverage matrix towards what that rubric zeroes.
5. **Long-form speech-to-speech latency evaluation** ([Speech-to-Speech-Latency](https://github.com/SakaiXue6666/Speech-to-Speech-Latency) + [OmniSTEval](https://github.com/pe-trik/OmniSTEval), both Apache-2.0). **W8c.**
   - Build our scorer on the §5.1 playback timeline: StartOffset, both EndOffsets, LAAL/YAAL and sentence-group lag.
   - Cross-check against OmniSTEval. Never use SimulEval code (CL).
6. **MFA** (English pretrained model + a Hindi model trained on our audio; FA-Bench, 2607.25581). **W5, W8b.**
   - Add "MFA trained on pilot code-mixed audio" to the W5 bake-off.
   - Use `english_mfa` v3.1 for Bulbul English, with Qwen3-ForcedAligner as runner-up.
   - Expect MFA to win on boundary error and the CTC aligners on confidence signals.
7. **NeMo-text-processing + indic-numtowords**. **W2, W4, W6, W8c.**
   - Verbalise numbers with lakh/crore and spoken variants (डेढ़, ढाई, पौने) for scripts and transcripts.
   - Use NeMo English inverse normalisation, extended by a few lines for "k", ₹ and "lakh crore", as the entity normaliser.
8. **IndicCMix** (MIT). **W2, W4.**
   - Mine its Devanagari-transliterated English words into a loanword lexicon ("क्रेडिट कार्ड" ↔ "credit card"), used both to convert transcripts and to check English words are restored correctly in W6.
   - Use rows as few-shot seeds for script generation.
9. **IndicConformer Hindi `.nemo` + NeMo Forced Aligner** (MIT / Apache-2.0). **W4, W5, W3.**
   - One clean-licence stack gives ASR text, word times and CTC confidence.
   - Use it as the first-pass transcriber and as the TTS-QA round-trip ASR (a different family from Bulbul and Parler).
   - Benchmark Indic-Transcribe-core (CL) against it before relying on it.
10. **Kyutai serving loop + DSM marker protocol** (moshi `server.py`, the `moshi-server` speech-to-text client). **W8a.**
    - Mirror the encode → step → decode loop with one 80 ms frame per step.
    - Model `input_end` on the DSM `Marker` flush.
    - Avoid both serving bugs (headline 7). Warm up CUDA graphs before `/health` returns 200.

**Runners-up:**
- InfiniSST (MIT) chunked pair builder.
- NaturalFlow-style DPO and HPO quality-gated latency reward, for an optional Part B post-training stage.
- 2609.37007 lossless codebook-level speculative decoding.
- Sarvam tts-robustness-benchmark prompts, to probe Bulbul on numbers.

---

## Traps

**T1. Licences we could not assign (§10 R5) or that bind outputs**
- **AGPL / copyleft code:**
  - whisper-timestamped (inside moshi-finetune `annotate.py`; Hibiki's tool);
  - SimulEval (CC BY-SA, archived), GEMBA (CC BY-SA), UniSpeech (CC BY-SA);
  - Stream-Omni (GPL), espeak-ng via Kokoro's Hindi path, the Stanford Parser via GCM (GPL);
  - num2words (LGPL).
- **NC weights:**
  - the Seamless family (also IndicSeamless and NLLB), Hibiki-Zero, J-Moshi, FLM-Audio, LLaMA-Omni 2;
  - F5-TTS, **XTTS-v2 (outputs too)**;
  - MMS_FA and ctc-forced-aligner's default model;
  - XCOMET, CometKiwi, BLASER 2.0, SONAR's Hindi speech encoder, NISQA weights.
- **NC metric dependency:** mwerSegmenter, and therefore StreamLAAL.
- **GPL weights:** Sarvam-Translate.
- **NOLIC:**
  - sarvamai `llm_intent_entity` and `llm_wer`; the `sarvamai` SDK on PyPI (scout B: no licence declared);
  - IndicF5 code, WhisperX's Hindi alignment model;
  - StreamUni, NAST-S2X, EASiST, SimulMEGA and TransVIP code;
  - FAST-CAP (custom NC).
- **BASE terms:** Gemma 3 (training on its outputs makes a derivative), Llama 3.x (naming). For anything that trains a model, prefer Apache LLMs: Qwen3.5, Gemma 4, Sarvam-30B/105B.
- **Mislabelled data:** Lahaja (card vs tag), a re-tagged SLR104 mirror, a vendor call-centre sample.

**T2. Cascade (§3B).** All of these are forbidden in the submitted system:
- Kyutai unmute, moshi-rag and invincible-voice;
- the Hinglish KAME experiment;
- SimulStreaming's LLM cascade, and every IWSLT Indic system;
- any outside ASR used to choose, rerank or time output, e.g. best-of-N at inference;
- Kyutai's semantic VAD (it produces text).

A transcribe-first text pass (StreamUni, StreamSpeech) is legal only **inside** the checkpoint. Using another speech-translation model as a speculative drafter is **A**.

**T3. NC models used to *build* Part A.**
- §10 R1 says "any open-weight model may be used to build the Part A dataset", but licence compliance is still ours.
- MMS_FA, NLLB and Seamless are CC BY-NC, and the dataset goes to a company.
- **A:** safest is to use them only for internal benchmarking and to build released records with MIT/Apache tools. Ask (Q3).

**T4. Bulbul defaults.**
- Always pass `model="bulbul:v3"`, `speaker="shubh"`, `language_code="en-IN"` (the REST name, matching §3B) and `speech_sample_rate=24000`.
- REST `output_audio_codec` defaults to wav; the WebSocket defaults to mp3, so set the codec explicitly there.
- Check the returned header.
- Don't touch `pitch` or `loudness` (⚠ conflicting docs; not freed by §3B).

**T5. Saaras.** No mode meets §3A (digits and Latin script appear, even a Latin "zero" in verbatim mode). It gives no word timestamps, takes at most 30 s over REST and 16 kHz PCM only.

**T6. Whisper normaliser.** It has no lakh, crore or "k" handling, which would mis-score "25 lakh". Its `BasicTextNormalizer` strips Devanagari vowel signs, so **never use it on Hindi**.

**T7. Scorer integrity (§10 R8).**
- Interpretation "drop" and "summarise" actions, filler, and keep-alive silence frames are all out.
- Silence frames also push all later speech back on the playback timeline.
- Send only real speech; gaps cost nothing extra.

**T8. EndOffset has two definitions** (last word vs last sample; source end = last Hindi word vs `valid_samples`). They can differ by seconds. Compute both until the organisers say which they use (Q5).

**T9. Platform.**
- torchaudio 2.9+ removed audio I/O and `apply_codec`; `forced_align` remains.
- `audioop` is gone in Python 3.13+ (we run 3.14), so simulate telephone codecs with ffmpeg (G.711, AMR-NB, GSM are available).
- fairseq and stable-ts are archived. IndicConformer `.nemo` needs the AI4Bharat NeMo fork.

**T10. Doesn't transfer.**
- Voice-transfer CFG (our target voice is fixed).
- Hibiki's alignment-aware TTS (Bulbul can't be word-constrained).
- CALM's speed-up (needs continuous latents).
- Using Kyutai or other TTS for targets (§3B: Bulbul only).
- Replay of the base model's synthetic data (Part A only).
- Lossy "tolerance" speculative decoding (I: not "verified").

---

## Questions for the organisers (new, from this research)

1. **RL and preference stages.** Hibiki-Zero (GRPO), NaturalFlow (DPO) and HPO post-train on the model's own rollouts, scored by reward models, and the FAQ allows reward models. Is an on-policy stage allowed when the rollout English audio is not Bulbul, given that only "supervised training pairs" must be Bulbul (§3B)?
2. **Third-party code.** §10 R5: may we ship permissively licensed third-party code (Kyutai MIT/Apache, NeMo Apache) as dependencies or vendored files with their notices, given that Sarvam claims IP in submitted code?
3. **NC models.** Under §10 R1 "any open-weight model", may non-commercial weights (MMS_FA, NLLB, Seamless) be used as tools to build the Part A dataset?
4. **Speculative decoding.** Must acceptance be exact (rejection sampling or greedy match), or is tolerance-based acceptance "verified" enough? May the draft be trained independently, or must it come from the checkpoint?
5. **Latency KPI.** Which end and start offsets (last word or last sample; `valid_samples` or last source word), and which output ASR or aligner, does the harness use?
6. **Bulbul.** Are `pitch` and `loudness` fixed at defaults? (§3B frees only pace and temperature; the docs disagree on whether v3 accepts them.)

---

## Status — 2026-10-14

**Done**
- This file: prior art organised by workstream, Top 10, traps, and six new organiser questions.
- Five parallel scouts: Kyutai family; non-Kyutai streaming S2ST; Sarvam plus past competitions; Indic data tooling; evaluation, serving and IWSLT.
- I re-verified every citation centrally: 136 arXiv IDs (all resolve, titles match), 105 GitHub repos (licence and activity), 84 HF models and datasets (licence and gating).
- I re-checked by hand the licence of Hibiki-Zero's weights (the scouts disagreed; the card text says CC BY-NC-SA 4.0), the Bulbul v4-flash / `shubh` trap, Starlette `receive_bytes`, the moshi-finetune memory and throughput figures, and the Human-1 abstract.

**Measured:** nothing. This was a read-only literature pass within a 1 GB budget (about 10 MB used). Every number above is the source's own claim, labelled S. The team still has to measure: aligner error on the W5 gold set, Mimi reconstruction of Bulbul and Hindi audio, Bulbul on lakh/crore, and ASR disfluency retention.

**Blockers:** none.

**Handoffs**

| Stream | Hand-off |
|---|---|
| W0 | Reduplication hyphen (चाय-वाय) and the `[OTP]` bracket ambiguity in IndicVoices L1 (§W4) |
| W1 | Lahaja licence ⚠; IndicVoices-R overlaps IndicVoices hours; Indic DiarBench as real Hindi dev audio; reject the SLR104 re-upload |
| W2 | IndicCMix, COMI-LINGUA, indic-numtowords spoken variants, Sarvam tts-robustness categories; avoid Gemma 3 / Llama for generation |
| W3 | Veena, Indic Parler, Indic-Speak (consented voices); XTTS and F5 are NC; no cloning of corpus speakers; SpeechArena ranks Bulbul v3 far above IndicF5 |
| W4 | Indic-Transcribe-core vs IndicConformer + NeMo Forced Aligner bake-off; NeMo Hindi normalisation; never score Hindi with Whisper's normaliser; disfluency retention unmeasured anywhere |
| W5 | Add trained MFA (code-mixed: 4.15 ms vs 38 ms); MMS_FA is NC (T3); FA-Bench numbers |
| W6 | Contextual-alignment maps (Top-10 #2); teacher style becomes model style; sentence-cut yes, drop/summarise no; Sarvam-Translate is GPL |
| W7 | Bulbul has no seed: archive everything. **Open point for the master session:** `research/` and `tasks/` are not §9.2 deliverables. Decide whether the submission commit carries them ("any other file" penalty) |
| W8a | Headline 7 bugs; DSM `Marker` pattern; `--static none`; measured-vs-theoretical latency gaps |
| W8b | Bulbul facts and T4; aligner choice (`english_mfa`, then Qwen3-ForcedAligner); stereo + `alignments` JSON pair format; duration ratio |
| W8c | KPI map, latency recipe, Sarvam rubric, dev sets and their licences |
| W8d | §8d.1 hypotheses and day-1 checklist; §8d.5–8d.7 recipes; the Q1 and Q4 answers gate the RL and speculative-decoding plans |
