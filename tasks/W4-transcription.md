# W4: Transcription of real audio

**Goal:** verbatim, spec-compliant Hindi transcripts for every real clip. Transcripts are a sampled item in the mid-term audit (12 pts, §7), and a format error fails the dataset gate (§8).

## PS rules (see W0 `SPEC.md` for the exact conventions)

- IndicVoices Level 1 ("as spoken"), disfluencies, repetitions and fillers included (§3A). In detail:
  - English words in Devanagari **without** bracketed Latin.
  - Numbers as spoken words.
  - Every sentence ends with sentence-final punctuation.
- Manual correction is allowed (§3A). Store corrections as archived patches so the rebuild stays reproducible.
- Real IndicVoices L1 transcripts add bracketed English (ओटीपी [OTP]) and use bracketed Latin filler tags ([umm]). **Both must be converted.**

## Research and measure

- **ASR benchmark** on 1–2 h of hand-checked pilot audio from W1. Candidates: Sarvam Saaras (₹30/h), IndicConformer, Whisper-large-v3 and its Hindi fine-tunes, MMS. Score on:
  - WER/CER;
  - does it write English words in Devanagari?
  - numbers;
  - **does it keep disfluencies, or clean them up?**
  - telephone vs wideband audio.
- **Reuse vs re-transcribe:** are corpus transcripts accurate enough to convert, or do we re-run ASR and merge?
- **Conversion pipeline:**
  - Strip bracketed Latin; map tags to Devanagari fillers or drop them.
  - Numbers → words exactly as spoken.
  - English words → Devanagari (IndicXlit or an LLM).
  - Sentence segmentation and punctuation.
- **Human correction:**
  - Tool: Label Studio or similar.
  - Which records get reviewed: entity-dense and low-confidence ones.
  - Measure throughput per annotator-hour, and run an agreement spot check.

## Done when

The pilot transcripts pass the validator, with measured error rates per source and a correction protocol written down.
