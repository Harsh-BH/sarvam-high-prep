# W1: Real-audio sourcing

**Goal:** know exactly how much usable, redistributable real Hindi audio exists, then pick and cut the real hours. The hidden test is **real** recordings (§6) and the base model is trained on **synthetic** data (§3B), so real audio is where fine-tuning gains most.

## PS rules

- Real recordings only from datasets whose licence allows redistribution; licence and consent are our responsibility (§3A, §10 R1).
- One speaker per clip; ≤60 s; each record is a contiguous `[start, end]` of one source file (§5.2).
- Overlapping cuts of the same recording count once toward hours (§3A). That means alternative segmentations are hour-free, but they must exist **before** the mid-term.
- The test set includes telephone audio (upsampled) and noise (§4). Real 8 kHz recordings upsampled to 24 kHz count as real records, not augmentation.

## Candidate corpora (licences checked 2026-10-13)

| Corpus | Licence | Notes |
|---|---|---|
| IndicVoices | CC BY 4.0 | 76% extempore, 15% conversational. **The conversational part is 8 kHz telephone.** Is the HF release L1 or L2? |
| SPRING-INX Hindi | CC BY 4.0 | 351 h; monologue + conversational; ~10% code-mixed; 8 kHz upsampled |
| Vaani (ARTPARK/IISc) | CC BY 4.0 | Spontaneous, image-prompted; only partly transcribed |
| Kathbath | CC0 / CC BY 4.0 | Read speech, with a noisy subset |
| MUCS Hindi (SLR103) | MS O-UDA | Read stories, 8 kHz |
| **Avoid** | | Gram Vaani (academic-only), Common Voice (no re-hosting), IndicTTS (custom licence), MUCS Hi-En SLR104 (CC BY-SA makes the whole dataset ShareAlike), Shrutilipi (rights chain unclear) |

## Research and measure, per corpus

- Hindi hours, speaker count, gender and region spread.
- **Segment-length distribution.** Are there long-form source files we can cut 10–60 s session-like clips from, or only short utterances?
- Telephone share and original sample rates.
- Code-mix rate: share of English words.
- Entity density: numbers, names, amounts per minute.
- Transcript level (L1 or L2), tag format, and quality on a 30-clip spot check.
- Conversational audio: is each side on its own channel, or mixed? If mixed, does diarisation (pyannote; check its licence and gating) give clean single-speaker cuts?
- Attribution text for `data_card.md`.

## Optional: own recordings

Only if the organisers allow it (PS analysis, question 1): a consent form, a phone-channel recording protocol, and scripts from W2.

## Done when

- An inventory table with the numbers above.
- A 2–5 h pilot pool of single-speaker 10–60 s clips with their existing transcripts, handed to W4.
