# W3: Hindi TTS and TTS QA

**Goal:** turn W2 scripts into clean synthetic Hindi audio with as many voices as possible, and catch every bad render before it reaches the dataset.

## PS rules

- 24 kHz mono 16-bit WAV; one speaker per clip; ≤60 s (§3A).
- **Hours count per generated file**, including the same text in another voice (§3A). Voice diversity is free when the texts differ, but script duplication costs full hours. One generated file = one clip.
- TTS output must be checked, and bad output filtered or regenerated: mispronounced names and numbers, skipped or repeated words, artefacts (§3A).
- Synthetic records carry `source.model` and `source.voice`; `source.corpus` = where the text came from (§5.2).
- Archive every kept output. The rebuild must never re-synthesise (§3A reproducibility).

## Engine candidates

- **Bulbul v3 hi-IN** (Sarvam):
  - ~37 voices; ₹30 per 10k characters (beta pricing); 2,500 characters per request; no seed.
  - **The default sample rate is 22,050 Hz, so always set `speech_sample_rate=24000`.**
  - Outputs can't be published (§10 R3).
  - Docs: docs.sarvam.ai.
- **Open-weight:** IndicF5, Indic Parler-TTS, others. Check licence, number of voices, and speed on an H100. Zero-shot voice cloning of real speakers raises a **consent** risk.

## Research and measure

- On 200 pilot scripts per engine, measure:
  - round-trip CER;
  - number and name accuracy;
  - code-mix pronunciation;
  - artefact rate;
  - how naturally fillers and self-corrections sound.
- **TTS input vs transcript:** do English words read better from Devanagari or Latin input? The transcript must be Devanagari and must match what was *actually* spoken.
- Prosody of digit strings and fractional numbers.
- Cost per hour of audio, including the regeneration rate.

## QA pipeline

- Round-trip ASR using a **different model family** from the TTS (otherwise their errors correlate).
- Entity match on numbers and names.
- Duration-per-character outliers.
- Skip/repeat detection via W5 alignment.
- Clipping, long-silence and artefact detectors.
- A regeneration loop. Rejected takes are never referenced.
- Tune the thresholds on a hand-labelled good/bad set.

## Done when

A TTS-choice memo with measured error rates, the QA pipeline with tuned thresholds, and the pilot audio handed to W5 and W6.
