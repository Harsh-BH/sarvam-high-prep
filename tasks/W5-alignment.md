# W5: Hindi forced alignment

**Goal:** accurate start and end times for every word, on both real and synthetic audio. These timings are audited at the mid-term (§3A). [Inference] They are probably also how training pairs place English against Hindi.

## PS rules

- Every transcript word gets times, including English words written in Devanagari, and fillers (§3A).
- Times are in seconds from the start of the **record**, not the source file (§5.2); subtract `source.start`.
- Ordered, non-negative, inside the clip (§5.2). We enforce `0 ≤ s_i < e_i ≤ s_{i+1}`, with the last end ≤ duration.
- The output must be reproducible (§3A), so run the aligner deterministically (on CPU) **and** archive its raw output.

## Research and measure

- **Gold set:** 20–30 min of hand-corrected word boundaries (Praat or Audacity labels). Cover wideband, telephone, synthetic, code-mixed, filler-heavy and 50–60 s clips.
- **Aligners to benchmark:**
  - torchaudio MMS-FA with uroman romanisation;
  - IndicWav2Vec CTC + CTC segmentation;
  - NeMo Forced Aligner with IndicConformer;
  - Montreal Forced Aligner (pretrained Hindi, or trained on our data).
- **Metrics:**
  - mean absolute boundary error;
  - % of boundaries within 50 ms and 100 ms;
  - failure rate on fillers, English-in-Devanagari, noise and long clips;
  - CPU runtime per audio hour.
- **Confidence scores** as a QA signal: low-confidence records go to review or get dropped. The same signal catches TTS skips and repeats for W3.
- **Post-processing** that guarantees ordered, non-overlapping times inside the clip without distorting good boundaries.

## Done when

An aligner module, a gold-set benchmark report, QA thresholds, and pilot timings that pass the validator.
