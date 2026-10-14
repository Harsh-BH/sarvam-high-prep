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

## Findings, 2026-10-14 (W5)

Tags: **S** = stated in the PS or a source; **I** = our inference; **A** = ambiguous. Measured numbers are ours.

### Decision

| | |
|---|---|
| Stage | `align` = `w5/align.py`, one line in `pipeline/run.py` STAGES. Stage contract, resumable, `validate.py --partial` 0 violations on both pilot runs (below) |
| Model | `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` @ `e2568c3f` (MIT): wav2vec2-base, Devanagari-character CTC, 20 ms frames. All characters of our 46.7k transcript words are in its vocabulary |
| Alignment | torchaudio `forced_align` (BSD-2) on emissions from 10 s windows (8 s kept, 1 s context each side, stitched frame-exactly; W3 found this model degrades on long inputs). Target = transcript characters with the model's `|` delimiter between words, so every word owns ≥ 1 frame and words never overlap |
| Post-processing | `fill_gaps` (below), then `finalize`: 3 decimals, last end clamped to `duration`, `0 ≤ s < e ≤ next s` asserted |
| Determinism | CPU, 8 threads, `MKL_CBWR=AUTO,STRICT`, oneDNN off. With oneDNN on, two processes differed in the 4th decimal of 2 of ~4,000 archived posteriors (timings identical); with it off, three runs were byte-identical (records, rejects, raw) |
| Archive | `raw/<id>.json` in the run folder (uploaded with the run): the Viterbi token spans (frames + posteriors), the greedy CTC hypothesis, the filled word frames, model + revision. Records carry `_archive.align = "raw/<id>.json"` (relative to the run) and `_qa.align = {regret, flags}`. The rebuild copies timings; it never re-aligns |

**Not chosen:**
- **IndicConformer 600M CTC** (MIT; the HF ONNX wrapper's CTC head, i.e. what NeMo Forced Aligner would align with): 80 ms frames; MAE 93 ms on the control, 35% of real segment edges spill > 100 ms. Matches FA-Bench's 90 ms for an 80 ms NFA model (research, S). Run on SageMaker (2.6 GB weights > W5's disk budget). Tokenised with greedy longest-match over its Hindi pieces, because the repo ships no SentencePiece model (I: a small handicap, not the 80 ms floor).
- **MMS-FA** (torchaudio, weights CC BY-NC, research T3): benchmark only. 24 ms raw MAE on the control, between our raw CTC (28) and our final timing (10).
- **MFA:** not run. (1) No pretrained Hindi acoustic model (research, S); training needs a conda Kaldi env (CPU build: 401 MB download, ~1.5 GB installed) while the shared disk is at 5.7 GB free. (2) The stage already measures 10 ms MAE on the control, below that control's own truth ambiguity (±16–50 ms), so an MFA gain could not be measured there, and real speech has no truth. Revisit only if Harsh's boundary check (below) finds systematic errors.
- **IndicWav2Vec-Hindi:** gated; our token gets HTTP 403 (needs Harsh's "Agree" click). Vakyansh is the same model class (Hindi wav2vec2, character CTC), so it adds little.

### How the benchmark gets its signal without a hand-labelled gold set

A 20–30 min hand-corrected gold set is hours of Praat work; the human budget is 30 min. So most signal comes from:

1. **Known-truth control set** (`data/w5/ctrl/`; S3 `stages/w5_ctrl_nc/20261014T093344Z-02f59e2`). MMS-TTS Hindi (`facebook/mms-tts-hin` @ `1d83b22`, VITS; **CC BY-NC: benchmark only, never a dataset input**) reads our own transcripts: the 35 most filler-heavy and 35 most English-heavy IndicVoices clips, 30 random ones, and all 50 W3 pilot texts; 9 real texts were dropped (render over 58 s, or a word with no character in the TTS vocabulary, e.g. ङ). Plus 12 clips joined to 45–58 s. 153 clips, 61 min, 9,872 words (256 fillers, 1,486 English-in-Devanagari). VITS predicts an integer duration (16 ms) for every character token, so every word's span is known. Two corrections, both checked on the audio:
   - VITS hands pause frames to the token next to the pause: a phrase-final है "lasts" 220 ms into digital silence (rec-950003, 3.41–3.63 s); a clause-initial तो "starts" 240 ms before its audio. Truth edges quieter than the clip floor + 15 dB are trimmed (SPEC §3: pauses belong to no word).
   - Between words VITS spends blank + space + blank tokens (median 96 ms). Where that gap has no silent frame (a fluent junction) the boundary may sit anywhere in it; elsewhere the truth is the character edge, widened by the adjacent blank token.
   - I: one synthetic voice, clean audio. It measures the aligner's mechanics (peaky CTC, long vowels, fillers, long clips, channel), not real-speaker variety.
2. **IndicVoices human segment edges** (642 real clips). S (SPEC §2.1): every IV clip carries the corpus's segment `start`/`end` (`_w1.segments`). W4's converter, run per segment, reproduces all 642 transcripts word for word, giving 1,165 segments with known first and last words. A word that starts before or ends after its segment, or sits inside a gap between segments, is an error. I: the edges include annotator padding, so this catches spill-over, not small errors.
3. **Perturbation consistency** on 150 real and 80 control clips: 0.37 s of silence prepended (every time must move by exactly 0.37 s), telephone (8 kHz, 300 Hz high-pass, 8-bit μ-law, back to 24 kHz), white noise at 10 dB SNR.
4. **Agreement between independent aligners** (MMS-FA, IndicConformer) on real and TTS speech.
5. **Corruption controls** for the QA thresholds: `cut` = one word plus the pauses around it removed from the audio, so the transcript holds an unspoken word, like a TTS skip (200 real, 50 TTS clips); `cutf` = the same for a filler (210 real, 10 TTS); `dup` = one word's audio repeated (speech the transcript lacks); 20 swapped transcripts.
6. **One optional human list** (Status, below).

### Results

Tables from `uv run w5/bench.py score` (raw copy: `data/w5/bench/score.md`). "w2v+fill (stage)" is exactly what the `align` stage writes; "w2v raw CTC" is the same Viterbi path without gap filling. MMS-FA real-speech rows cover the first 80 real clips; IndicConformer covers all.

#### Control set (known truth, junction-tolerant): boundary error

| aligner / condition / words | MAE ms | ≤50 ms % | ≤100 ms % | >200 ms % |
|---|---|---|---|---|
| w2v+fill (stage) / clean / all | 10.1 | 92.8 | 97.4 | 0.67 |
| w2v+fill (stage) / clean / hindi | 9.9 | 93.0 | 97.7 | 0.66 |
| w2v+fill (stage) / clean / english | 8.4 | 93.7 | 97.0 | 0.37 |
| w2v+fill (stage) / clean / filler | 25.2 | 84.4 | 92.0 | 2.54 |
| w2v+fill (stage) / clean / long | 13.2 | 91.5 | 96.3 | 1.70 |
| w2v+fill (stage) / tel / all | 11.7 | 92.1 | 97.2 | 0.96 |
| w2v+fill (stage) / noise / all | 24.0 | 84.3 | 94.4 | 1.34 |
| w2v+fill (stage) / shift / all | 11.0 | 92.2 | 97.0 | 0.68 |
| w2v raw CTC / clean / all | 28.0 | 81.3 | 94.1 | 1.36 |
| w2v raw CTC / clean / hindi | 28.5 | 81.2 | 93.6 | 1.48 |
| w2v raw CTC / clean / english | 22.4 | 85.4 | 97.6 | 0.54 |
| w2v raw CTC / clean / filler | 44.6 | 61.1 | 88.3 | 2.15 |
| w2v raw CTC / clean / long | 27.7 | 81.8 | 94.6 | 1.39 |
| w2v raw CTC / tel / all | 28.3 | 81.3 | 93.7 | 1.36 |
| w2v raw CTC / noise / all | 29.8 | 80.4 | 93.2 | 1.50 |
| w2v raw CTC / shift / all | 28.9 | 80.7 | 93.7 | 1.40 |
| MMS-FA (NC) / clean / all | 22.6 | 86.2 | 96.5 | 1.14 |
| MMS-FA (NC) / clean / hindi | 22.5 | 86.3 | 96.5 | 1.19 |
| MMS-FA (NC) / clean / english | 20.5 | 88.4 | 97.9 | 0.64 |
| MMS-FA (NC) / clean / filler | 40.3 | 70.3 | 90.2 | 2.54 |
| MMS-FA (NC) / clean / long | 22.9 | 86.6 | 96.5 | 1.14 |
| IndicConformer CTC / clean / all | 92.6 | 45.0 | 62.3 | 14.18 |
| IndicConformer CTC / clean / hindi | 94.1 | 44.2 | 61.5 | 14.56 |
| IndicConformer CTC / clean / english | 85.4 | 48.8 | 65.8 | 12.62 |
| IndicConformer CTC / clean / filler | 86.3 | 47.9 | 64.5 | 11.13 |
| IndicConformer CTC / clean / long | 97.2 | 43.5 | 60.7 | 15.65 |

Signed error vs the strict (character-token) truth, p10 / p50 / p90 ms:
- w2v+fill (stage): start -76 / -32 / -4; end +15 / +55 / +105
- w2v raw CTC: start -20 / +20 / +80; end -65 / +5 / +35
- MMS-FA (NC): start -16 / +20 / +64; end -20 / +35 / +80
- IndicConformer CTC: start -224 / -16 / +189; end -175 / +35 / +240

#### Real speech: IndicVoices human segment edges (642 clips)

| aligner | edges | outside segment >100 ms % | >250 ms % | words inside inter-segment gaps |
|---|---|---|---|---|
| w2v+fill (stage) | 2328 | 0.47 | 0.04 | 0 of 35075 |
| w2v raw CTC | 2328 | 0.21 | 0.04 | 0 of 35075 |
| MMS-FA (NC) | 312 | 1.92 | 1.28 | 0 of 4695 |
| IndicConformer CTC | 2328 | 34.92 | 15.81 | 239 of 35075 |

#### Consistency under perturbation (|boundary - clean boundary|)

| aligner / set / condition | boundaries | median ms | ≤20 ms % | ≤50 ms % | >200 ms % |
|---|---|---|---|---|---|
| w2v+fill (stage) / real / shift | 15844 | 10 | 93.4 | 97.5 | 0.08 |
| w2v+fill (stage) / real / tel | 15844 | 0 | 88.4 | 97.0 | 0.14 |
| w2v+fill (stage) / real / noise | 15844 | 20 | 53.3 | 80.9 | 0.69 |
| w2v+fill (stage) / ctrl / shift | 6944 | 10 | 97.9 | 98.6 | 0.39 |
| w2v+fill (stage) / ctrl / tel | 11942 | 0 | 87.1 | 93.9 | 1.55 |
| w2v+fill (stage) / ctrl / noise | 11942 | 40 | 32.0 | 62.1 | 2.80 |
| w2v raw CTC / real / shift | 15844 | 10 | 99.1 | 99.6 | 0.04 |
| w2v raw CTC / real / tel | 15844 | 0 | 94.7 | 99.7 | 0.06 |
| w2v raw CTC / real / noise | 15844 | 0 | 84.5 | 98.0 | 0.25 |
| w2v raw CTC / ctrl / shift | 6944 | 10 | 99.6 | 99.8 | 0.14 |
| w2v raw CTC / ctrl / tel | 11942 | 0 | 96.5 | 99.9 | 0.05 |
| w2v raw CTC / ctrl / noise | 11942 | 0 | 90.1 | 99.4 | 0.14 |

#### Agreement between independent aligners (clean)

| pair / set | boundaries | median ms | ≤50 ms % | ≤100 ms % | >200 ms % |
|---|---|---|---|---|---|
| w2v+fill (stage) vs MMS-FA (NC) / real | 9390 | 20 | 73.5 | 95.6 | 1.00 |
| w2v+fill (stage) vs IndicConformer CTC / real | 70150 | 140 | 18.6 | 36.0 | 37.45 |
| MMS-FA (NC) vs IndicConformer CTC / real | 9390 | 140 | 19.3 | 38.0 | 35.03 |
| w2v raw CTC vs MMS-FA (NC) / real | 9390 | 20 | 80.4 | 97.6 | 0.61 |
| w2v+fill (stage) vs MMS-FA (NC) / syn | 3496 | 40 | 65.9 | 98.6 | 0.00 |
| w2v+fill (stage) vs IndicConformer CTC / syn | 3496 | 120 | 24.7 | 46.9 | 22.03 |
| MMS-FA (NC) vs IndicConformer CTC / syn | 3496 | 100 | 24.3 | 48.7 | 20.97 |
| w2v raw CTC vs MMS-FA (NC) / syn | 3496 | 20 | 80.0 | 99.2 | 0.06 |

#### QA rules on clean and corrupted audio (stage thresholds)

| set / condition | records | rejected % | align_regret | unspoken | filler_short | edited word flagged % |
|---|---|---|---|---|---|---|
| real / clean | 642 | 8.6 | 0 | 41 | 14 | – |
| real / cut | 200 | 50.5 | 1 | 97 | 3 | 30.5 |
| real / cutf | 211 | 27.0 | 0 | 19 | 38 | 16.6 |
| real / dup | 60 | 8.3 | 0 | 5 | 0 | 0.0 |
| syn / clean | 50 | 0.0 | 0 | 0 | 0 | – |
| syn / cut | 50 | 70.0 | 0 | 35 | 0 | 62.0 |
| syn / cutf | 10 | 40.0 | 0 | 0 | 4 | 40.0 |
| syn / dup | 50 | 0.0 | 0 | 0 | 0 | 0.0 |
| ctrl / clean | 153 | 11.8 | 0 | 10 | 8 | – |

#### Filler durations after gap filling (ms, p10 / p50 / p90; share < 60 ms)

- real / clean: अ n=395: 120 / 180 / 320, <60 ms 4%; आ n=72: 80 / 110 / 180, <60 ms 0%; उम्म n=46: 210 / 390 / 800, <60 ms 0%; ए n=20: 80 / 160 / 220, <60 ms 0%; हम्म n=10: 208 / 330 / 572, <60 ms 0%
- syn / clean: अ n=6: 160 / 160 / 180, <60 ms 0%; आ n=5: 100 / 100 / 132, <60 ms 0%; उम्म n=3: 288 / 320 / 320, <60 ms 0%; ए n=1: 140 / 140 / 140, <60 ms 0%; हम्म n=1: 280 / 280 / 280, <60 ms 0%; अं n=1: 280 / 280 / 280, <60 ms 0%
- ctrl / clean: अ n=191: 80 / 140 / 220, <60 ms 4%; उम्म n=30: 220 / 250 / 344, <60 ms 0%; आ n=17: 100 / 140 / 188, <60 ms 0%; ए n=11: 140 / 160 / 200, <60 ms 0%; हम्म n=6: 240 / 290 / 370, <60 ms 0%; अं n=1: 280 / 280 / 280, <60 ms 0%
- real / cutf: अ n=144: 60 / 100 / 180, <60 ms 14%; आ n=36: 40 / 60 / 90, <60 ms 39%; उम्म n=21: 100 / 140 / 220, <60 ms 0%; ए n=8: 54 / 80 / 98, <60 ms 12%; हम्म n=2: 102 / 110 / 118, <60 ms 0%
- syn / cutf: अ n=4: 32 / 60 / 74, <60 ms 50%; आ n=3: 60 / 60 / 60, <60 ms 67%; उम्म n=2: 120 / 120 / 120, <60 ms 0%; ए n=1: 100 / 100 / 100, <60 ms 0%

**Reading the tables:**
- On known truth the stage is the most accurate aligner tested: 10 ms MAE, 92.8% of boundaries within 50 ms, 0.67% off by more than 200 ms. English-in-Devanagari words are as good as Hindi ones (8.4 ms). Fillers are the weak spot (25 ms; 2.5% gross), then 45–58 s clips (13 ms; 1.7% gross).
- Telephone audio costs almost nothing (11.7 ms). White noise at 10 dB SNR costs the most (24 ms), because the noise floor moves the silence decision in gap filling; even then the stage beats raw CTC (29.8 ms).
- On real speech the stage never places a word inside a gap between IndicVoices' human segments, and 0.47% of segment edges spill more than 100 ms past them (raw CTC 0.21%; IndicConformer 35%).
- MMS-FA agrees with the stage within 50 ms on 73.5% of real boundaries and within 100 ms on 95.6% (the raw paths agree better, 80.4%, because both are peaky CTC). IndicConformer agrees with neither (18–19% within 50 ms), consistent with its 93 ms control error.

### Post-processing: gap filling

- **Problem** (measured on the control): CTC puts each character on one peaky frame and leaves the rest to blank. Raw word starts are late (median +20 ms) and ends have a long early tail (p10 −65 ms). The gross errors (> 200 ms; median 305 ms, max 632 ms) sit on long vowels: phrase-final है/हैं/हूँ end early, vowel-initial और/तो/अ start late (है 63, और 41, हैं 35, तो 24 of them).
- **Rule** (`align.fill_gaps`): a frame is silent if quieter than the clip's 10th-percentile level + 10 dB. A gap between two words with no silent frame is split at its midpoint. Otherwise each word's edge moves through non-silent frames up to the silence, at most 100 ms (also before the first and after the last word). Pauses stay outside every word (SPEC §3).
- **Effect:** control MAE 28.0 → 10.1 ms, ≤50 ms 81.3 → 92.8%, gross 1.36 → 0.67%; fillers 44.6 → 25.2 ms. Cost: real segment-edge spill > 100 ms 0.21 → 0.47%; boundaries move more under noise (real, ≤50 ms of the clean timing: 98.0 → 80.9%).
- **Parameters**, picked on the control and checked against the real segment edges: the level threshold (6–20 dB) changes MAE by < 0.5 ms; pause-side caps of 60–160 ms are within 1.1 ms MAE of each other, and longer caps spill further past human segment edges (0.47% at 100 ms, 2.23% at 500 ms); an uncapped midpoint split beat a capped one by 0.6 ms.
- I: after filling, signed errors against the strict truth are start −32 ms, end +55 ms (median). That is half of VITS's 96 ms blank+space gap at fluent junctions, which real fluent speech does not have. A, not measured: whether real-speech boundaries are this good.
- **Guarantees** (§5.2, SPEC §3): every word owns ≥ 1 CTC frame and the delimiter ≥ 1 more, so words are ordered and non-overlapping before filling; filling never crosses a neighbour; `finalize` rounds to 3 decimals, clamps the last end to `duration` and raises on an empty span. The validator found 0 `word_times`, `words_match` or `sentence_times` violations on 637 records.

### QA rules → rejects (design §3.3: drop, never review)

| Rule | Test | Threshold | Evidence |
|---|---|---|---|
| `align_infeasible` | Transcript needs more CTC frames than the audio has | — | 0 on the pilots |
| `align_regret` | (greedy-path − forced-path log-prob) / frames | > 0.5 | Clean max: real 0.38, TTS 0.17, control 0.32. 20 swapped transcripts: 1.06–2.12 |
| `unspoken` | Non-filler word whose every character has posterior < p, squeezed to its minimum frames + k, with ≥ c characters | TTS: p 0.03, k 1, c 1. Real: p 0.003, k 0, c 2 | Grid below |
| `filler_short` | Filler (SPEC R9) shorter than 60 ms after gap filling | 60 ms (W3) | Filler section |

`unspoken` operating points (share of **clean** records rejected = cost; share of deleted words caught = recall, from the `cut` controls):

| p / k / c | real clean rejected (hours) | TTS clean | control clean | deleted word caught, real / TTS |
|---|---|---|---|---|
| **0.003 / 0 / 2 (real)** | **6.5% (7.8%)** | 0.0% | 0.0% | 31% / 40% |
| 0.01 / 0 / 2 | 8.9% (10.1%) | 0.0% | 0.7% | 38% / 50% |
| 0.01 / 1 / 1 | 15.1% (16.0%) | 0.0% | 3.3% | 46% / 58% |
| **0.03 / 1 / 1 (TTS)** | 21.5% (21.6%) | **0.0%** | 6.5% | 52% / **62%** |

- Why two settings: no setting falsely flags the clean TTS pilot, so TTS gets the lenient one (62% of deleted words caught: W3's skips). Real telephone conversation reduces short function words (ओ, औ, वो, है, ना) until the model hears nothing, so the cost on real audio climbs fast and real records get the strict one. A: whether those real flags are reduced-but-spoken words or transcript errors; W4's ASR gate bounds reference word errors at 0.53%, and at p 0.01 / k 1 / c 1 the rule flags 0.39% of real non-filler words. Harsh's list (15 items) measures it.
- Posterior means don't work: 1% of clean words score ~0 (mostly fillers, below), so per-word mean confidence flags 24–47% of clean records.
- **Not detectable by the aligner:** repeated speech (`dup`: 0% caught on TTS and real). Viterbi absorbs the extra audio into the neighbouring word. W3's round-trip ASR must keep catching repeats.
- On the control, 11.8% of clips trip the TTS-level rules. They are genuine: VITS renders a bare अ or a fragment (ब, उ, न) in 13–30 ms, i.e. barely at all.

### Fillers (handoff to W3)

- **The model never emits fillers.** 81% of the अ that IndicVoices' human transcribers wrote (319 of 395) score a character posterior below 0.05, so no posterior test can verify a filler; only its duration and level can.
- **Durations after gap filling** (ms, p10 / p50 / p90): real अ 120 / 180 / 320 (n = 395), आ 80 / 110 / 180, उम्म 210 / 390 / 800, ए 80 / 160 / 220, हम्म 208 / 330 / 572. W3's pilot: अ 160 / 160 / 180 (n = 6), उम्म 288 / 320 / 320. MMS-TTS often renders a bare अ in 13–30 ms (W3's worry, confirmed for a VITS voice).
- **The 60 ms rule** costs 2.6% of real records and 0% of TTS ones, but catches only 16% (real) and 40% (TTS, n = 10) of fillers whose audio was deleted. Gap filling hands a missing filler the junction frames around it (median 100 ms). So the rule catches fillers rendered too short, not missing ones. W3: keep preferring उम्म/हम्म-type fillers (as already planned) and keep round-trip ASR; don't rely on W5 to prove a filler was spoken.

### Pilot timings (stage runs, S3)

| Input | Run | Where | Records in | Kept | Rejected | Kept hours | `validate.py --partial` |
|---|---|---|---|---|---|---|---|
| `stages/tts` LATEST (W3 pilot) | `stages/align/20261014T105835Z-286b0f5` | laptop CPU | 50 | 50 | 0 | 0.179 | 0 violations |
| `stages/transcribe` LATEST (IndicVoices, cut audio) | `stages/align/20261014T110042Z-286b0f5` (= `align/LATEST`) | ml.c5.4xlarge | 642 | 587 | 55: `unspoken` 41, `filler_short` 14 | 3.131 of 3.498 (10.5% of hours rejected) | 0 violations |

- 30,977 real and 1,748 synthetic words timed.
- **Reproducibility:** three laptop runs byte-identical (records, rejects, raw). The SageMaker run (different CPU) and the laptop benchmark run give identical timings for all 30,977 real words.
- **Runtime:** 642 clips (3.50 h) in 7.2 min on ml.c5.4xlarge with 8 threads, including the S3 download and validation: about 2 min per audio hour. I: 200 h ≈ 7 h on one instance ≈ $6. Laptop numbers (400–1,000 s per audio hour) were measured with 3–5 jobs sharing 16 cores and aren't comparable. IndicConformer ran at 129 s per audio hour on the same instance type.

### Models, tools and APIs used (§10 R1–R2)

| Item | Revision | Licence | Used for | In released data? |
|---|---|---|---|---|
| vakyansh-wav2vec2-hindi-him-4200 (Harveenchadha) | `e2568c3f` | MIT | the `align` stage | **yes**: the timings |
| torchaudio `forced_align`, `resample` | 2.11 (laptop) | BSD-2 | the `align` stage | yes |
| facebook/mms-tts-hin (VITS) | `1d83b223` | **CC BY-NC 4.0** | control-set audio | **no**, benchmark only (T3) |
| torchaudio MMS_FA bundle + uroman | MMS_FA `ctc_alignment_mling_uroman`; uroman 1.3.1.1 | **CC BY-NC** weights; uroman MIT-style with attribution | baseline aligner | **no**, benchmark only |
| ai4bharat/indic-conformer-600m-multilingual (CTC head) | `2a77d305` (bucket copy `raw/_models/…`) | MIT | baseline aligner (SageMaker) | no |
| W4 `w4/convert.py` (`clean`, `nasal`, `spell_acronyms`) | this repo | ours | mapping words to IndicVoices segments | no |

No paid or sampled API was called (no Sarvam, Bedrock or LLM calls), so nothing went through `archive_call`.
The MMS-TTS control renders are sampled generation (VITS noise), seeded per clip (torch seed = the clip's number) and stored once in S3 `stages/w5_ctrl_nc/20261014T093344Z-02f59e2` (audio + truth). They are a benchmark fixture, not a dataset input, so they did not go through `archive_call`; re-rendering would give different audio, so the S3 copy is the fixture.

### Spend

| Date | Item | Billable | Cost |
|---|---|---|---|
| 2026-10-14 | SageMaker ml.c5.4xlarge (IndicConformer ×2, `align` real) | 289 + 655 + 574 s | $0.34 at $0.816/h |
| 2026-10-14 | SageMaker ml.m5.4xlarge (IndicConformer, TTS pilot) | 169 s | $0.05 at $0.97/h |
| | S3: 171 MB control set + run outputs | | < $0.01/month |
| | Sarvam, Bedrock, any LLM | none | $0 |
| **Total** | | | **≈ $0.39 of the $15 cap** |

## Status — 2026-10-14

**Done:**
1. **`align` stage** (`w5/align.py`), registered in `pipeline/run.py`, with a runnable check (`uv run w5/test_align.py` → `ok: align`). Stage contract, resumable, deterministic on CPU, raw Viterbi output archived per record.
2. **Pilot timings pass the validator:** real 587 of 642 kept, TTS 50 of 50, 0 violations (runs above).
3. **Benchmark without a hand-labelled gold set** (`w5/bench.py`): known-truth TTS control, IndicVoices human segment edges, perturbation consistency, cross-aligner agreement, corruption controls. Four aligners: vakyansh wav2vec2 (chosen), MMS-FA, IndicConformer CTC; MFA not run (reasons above).
4. **QA thresholds and reject rules**, calibrated on the controls.
5. **Filler-duration report** for W3.
6. **One optional listening list** for Harsh (below).

**Measured headline:** control-set MAE 10 ms, 92.8% of boundaries within 50 ms, 97.4% within 100 ms, 0.67% off by more than 200 ms (raw CTC 28 ms, MMS-FA 23 ms, IndicConformer 93 ms). On real speech: 0 words inside IndicVoices' inter-segment gaps, 0.47% of segment edges spill more than 100 ms. QA rejects 8.6% of real records (10.5% of hours) and 0% of the TTS pilot; it catches 62% of deleted words in TTS audio.

**Spend:** ≈ $0.39 of the $15 cap (SageMaker CPU, table above). No Sarvam, Bedrock or LLM calls.

**Disk:** `data/w5` is 206 MB now; the peak was about 1.5 GB (MMS-FA 1.2 GB and MMS-TTS 0.14 GB weights, both deleted, re-fetchable at the pinned revisions). The worktree's launcher cache `data/cache` is 31 MB: its cut-audio folder is a symlink to the main checkout's copy instead of a 600 MB re-download. The control set (171 MB) is kept locally for re-scoring; its S3 copy is `stages/w5_ctrl_nc/20261014T093344Z-02f59e2`.

**Blockers:** none.

**Handoffs:**
- **Master / W7 (pipeline):**
  1. `stages/align/LATEST` points at the real run. The TTS run is `stages/align/20261014T105835Z-286b0f5`. One LATEST per stage can't hold both the real and the synthetic path: W6 needs both run ids, or the launcher needs per-path LATEST or one merged input.
  2. SageMaker runs record `"where": "local"` in `run.json` (`run_local` writes it inside the job).
  3. `run.py --local` caches under the **worktree's** `data/cache`, so a child re-downloads inputs the main checkout already holds.
  4. Rebuild: copy `hi_words` and `sentences[].hi_start/hi_end` from `stages/align/<run>/records.jsonl`; never re-align. `_archive.align` is `raw/<id>.json` relative to that run folder. Strip `_qa`, `_archive` (and `_w5bench`, which only `align_bench` writes) at release.
  5. `align_bench` and `w5_ctrl_nc` are benchmark-only (the control audio is CC BY-NC). Never feed them downstream.
  6. At 200 h, run `align` with `--sm ml.c5.4xlarge`: about 2 min per audio hour, so ≈ 7 h and ≈ $6 on one instance. `sm_entry` installs the stage's unpinned PEP 723 deps (`transformers` latest) into the image; pin them if a release breaks the job.
- **W3:** filler durations and the 60 ms rule (filler section): the rule catches fillers rendered too short, not missing ones (16–40% recall). The TTS `unspoken` rule catches 62% of skipped words at 0% cost on the clean pilot. Repeats are invisible to the aligner (0% on `dup`), so round-trip ASR stays the repeat check.
- **W1 / W4:** 55 real rejects (`unspoken` 41, `filler_short` 14) in `stages/align/20261014T110042Z-286b0f5/rejects.jsonl`; each `detail.flags` holds the flagged word indices. Their IDs are burned (design §3.3). If Harsh's list shows the flagged words were mostly spoken, loosen `UNSPOKEN[False]` rather than review.
- **W6:** sentence spans (`hi_start`, `hi_end`) come from the words; sentence splits and text are unchanged.
- **W8b:** any CTC aligner has the same peaky-boundary problem on English targets; the control method (a TTS that exposes durations, plus pause trimming) transfers.

**New question for W0's organiser list (A):** what tolerance does the mid-term audit use for Hindi word timing, and does a word span include adjacent pauses or closures? SPEC §3 says pauses belong to no word [OURS].

**One list for Harsh (optional, ≈ 15 min):** open `data/w5/human/index.html` in a browser (55 items, 42 s of audio in total), write each answer in the `answer` column of `data/w5/human/answers.tsv`, then run `uv run w5/bench.py human-score`.
- 30 `bound` items (12 telephone, 12 wideband, 6 TTS): play the word alone and answer ok / cut / extra. This is the only real-speech measure of boundary accuracy.
- 15 `spoken` items: words the `unspoken` rule rejected. Was the word spoken (y / n)? Mostly y means the real threshold drops good records.
- 10 `filler` items (5 `filler_short` rejects, 5 kept fillers): is a filler audible (y / n)?

**Owned files:** `w5/align.py`, `w5/test_align.py`, `w5/bench.py`, `tasks/W5-alignment.md`, and two STAGES lines in `pipeline/run.py`. Data (gitignored): `data/w5/{ctrl,bench,stage_real,human,ic_meta,smoke}`.
