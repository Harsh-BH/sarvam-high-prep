# W8c: held-out split, eval harness, SeamlessStreaming baseline

**Goal (learning version of W8c in W8-partb-prep.md):** measure a streaming Hi→En S2S system the same way before and after fine-tuning (§8 G2), on held-out data nobody trains on. Sarvam gives no KPI feedback (§6), so the KPIs of §7 get proxies.

Owner files: `w8c/`, `data/w8c/`, this file. Base model (Harsh's decision): SeamlessStreaming (Meta `seamless_communication`, **CC BY-NC 4.0 weights: NC, private eval only, never publish outputs**).

## Split rule

Code: `w8c/split.py` (`uv run w8c/split.py`). Files: `w8c/splits/{train,dev,test}.txt` + `heldout.json` (git), copied to `data/w8c/splits/`.

- **Train** = release/pilot-v2 minus every record `is_heldout()` flags.
- **Held-out synthetic:** whole voices, per engine, in crc32(model/voice) order, until ≥ 10% of that engine's pilot-v2 hours. Each pilot-v2 script has exactly one take, so whole scripts go with their voice. A cloned OmniVoice voice is eligible only if the IndicVoices person it imitates is not a pilot-v2 real speaker. Voices go to dev until dev holds ⅓ of the held-out synthetic hours, then to test.
- **Held-out real:** IndicVoices Hindi *valid* (data/w1/iv_full), cut with W1's cutter and drop rule (`w1/cut_pilot.file_candidates`, pinned ffmpeg).
  - Eligible speakers: not a pilot-v2 real speaker, not a w3/voices.tsv reference speaker.
  - Speaker → dev if crc32(speaker) % 3 == 0, else test. Dev and test are speaker-disjoint.
- **Future records** (scale sessions) are train unless `is_heldout(record)` says otherwise. It flags:
  - a dev/test id;
  - a held-out person: a real `speaker_id`, or an `iv-<speaker>` voice that imitates one;
  - a held-out (model, voice);
  - a held-out script pid (`_w2.pid` / `_w3.pid`).
- **IDs:** `rec-800000`+ from `w8c/ids.tsv`. This is the SPEC §4 reserved range, and that file is its only allocator. It never collides with W1/S1 (`w1/ids.tsv`) or W3/S2 (`w3/ids.tsv`). These clips are eval data, not Part A records.
- **FLEURS hi_in test** (CC BY 4.0, read speech): a third, eval-only set; English references joined from en_us on the FLoRes sentence id.

**Finding: IndicVoices valid is NOT speaker-disjoint from train.**
- 255 of the 483 valid speakers occur in our 2,273-file IV-train sample, all 255 with identical gender/age/district.
- 96 of the 270 pilot-v2 real speakers occur in valid, as do 26 of the 31 OmniVoice reference speakers.
- So "valid = unseen speakers" is false. Disjointness comes from the speaker filter above, not from IV's split.

## Numbers (2026-10-15)

| | synthetic (pilot-v2) | real (IV valid, before stages) |
|---|---|---|
| dev | 33 rec, 0.193 h: bulbul kavya, OmniVoice iv-S4257045200388748, iv-S4258798800383835 | 101 clips, 0.626 h, 49 speakers |
| test | 45 rec, 0.298 h: OmniVoice iv-S4258214800384862, iv-S4259922900308772, magpie Aria | 187 clips, 1.124 h, 76 speakers |
| train | 1,010 of 1,088 pilot-v2 records, 6.090 h | |

- **Supply limit:** every eligible valid clip is used.
  - The cutter finds 1.75 h of 10–60 s single-speaker windows in 9.3 h of eligible raw valid audio. Breaks: 609 loud gaps, 478 second-voice/partner segments, 84 disfluency segments, 79 conversations without a partner JSON.
  - So real dev/test land below the 1 h / 2 h target. FLEURS test adds read speech on top.
- **First try:** also excluding the 1,162 IV-train-sample speakers (so S1 loses no train speakers) left only 0.95 h. Dropped: S1 just skips the held-out speakers' train files via is_heldout.

## Eval sets

`w8c/sets.py` builds them into `data/w8c/sets/` and uploads to `stages/w8c_sets/<run>/` (no LATEST: pass the path). Records keep the SPEC §5 fields; audio is 24 kHz mono s16.

| Set | Source | Records | Hours |
|---|---|---|---|
| dev | 91 real IV-valid + 33 synthetic (OmniVoice 28, bulbul 5); 48 speaker/voice ids | 124 | 0.747 (real 0.554) |
| test | 154 real IV-valid + 45 synthetic (OmniVoice 42, magpie 3); 75 speaker/voice ids | 199 | 1.208 (real 0.911) |
| fleurs | FLEURS hi_in test @ google/fleurs 70bb2e8, en refs = en_us raw transcription of the same FLoRes id | 418 | 1.342 |

Real held-out records went through the Part A stages with explicit paths:

| Stage run | In | Kept | Rejected |
|---|---|---|---|
| `stages/w8c_cut/20261014T204257Z-197c974` (seed) | 288 | 288 | |
| `stages/audio_qc/20261014T204346Z-197c974` | 288 | 280 | level 6, snr 1, silence 1 |
| `stages/transcribe/20261014T204437Z-197c974` | 280 | 279 | chars 1 |
| `stages/w8c_filter/20261014T205156Z-197c974` (seed) | 279 | 277 | 2 with U+090D (align KeyError, fixed on main since) |
| `stages/align/20261014T205202Z-197c974` (ml.c5.4xlarge spot) | 277 | 250 | unspoken 21, filler_short 6 |
| `stages/translate/20261014T210035Z-197c974` (Bedrock, W6 engines) | 250 | 250 | |
| `stages/asr_check/20261014T210456Z-197c974` (ml.c5.2xlarge spot) | 250 | 245 | P1 3, P2 2 |

- Sets run: `stages/w8c_sets/20261014T221953Z-e398964`.
  - Its fleurs.jsonl and audio are byte-identical (ETag) to `stages/w8c_sets/20261014T205358Z-81fbe04`, the run the FLEURS evals used.
- Speakers: dev ∩ test = dev ∩ train = test ∩ train = ∅ (checked on speaker_id).
- dev.txt/test.txt keep the 366 candidate ids. The 43 rejected along the way stay held out, since their ids are burned.

## Harness

**System-output contract.** Any system writes, per record:
- `OUT/<id>.json`: `{"id", "frames": [{"t", "t0", "n"}], "text", "text_events": [{"t", "t0", "text"}], "failed"}`;
- `OUT/<id>.wav`: the frames' audio concatenated as emitted (24 kHz mono s16).

Times:
- `t` is arrival in seconds, with 0 = Hindi frame 0 sent (§5.1).
- `t0` is the same with zero compute, so before/after comparisons don't depend on the GPU.

**Runner** (`w8c/run_seamless.py`):
- Input simulation:
  - SeamlessStreaming s2st without VAD, plus the detokenizer branch for the text track.
  - Our 24 kHz clip is sent as 80 ms frames, frame k at k × 80 ms.
  - The agent reads 320 ms blocks resampled to 16 kHz (`torchaudio.functional.resample`, defaults).
  - A block is available when its last frame is sent.
- Output timing:
  - `t = max(available, previous compute end) + compute`: lag piles up when the model is slower than real time.
  - Vocoder audio (16 kHz) → 24 kHz, cut into frames of ≤ 1 s.
  - `failed` if the model hasn't finished 30 s after input_end (§5.1 R3).
- Settings (agreed with w8d):
  - fp16, decision_threshold 0.5, min_starting_wait_w2vbert 192, no_early_stop;
  - max_len_a 1, max_len_b 200: the text decoder's own defaults. streaming_evaluate's 0/100 truncates 60 s records.
- `--ckpt` = a fine-tuned monotonic decoder `{"model": state_dict}`, loaded strict over the base.

**Scorer** (`w8c/eval.py score --set SET.jsonl --sys OUT`; `uv run w8c/eval.py selftest` checks it on toy outputs):
- Output ASR: Whisper large-v3-turbo (MIT) with word timestamps, run on `<id>.wav`, cached in `asr.jsonl`.
  - Word times map onto the §5.1 playback timeline (frame k plays at max(t_k, end of frame k−1)).
- Translation:
  - ASR-BLEU / ASR-chrF++: sacrebleu on lowercased text with punctuation stripped.
  - ASR-COMET: Unbabel/wmt22-comet-da (Apache-2.0), src = Hindi.
- Entities:
  - Numbers: W6's parser `w6/qa.en_numbers` (lakh/crore, digit strings, 25k = 25,000).
  - Names: capitalised non-initial reference words.
  - Score = found / gold; 1 when a record has none.
- Latency (s):
  - `start_offset`: first output word start − first Hindi word start.
  - `end_offset`: last output word end − last Hindi word end.
  - `laal`: LAAL over output words, delay = word end.
  - `sent_lag`: mean of (word end − end of its source sentence); output word i is mapped to the reference word at the same relative position.
  - Each is reported for `t` and `t0`.
- Text track (S2TT): the same metrics on `text`, with delay = emission time.
- Failed session or no output: worst values (empty hypothesis; delays = source end + 30 s).
- Skipped:
  - intent (LLM judge): add when a fine-tune changes semantics we can't see in COMET/entities;
  - audio quality (UTMOS/DNSMOS): add when the vocoder or T2U changes; the first fine-tune changes only the text decoder.

**SageMaker job** (`w8c/sm_eval.py`):
- Runs the runner, then the scorer, on each set in one GPU job.
- Image: pytorch-training 2.2.0-gpu-py310-cu121. fairseq2 0.2.1 has no py3.12/torch 2.9 build; fairseq2n comes from Meta's pt2.2.0/cu121 index, and seamless_communication is installed `--no-deps` @ 9a081e9.
- Outputs go to `stages/w8c_eval/<run>/<set>/`: per-record outputs, `asr.jsonl`, `per_record.jsonl`, `scores.json`; plus `run.json`.
- Resumes per record after a spot interruption.

## Handoff to w8d

```
uv run w8c/sm_eval.py --sets stages/w8c_sets/20261014T221953Z-e398964 --ckpt s3://sarvam-parta-114872709824/w8d/<run>/ckpt/decoder.pt \
    --name <run> --instance ml.g5.xlarge [--only dev] [--limit N] [--spot]
```
- Same sets, runner, settings and scorer as the baseline; only the decoder weights differ.
- Compare `stages/w8c_eval/<run>/<set>/scores.json` with the baseline's (base: `stages/w8c_eval/20261014T222707Z-e398964` dev+test, `stages/w8c_eval/20261014T210742Z-46901b9` fleurs), and use `t0` latencies for the before/after.
- `--only dev` for early stopping; `--limit N` for a smoke test.
- The first FT's text track used the v1 joiner. Run the base once more with the current runner (`--name base-v2`, no `--ckpt`) before comparing text-track numbers of new checkpoints. Speech-track baselines stand as they are.
- Quotas:
  - ml.g5.xlarge allows 1 instance; ml.g5.2xlarge is the same A10G and keeps `t` comparable.
  - The account allows 10 spot training instances in total; main's sm.py waits up to an hour for a slot.
- One dev+test run costs ≈ $2.7–3.1 on-demand and ≈ 95–100 min; fleurs ≈ $2 and 70 min.

## Baseline vs first fine-tune (2026-10-15)

- **Systems:**
  - Base: SeamlessStreaming (seamless_streaming_unity + seamless_streaming_monotonic_decoder + vocoder_v2).
  - FT: w8d's decoder `w8d/20261014T205030Z-4b7a960/ckpt/decoder.pt`: LoRA r16 merged, 800 steps × 8 on train.txt; T2U and vocoder unchanged.
- **Runs** (all sets ran with the same runner settings):

  | | dev + test | fleurs |
  |---|---|---|
  | base | `stages/w8c_eval/20261014T222707Z-e398964` (g5.xlarge) | `…/20261014T210742Z-46901b9` (g5.xlarge) |
  | FT | `…/20261014T222712Z-e398964` (g5.2xlarge, same A10G) | `…/20261014T210815Z-46901b9` (g5.2xlarge) |

- **Failed sessions:** 0 of 741 for both.
- **Latency:** in seconds, on the zero-compute timeline `t0`. On the A10G, compute adds ≈ 0.4–0.5 s to LAAL (`t`).

| Metric | dev base | dev FT | test base | test FT | fleurs base | fleurs FT |
|---|---|---|---|---|---|---|
| ASR-BLEU | 20.0 | 32.7 | 16.8 | 26.8 | 27.8 | 24.9 |
| ASR-chrF++ | 42.9 | 56.5 | 40.1 | 52.4 | 53.7 | 53.7 |
| ASR-COMET | 0.724 | 0.757 | 0.711 | 0.749 | 0.821 | 0.824 |
| number acc. (speech) | 0.44 | 0.52 | 0.35 | 0.51 | 0.72 | 0.81 |
| name acc. (speech) | 0.49 | 0.54 | 0.44 | 0.52 | 0.66 | 0.70 |
| S2TT BLEU | 21.0 | 38.1 | 17.2 | 31.5 | 25.3 | 22.1 |
| S2TT COMET | 0.748 | 0.805 | 0.729 | 0.777 | 0.830 | 0.823 |
| number acc. (text) | 0.53 | 0.74 | 0.52 | 0.75 | 0.78 | 0.80 |
| output length / reference (speech) | 0.75 | 0.95 | 0.75 | 0.96 | 1.00 | 1.05 |
| start offset | 3.85 | 3.53 | 3.88 | 3.48 | – | – |
| end offset (last word) | 2.45 | 4.03 | 2.49 | 3.90 | 2.57 | 2.47 |
| LAAL | 5.07 | 5.00 | 5.11 | 4.82 | 4.56 | 4.26 |
| sentence lag | −1.00 | 0.59 | −0.71 | 0.83 | – | – |
| text LAAL | 3.63 | 3.02 | 3.77 | 3.22 | 2.74 | 2.38 |

**Test, real IndicVoices only (154 records):**
- ASR-COMET 0.716 → 0.742. Paired bootstrap Δ +0.026, 95% CI [+0.014, +0.038]; FT better on 107 of 154 records.
- ASR-BLEU 18.1 → 23.5; numbers 0.44 → 0.71.

**Test, synthetic held-out voices (45 records):**
- ASR-COMET 0.691 → 0.773, Δ +0.082 [+0.061, +0.105].
- ASR-BLEU 12.9 → 35.2.

**FLEURS** (read Wikipedia, human references):
- ASR-COMET Δ +0.003 [−0.004, +0.011]: no change.
- S2TT COMET Δ −0.007 [−0.012, −0.002].
- BLEU −3.

**Reading it:**
- **The base under-translates our data.**
  - Its speech output is 0.75× the reference length on dev/test and 0.60× on the dense synthetic scripts, against 1.00× on FLEURS.
  - It drops clauses and the ends of utterances in 10–60 s multi-sentence speech. That explains its low end offset and negative sentence lag: it stops talking early. It also explains most of its low BLEU.
  - The fine-tune fixes length (0.95×). Its later end offset is the price of saying the rest; LAAL, which corrects for length, is flat to better.
- **Part of the BLEU gain is style.**
  - Dev/test references are W6 LLM translations, the same style as the training targets (lakh/crore kept, W6 STYLE.md). Synthetic dev/test are W2 scripts from the same generator as the training scripts, so their gain is the most in-domain.
  - FLEURS has independent human references. There BLEU drops and COMET doesn't move, which is what a style shift without a general quality gain looks like.
  - COMET on held-out real speakers (+0.026, CI clear of 0) is the honest headline.
- **Entities:** number accuracy rises on every set, including FLEURS (+0.09). The training data is number-dense (PS §3A), and the decoder learned to say them.
- **Text track:** joined with the v1 runner, which split ≈ 0.6% of words at detokenizer pieces (fixed in 06ddbd9). Base and FT share it, so their comparison is fair. Future runs use the fixed runner: rerun the base once (`--name base-v2`, no `--ckpt`) to compare text-track numbers. The speech track doesn't depend on it.
- **Small sets:** dev 124 / test 199 records, 0.75 h / 1.21 h. Differences under ~0.01 COMET aren't resolvable (see the CIs).

## Spend

| Date | What | Billable | Cost |
|---|---|---|---|
| 2026-10-15 | `align` ×2, ml.c5.4xlarge spot (1st failed on U+090D) | 95 s billable | ≈ $0.02 |
| 2026-10-15 | `asr_check`, ml.c5.2xlarge spot (1 interruption) | 954 s | ≈ $0.11 |
| 2026-10-15 | `translate`, Bedrock via archive_call, stream w6 (798 calls, 250 records) | 131.4 INR ledger (W6 says it over-counts ~1.5×) | ≈ $1.56 |
| 2026-10-15 | smoke eval, ml.g5.xlarge on-demand ($1.69/h) | 539 s | $0.25 |
| 2026-10-15 | fleurs base, ml.g5.xlarge on-demand | 4,150 s | $1.95 |
| 2026-10-15 | fleurs FT, ml.g5.2xlarge on-demand ($1.82/h) | 4,143 s | $2.09 |
| 2026-10-15 | dev+test base, ml.g5.xlarge on-demand | 5,697 s | $2.67 |
| 2026-10-15 | dev+test FT, ml.g5.2xlarge on-demand | 6,125 s | $3.10 |
| | **Total** (+ < $0.1 S3) | | **≈ $11.8 of $15** |

- Prices come from the AWS Pricing API (ap-south-1, SageMaker training).
- Spot was unavailable for the GPU jobs: the account's 10-spot cap was held by scale-real, and g5.xlarge quota is 1 instance.

## Status — 2026-10-15

**Done:**
1. **Split** (merged to main as 347e78e).
   - train 1,010 pilot-v2 ids (6.09 h). Held out: 6 whole voices (78 records), plus 288 speaker-disjoint IndicVoices-valid clips (rec-800000+).
   - `is_heldout()` is used by S1 and S2. S1 filters all 483 valid speakers, a superset; S2 renders only train voices.
   - Finding: IndicVoices valid is not speaker-disjoint from train.
2. **Eval sets** frozen in `stages/w8c_sets/20261014T221953Z-e398964`:
   - dev 124 records / 0.747 h; test 199 / 1.208 h; fleurs 418 / 1.342 h.
   - Real references come from the Part A stages; 245 of 288 clips survived them.
3. **Harness:**
   - `w8c/eval.py`: ASR-BLEU/chrF++/COMET, numbers + names, start/end offset, LAAL and sentence lag on the §5.1 playback timeline, and the S2TT track. `selftest` passes.
   - `w8c/run_seamless.py`: simulated real-time SeamlessStreaming, base or fine-tuned decoder.
   - `w8c/sm_eval.py`: one SageMaker job.
4. **Baseline + first fine-tune** on all three sets; tables above.
   - Held-out real test ASR-COMET +0.026 (CI +0.014 to +0.038).
   - The base under-translates our long multi-sentence clips (0.75× reference length); the fine-tune fixes that.
   - On FLEURS (human references) there's no quality gain and BLEU falls 3: part of the in-domain BLEU jump is the W6 style.
5. **Handoff** to w8d: the command above. The scores were sent to the w8d session.

**Spend:** ≈ $11.8 of the $15 cap (table above). Credits covered all of it: NET $0.00 at the last check.

**Blockers:** none.
- One fix went to main via the master: w5/align.py crashed on U+090D and now rejects it with rule align_vocab.
- One quota note: only 1 ml.g5.xlarge training instance.

**Not done / next:**
- intent judge and UTMOS (skipped; see Harness);
- a `base-v2` text-track rerun when the next checkpoint is evaluated;
- dev/test stay below the 1 h / 2 h target: eligible IV-valid supply is 1.75 h before attrition.

**Optional list for Harsh (≈ 10 min):**
- Open `data/w8c/review/review.tsv`: 8 held-out real test records, the 4 biggest COMET gains and the 4 biggest losses.
- For each id, play `<id>.hi.wav` (Hindi), then `<id>.base.wav` and `<id>.ft.wav` (English, ≈ 10–30 s each).
- Write `base`, `ft` or `same` in the verdict column.
- It checks that COMET's direction matches your ears on our data.

**Local disk:**
- `data/w8c/` now holds only the jsonl files, splits, logs and `review/` (12 MB).
- Clip, set and smoke audio were deleted; all of it is in S3.
