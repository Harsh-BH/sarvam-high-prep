# S1: scale real Hindi audio to ~+30 h

**Goal:** about 30 more hours of real, complete, validated records (real total ≈ 33 h). Source: IndicVoices Hindi *train* archives only; never the valid split, whose speakers are w8c's test set.

**Owner files:**
- `scale/real/`: `cut_iv.py` (the `cut` stage), `test_cut_iv.py`, `chain.sh` (one shard through the stages), `report.py` (held-out check, id register, stats), `valid_speakers.txt`
- `data/scale-real/` (logs, chain run logs)
- this file
- One STAGES line in `pipeline/run.py` (`cut`) and the cap constant in `w6/translate.py`.

## Design

**Ingest + cut, off the laptop.** `cut` is a stage (`scale/real/cut_iv.py`, registered in `pipeline/run.py`), run as a SageMaker CPU job (ml.m5.large spot):
- It streams one `v{1..5}_Hindi_train.tgz`. Every connection must carry the pinned ETag (`w1/cut_pilot.IV_REV`). A dropped connection reconnects with a Range header at the same byte, so a 50 GB stream never restarts.
- Members come sorted, JSON before WAV, and both sides of a call next to each other. The stage buffers one recording group and reads a WAV only if its file is wanted.
- **Cutting** uses W1's code unchanged: `file_candidates` (one-speaker rule, partner timings, the stammers / uh-huh / ugh drop rule), `make_record`, and the pinned ffmpeg.
- **`used` spans:** every span already in `w1/ids.tsv`, burned ones included, so nothing is cut twice.
- **raw/ ingest:** every source a kept clip comes from goes to `raw/indicvoices/<member>` write-once, with its JSON and sha256 sidecars, through `ingest_iv._put`. It's the same bytes the clip was cut from, so `pipeline/rebuild.py` can re-cut it.
- **Held out:** every IV-valid speaker (483, `valid_speakers.txt`), every IV person a `w3/voices.tsv` voice clones, and anything w8c's `is_heldout()` flags.
  - IV train and valid **share speakers**: 255 of the 1,162 speakers in W1's train sample are also in valid.
  - This set is a superset of w8c's 129 held-out persons, so it stays safe if w8c grows its set.
- **Quotas:** hours per scenario per shard, and at most `--speaker-cap` = 240 s of clips per speaker per shard. Streaming stops once every quota is full.
- **Shards:** 5 archives × 2 speaker halves (crc32(speaker) % 2). Each shard is its own cut run, so the chain below shards naturally.
  - **Ids:** each shard takes a block of 1,000 starting at `--id-base`; gaps are fine (SPEC §4). The job writes `ids_new.tsv`, and `report.py ids` appends those rows to `w1/ids.tsv`.

**Per shard** (`scale/real/chain.sh <tag> stages/cut/<run>`):

| Step | Where | Why |
|---|---|---|
| held-out check (`report.py heldout`) | local, records only | a shard with any held-out record stops here |
| `audio_qc` → `dedupe` → `transcribe` | local, one shard's audio (≈0.6 GB) on disk at a time, deleted after | light CPU; the punctuation model is already in the HF cache |
| `align` | SageMaker ml.c5.2xlarge spot | CPU-heavy |
| `asr_check` | SageMaker ml.c5.2xlarge spot | ≈0.8 h per audio hour on CPU: a ≈3.5 h shard takes ≈3 h |
| `translate` | local (Bedrock) | the SageMaker role has no Bedrock permission |

**Order change vs the brief:** `asr_check` runs **before** `translate`.
- The gate reads only the transcript, so the result is the same.
- About 3% of records (asr_check's drops) no longer cost Bedrock tokens, and Bedrock is most of the budget.
- The final, complete run of each shard is its `translate` run.

## Budget

Measured from W6's archived usage on the real pilot: translation costs **$0.644 per audio hour** at list price. W6's archive *estimate* reads $1.07/h, an over-count of 1.66×.

| Item | Estimate |
|---|---|
| Bedrock, ≈30 h through `translate` | ≈ $19.5 |
| SageMaker: cut (10 × m5.large spot), align and asr_check (c5.2xlarge spot) | ≈ $4–5 |
| S3: ≈10 GB raw/ + ≈7 GB clips | ≈ $0.4/month |
| **Total** | **≈ $24–25** of the $25 cap |

`w6/translate.py` `CAP_INR` is raised from 2,500 to 6,000. That cap is on the *shared* `w6` ledger, in estimate units. S1's real spend is computed from the usage in its own archived responses.

## Log

### 2026-10-14, cut round 1 (10 shards)

| Shard | Run | Clips | Conv h | Ext h | Read h | Streamed | Wall |
|---|---|---|---|---|---|---|---|
| v1p0 (smoke) | `stages/cut/20261014T205116Z-d48bdc4` | 405 | 0.24 | 1.61 | 0.40 | 14.3 GB (all) | 497 s |
| v1p1 | `…205938Z-87daaf7-v1p1` | 440 | 0.48 | 1.55 | 0.40 | 14.3 GB | 695 s |
| v2p0 | `…-v2p0` | 285 | 0.10 | 1.55 | 0.23 | 17.9 GB | 806 s |
| v2p1 | `…-v2p1` | 284 | 0.08 | 1.55 | 0.16 | 17.9 GB | 893 s |
| v3p0 | `…-v3p0` | 580 | 0.88 | 1.55 | 0.40 | 54.7 GB | 691 s |
| v3p1 | `…-v3p1` | 547 | 0.89 | 1.55 | 0.40 | 54.7 GB | 943 s |
| v4p0 | `…-v4p0` | 433 | 0.61 | 1.55 | 0.41 | 41.6 GB | 1,073 s |
| v4p1 | `stages/cut/20261014T210058Z-b2f3dca-v4p1` | 480 | 0.80 | 1.56 | 0.40 | 41.6 GB | 604 s |
| v5p0 | `…-v5p0` | 356 | 0.10 | 1.55 | 0.40 | 21.6 GB | 343 s |
| v5p1 | `…-v5p1` | 387 | 0.10 | 1.55 | 0.40 | 21.6 GB | 940 s |
| **Total** | | **4,197** | **4.28** | **15.59** | **3.61** | | **23.48 h** |

(`…` = `stages/cut/20261014T205938Z-87daaf7`.) All runs: validator 0 violations; `report.py heldout` 0 of 4,197.

- **Conversation is scarce.** Each sorted archive holds call uuids at its two ends and numeric extempore/read files in a long middle block. W1's first-500 MB sample was call-heavy for that reason, and so misleading. Under the strict one-speaker rule, a call side yields 1–7 s of clips on average. Every shard streamed its whole archive (the Conversation quota never filled), so **4.28 h is all the Conversation in IV Hindi train** under this rule and this held-out set.
- **Held-out exclusion is costly.** 16,348 file visits were skipped as held-out: about a quarter of the files in each half, because IV-valid speakers record heavily in train too.
- **Extempore and Read** filled their quotas early, so most of each archive was streamed past. Candidates in the processed files alone came to 38.5 h of Extempore.
- Streaming ran at 20–90 MB/s per job.
- `w1/ids.tsv` +4,197 rows (commit ddb2ee7).

**Round 2 (top-ups):** each shard is continued (`--continue`), with quota Extempore 1.0 h + Read 0.15 h, ids from rec-011001. The target is ≈35 h cut, so that ≈30 h survive the stages.

### 2026-10-15, cut round 2 (top-ups) and the stages

**Top-ups.** 10 shards, each continuing its round-1 run with `--continue`. They ran on-demand ml.m5.large to leave the shared spot pool free. Result: 1,887 clips, 11.14 h (Extempore 9.98, Read 1.16), 0 violations. `w1/ids.tsv` +1,887 rows (rec-011001…rec-020194, commit 24408a0).

**Stages.** `scale/real/chain.sh` ran per shard, 20 shards. Two changes along the way:
- **Spot cap raised.** The master approved 7 spot instances, provided account-wide spot use stays ≤ 8. Each SageMaker submit now holds a submit lock and waits while `spot_used` ≥ 8.
- **Bug fixed (ad84083).** The backgrounded `run.py` inherited the submit-lock file descriptor and kept the lock for its whole job. That stalled every other chain for about 1.5 h. Cost: wall time only. Fix: `{sl}>&-` on the child.

## Status — 2026-10-15

**Done.** +30.005 h of real, complete records: 5,384 records from 792 speakers, IndicVoices Hindi *train* only. The real total is now 3.074 + 30.005 = **33.08 h**.

**Checks:**
- Every stage run: `validate.py --partial` 0 violations.
- `report.py check` across all 20 complete runs:
  - 0 duplicate ids, and every id registered in `w1/ids.tsv` with its exact span;
  - 0 overlapping spans, pilot-v2's real records included;
  - 0 held out by w8c's `is_heldout()`;
  - 0 already in a release, and 0 records missing `en_text` or `hi_words`.
- **Rebuild:** 40 sampled clips (2 per shard) were re-cut from `raw/` with the pinned ffmpeg, after checking the sha256 sidecar. 40/40 are byte-identical to the stage audio.

**Hours kept at each stage** (`uv run scale/real/report.py stats`):

| Stage | Records | Hours | Lost h | Rejects by rule |
|---|---|---|---|---|
| cut | 6,084 | 34.622 | – | held-out files skipped at cut: 16,348 file visits in round 1 (not records) |
| audio_qc | 5,976 | 34.051 | 0.571 | silence 59, snr 20, level 19, clipping 10 |
| dedupe | 5,976 | 34.051 | 0 | – |
| transcribe | 5,960 | 33.930 | 0.121 | chars 16 |
| align | 5,511 | 30.663 | 3.267 | unspoken 284, filler_short 149, align_vocab 11, align_regret 5 |
| asr_check | 5,384 | 30.005 | 0.658 | P1 72, P2 55 |
| translate | 5,384 | 30.005 | 0 | – (1,807 records took the repair round) |

86.7% of cut hours survive. align is the main loss (9.6%), close to the pilot's 10.5%.

**Mix of the kept hours:**
- By scenario: Extempore 22.003 h, Read 4.520 h, Conversation 3.482 h (8 kHz telephone). `audio_qc` labels 4.288 h as telephone.
- Speakers: 2,877 male / 2,490 female / 17 other (by record), from 5 states.
- Duration p10/50/90: 11.0 / 16.0 / 36.6 s (range 10.0–59.8 s).
- Conversation is limited by supply. Under W1's strict one-speaker rule and this held-out set, IV Hindi train holds about 4.3 h of conversation clips, and all of it was cut.

**Spend (list price; AWS credits, NET $0.00 at the last `credits.sh`. Cost Explorer lags ~24 h, so this run isn't in it yet):**

| Item | Jobs | Instance | Training s | Billable s | Cost |
|---|---|---|---|---|---|
| cut round 1 (spot) | 10 | ml.m5.large, ml.c5.xlarge | 9,140 | 3,154 | $0.13 |
| cut round 2 (on-demand) | 10 | ml.m5.large | 5,380 | 5,380 | $0.18 |
| align (spot) | 20 | ml.c5.2xlarge | 9,339 | 2,810 | $0.32 |
| asr_check (spot) | 20 | ml.c5.2xlarge | 68,424 | 20,438 | $2.32 |
| Bedrock, translate (Mistral Large 3 + gpt-oss-120b judge), from archived token usage | ≈ 16.5k calls | – | – | – | $18.54 |
| **Total** | | | | | **$21.49** of the $25 cap: **$0.72 per kept hour** |
| S3, new storage | | | | | +19.4 GB `raw/indicvoices` and ≈ 6 GB clip audio: ≈ $0.65/month |

- Sarvam: ₹0.
- The local stages (audio_qc, dedupe, transcribe, translate) ran on the laptop and cost $0.
- Peak local disk was ≈ 0.6 GB: one shard's audio at a time, deleted after use. Nothing is left in the cache.

**Final runs for assembly.** These 20 `translate` runs are whole records: every field, plus `_w1`, `_qa` and `_archive` to strip.

```
stages/translate/20261014T224328Z-0358465-v1p0
stages/translate/20261015T003716Z-ad84083-v1p0b
stages/translate/20261014T225802Z-0358465-v1p1
stages/translate/20261015T005958Z-ad84083-v1p1b
stages/translate/20261014T235953Z-ad84083-v2p0
stages/translate/20261015T012252Z-ad84083-v2p0b
stages/translate/20261015T005341Z-ad84083-v2p1
stages/translate/20261015T014807Z-ad84083-v2p1b
stages/translate/20261014T232321Z-0358465-v3p0
stages/translate/20261015T020327Z-ad84083-v3p0b
stages/translate/20261015T021439Z-ad84083-v3p1
stages/translate/20261015T022512Z-ad84083-v3p1b
stages/translate/20261015T012834Z-ad84083-v4p0
stages/translate/20261015T023210Z-ad84083-v4p0b
stages/translate/20261015T015325Z-ad84083-v4p1
stages/translate/20261015T023814Z-ad84083-v4p1b
stages/translate/20261014T223844Z-0358465-v5p0
stages/translate/20261015T025014Z-dc2f2a3-v5p0b
stages/translate/20261015T014001Z-ad84083-v5p1
stages/translate/20261015T024419Z-ad84083-v5p1b
```

- Each run's audio is its shard's cut run (`run.json` `audio_from`; `data/scale-real/chains/<tag>.cut`).
- The full chain per shard is in `data/scale-real/chains/<tag>.tsv`.

**Blockers:** none.

**Handoffs:**
- **W7 / master (assembly):**
  - Pass the 20 runs above to `w7/assemble.py`, together with pilot-v2's runs. Ids don't collide with pilot-v2 (checked).
  - Real sources are in `raw/indicvoices/…` with sha256 sidecars, so `rebuild` needs nothing new.
  - The last stage is now `translate`, not `asr_check`.
- **w8c:**
  - S1 never used an IV-valid speaker (all 483), nor any IV person a W3 voice clones. You can grow the held-out set from IV-valid speakers without leaking into S1's data.
  - The 17 "Other"-gender records are ordinary IV speakers.
- **Master (shared code, not edited by S1 beyond the lines named above):**
  1. `run.run_local` downloads the audio even for text-only stages (`transcribe`, `translate`). Together with the SageMaker role having **no Bedrock permission**, that forced `translate` to run locally with a per-shard audio download. A Bedrock grant on `sarvam-sm-role` (Converse / InvokeModel on the two model ids) would let it run on SageMaker.
  2. SageMaker `_ckpt/` copies under `stages/<stage>/<run>/_ckpt/` duplicate each run's outputs in S3. For S1's cut runs that is ≈ 6 GB of clip audio, ≈ $0.15/month. They are safe to delete once you're happy with the runs.
  3. `w4/asr_run.hypotheses()` still holds each model's output file open until the model finishes (W7's handoff). No spot interruption hit S1, but one would lose that model's progress.
  4. The account-wide spot limit (10 instances across all training jobs) is the real limit on throughput. `asr_check` takes 68 k training seconds for 30 h of audio.
  5. My `w6/translate.py` change is only `CAP_INR` 2,500 → 6,000, on the shared `w6` estimate ledger.
- **More real hours, if wanted:**
  - Extempore and Read supply is large: most of each archive was streamed past with the quota already full. Another `--continue` round would add ≈ 1 h per shard per hour of quota, at ≈ $0.72 per kept hour.
  - More **Conversation** needs a policy change: W1's audibility-gated one-speaker variant (≈ 2.5× yield), or IV-valid speakers that w8c doesn't hold out.

**Optional list for Harsh (≈ 5 min of audio, ≤ 10 min of time).** `data/scale-real/listen.tsv` lists 20 new conversation (telephone) clips, picked by crc32. For each, fetch the clip with `aws s3 cp <s3 key> . --profile harsh-admin`, listen, and write ok / 2nd voice / bad text in the last column. It spot-checks the one-speaker rule on the new conversation clips, the only rule no model can check (SPEC §1). Skip it if the pilot listening already covered this.
