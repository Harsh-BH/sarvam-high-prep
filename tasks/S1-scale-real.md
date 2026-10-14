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
