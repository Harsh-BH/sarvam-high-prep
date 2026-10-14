# S3 storage, ingestion and cleaning pipeline: design

2026-10-14. Status: approved in chat, section by section; awaiting review of this document. Owner: master session → W7 after handover.

## Goal

One storage layer and two pipelines that every stream reads and writes:

- **Ingestion** gets every input we depend on into S3, pinned and immutable: corpus audio and transcripts, every Sarvam/LLM output, every human correction.
- **Cleaning** turns that into validated Part A records: licence gate, one-speaker QC, cutting, audio QC, dedupe, transcript conversion, TTS QA.

**Success means:**

- Anyone with bucket access rebuilds every record from `raw/` + `archive/` and gets identical bytes where possible (§3A).
- `validate.py` passes on every stage output and on the release (§5.2, §8).
- Nothing private can be published (§10 R3).
- The mid-term snapshot's IDs, `source_file` names and audio never change (§3B, §8 G3).

**Non-goals:** W5 alignment, W6 translation and W7 assembly internals (they use the same stage format), Part B training, augmentation (training-time only, §3A), a managed DAG or dashboard.

## Decisions taken

| Decision | Choice | Why |
|---|---|---|
| Storage | S3, one bucket, ap-south-1 | User's choice; same region as the GPU access, so transfer is free |
| GPU compute | SageMaker training jobs in ap-south-1 | The `harsh` account has SageMaker GPU quota (ml.g5.\*, ml.g4dn.\* incl. spot, ml.g6.\*, ml.g6e.12xlarge); EC2 GPU quota is 0 in every region |
| Part B compute | Modal (organiser credits) or a P-family quota request | No H100/P-family SageMaker quota exists |
| Pipeline shape | Approach A: plain stage CLIs + fixed S3 layout + one launcher | Fits ~9 stages and 6 weeks; reuses stream code; rebuild stays portable. Rejected: managed DAG (heavy, AWS-locked), DVC/LakeFS (extra tool, overlaps versioning + archive) |

## 1. Storage

**Bucket:** `sarvam-parta-114872709824`, ap-south-1.

**Protection:**

- Block Public Access on the bucket **and at account level** (the account has none today).
- Default SSE-S3 encryption. Versioning on.
- Bucket policy: deny any request without TLS (`aws:SecureTransport = false`).
- Write-once archive: the archive writer uses `PutObject` with `IfNoneMatch: *`, so a second write to the same key fails. Only the admin may delete under `archive/`, `raw/`, `release/`.

**Layout:**

```
raw/<corpus>/<upstream path>               source files we cut from, byte-for-byte, never modified
raw/_catalog.jsonl                         corpus rows: licence, licence URL, attribution, status OK/HOLD/TRAP
                                           file rows: upstream rev (ETag / HF commit), sha256, size
archive/api/<service>/<sha256(request)>.json|.wav
                                           every Sarvam/LLM call: canonical request + raw response, write-once
archive/human/<stage>/<date>-<who>.jsonl   human review patches, write-once
stages/<stage>/<run>/records.jsonl         SPEC §8 stage file
stages/<stage>/<run>/rejects.jsonl         excluded records + the rule that excluded them
stages/<stage>/<run>/audio/                if the stage produces audio
stages/<stage>/<run>/run.json              git sha, params, input run, instance, cost
stages/<stage>/LATEST                      the run the next stage reads
release/<tag>/manifest.jsonl + audio/      frozen snapshots; release/midterm/ is the G3 freeze
```

- Lifecycle: noncurrent versions under `stages/` expire after 30 days. `raw/`, `archive/`, `release/` never expire.
- Git holds code and small registries (`w1/ids.tsv`). S3 holds bytes. Local `data/` is a disposable cache (`aws s3 sync`).

**Access:**

- No root keys. One person works on this (Harsh), through the IAM user `harsh-admin` (profile `harsh-admin`); it is the only principal besides root allowed to delete under `archive/`, `raw/`, `release/`. No team group: add one only if teammates join.
- New SageMaker execution role `sarvam-sm-role` with the same bucket scope. `AMCSageMakerExecutionRole` belongs to another project and is not reused.

**Size and cost:** ~150–350 GB (≤35 GB final dataset, ~100–300 GB kept sources, ~10 GB synthetic takes) at $0.025/GB-month ≈ $4–9/month. Same-region transfer to SageMaker is free; egress elsewhere is $0.109/GB after the first 100 GB/month.

## 2. Ingestion

### 2.1 Corpus audio and transcripts

- **Licence gate first.** `raw/_catalog.jsonl` corpus rows come from `research/datasets.md` and `tasks/W1-real-audio.md`. Ingestion refuses any corpus whose status is not `OK` (today: Kathbath HOLD; Gram Vaani, Shrutilipi, YODAS, IndicVoices-R TRAP/excluded). The same rows generate `data_card.md` (§10 R1).
- **One adapter per source type:**
  - tar over HTTP with ranged reads (IndicVoices; W1 `tar_scan.py`);
  - Hugging Face parquet pinned to a commit sha, never `main` (Vaani, Lahaja, Rural Women);
  - plain HTTP / zip (SPRING-INX, MUCS).
- **Two passes:** (1) index metadata only (JSONs, parquet columns without audio) and prefilter on licence, single speaker, verification flags, duration; (2) fetch only the selected source files and stream them into `raw/`. Local disk is never the staging area.
- **Pinning:** each file row records upstream URL, rev, sha256, size. Re-running skips keys already present with the same sha256.
- **Runs on** CPU: a SageMaker CPU job in ap-south-1, or a laptop for small runs.

### 2.2 API outputs

`archive_call(service, request) -> response`, the one helper every stream uses for Sarvam (Bulbul, Saaras) and LLM calls:

1. `key = sha256(canonical JSON of request)`, where the request includes model, voice, params, text and a `take` number (deliberate regeneration = next take).
2. If `archive/api/<service>/<key>` exists, return it. No call, no spend.
3. Otherwise call the API, write request + raw response with `IfNoneMatch: *`, then return. A crash after the call never loses a paid output.
4. Retries with backoff; never stores a partial response.
5. Spend per stream is summed from archived metadata; `archive_call` refuses new calls once a stream's cap is reached.

### 2.3 Human corrections

The review tool exports patch files to `archive/human/<stage>/`, write-once. Patches are never edited: a newer patch for the same `id` supersedes the older one; both are kept.

### 2.4 Provenance and check

- Every record carries `_rev` (its `raw/` key + sha256) and `_archive` (the archive keys each field came from).
- `ingest verify` recomputes the sha256 of every `raw/` object against the catalog.

## 3. Cleaning

**Stage contract:**

- Input: the previous stage's `records.jsonl` (via `LATEST`). Output: `records.jsonl` + `rejects.jsonl` (+ `audio/`).
- Joined on `id`. Nothing is dropped silently: every exclusion lands in `rejects.jsonl` with its rule.
- A rejected ID is burned, never reused (SPEC §4).
- `validate.py --partial` gates every stage output.

### 3.1 Real path

| # | Stage | Runs on | Does | Existing code |
|---|---|---|---|---|
| C1 | `select` | CPU | Catalog `OK` only, one `speaker_id` per file, no chatter flags, Hindi | W1 `cut_pilot.py` |
| C2 | `segment` | CPU | One-speaker QC: drop tagged second-voice / unintelligible segments, partner overlap, loud gaps; pick contiguous 10–60 s windows | W1 `runs()` |
| C3 | `cut` | CPU | soxr resample to 24 kHz mono s16, sample-exact trim, pinned ffmpeg; IDs from `w1/ids.tsv` | W1 `cut()` |
| C4 | `audio_qc` | CPU | Clipping ratio, SNR estimate, silence share, level, duration. Telephone is a label, not a reject (§4) | new |
| C5 | `dedupe` | CPU | Flag identical intervals of the same `source_file`. Cross-corpus duplicates are blocked at the catalog | new |
| C6 | `transcribe` | CPU | Corpus transcript → SPEC §2; ear-flags go to the review queue. IndicVoices L1 is done; SPRING-INX and Vaani first need a Latin→Devanagari + digits→words step (Aksharantar / IndicCMix lexicon) | W4 `convert.py` |
| C7 | `asr_check` | GPU (SageMaker g5) | Route to review priorities P0–P3 (W4 protocol). Routing uses only ASRs not trained on the corpus (vakyansh, Whisper-turbo); IndicConformer (best CER, possibly trained on IndicVoices) is the reviewer's draft, not a router. No ASR keeps fillers, so ASR text never replaces the corpus transcript | W4 `asr_run.py`, `asr_score.py`, `review_list.py` |
| C8 | `apply_patches` | CPU | Apply `archive/human/` patches (from the audit, §3.3); a `recut` sends the clip back to C3 under a **new** ID | W4 `review_list.py` |

### 3.2 Synthetic path

| # | Stage | Runs on | Does |
|---|---|---|---|
| S1 | `scripts` | API | W2 script generation through `archive_call` |
| S2 | `tts` | API / GPU | Bulbul through `archive_call`, or an open-weight engine on SageMaker; every take archived |
| S3 | `tts_qa` | GPU | W3 `qa.py`: round-trip ASR from a different family, entity match, duration/char, clipping, long silence, skip/repeat. Fail → next `take`. Rejected takes stay archived, never referenced. `hi_text` = what the TTS actually said |

Both paths then feed W5 → W6 → W7 through the same stage format.

### 3.3 Review policy (decided 2026-10-14: one human, no annotators)

- **Drop instead of review.** C2 treats `[stammers]` `[uh-huh]` `[ugh]` like `unintelligible`: drop the segment and break the clip there, so the audio around it survives in other clips. Existing clips with these flags (W4's P0) and clips where independent ASRs disagree with the reference (P1) are retired to `rejects.jsonl`; their IDs are burned.
- **Entity check is automatic.** P2 compares number words and English-in-Devanagari words with the IndicConformer draft; no listening.
- **One fixed audit.** A deterministic, stratified ≈1 h sample (scenario × 8 kHz/wideband), listened to once by Harsh, measures transcript accuracy for the mid-term report (§7 audits transcripts). Corrections from it go through C8.
- **Why:** at W4's measured rate a full review is ≈1.2 listening-hours per audio-hour (≈240 h for 200 h); this policy needs ≈5 h total. Supply (≈600 h IndicVoices) covers the ≈18% of audio lost.

### 3.4 Open decisions (stage parameters, not code)

| Parameter | Stage | Options | Evidence |
|---|---|---|---|
| Conversation one-speaker rule | C2 | strict / relaxed | 3.10 h vs 7.88 h of conversation in W1's sample |
| Minimum clip duration | C2 | 10 s / lower | Vaani segments are mostly <10 s, none >~20 s (W1 round 2) |

## 4. Execution

### 4.1 Launcher

```
uv run pipeline/run.py <stage> --in <prev run> [--local | --sm <instance> [--spot]]
```

- `--local`: sync input to the cache, run the stage CLI, upload outputs, move `LATEST`.
- `--sm`: the same CLI as a SageMaker training job: AWS PyTorch Deep Learning Container image, code tarball at `s3://…/code/<git-sha>.tar.gz`, S3 input channel, partial output in `/opt/ml/checkpoints` (CheckpointConfig → S3) so spot interruptions resume.
- Every run writes `run.json` (git sha, params, input run, instance, cost).
- `LATEST` moves only when the stage completes **and** `validate.py --partial` passes. Downstream never reads a half-written run.
- One pinned static ffmpeg binary (sha256 recorded) is used by C3 and `rebuild` on every machine.

### 4.2 Rebuild (§3A)

`rebuild release/<tag>`:

- Real: fetch from `raw/`, verify sha256, re-cut with the pinned ffmpeg, compare the WAV.
- Synthetic: copy the archived take. Never re-synthesise.
- Text: stage files + `archive/human/` patches.
- Report: % byte-identical and a diff list. If `raw/` is lost, refetch from the pinned upstream rev.

### 4.3 Failure handling

| Failure | Handling |
|---|---|
| Bad record | `rejects.jsonl` with reason; stage continues |
| Stage crash / spot interruption | Re-run the same run; IDs already in the output are skipped |
| API error | `archive_call` retries with backoff; no partial writes |
| Spend over cap | `archive_call` refuses; stream asks for a raise |

### 4.4 Testing

- `test_validate.py`: the format contract (exists).
- Pilot run: every stage on the 50-clip pilot locally; `validate.py` passes at the end.
- Rebuild test: the pilot snapshot rebuilds 100% byte-identical (W1 already 21/21 on re-cuts).
- `archive_call` test: two identical requests → exactly one API call.
- SageMaker smoke test: one real `asr_check` run on the pilot through `--sm`.

### 4.5 Cost

| Item | Estimate |
|---|---|
| S3 storage | $4–9/month |
| GPU (C7, S2 open-weight, S3) | 20–60 h × ml.g5.xlarge $1.69/h ≈ $35–100; ml.g4dn.xlarge $0.81/h or spot for ASR |
| CPU jobs | small |
| Sarvam API | tracked per stream; Part A ≤30–40% of the $1,000 credit |

## 5. One-time setup

**Done 2026-10-14** (approved by Harsh), verified live: plain put to `archive/` → AccessDenied; put with `If-None-Match: *` → OK; repeat → PreconditionFailed; non-TLS → 403. Policies pass IAM Access Analyzer with 0 errors. Remaining: deactivate the root access keys (Harsh's call). The `sarvam-team` group was created and then removed (2026-10-14): Harsh works alone.


1. Bucket with BPA, SSE-S3, versioning, TLS-only policy, delete protection on `archive/` `raw/` `release/`, lifecycle on `stages/`.
2. Account-level Block Public Access.
3. IAM user `harsh-admin` + profile, replacing the root keys.
4. SageMaker role `sarvam-sm-role`, scoped to the bucket.

## 6. Repo placement

New code lives in `pipeline/` (`run.py`, `archive_call.py`, `ingest/`, `rebuild.py`). Stage logic stays in the stream folders (`w1/`, `w3/`, `w4/`). This file and `tasks/` are planning docs, not §9.2 deliverables: they stay out of the submission repo.
