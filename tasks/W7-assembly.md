# W7: Assembly, reproducibility and release

**Goal:** turn stage outputs into the submitted Part A dataset, reproducibly. At stake: the reproducibility points (3 pts), the mid-term gate (§8), and disqualification under §10 R7 if a re-run fails.

## PS rules

- The pipeline must rebuild every record from the source audio plus the archived transcripts, translations and TTS outputs. Byte-identical where possible; equivalent where not, e.g. LLM non-determinism (§3A).
- Deliverables (§9.2):
  - `.wav` files + `.jsonl` manifest in a **private** HF dataset repo shared with Sarvam;
  - the code as a Git commit hash;
  - `IITcode_midterm_report.pdf`, ≤4 pp + 1 p refs.
- A private HF dataset holding our source audio is allowed (§11 FAQ, §9.2). **Any other extra file is penalised** (§9.2).
- G3 compares the final dataset with the mid-term one (§8). No new IDs, and no source interval may grow.

## Build

- **Two pipelines:**
  - `generate`: non-deterministic; calls ASR, LLM and TTS; writes archives.
  - `rebuild`: deterministic; archives + source audio → WAVs + JSONL; **no API calls**.
- **Pinning:**
  - HF dataset revisions;
  - a dependency lockfile;
  - fixed resampler and ffmpeg versions;
  - fixed JSON float formatting and key order;
  - WAVs written without tool-version metadata chunks.
- **Validator** (from W0) in CI on every build, plus the hours counter.
- **Statistics for the mid-term report:**
  - hours by source and by real vs synthetic;
  - duration histogram; speaker counts;
  - entity and phenomenon counts (from W2 metadata); code-mix rate;
  - TTS rejection rate; aligner QA results;
  - English/Hindi duration ratio once W8b exists.
- **Freeze:** tag the HF revision submitted at the mid-term. Every later change is a diff against it.

## Research

- HF repo limits: files per folder (~24k clips for 200 h of 30 s clips) and storage backend (LFS vs Xet).
- Sharing a private repo with Sarvam's account or org.
- Modal volumes and images for running the pipeline.

## Done when

The pilot goes end to end into a private HF repo, passes the validator, and `rebuild` reproduces it byte for byte on a clean machine.

## Status — 2026-10-14

Purpose changed mid-task (master, from Harsh): **personal learning project, nothing submitted. Step 4 (Hugging Face repo + upload) is skipped entirely**; HF limits were not researched. Steps 1–3 and 5 are done.

**Done:**
1. **Gate (step 1):** `asr_check` over the complete real set on SageMaker: `stages/asr_check/20261014T115647Z-c478d29`
   (input `stages/translate/20261014T112157Z-2bef5a6`). **571 kept, 16 retired (P1 13, P2 3), 3.074 h, `validate.py
   --partial` 0 violations.** LATEST moved. It matches W4's local gate on the same ids except `rec-000580`, which W4 retired
   (P1) and this run keeps: the CPU/fp32 run lands it just under the threshold. The three model hypotheses are archived
   with the run (`asr/*.jsonl`).
   - Instance: ml.g5.xlarge waited for capacity (stopped after 10 min), and ml.g4dn.xlarge spot hit "Insufficient
     capacity" for 37 min (stopped). Both billed $0. The run used **ml.c5.2xlarge spot**: the stage's main model,
     IndicConformer, runs on CPU onnxruntime anyway, and `asr_run` falls back to CPU fp32.
   - Timings: vakyansh 544 s, Whisper-turbo 5,644 s (9.6 s/clip), IC-CTC about 45 min. One spot interruption at 31 min.
     Training 6,556 s, billable 2,096 s.
2. **Synthetic provenance (step 2):** `uv run w7/archive_w3.py` (test: `uv run w7/test_archive_w3.py`):
   - 764 files to `archive/w3/` (287 MB), all write-once;
   - 50 ids bound under `archive/tts/` (30 MB);
   - ₹52.78 of Bulbul spend in the ledger.
   - `rebuild` now copies synthetic audio from there and never re-synthesises.
3. **Assembly (step 3):** `uv run w7/assemble.py pilot-v1 --runs stages/asr_check/20261014T115647Z-c478d29
   stages/translate/20261014T112115Z-28cd27c` → **`release/pilot-v1`: 621 records (571 real + 50 synthetic), 3.253 h,
   full `validate.py` 0 violations**, no `_` keys, sorted by id (SPEC §6). 562 MB of audio; stage files frozen under
   `inputs/`. Took 5 min on the laptop.
   - **Rebuild on a clean machine:** `uv run pipeline/run.py rebuild --in stages/asr_check/20261014T115647Z-c478d29 --audio-from stages/asr_check/20261014T115647Z-c478d29 --sm ml.m5.large --spot -- pilot-v1` → `stages/rebuild/20261014T143034Z-c57d330/rebuild.json`: **621/621 WAVs byte-identical, manifest byte-identical**, from the bucket alone (real records re-cut from `raw/` with the pinned ffmpeg; synthetic records from `archive/tts/`; stage files from `inputs/`). 3 min 10 s on 2 vCPUs (§3A, §10 R7).
4. **Data card (step 5):** `data_card.md` (draft) with licences/attribution from `pipeline/corpora.json`, every model
   and API that produced a released field (§10 R2), and the statistics from `uv run w7/stats.py pilot-v1`. All stats come
   from the bucket alone. Headline:
   - hours: IndicVoices Conversation 8 kHz 1.252, Extempore 1.578, Read 0.244; Bulbul 0.123, Magpie 0.056;
   - durations: p50 14.6 s, range 9.0–59.8 s; 270 real speakers (F 127 / M 141 / other 2) from 5 states; 15 TTS voices;
   - code-mix: 12.5% of real words are English (IndicVoices-glossed, a lower bound); synthetic has 272 English spans
     and 176 entities in 1,748 words; fillers 366 / 11; immediate repeats 190 / 18;
   - TTS reject rate: Bulbul 18.1%, Magpie 11.0%, Parler 40.0% (reproduces W3);
   - aligner regret p50/p90: 0.069/0.148 real, 0.050/0.088 synthetic; align rejects 55 (unspoken 41, filler_short 14);
   - translation: 179 real and 3 synthetic records carry ≥ 1 critical flag after repair (kept, per W6).
5. **Step 4 skipped** (no HF repo, no upload, no HF research) on Harsh's change of purpose.

**Spend (AWS, list price; credits cover it all, net $0.00 at 20:10 IST):**

| Item | Billable | Price | Cost |
|---|---|---|---|
| `asr_check`, ml.g5.xlarge (waited for capacity, stopped) | 0 s | – | $0 |
| `asr_check`, ml.g4dn.xlarge spot (insufficient capacity, stopped) | 0 s | – | $0 |
| `asr_check`, ml.c5.2xlarge spot | 2,096 s | $0.408/h | $0.24 |
| `rebuild`, ml.m5.large spot | 109 s | $0.121/h | $0.004 |
| S3: +287 MB `archive/w3/`, +30 MB `archive/tts/`, +565 MB `release/pilot-v1/` | – | $0.025/GB-mo | ≈ $0.02/month |
| **Total** | | | **≈ $0.25** of the $15 cap. Sarvam ₹0, Bedrock $0 |

**Disk:** `data/w7` holds 68 MB (`assemble/pilot-v1`: 50 synthetic WAVs + symlinks into the shared cut cache). Scratch
from the dry runs and the local rebuild smoke test (≈ 310 MB) is deleted. The worktree's `data/tools` is a symlink
to the main checkout's pinned ffmpeg, not a 309 MB copy.

**One list for Harsh: empty.** Nothing in W7 needs ears or eyes. The byte-for-byte rebuild and the validator are
the checks.

**Handoffs:**
- **Master (shared code I didn't edit):**
  1. `w4/asr_run.py hypotheses()` holds each model's output file open until the model finishes. SageMaker syncs a
     checkpoint file only once it's closed, so a spot interruption loses that model's whole progress (seen: about
     20 min of Whisper lost at 18:01). Fix: open/append/close per line, or every N lines.
  2. `rebuild.tts_audio()` is a sibling of `ffmpeg.cut()`. Move it into `pipeline/ffmpeg.py` if W3's bulk run
     should import it from there; W3 must make synthetic `audio/` with it (pinned ffmpeg), never with `w3/pilot.py
     to24k` (system ffmpeg, ±1 LSB on resampled takes).
  3. `release/pilot-v0` predates the frozen `inputs/`. Its provenance names local paths, and rebuild still accepts
     them (40/40 identical, smoke-tested through the new `rebuild` stage).
  4. `sm.py` hard-codes `MaxRuntimeInSeconds` 6 h. At 200 h, `asr_check` on this CPU instance would take about 64× this
     run's ≈ 2.5 h of decoding (≈ 160 instance-hours, ≈ $20 on spot): it needs `onnxruntime-gpu` + a GPU that gets capacity, or sharding
     into < 6 h jobs.
- **W3:** every future kept take goes to `archive/tts/<id>.wav` + sidecar (`w7/archive_w3.py bind()`); raw responses
  go through `archive_call`. `data/w3/archive/` is now fully in S3 and can be deleted locally when W3 agrees
  (277 MB).
- **W4:** `rec-000580` passes P1 on the CPU run but failed it on yours (GPU fp16). Borderline; no action unless the
  audit says otherwise.

**Files (committed on `stream/w7`):** `w7/archive_w3.py`, `w7/test_archive_w3.py`, `w7/assemble.py`,
`w7/rebuild_stage.py`, `w7/stats.py`, `data_card.md`, `pipeline/rebuild.py`, `pipeline/release.py`,
`pipeline/test_rebuild.py`, one STAGES line in `pipeline/run.py`. Checks: `uv run pipeline/test_rebuild.py`,
`uv run w7/test_archive_w3.py`, `uv run pipeline/test_run.py`, `uv run pipeline/test_sm.py`.

### Design decisions (W7)

| Decision | Choice | Why |
|---|---|---|
| Where a synthetic take lives | `archive/<source_file>` = `archive/tts/<id>.wav`, write-once, + `.meta.json` sidecar `{sha256, take, render}` | Mirrors real (`raw/<source_file>` + sha256 sidecar), so rebuild resolves both the same way. Write-once binds an id to its audio forever (G3) |
| W3's pre-`archive_call` renders | Verbatim under `archive/w3/<engine>/<key>.wav\|.json` (all 382, kept or rejected) + dated `renders-`/`qa-` snapshots; Bulbul spend markers under `archive/spend/w3/` (₹52.78, so `archive.spent("w3")` is honest) | Paid outputs are never lost; design §2.2 "every take archived". Not re-keyed into `archive/api/` (their request format predates `archive_call`'s `take` convention) |
| Synthetic release audio | `rebuild.tts_audio(take)`: canonical 44-byte header, soxr resample with the **pinned** ffmpeg. `assemble` makes it with the same function rebuild uses | W3's `to24k` used the system ffmpeg (n9.0.1): Bulbul (24 kHz, no resample) is byte-identical either way; the 17 Magpie takes (22,050 Hz) differ by **±1 LSB** (RMS 0.02 vs signal RMS ≈2,000), same length. Equivalent for W5's timings; byte-identical only with the pinned binary |
| What rebuild re-merges | `release.snapshot` freezes its input stage files under `release/<tag>/inputs/<n>.jsonl`; provenance names those keys + their origin runs | `stages/` is mutable and its noncurrent versions expire in 30 days; `release/` never expires. Removes the "local stage-file paths" ponytail, so a clean machine needs only the bucket |
| Joining real + synthetic | `w7/assemble.py`: whole-record runs; an id in two runs is refused | `release.merge()` unions fields: passing `translate` (587) and `asr_check` (its subset) together would silently resurrect gate-dropped ids |
| Clean-machine rebuild | `rebuild` registered as a stage (`w7/rebuild_stage.py`, one STAGES line), run with `--sm ml.m5.large` | `sm_entry` only runs STAGES; input ignored, report in `rebuild.json`, exit 1 on any diff |

## Status — 2026-10-15: release/pilot-v2

- `uv run w7/assemble.py pilot-v2 --runs stages/asr_check/20261014T115647Z-c478d29 stages/translate/20261014T112115Z-28cd27c stages/translate/20261014T175905Z-7250bf3` → **1,088 records, 6.581 h**, 0 violations on the full validator:

  | Source | Records | Hours |
  |---|---|---|
  | Real, IndicVoices (gated) | 571 | 3.074 |
  | Synthetic, OmniVoice (W3 round 2, W2 pilot scripts) | 467 | 3.328 |
  | Synthetic, Bulbul v3 (round 1) | 33 | 0.123 |
  | Synthetic, Magpie 357M (round 1) | 17 | 0.056 |

- Synthetic ids are now bound to their kept take at release time (`assemble.bind_take`, write-once). Nine ids already bound by an abandoned W3 run to other audio were burned and left out: rec-400051, 400052, 400054, 400055, 400056, 400058, 400059, 400060, 400062.
- **Clean-machine rebuild:** `stages/rebuild/20261014T182330Z-4c18ee7` on SageMaker ml.m5.large spot (119 s billed): **1,088/1,088 WAVs and the manifest byte-identical**, from the bucket alone.
