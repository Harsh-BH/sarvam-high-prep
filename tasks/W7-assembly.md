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

<RESULTS>

### Design decisions (W7)

| Decision | Choice | Why |
|---|---|---|
| Where a synthetic take lives | `archive/<source_file>` = `archive/tts/<id>.wav`, write-once, + `.meta.json` sidecar `{sha256, take, render}` | Mirrors real (`raw/<source_file>` + sha256 sidecar), so rebuild resolves both the same way. Write-once binds an id to its audio forever (G3) |
| W3's pre-`archive_call` renders | Verbatim under `archive/w3/<engine>/<key>.wav\|.json` (all 382, kept or rejected) + dated `renders-`/`qa-` snapshots; Bulbul spend markers under `archive/spend/w3/` (₹52.78, so `archive.spent("w3")` is honest) | Paid outputs are never lost; design §2.2 "every take archived". Not re-keyed into `archive/api/` (their request format predates `archive_call`'s `take` convention) |
| Synthetic release audio | `rebuild.tts_audio(take)`: canonical 44-byte header, soxr resample with the **pinned** ffmpeg. `assemble` makes it with the same function rebuild uses | W3's `to24k` used the system ffmpeg (n9.0.1): Bulbul (24 kHz, no resample) is byte-identical either way; the 17 Magpie takes (22,050 Hz) differ by **±1 LSB** (RMS 0.02 vs signal RMS ≈2,000), same length. Equivalent for W5's timings; byte-identical only with the pinned binary |
| What rebuild re-merges | `release.snapshot` freezes its input stage files under `release/<tag>/inputs/<n>.jsonl`; provenance names those keys + their origin runs | `stages/` is mutable and its noncurrent versions expire in 30 days; `release/` never expires. Removes the "local stage-file paths" ponytail, so a clean machine needs only the bucket |
| Joining real + synthetic | `w7/assemble.py`: whole-record runs; an id in two runs is refused | `release.merge()` unions fields: passing `translate` (587) and `asr_check` (its subset) together would silently resurrect gate-dropped ids |
| Clean-machine rebuild | `rebuild` registered as a stage (`w7/rebuild_stage.py`, one STAGES line), run with `--sm ml.m5.large` | `sm_entry` only runs STAGES; input ignored, report in `rebuild.json`, exit 1 on any diff |
