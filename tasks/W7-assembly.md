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
