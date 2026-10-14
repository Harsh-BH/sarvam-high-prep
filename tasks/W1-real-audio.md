# W1: Real-audio sourcing

**Goal:** know exactly how much usable, redistributable real Hindi audio exists, then pick and cut the real hours. The hidden test is **real** recordings (§6) and the base model is trained on **synthetic** data (§3B), so real audio is where fine-tuning gains most.

## PS rules

- Real recordings only from datasets whose licence allows redistribution; licence and consent are our responsibility (§3A, §10 R1).
- One speaker per clip; ≤60 s; each record is a contiguous `[start, end]` of one source file (§5.2).
- Overlapping cuts of the same recording count once toward hours (§3A). That means alternative segmentations are hour-free, but they must exist **before** the mid-term.
- The test set includes telephone audio (upsampled) and noise (§4). Real 8 kHz recordings upsampled to 24 kHz count as real records, not augmentation.

## Candidate corpora (licences checked 2026-10-13)

| Corpus | Licence | Notes |
|---|---|---|
| IndicVoices | CC BY 4.0 | 76% extempore, 15% conversational. **The conversational part is 8 kHz telephone.** Is the HF release L1 or L2? |
| SPRING-INX Hindi | CC BY 4.0 | 351 h; monologue + conversational; ~10% code-mixed; 8 kHz upsampled |
| Vaani (ARTPARK/IISc) | CC BY 4.0 | Spontaneous, image-prompted; only partly transcribed |
| Kathbath | CC0 / CC BY 4.0 | Read speech, with a noisy subset |
| MUCS Hindi (SLR103) | MS O-UDA | Read stories, 8 kHz |
| **Avoid** | | Gram Vaani (academic-only), Common Voice (no re-hosting), IndicTTS (custom licence), MUCS Hi-En SLR104 (CC BY-SA makes the whole dataset ShareAlike), Shrutilipi (rights chain unclear) |

## Research and measure, per corpus

- Hindi hours, speaker count, gender and region spread.
- **Segment-length distribution.** Are there long-form source files we can cut 10–60 s session-like clips from, or only short utterances?
- Telephone share and original sample rates.
- Code-mix rate: share of English words.
- Entity density: numbers, names, amounts per minute.
- Transcript level (L1 or L2), tag format, and quality on a 30-clip spot check.
- Conversational audio: is each side on its own channel, or mixed? If mixed, does diarisation (pyannote; check its licence and gating) give clean single-speaker cuts?
- Attribution text for `data_card.md`.

## Optional: own recordings

Only if the organisers allow it (PS analysis, question 1): a consent form, a phone-channel recording protocol, and scripts from W2.

## Done when

- An inventory table with the numbers above.
- A 2–5 h pilot pool of single-speaker 10–60 s clips with their existing transcripts, handed to W4.

## Findings, 2026-10-14 (W1)

Everything below was **measured** on files on disk unless it is marked *(paper)* or *(card)*. Scripts are in `w1/`; data is in `data/w1/` (gitignored).

- `tar_scan.py` streams a remote `.tar.gz` once, logs every member's duration and keeps a sample.
- `inventory.py` produces the per-corpus numbers.
- `cut_pilot.py` is the clip cutter.
- `asr_check.py` is the CTC-ASR QA.

### Inventory

**A. Access, licence, scale**

| Corpus | Licence (where checked) | Access | Hindi hours | Speakers | Verdict |
|---|---|---|---|---|---|
| **IndicVoices**, long-form release | CC BY 4.0 ([HF card](https://huggingface.co/datasets/ai4bharat/IndicVoices): `license: cc-by-4.0` and badge). The gate has no `extra_gated` terms; it is click-through only | **Ungated** `https://iv-release.objectstore.e2enetworks.net/dmu_release/v{1..5}_Hindi_{train,valid}.tgz`, linked from the HF README. Range requests work, ~45 MB/s. Train: 14.3 + 17.9 + 54.7 + 41.6 + 21.6 = **150 GB** gz. Valid: 1.4 GB | Valid: **11.8 h raw / 7.4 h transcribed** (all 553 files). Train sample (first 500 MB of each release): 92.2 h raw / 42.4 h transcribed. Full Hindi transcribed ≈ 600 h [INF: HF `hindi` train has 445,160 segments vs valid's 5,530 (card), × valid's 7.43 h] | Valid: 483 speakers, M 276 / F 274 / other 3. **Hindi belt only:** UP 199, MP 163, RJ 131, BR 60 files | **Primary source** |
| **SPRING-INX** Hindi | "released under CCBY 4.0 license", from the asr.iitm.ac.in JS bundle (the page is a SPA); also CC BY 4.0 on AIKosh for other languages | **Ungated** `https://asr.iitm.ac.in/SPRING_INX/data/SPRING_INX_Hindi_R{1,2}.tar.gz` (R1 26.8 GB, R2 13.8 GB, ~5–18 MB/s). R2 streamed in full | R2: **166.2 h raw / 157.6 h transcribed**, 667 recordings. R1 not streamed. Paper: Hindi total 351 h | Paper: 764 M / 639 F. The release has no speaker IDs: `utt2spk` is per utterance | **Monologues: yes. Conversations: no** (mixed channel) |
| MUCS 2021 SLR103 | OpenSLR names only a link (`msropendata-web-api.azurewebsites.net/licenses/f1f352a6…`), which **no longer resolves** (DNS fails 2026-10-14). The task file says MS O-UDA; **unverified** | Ungated OpenSLR. Train 4.4 GB, test 258 MB | Test 5.56 h (measured). Train 95.05 h (OpenSLR page) | Test 19; train 59 (page) | **Reject**: read, short |
| FLEURS `hi_in` | CC BY 4.0 (HF tag) | Ungated HF | **8.71 h**, 2,777 utterances | 1,577 M / 1,200 F utterances; no speaker IDs | **Reject**: read, single-utterance files |
| Lahaja | **Conflict:** HF tag `mit`, README "CC BY 4.0". Both allow redistribution | Gated (`auto`) | 12.5 h *(card)* | 132 speakers, 83 districts *(card)* | Better as W8c's eval set (it's a benchmark) (measured in Round 2) |
| Kathbath | **Conflict, verified by the scout** (`research/datasets.md`): CC BY 4.0 on HF vs **CC BY-SA 4.0 on AIKosh**. The README also says CC0 for packaging | Gated (`auto`); Hindi 16.6 GB parquet | 1,684 h across 12 languages *(card)* | 1,218 across 12 languages *(card)* | **HOLD for Part A** until the licence is resolved (W0 / organiser Q). Read IndicCorp sentences, short, so low value anyway (measured in Round 2) |
| Vaani | CC BY 4.0 (HF tag). The gate collects name, company, purpose | Gated (`auto` + form). `audio/Hindi` is 10.1 M utterances in **1.7 TB** of parquet (card YAML) | **1,003.78 h transcribed Hindi** in [Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) (CC BY 4.0, gated `auto`). Scout figure, not re-measured | 156K across all languages *(card)* | **Candidate #2 for real noisy audio.** [Vaani-Noise-Event](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset) tags can *select* real noisy clips (scout). Image-prompted spontaneous speech, utterance-level. Session shape measured in **Round 2** below: short segments, mostly < 10 s |
| Rural Women ASR v2 (AI4Bharat) | CC BY 4.0 in the README (scout); consent statement UNVERIFIED | **Open**, no gate (31.4 GB Hindi) | ≈87 h of chunks (scout); unique-source hours lower | 99.7% female; rural MP/UP/BR; phone mic; 48 kHz (scout) | **New candidate:** IV schema, so `load_iv`/`cut_pilot` nearly apply. Not sampled by W1 yet (disk budget). Phone-recorded, close to §4 |
| IndicVoices-R | CC BY 4.0 | Gated | 74.6 h (paper, via scout) | — | **Excluded.** It is the same speech as IV, enhanced into new files. **Never cut the same speech from both IV and IV-R**: hours would be double-counted (§3A union is per source file) |

**B. Measured properties**

| Corpus / part | Native rate | Long-form? File dur p10/50/90 | Segment dur p10/50/90 | 10–60 s windows (gap ≤ 2 s) | Code-mix | Number words / speech-min | Transcript level and tags |
|---|---|---|---|---|---|---|---|
| IV conversation (valid) | **8 kHz** (123/123) | Yes, 129 / 194 / 256 s | 0.5 / 2.1 / 8.0 s | 1.32 h of 6.69 h raw | 12.8% | 6.3 | See note 1 |
| IV extempore (valid) | 44.1 kHz | Yes, 9 / 70 / 202 s | 2.9 / 8.2 / 15.8 s | **4.18 h of 4.54 h** | 8.3% | 2.6 | See note 1 |
| IV read (valid) | 44.1 kHz | No, 1.1 / 5.1 / 22 s | — | 0.36 h of 0.57 h | **19.2%** | **17.5** (payment / Umang / IRCTC commands) | Note 1, plus the prompt text |
| SPRING-INX R2, M | 16 kHz file; **~15% band-limited ≤ 4 kHz** (5/33 sampled files, i.e. upsampled narrowband) | Yes, 598 / 620 / 910 s | 9.2 / 12.9 / 14.3 s, **contiguous** chunks | **60.1 h of 62.3 h** | 5.3% (Latin tokens) | 2.7 | See note 2 |
| SPRING-INX R2, C2/C3/C | 16 kHz; ~5% band-limited (C2, 1/21) | Yes, 833 / 931 / 1,712 s | 1.5 / 5.6 / 13.4 s | (mixed speakers) | 8.8% | 2.7 | See note 2 |
| MUCS test | 8 kHz | No, 3.6 / 5.1 / 7.0 s | = file | 0.02 h (8 of 3,843 utterances ≥ 10 s) | 0% | 1.9 | Prompt text; 386 sentences read ~10× each |
| FLEURS `hi_in` | 16 kHz | No, median 10.6 s, max 36 s | = file | (57% of utterances ≥ 10 s, but one sentence each) | 4% of utterances contain Latin | — | Raw and normalised prompt text; 18.9% contain ASCII digits |

1. IndicVoices tags and transcript levels:
   - `verbatim` is **L1**: non-standard spellings are kept (`नमश्कार`, `बोहोत`, `सून्य`, `कित्ते`), English is in Devanagari, there are no Latin glosses and no punctuation.
   - `normalized` is L2: standard spelling plus `[Latin]` glosses after English words.
   - Tags come in both `[x]` and `<x>` forms, 127 distinct. The most frequent are `inhaling`, `noise`, `persistent-noise-start/end`, `uhh`, `talking`, `unintelligible`, `hmm`, `stammers`, `umm`. There are 7.5–13 tags per 100 words.
   - Every released file's `verification_report.decision` is `excellent` (2,826/2,826). File-level flags are set as follows: `noise_persistent` 20%, `noise_intermittent` 16%, `echo_present` 3%, `chatter_*` 4%.
2. SPRING-INX tags and transcript level:
   - Kaldi format (`text`, `segments`, `utt2spk`), with no tags (the paper says they were removed).
   - **English words are in Latin script** ("electrical branch", "last year"), so the text is at least partly L2. W4 must transliterate.

**Code-mix rate:** IV's is exact: `[Latin]` glosses in `normalized` divided by words. SPRING's counts Latin tokens.

**Entity density** is a heuristic: number and currency words per minute of transcribed speech. Names aren't counted.

**Conversational channel layout:**

- **IndicVoices: separate channels.** Each call ships as `<uuid>_0.wav` + `<uuid>_1.wav`.
  - Measured on 622 complete pairs: identical durations (diff p90 = 0.00 s) and never the same `speaker_id` (0/622).
  - Where only the partner speaks (7,404 regions), this channel sits at a median of **−50 dB** re. the speaker; 20% is digital silence. But **19.4% of those regions are above −30 dB and 8.6% above −20 dB**: the partner leaks into this channel in a minority of files.
  - 32 of 2,273 files (1.4%) carry `speaker_id` 1 segments.
  - So W0's inference that "conversational files likely hold both parties in one channel" is **mostly false, but not safe to ignore**. See the one-speaker QC below.
- **SPRING-INX: mixed in one channel**, with contiguous segments that alternate between speakers, and no speaker labels. Usable only after diarisation.
  - pyannote `speaker-diarization-3.1` and `segmentation-3.0` are MIT but gated (`auto`). `speaker-diarization-community-1` is CC BY 4.0 and gated (all checked on the HF API).
  - Not needed now: IV channels and SPRING monologues already give far more single-speaker audio than we need.

**Attribution text (for `data_card.md`):**

- *IndicVoices*: Javed T. et al., "IndicVoices: Towards building an Inclusive Multilingual Speech Dataset for Indian Languages", ACL Findings 2024 (arXiv 2403.01926). AI4Bharat, funded by Bhashini, MeitY and Nilekani Philanthropies. CC BY 4.0, https://creativecommons.org/licenses/by/4.0/.
- *SPRING-INX*: Nithya R. et al., "SPRING-INX: A Multilingual Indian Language Speech Corpus by SPRING Lab, IIT Madras" (arXiv 2310.14654). Funded by MeitY. CC BY 4.0.

**Stated / inferred / ambiguous:**

- STATED (§3A, §10 R1): the licence must allow redistribution. CC BY 4.0 allows it with attribution, so §9.2 `data_card.md` must carry the attribution text above.
- INFERRED: the long-form IV archives carry the dataset's CC BY 4.0. The README that documents them sits under that licence, but the archives have no licence file of their own.
- AMBIGUOUS: whether IV *valid* may go into Part A. Licence-wise it may. Keeping it out leaves a clean real dev set (handoff to W8c, below).

### One-speaker QC (W0 handoff item 1)

The cutter (`w1/cut_pilot.py`, `runs()`) applies, per long-form file:

1. **Exclude whole files** with more than one `speaker_id` or a `chatter_*` verification flag.
2. **Drop segments, and break clips at them,** when a segment:
   - carries a second-voice or unintelligible tag (`talking`, `whisper`, `yelling`, `singing`, `unintelligible`); or
   - in a conversation, is **talked over by the partner by even 0.1 s**, according to the partner file's segment times.
3. **Break at gaps** (untranscribed time) that may hold a voice:
   - In general: louder than −30 dB re. the speaker **and** ≥ 6 dB over the file's noise floor (the p10 frame level). Field noise is not a voice.
   - If the partner talks in the gap, the gap must stay within 6 dB of the floor or below −45 dB re. the speaker.
4. **Skip conversation files whose partner JSON isn't on disk** (295 in the sample).

**Cost on the train sample:** 1,018 candidate clips, **5.39 h**: conversation 3.10 h, extempore 2.04 h, read 0.25 h.

- 12,253 segments were dropped and 8,881 gaps broke a run.
- Dropping a segment only when the partner is *audible* in this channel would give 7.88 h of conversation instead of 3.10 h. I chose the strict rule. Relaxing it is a team call.

**An ASR check can't prove "no second voice". The positive control failed:**

- Setup: CTC-ASR over 20 control clips cut *across* partner-only regions that the energy test called audible, from −30 dB up to +1 dB.
- It emitted **zero** characters in those regions.
- Inspection showed why. rec-900005's "+1 dB partner" is a line tone (74% of energy in 1–4 kHz, peak 1.4 kHz). rec-900001's −4 dB region is speech-shaped (peak 503 Hz) but the model decodes nothing from it: AEC-residual or echo-like leak.
- So the energy and partner-timing rule above is the guard, and ASR is only a backstop for untranscribed speech by the speaker.
- **A human should still listen** to the 20 pilot conversation clips (`data/w1/pilot/audio/`, `scenario` = Conversation). SPEC §1 makes one-speaker QA manual.

### Pilot and pool (handed to W4)

| Set | Clips | Hours (validator, union) | Mix | Duration p10/50/90 | Speakers |
|---|---|---|---|---|---|
| `data/w1/pilot/` (`manifest.jsonl` + `audio/`) | 50 (`rec-000001`… in pick order) | 0.315 h | 20 conversation (8 kHz), 20 extempore, 10 read (44.1 kHz) | 11.1 / 15.3 / 43.8 s | 40 |
| `data/w1/pool/` (includes the pilot) | **711** | **4.005 h** over 547 long-form files | 402 conversation, 245 extempore, 64 read | 10.9 / 15.1 / 41.5 s | 325 |

**Record format.** SPEC §8 stage format with SPEC §4 naming:

- `source_file = indicvoices/Hindi/v4/train/<uuid>_0.wav`. This is the long-form file as the archive distributes it, with offsets in it.
- `speaker_id = spk-indicvoices-<S…>`.
- `_rev` = archive URL + ETag.
- `_orig_text` = joined IV `verbatim`.
- `_w1.segments` = per-segment clip-relative times with both `verbatim` and `normalized` text.
- `_w1` also carries `orig_sample_rate`, scenario/task, gender, age, state, district and verification flags.
- `sentences` is **not** written, because `hi_text` is W4's (SPEC §8).

**IDs** come from `w1/ids.tsv`, which must be committed: it is the allocation register, and IDs are never reused (SPEC §4).

- `rec-000001`–`rec-000711` are allocated.
- Earlier dry runs, never handed out, used the same numbers before the register existed. Nothing downstream holds them.

**Validator** (`uv run validate.py … --partial`): pilot and pool both give **0 violations**, so there are no audio, timing or source failures.

- With the raw IV text in `hi_text` (first dry run), the 50 pilot records gave the expected W4 workload: `hi_final` 109, `hi_chars` 77 (brackets, Latin tags, `,`, `-`), `hi_nfc` 36 (precomposed nukta letters), `hi_space` 32.

**Rebuild:**

- Clip = ffmpeg soxr resample of the *whole* long-form file to 24 kHz, then `atrim` on sample indices, with bit-exact flags.
- **21/21 re-cut clips are byte-identical** (ffmpeg n9.0.1).
- The source bytes are pinned by archive ETag. Re-fetching needs only a range request on the first 500 MB of each archive, because that is where every pool file came from.

**Disk:** `data/w1` is at 5.9 GB of 10 GB. I deleted 3.06 GB of IV-train WAVs that no pool clip uses; their JSONs are kept.

### Transcript-quality spot check

**Method.** `w1/asr_check.py` runs `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` (MIT, wav2vec2-base, 4,200 h Hindi; greedy CTC on CPU) over the whole pool. It computes CER against IV `verbatim` (tags stripped) and flags speech outside the transcribed segments. I also read 30 pilot clips side by side (10 conversation, 10 extempore, 10 read).

**Caveat:** this checks the reference against an *independent model*, not against ears.

| Pool part | Clips | CER p50 / p90 | Untranscribed-speech runs (≥ 4 chars outside segments) |
|---|---|---|---|
| Conversation (8 kHz) | 402 | 0.152 / 0.596 | 0 |
| Extempore | 245 | 0.093 / 0.172 | 0 |
| Read | 64 | 0.073 / 0.142 | 0 |
| SPRING-INX R2 M (30 windows, scratch) | 30 | 0.159 / 0.526 | 0 |

The SPRING CER is an upper bound: its Latin English words are stripped from the reference.

**Reading the 30 pilot clips.** All 30 references are consistent with the audio. The disagreements fall into three groups:

1. ASR errors, mostly on 8 kHz audio and entities (`आर वन फ़ाइव` → `अरुण फाइप`).
2. IV's deliberate L1 spellings (`बोहोत`, `इशलिए`, `हे`, `शंगीत`).
3. Disfluencies the reference keeps (`शंगीत संगीत संगीत`).

Three items need ears:
- rec-000001 and rec-000022: the ASR loses the final phrase.
- rec-000021: the reference has `फैन` where both ASRs hear `फ्रेंड`.

**Read-task `verbatim` carries prompt punctuation and inconsistent acronym forms** (`पी.आर.ए.एन.`, `है,`, `ढूँढें।` vs `पी आर ए एन`). This is W4's problem.

**The worst conversation clips are mostly an ASR failure, not a transcript failure.** 18 of 402 have CTC CER > 0.8, with near-empty hypotheses even though in-segment energy (58–73 dB) and spectrum (~70% of energy in 300–1,000 Hz) look like the good clips. Re-run with `openai/whisper-large-v3-turbo` (MIT, `language=hi`, scratch script):
- On **14 of the 18**, Whisper matches the reference (CER 0.05–0.28).
- Two more (rec-000142, rec-000442) match apart from digits.
- **rec-000046:** both models hear a menu list (`…फ्लोरल…फिश टिक्का…बीपीक्यू प्लेट…`) that the reference omits. It is the only clip in the pool where the hypothesis is > 1.25× the reference length. **Incomplete transcript; W4 must fix or drop it.**
- **rec-000434:** possible mismatch after the first sentence.
- **rec-000177:** Whisper loops on noise. Undecidable.

**Lessons:**
- The vakyansh CTC model fails silently on about 4.5% of IV 8 kHz channels, so don't use it alone as W4's telephone QA.
- Whisper-turbo handles those channels but writes digits, so normalise before computing CER.

**Bug found on the way:** this checkpoint emits `<s>` as its CTC blank, although `config.pad_token_id` is `<pad>`. `asr_check.py` treats every special token as blank.

### Open questions for organisers (add to W0's list)

1. Are IndicVoices' **long-form** archives (the `dmu_release` URLs) covered by the dataset's CC BY 4.0? They are linked from the card but carry no licence file. [Ask AI4Bharat if Sarvam can't answer]
2. Is a real 8 kHz telephone recording, upsampled to 24 kHz, an acceptable "real record"? This is SPEC Q11; §4 says lower-bandwidth recordings are upsampled. It decides whether 57% (valid) to 77% (train sample) of IV raw hours are usable.
3. May the dataset hold a clip where the other party is faintly audible but unintelligible (channel leak), or must one-speaker mean acoustically clean (§3A)? This decides between 3.1 h and 7.9 h of conversation yield per ~90 h raw.

### Blockers and asks

- **HF token** (asked at the start of this session; nothing in `.env` yet). It's only needed to measure Kathbath, Lahaja and Vaani on samples. IndicVoices and SPRING-INX need no token.
- **Not done:**
  - SPRING-INX R1 (26.8 GB): not streamed, since streaming means pulling the whole tarball. A transcripts-only scan writes ≈ 0 to disk but downloads 26.8 GB. Ask before running it.
  - Human listening QA.

### Handoffs

- **W4:**
  - `data/w1/pool/manifest.jsonl` (711 clips, 4.0 h) and `data/w1/pilot/` (50).
  - Source text is in `_orig_text` and `_w1.segments[].verbatim`.
  - `normalized` gives the Latin spelling of each English word (for R6/R8), e.g. `आई [I] आर [R] सी [C] टी [T] सी [C]`.
  - Tag-to-Devanagari mapping as in SPEC §2.4. Expect both `[x]` and `<x>` forms.
  - Clips with `<unintelligible>` or second-voice tags are already excluded.
- **W5:** clip-relative segment times in `_w1.segments` are a free coarse alignment. IV boundaries are tight, which is why clips are padded 0.05–0.2 s.
- **W6:** read-task clips (`task_name` = Digital Payment / Umang / Bigbasket / Alexa Commands) are the entity-dense ones: 17.5 number words per minute, versus 2.6 in extempore and 6.3 in conversation.
- **W7:**
  - The rebuild needs only `cut_pilot.cut(source, start_ms, end_ms)` plus the archive at `_rev`.
  - Hosting the used long-form files in a private HF dataset (§11 FAQ) avoids depending on the E2E object store staying up.
- **W8c:** IndicVoices Hindi **valid** has 553 recordings, 11.8 h raw, 483 speakers, with L1 `verbatim` and a 57% telephone share. It's on disk in `data/w1/iv_full/`, kept out of the pool, and works as a real-audio dev set. Lahaja (gated) is the other candidate.
- **W0:**
  - Your inference about IV conversations needs a correction: IV conversations are per-party files, but with measurable leak (numbers above).
  - Suggest adding to SPEC §1 that one-speaker QA for conversation clips uses partner timings.

## Status — 2026-10-14

**Done:**

- **Inventory:** IndicVoices, SPRING-INX, MUCS and FLEURS measured on real samples. Lahaja, Kathbath and Vaani from cards only. Rural Women, Vaani transcription and IV-R from the scout.
- **IV conversation layout** verified on 622 `_0`/`_1` pairs (W0 handoff 1).
- **One-speaker QC** built, measured, and positive-controlled. The control showed that ASR can't serve as the second-voice guard.
- **Pilot:** 50 clips, 0.315 h. **Pool:** 711 clips, 4.005 h, single-speaker, 10–60 s, 24 kHz mono s16, with existing transcripts.
- Both pass `validate.py --partial` with 0 violations. 21/21 sampled re-cuts are byte-identical. IDs `rec-000001`–`rec-000711` are in `w1/ids.tsv`.
- 30-clip spot check, plus ASR QA over the whole pool with a Whisper cross-check.

**Numbers that drive planning:**

- **Supply.** IndicVoices Hindi long-form is 150 GB, ~600 h transcribed [INF]. SPRING-INX R2 has 62 h of monologue, with 60 h in 10–60 s windows (R1 unscanned). Vaani has 1,003.78 h transcribed (scout).
- **Supply is not the limit.** Real single-speaker 10–60 s audio exceeds the 200 h cap several times over. What binds is W4's conversion throughput and the strictness of the conversation one-speaker rule.
- **Yield after QC** on the IV train sample: 5.39 h of clips per 92 h raw.
  - Extempore yields 70% of raw.
  - Conversation yields 3.5% (strict rule; 8.9% with the audibility-gated variant, partner JSON permitting).

**Open questions:** see "Open questions for organisers" above (IV long-form licence, 8 kHz upsampled as real, faint-leak policy). Also the Kathbath licence conflict (HOLD).

**Blockers:** none for W1's deliverable.

- HF token: still absent. It only blocks sampling Vaani, Kathbath, Lahaja and the scout's gated sets.
- SPRING-INX R1 and Rural Women ASR v2 (31 GB, open) need streaming bandwidth. Disk stays near zero, but ask before pulling.

**Next, if W1 continues:**

1. A human listens to the 20 pilot conversation clips plus rec-000001/021/022/046/434.
2. Add a SPRING-M loader to `cut_pilot.py`. SPRING monologues are single-speaker 16 kHz with Latin-script English (more W4 work).
3. Sample Rural Women and Vaani for session shape.
4. Decide the conversation one-speaker rule with W0/organisers (Q above).

**Owned files:**

- `w1/`: `tar_scan.py`, `inventory.py`, `cut_pilot.py`, `asr_check.py`, `ids.tsv`. Commit `ids.tsv`.
- `data/w1/` (gitignored): `pilot/`, `pool/` (with `asr_check.tsv`), `iv_full/` (valid, for W8c), `iv_train/` (pool sources + all JSONs), `spring/` (R2 transcripts + 1.5 GB audio sample), `fleurs/`, `mucs/`. Total 5.9 GB.

## Round 2: gated sets measured, 2026-10-14

With HF access to Vaani-transcription-part, Vaani-Noise-Event, Kathbath and Lahaja, I measured all four on samples.

**Method.** `w1/hf_sample.py` does ranged parquet reads only. **No audio was written to disk:** clips are decoded in memory. There are two modes:

- **Audio sample:** a few row groups per shard, giving format, sample rate and duration from the embedded bytes.
- **`--columns` (metadata only):** exact byte ranges of the chosen column chunks (e.g. `audio.path`, `transcript`, `duration`) across every row group. That's about 0.15 MB per 480 MB shard.

The metadata indexes stay in `data/w1/hf/` (72 MB, no audio), and W4 can use them to pick segments. Full-corpus indexing ran at roughly 2–10 min per shard because of per-request overhead, so I stopped where the answer was clear. Coverage is stated per row.

**Shards are clustered** (by state, split, gender, language or annotation quality), so shares from partial reads aren't corpus-wide. Card totals are authoritative where given.

| Set | Sampled | Native format | Unit and session shape | Duration p10 / 50 / 90 / max | ≥ 10 s | Transcript level and tags | Code-mix | Number words / min | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| **Vaani-transcription-part, Hindi** (1,003.78 h, 661,593 rows, card) | Audio: 990 rows from 4 shards. Metadata: all 41 test shards = 92,734 segments, **146.2 h**, 9,956 speakers, 26 states / 129 districts | WAV 16 kHz mono (990/990) | See note 1 | 2.6 / 5.1 / 9.5 / 20.5 s | **7.7% of segments**, 24.0 h of 146 h (if representative ≈ 165 h corpus-wide [INF]) | See note 2 | 10.9% `{gloss}` / words | 6.6 | See note 3 |
| **Vaani-Noise-Event, Hindi** (154.6 h all languages, card) | Audio: 364 Hindi rows from 3 shards. Metadata: 28/182 shards = 13,943 rows, 4,464 Hindi = 7.8 h | WAV 16 kHz | **The same segments as Vaani-transcription-part** (same filename scheme, offsets in ms) | – / 5.8 / 9.9 / 20.3 s | 9.2% | Re-annotated: spans (`<horn> … </horn>`), commas, `<PAUSE>`. Only 2.3% of joined transcripts are byte-identical to the transcription-part text | – | – | See note 4 |
| **Kathbath, Hindi** | Audio: 200 rows from 2 shards. Metadata: 15/34 shards = 43,004 rows, 59.7 h (these shards are all female, 53 speakers) | FLAC 16 kHz mono | **Read IndicCorp sentences**, one per file; every sentence unique (43,003/43,004) | 3.5 / 4.8 / 6.6 / 17.6 s | **0.4%** | Prompt text; no tags, no Latin, no digits | 0 | 1.95 | **HOLD stands** (licence). Even cleared, it's unusable for 10–60 s sessions |
| **Lahaja** | **All** 6,152 rows (metadata) + 200 rows of audio | WAV 16 kHz mono | IV tooling: segments of 2,243 recordings (IV-style numeric `fname`). Segments only, no long-form | 1.1 / 6.3 / 14.4 / 27.8 s | 26% (6.2 h) | IV-style `verbatim` (L1, `[uhh]`, `[inhaling]`; 2.2 tags/100 words) + `normalized` with `[Latin]` | 15.2% | 6.3 | **Eval only (W8c)**: 12.32 h, 132 speakers, 83 districts, mostly non-native Hindi speakers |

1. **Vaani segments.** The filename encodes `…_<speaker>_<seq>_<image>_<start ms>_<end ms>.wav`. Each segment is a cut of one image-description recording; the parent isn't distributed here.
   - The parent recording is at most ~30 s (p50 of the maximum segment end is 13.9 s). 88% of parents contribute a single transcribed segment.
   - Gaps between a parent's transcribed segments: p50 1.1 s, p90 9.9 s.
2. **Vaani transcript level.** Spontaneous L1-style text. English is in Devanagari with a `{latin}` gloss (`जंगल {jungle}`). Tags are paired spans (`<noise> … </noise>`, `<static noise>`) plus `[breathing]` and `<pause>`.
   - 8.1% of segments carry second-voice or unintelligible tags.
   - **24.1% of segments end in `--`** (99% at the segment end): the speaker was cut off mid-utterance.
3. **Vaani verdict: a usable real source, but only as short records.**
   - A record = one segment file as distributed (`source_file` = that file, `start` 0). Only segments ≥ 10 s count: 7.7%, minus those with a trailing `--` or a second-voice tag.
   - Windows over the parent recordings would give 35 h per 146 h, but only 12–17 s long. That needs the main Vaani repo, which **still returns 403**: accept the terms at https://huggingface.co/datasets/ARTPARK-IISc/Vaani if we want that.
4. **Noise-Event verdict: a selector over Vaani segments, not a separate source.**
   - **The join is on the audio filename.** 555 of 4,464 Hindi noise clips (12.4%) are in the test-split index alone. If every clip were in the transcription set, 17.8% (the test split's share of rows) would match, so ≈ 70% are [INF].
   - Noise timestamps are relative to that same segment, so they carry over to a record cut from it unchanged.
   - Hindi rows in the indexed part: 60% `verified_timestamps`, 29% unverified, 11% none (the card gives 12% verified overall, so these shards are quality-clustered).
   - Categories: `human_non_speech` 2,787, `vehicle_traffic` 1,260, `animal` 846, **`baby_child` 521**, `phone_signal_alarm` 225, `singing_music` 144.
   - 22.5% have a second voice (a tag or `baby_child`).
   - **Usable as records:** clips ≥ 10 s with no second voice and no trailing `--` are 293 of 4,464, i.e. 1.0 h of 7.8 h (13%).

**What this changes for Part A:**

- **IndicVoices stays the primary real source.** It has long-form, single-speaker 10–60 s windows, and SPRING-INX monologues are the second.
- **Vaani's 1,003.78 h is mostly not session-shaped.** About 1 in 13 segments is ≥ 10 s, a quarter end mid-utterance, and none exceeds ~20 s.
  - It is useful for short, noisy, regionally diverse real records (26 states in one split vs IV's Hindi belt). Noise-Event lets us pick the noisy ones.
  - §3A has no minimum clip length, but §4 sessions run 10–60 s [INF: short records are allowed but match the test less well].
- **Lahaja → W8c** as an accent-diverse real eval set (and not in Part A, to keep it clean).
- **Kathbath:** HOLD, low value.

### Status, 2026-10-14 (round 2)

**Done:** all four gated sets measured on samples, and the inventory rows above filled. `w1/hf_sample.py` added (ranged audio sampling plus the metadata-only `--columns` mode).

- Disk: +72 MB (metadata only). `data/w1` total ≈ 6.0 GB. No GPU used.

**Not done:**

- Full-corpus Vaani and Noise-Event indexes, which would give exact corpus-wide shares and the exact join rate. They need roughly 2–4 h of throttled requests. Run them overnight if the exact numbers matter:
  - `HF_THREADS=16 uv run … w1/hf_sample.py "<glob>" --columns audio.path --out …`
- Kathbath index stopped at 15/34 shards.

**Asks:**

- Accept the terms on the **main** Vaani repo (403 today) if we want 12–17 s windows over parent recordings.
- W0: add a Vaani `source_file` rule to SPEC §4. I propose `vaani/<segment filename>`, with the parent recording and offsets in `_w1`, so hours union on the segment files is exact and parents are traceable.

## Round 3: drop-not-review recut, 2026-10-14

**Policy** (`docs/superpowers/specs/2026-10-14-s3-ingestion-cleaning-design.md` §3.3): one human, no annotators, so any clip that would need a listener is **dropped**, not reviewed.

**Cutter change** (`w1/cut_pilot.py`):

- **New default.** Segments tagged `stammers`, `uh-huh` or `ugh`, in either `[..]` or `<..>` form and any case, now behave like `unintelligible`: the segment is dropped and the clip breaks there. `--keep-disfluency-tags` turns this off.
  - The match is exact: `[laugh]`, `[ughh]` and `[uh-uhh]` don't match. Self-checks cover this.
- **`w1/retired.tsv`** is the new tracked **burn register** (id + reason). Commit it with `ids.tsv`.
  - A burned id is never issued again, and a normal (non-recut) run skips any span whose registered id is burned.
- **`--recut`** does the following:
  - Moves `manifest.jsonl` to `manifest.v1.jsonl` and `pilot/manifest.jsonl` to `pilot/manifest.v1.jsonl`.
  - Keeps every v1 clip not in `retired.tsv` **byte-for-byte** (same id, same span).
  - Writes retired clips to `pool/rejects.jsonl` with `_reject_reason`, and deletes their audio. It can be re-cut from the source span if ever needed.
  - Cuts **replacements** from the long-form files of the retired clips under the new rule:
    - segments under kept clips count as used, so they are never cut twice and pads stop at their edge;
    - only windows that overlap a retired span are kept;
    - each replacement gets a fresh id from `ids.tsv`.
  - A retired clip with nothing droppable inside (rec-000046, incomplete transcript) is **burned whole**, and nothing is re-cut from its span.
- **Retired:** W4's P0 list (`data/w4/review.tsv`), 103 ids. That's the 102 clips carrying stammers (84), ugh (12) or uh-huh (7) tags, one with both, plus rec-000046.

**Before and after:**

| Scenario (native rate) | v1 clips / h | Retired clips / h | Kept | Replacements clips / h | v2 clips / h |
|---|---|---|---|---|---|
| Conversation (8 kHz) | 402 / 1.716 | 55 / 0.263 | 347 | 9 / 0.044 | 356 / 1.497 |
| Extempore (44.1 kHz) | 245 / 2.041 | 48 / 0.461 | 197 | 31 / 0.200 | 228 / 1.780 |
| Read (44.1 kHz) | 64 / 0.248 | 0 / 0.000 | 64 | 0 / 0.000 | 64 / 0.248 |
| **Total** | **711 / 4.005** | **103 / 0.724** | **608** | **40 / 0.244** | **648 / 3.526** |

- **Net loss:** 0.479 h, **12.0% of pool hours**, within the ≈18% the spec budgets. Speakers fall from 325 to 301.
- **Replacements** are `rec-000712`–`rec-000751`, 10.6–41.6 s each. Burned ids are `rec-…` per `w1/retired.tsv` (inside 000001–000711).
- **Pilot:** 50 → **40** (10 pilot clips retired; no replacements added). The old list is `pilot/manifest.v1.jsonl`.

**Validator** (`uv run validate.py … --partial`):

- Pool: **648 records, 0 violations**, real 3.526 h over 510 source files.
- Pilot: 40 records, 0 violations, 0.243 h.
- `manifest.v1.jsonl` no longer validates, because the rejected audio is gone. That's by design: it's an archive.

**Checks** (all pass):

- The 608 kept lines are byte-identical to v1.
- `rejects.jsonl` is exactly the burned set, every reject has a reason, and no burned id is in the manifest.
- Every burned id stays registered in `ids.tsv`; 751 rows, all unique.
- Zero replacements carry a droppable or second-voice tag.
- Zero overlapping clips within any source file, and nothing overlaps rec-000046's span.
- `pool/audio` and `pilot/audio` contain exactly the manifest's files.
- **Replacement sanity:** the CTC-ASR check on the 40 replacements (`pool/asr_check_replacements.tsv`) gives CER p50 0.072 (conversation, 9 clips) and 0.075 (extempore, 31 clips), with **0** untranscribed-speech runs, so the offsets are right.
  - `pool/asr_check.tsv` still describes v1. I left it in place in case W4 reads it by path.

### Status, 2026-10-14 (round 3)

**Done:** drop rule (default on, with a flag), burn register, recut. Pool v2 is 648 clips / 3.526 h and validates with 0 violations.

**Handoffs:**

- **W4:** P0 is resolved by retirement. The 40 replacements need your normal pass: conversion, then the P1 ASR-disagreement and P2 entity checks. Under §3.3, a P1 hit means the clip is retired here too: send me the ids and I'll add them to `retired.tsv` and re-run `--recut`.
- **W7:** ids listed in `w1/retired.tsv` are burned for good. The G3 freeze should treat them as "never released", not "removed".

**Note:** from now on, fresh cuts drop these tags at cut time, so this P0 class shouldn't recur.

## Round 4: pinned-ffmpeg cut stage (S3 plan Task 10), 2026-10-14

**Cutter change** (`w1/cut_pilot.py`; not committed, the master commits):

- **All audio goes through `pipeline/ffmpeg.cut`, the pinned BtbN `n9.0.2-17-g2a571b6068` build, sha256-checked.**
  - `cut()` is now a thin ms→s wrapper over it, so fresh cuts, `--recut` and the stage mode all use the pinned build. The system ffmpeg is no longer called.
  - The ms→s→ms round trip is exact; a self-check covers it.
- **Stage contract mode:**
  - `uv run w1/cut_pilot.py --in RECORDS.jsonl --audio-dir DIR --out OUT` re-cuts each record's existing `[source.start, source.end]` from `DIR/<source.source_file>`.
  - `DIR` mirrors the bucket's `raw/` layout, as `pipeline/rebuild.py` does. Locally, `data/w1/raw/indicvoices → ../iv_train` (a symlink in gitignored `data/`).
  - Outputs follow the contract:
    - `OUT/records.jsonl`: input lines verbatim, so ids, intervals and `_rev` are unchanged.
    - `OUT/audio/<id>.wav`.
    - `OUT/rejects.jsonl`: `{"id","stage":"cut","rule","detail"}`. Rule `retired` holds every burned id from `w1/retired.tsv` with its reason, and `source_missing` is for a missing source.
  - Resume and torn-line repair work as in `pipeline/audio_qc.py`.
  - A PEP 723 header (numpy) makes `uv run w1/cut_pilot.py` work.

**Re-cut of the whole pool** into `data/w1/pool_pinned/`, from `data/w1/pool/manifest.jsonl` (648 clips) in 30 s:

| Check | Result |
|---|---|
| `records.jsonl` vs `pool/manifest.jsonl` | **byte-identical**: same 648 ids, intervals, `_rev` |
| `rejects.jsonl` | 103, all `cut`/`retired`, exactly the `w1/retired.tsv` set |
| `validate.py --partial --audio-dir data/w1/pool_pinned` | **648 records, 0 violations**, real 3.526 h over 510 source files |
| `pipeline/ffmpeg.cut` called twice per clip, directly from the record's seconds | **648/648 identical to each other and to `pool_pinned/audio`** |
| Resume (same command again) | no-op: `already done: 751`, files unchanged |
| vs the old `pool/audio` (system ffmpeg n9.0.1) | **0/648 byte-identical**: every clip differs by at most **1 LSB**, on 0.1% of samples (p50; max 0.2%) |

**Consequences:**

- **`data/w1/pool_pinned/` supersedes `data/w1/pool/audio` for anything rebuilt or released.** The old audio can never be reproduced by `rebuild.py`.
- Text stages keyed by id carry over unchanged: W4's transcripts and review list.
- Audio-derived measurements on the old audio differ by at most 1 LSB, which is acoustically irrelevant. Recompute anything that stores audio hashes.
- Disk: +585 MB (`pool_pinned`). The machine's free space is down to 8.4 GB because of other streams. `data/w1/pool/audio` (≈ 0.6 GB) can be deleted once the master has seeded `stages/cut/` from `pool_pinned`.

### Status, 2026-10-14 (round 4)

**Done:** cutter on the pinned ffmpeg, plus a stage-contract mode. `data/w1/pool_pinned/` holds 648 records with audio and 103 retired rejects. It validates with 0 violations, and every clip is reproduced byte-for-byte by `pipeline/ffmpeg.cut`.

**For the master** (Task 10 Step 3):

- Seed `stages/cut/<run>/` from `data/w1/pool_pinned/` with `s3.upload_dir`, not from `pool/` or `pilot/`. Those still hold system-ffmpeg audio.
  - The 40-clip pilot is a subset of these ids. Its pinned audio is `pool_pinned/audio/<id>.wav` for the ids in `data/w1/pilot/manifest.jsonl`.
- Files to commit (all still uncommitted): `w1/cut_pilot.py` (rounds 3–4), `w1/ids.tsv` (+40 replacement ids, round 3), `w1/retired.tsv` (burn register, round 3), `w1/hf_sample.py` (round 2), `tasks/W1-real-audio.md`.
