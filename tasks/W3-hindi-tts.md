# W3: Hindi TTS and TTS QA

**Goal:** turn W2 scripts into clean synthetic Hindi audio with as many voices as possible, and catch every bad render before it reaches the dataset.

## PS rules

- 24 kHz mono 16-bit WAV; one speaker per clip; ≤60 s (§3A).
- **Hours count per generated file**, including the same text in another voice (§3A). Voice diversity is free when the texts differ, but script duplication costs full hours. One generated file = one clip.
- TTS output must be checked, and bad output filtered or regenerated: mispronounced names and numbers, skipped or repeated words, artefacts (§3A).
- Synthetic records carry `source.model` and `source.voice`; `source.corpus` = where the text came from (§5.2).
- Archive every kept output. The rebuild must never re-synthesise (§3A reproducibility).

## Engine candidates

- **Bulbul v3 hi-IN** (Sarvam):
  - ~37 voices; ₹30 per 10k characters (beta pricing); 2,500 characters per request; no seed.
  - **The default sample rate is 22,050 Hz, so always set `speech_sample_rate=24000`.**
  - Outputs can't be published (§10 R3).
  - Docs: docs.sarvam.ai.
- **Open-weight:** IndicF5, Indic Parler-TTS, others. Check licence, number of voices, and speed on an H100. Zero-shot voice cloning of real speakers raises a **consent** risk.

## Research and measure

- On 200 pilot scripts per engine, measure:
  - round-trip CER;
  - number and name accuracy;
  - code-mix pronunciation;
  - artefact rate;
  - how naturally fillers and self-corrections sound.
- **TTS input vs transcript:** do English words read better from Devanagari or Latin input? The transcript must be Devanagari and must match what was *actually* spoken.
- Prosody of digit strings and fractional numbers.
- Cost per hour of audio, including the regeneration rate.

## QA pipeline

- Round-trip ASR using a **different model family** from the TTS (otherwise their errors correlate).
- Entity match on numbers and names.
- Duration-per-character outliers.
- Skip/repeat detection via W5 alignment.
- Clipping, long-silence and artefact detectors.
- A regeneration loop. Rejected takes are never referenced.
- Tune the thresholds on a hand-labelled good/bad set.

## Done when

A TTS-choice memo with measured error rates, the QA pipeline with tuned thresholds, and the pilot audio handed to W5 and W6.

## Findings, 2026-10-14 (W3)

Everything below was **measured** on files in `data/w3/` (gitignored) unless it is marked (S) = source's own claim, or INFERRED. Scripts are in `w3/`; each one carries an `assert` self-check that runs on every invocation.

| File | What it does |
|---|---|
| `w3/probe.tsv` → `w3/probe.py` | 100 provisional probe items → `data/w3/probe.jsonl`. Markup `{देवनागरी\|latin}` marks English words and `[..]` marks entity spans. Commas are TTS pause hints, not transcript. Every `hi_text` passes the `validate.py` SPEC §2 rules. **A bake-off instrument, not W2 content** |
| `w3/render.py` | Bulbul / Magpie / Parler renderer, archive-first. The key is a hash of (engine, voice, params, input text). Writes `data/w3/archive/<engine>/<key>.wav` (bytes as returned) + `.json` (request without the key, `request_id`, timing), then indexes it in `data/w3/renders.jsonl`. Never re-requests an existing key |
| `w3/qa.py` | `asr` (cached hypotheses per model) · `score` → `qa.tsv` · `controls` + `report` (positive controls) · `summary` (tables below) · `spotcheck` (the list for human ears) |
| `w3/pilot.py`, `w3/ids.tsv` | QA-kept renders → synthetic stage records. Ids are allocated once per archive key, from `rec-400000` |

**Probe set** (`data/w3/probe.jsonl`): 100 items over 8 domains plus 10 stress items; 11,981 characters (no spaces); 357 entity spans; 511 English spans. Tags: lakh/crore, डेढ़/ढाई/सवा/साढ़े/पौने, irregular 21–99, digit strings (Hindi and English digits, डबल/ट्रिपल), dates/times, names/places/brands, acronyms, fillers, self-corrections, negation, tag ना, conditionals. Renders run 10–20 s (p50 12 s), shorter than the 10–30 s target. Long-form (3 items joined, ~40 s) is queued for Bulbul only and **not yet rendered** (see Spend).

### Engines tested

| Engine | Licence (checked 2026-10-14) | Voices used | How it ran (pinned) | Output |
|---|---|---|---|---|
| **Bulbul v3** hi-IN | API. §10 R2 (declare), §10 R3 (never publish) | 12 of 37: ritu priya simran kavya ishita roopa / aditya rahul amit manan rohan kabir | REST, `model="bulbul:v3"`, `speech_sample_rate=24000`, `output_audio_codec=wav`, default temperature 0.6 (plus a 0.3 test), pace 1.0 | 24 kHz mono s16 |
| **Magpie TTS Multilingual 357M** (NVIDIA) | NVIDIA Open Model License, text fetched: "Models are commercially usable… NVIDIA does not claim ownership to any outputs". Attribution is required only when redistributing the model. **Rights terminate if a guardrail is bypassed.** Ungated | All 5 fixed voices (Aria Jason John Leo Sofia). The card says "4 proprietary voices and 1 public voice", all English speakers. Zero-shot cloning was removed from this release (S) | NeMo-Speech.cpp v0.2.0 prebuilt **CUDA** archive (SHA-256 checked), GGUF `v2607` f16 at HF rev `19806879`, NanoCodec rev `fc00890b`, `--language hi --seed 1`. **Seeded, so reproducible**, unlike Bulbul | 22,050 Hz s16 |
| **Indic Parler-TTS** (AI4Bharat) | Apache-2.0, gated (access granted) | Rohit, Divya, Aman, Rani (the card's 4 Hindi speakers, chosen by name in the text description) | `parler-tts` git `d108732`, model rev `660bc53c`, fp16 on the RTX 3050, `do_sample`, `torch.manual_seed(1)` | 44.1 kHz |

**Not run:**
- IndicF5: code has no licence; cloning only.
- pocket-tts-hindi (CC BY 4.0, 219 MB, CPU): cloning only, so it needs a consented voice prompt. A team member's own voice would do. It's a cheap route to more voices.
- VoxCPM2 (Apache-2.0, Hindi, voice design without cloning): its 5 GB of weights exceed this budget.
- Veena: 7.5 GB.
- XTTS / F5: NC.
- OmniVoice: weights CC BY-NC.

### Round-trip ASR (the QA instrument)

- **ASR 1: IndicConformer-600M multilingual** (MIT), RNNT, CPU ONNX, using W4's local copy. It's the best measured Hindi ASR (W4). It never emits Latin or digits.
- **ASR 2: vakyansh wav2vec2 Hindi CTC** (MIT).
  - Both ASRs come from different families than all three TTS engines.
  - INFERRED: IndicConformer and Parler both come from AI4Bharat with IndicVoices-like data, so their errors may correlate on Parler. vakyansh is the independent check there.
- **New ASR failure mode:** vakyansh silently drops everything after some pause in clips longer than ~10 s.
  - Rate: 4/102 Bulbul, 11/140 Magpie and 23/140 Parler renders came out at under 70% of the reference length (IndicConformer: 0, 1, 0).
  - Decoding ≤ 7 s chunks, cut at the quietest 20 ms frame, fixes it (CER 0.31–0.62 → 0.007–0.08 on the six I checked). `qa.py` now chunks.
  - This is probably W1's "the ASR loses the final phrase" (`w1/asr_check.py` decodes whole clips).
- **Fillers can't be verified by ASR.** W4 measured 0% filler retention, and both ASRs here drop or rewrite अ/अं/उम्म/हम्म too. QA therefore never counts a filler as a skip, and filler clips go to human ears.
- **Disagreement saved two false conclusions:**
  - Magpie p001 looked like babble under vakyansh (CER 0.62), but IndicConformer heard it at CER 0.018.
  - Parler "stops after the first sentence" was vakyansh truncation too.

### Bake-off results (Devanagari input, default params, single items)

Measured. `CER` = round-trip CER against the input transcript, after nukta/candrabindu/visarga folding. Entity and English-span hits use a number-aware match, where one wrong number word is a miss. A defect counts only when **every** non-truncated ASR shows it.

| Engine | Renders | Audio h | CER IndicConformer p50 / p90 | Entity hit | Number-entity hit | English-span hit | Agreed skip | Agreed insert | QA reject | s/char p50 | Wall s per audio s |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Bulbul v3 | 72 | 0.26 | 0.016 / 0.048 | 96.6% | 95.3% | 98.9% | 4.2% | 0.0% | **18.1%** | 0.091 | 0.28 (API) |
| Magpie 357M | 100 | 0.33 | **0.010 / 0.027** | **98.6%** | **98.2%** | **99.2%** | 1.0% | 7.0% | **11.0%** | 0.095 | 0.78 (RTX 3050, model load per clip) |
| Indic Parler | 100 | 0.34 | 0.020 / 0.077 | 94.1% | 92.7% | 97.1% | 7.0% | 5.0% | **40.0%** | 0.095 | 1.29 (RTX 3050 fp16) |

Reject reasons (renders can have several):
- Bulbul: entity miss 8, English-span miss 4, CER 3, skip 3.
- Magpie: insert 7, entity miss 5, English-span miss 4, skip 1, CER 1.
- Parler: entity miss 19, CER 15, English-span miss 13, skip 7, insert 5, leading/trailing silence > 1 s 4, ≥ 2 substitutions 2, pause 1.

**The Bulbul vs Magpie gap (18% vs 11%) is within noise:** n = 72/100 gives a binomial SE of ≈4–5 points. Per voice, n = 6 for Bulbul, too few to rank voices. Parler is clearly worse.

By phenomenon (number-entity hit for numeric tags, entity hit for names; then QA reject):

| Tag | Bulbul | Magpie | Parler |
|---|---|---|---|
| digit strings | 92.5% · 12.5% (n 24) | 95.2% · 22.6% (31) | 85.5% · 51.6% (31) |
| lakh | 93.3% · 35.7% (14) | 100% · 5.9% (17) | 97.5% · 29.4% (17) |
| डेढ़/ढाई/सवा/साढ़े/पौने | 94.4% · 28.6% (14) | 100% · 6.2% (16) | 96.2% · 31.2% (16) |
| 21–99 | 98.2% · 9.1% (22) | 97.4% · 8.0% (25) | 96.2% · 32.0% (25) |
| dates | 90.0% · 28.6% (14) | 100% · 6.2% (16) | 95.3% · 56.2% (16) |
| times | 92.5% · 28.6% (14) | 97.4% · 23.1% (13) | 94.9% · 30.8% (13) |
| acronyms (spelled letters) | 87.5% · 16.7% (18) | 86.2% · **37.5%** (16) | 86.2% · 50.0% (16) |
| names (entity hit) | 98.4% · 17.9% (28) | 97.7% · 15.6% (32) | 92.4% · 56.2% (32) |
| self-corrections | 95.5% · 22.2% (18) | 100% · 5.3% (19) | 98.0% · 31.6% (19) |

**Failure modes seen.** Each was confirmed by both ASRs unless stated; none has been confirmed by ear yet.
- **Bulbul drops spans.** p009/simran says "सम है" for "सम एश्योर्ड बीस लाख है", losing an entire amount. It did this with both the Devanagari and the Latin input, which makes it reproducible for that text and voice.
- **Bulbul number words (ASRs disagree, so this needs a listen):**
  - p013/ritu इकतीस: IndicConformer hears इक्कीस, vakyansh hears "एक तस".
  - p007/ritu "सात" in a PNR: IndicConformer hears साठ.
- **Magpie inserts a short syllable between spelled digits/letters.** p003: "छह चार, आठ एक" → "चार ए आठ" / "चार टी आठ". That's 16% of digit renders and 25% of acronym renders.
- **Magpie's voices are English speakers reading Hindi** (card, S). Intelligibility is measured; accent and naturalness are **not**.
- **Parler:**
  - more entity misses;
  - 4% leading/trailing silence over 1 s;
  - name-based speaker control looks unreliable: "Divya" (female on the card) has a median F0 of 137 Hz.
- **Voice gender can't be read from the names.** Median F0 (autocorrelation):
  - Magpie: "Aria" 123 Hz and "Sofia" 130 Hz, but "Jason" 179, "John" 181 and "Leo" 200 Hz. Either the CLI maps names to different speaker indices than `speakers.json`, or my estimator is wrong.
  - Bulbul manan (167 Hz) and rohan (185 Hz) are ambiguous.
  - **Hindi verbs inflect for speaker gender** (rec-400000: male voice aditya says "बोल रही हूँ"). So voice gender must be verified by ear and matched to the script.

### Devanagari or Latin input for English words? **Devanagari.**

Measured on pairs with the same text and voice, only the input script changed:

| Engine | Pairs | English-span hit Dev → Lat | CER p50 Dev → Lat | QA reject Dev → Lat |
|---|---|---|---|---|
| Bulbul | 18 | 98.5% → 97.0% | 0.011 → 0.016 | 22.2% → 38.9% |
| Magpie | 40 | 99.0% → 94.3% | 0.009 → 0.022 | 12.5% → 45.0% |
| Parler | 40 | 96.5% → 84.8% | 0.018 → 0.050 | 42.5% → 77.5% |

- **Mechanism.** With Latin input the engine picks its own pronunciation:
  - Bulbul reads "September" as सेप्टेम्बर where the script's Hindi-English form is सितंबर (p006);
  - "Amazon" as एमेज़ॉन (p015).
- §3A requires the transcript to be what was *actually said*, so Latin input makes the transcript unknowable before synthesis. Every render would need its English words re-transcribed.
- Caveat: the reference is the Devanagari spelling, so the metric favours Devanagari by construction where the two pronunciations differ. The decision stands anyway: **the transcript is fixed only if the input pins the pronunciation.**

### Other settings

- **Bulbul temperature 0.3 vs the default 0.6** (12 paired renders): CER p50 0.017 → 0.003; QA reject 2/12 → 0/12. INFERRED: use 0.3 for Hindi. Naturalness is unmeasured; §3B fixes only the *English* target settings, so the Hindi side is free.
- **Numbers:** verbalise them in the script. All three engines read pre-verbalised लाख/करोड़/डेढ़/पौने/21–99 at 93–100% number-entity hit. Bulbul v3 auto-preprocesses its input (docs, S), so digits in the input would make the spoken form unknowable, like Latin.
- **Commas** in the input are pause hints: they aren't words (SPEC R1), so they're stripped from the transcript.
- **Long-form and the 60 s cap:** **not measured** (Bulbul credits ran out before the long-form jobs).

### Cost per audio hour

- **Bulbul**, measured:
  - 13.5 billed characters per audio second → **₹146 per audio hour** at list price (₹30 / 10K chars; billing-unit count unconfirmed against the dashboard).
  - At the measured 18% QA reject: ≈₹178 per *kept* hour, so 100 kept hours ≈ ₹18K. That's less if temperature 0.3 really cuts rejects.
  - Mean API latency 3.5 s per request (0.28 s per audio s).
- **Magpie:** 0.78 wall-s per audio-s on the shared RTX 3050 (a process per clip, model load included). The CPU build ran at 4–12× slower than real time under the shared load. H100: unmeasured.
- **Parler:** 1.29 s per audio s on the RTX 3050 in fp16.

### QA pipeline and thresholds

There are no ears here, so the hand-labelled good/bad set is replaced by **positive controls** (`qa.py controls`). They are 231 known-bad copies of 40 renders that both ASRs heard at CER ≤ 0.05:
- **skip:** 0.6 s cut out;
- **repeat:** 0.9 s duplicated;
- **pause:** 2.5 s of silence inserted;
- **clip:** ×6 gain;
- **num:** one number word in the reference changed to a neighbour (इकतीस → बत्तीस), i.e. "the TTS said the wrong number";
- **en:** one English word in the reference swapped for another.

Controls are never records.

| Rule (`w3/qa.py`) | Threshold | Why |
|---|---|---|
| Round-trip CER, best ASR | > 0.06 → reject | At 0.15 only 20/40 repeats were caught; at 0.06, 33/40 |
| Entity span (number-aware) | missed by every ASR → reject | A fuzzy distance alone calls one wrong digit in a 12-digit string a hit (2/35 edits) |
| English span | missed by every ASR → reject | Without it, 2/37 English swaps were caught; with it, 36/37 |
| Agreed skip (ref words with fewer hyp words, fillers excluded) | any → reject | — |
| Agreed insert (±1 word position) | any → reject | Catches Magpie's inter-digit syllables |
| Agreed word substitution | ≥ 2 → reject; 1 → human review | Same-value number spellings (छह/छः) and fillers excluded |
| ASR truncation guard | an ASR under 70% of the reference length, while another is ≥ 85%, is ignored for that render | vakyansh failure above |
| Seconds per char vs the engine-voice median | outside 0.6–1.6× | Never fired on real renders |
| Longest internal pause / leading or trailing silence | > 1.5 s / > 1.0 s | — |
| Clipping | > 0.01% of samples at full scale | — |

| Control | Caught |
|---|---|
| clip | 40/40 |
| pause | 40/40 |
| number swap | 33/34 |
| English swap | 36/37 |
| skip | 35/40 |
| repeat | 33/40 |

- **False-reject upper bound:** 12% of the 213 Devanagari renders that IndicConformer heard at CER ≤ 0.03 still get rejected. That set still holds real defects, so 12% is an upper bound.
- **Regeneration loop: not built yet.** Bulbul has no seed, so a retake needs a `take` field in the archive key that is *not* sent to the API. Add it with the bulk run; rejected takes stay archived and are never referenced (SPEC §4).

### Human spot-check list (`data/w3/spotcheck.tsv`, 41 items; `qa.py spotcheck` regenerates it)

- **21 × voice:** one clip per engine-voice, for gender (the F0 contradictions above) and for accent, especially Magpie's English-speaker voices reading Hindi. Before any Magpie record is released, also run a 10-clip naturalness listen.
- **14 × filler:** pilot records whose transcript has अ/अं/उम्म/हम्म. Was the filler actually voiced? No ASR can tell.
- **6 × substitution:** kept renders where both ASRs heard one word differently.
- **Also needs ears:** the Bulbul number cases where the two ASRs disagree (p013 ritu, p007 ritu).

### Pilot audio, handed to W5 and W6

`data/w3/pilot/` holds 50 records (33 Bulbul, 17 Magpie; 10.7 min; 0.179 h synthetic):
- ids `rec-400000`–`rec-400049`, in `w3/ids.tsv`;
- `uv run validate.py data/w3/pilot/manifest.jsonl --audio-dir data/w3/pilot --partial` → **50 records, 0 violations**.

What's in each record:
- **Text and source:** `hi_text` is the input transcript. A render was kept only if QA found no skip, insert, substitution or entity miss. `source.corpus = "w3-probe-v0"` (provisional text, not W2's).
- **Audio:**
  - `tts/<id>.wav` is the archived take, byte for byte;
  - `audio/<id>.wav` is 24 kHz s16 with a canonical header. Bulbul audio is copied as-is; Magpie's 22,050 Hz is resampled with ffmpeg `aresample=resampler=soxr`.
- **Speaker id:** `speaker_id = spk-tts-<model>-<voice>`.
- **Internal fields:** `_w3` (pid, archive key, input text, entities), `_qa`, `_archive`.
- **Selection:** QA keep and no review flag; one render per text; Devanagari input first; at most 5 per voice; Parler excluded. Re-running keeps existing ids first, so ids never churn.

## TTS-choice memo (2026-10-14)

1. **Primary engine: Bulbul v3, hi-IN, Devanagari input, temperature 0.3.**
   - Its voices are Indian and the hidden test is real Indian speech (§4, §6).
   - It's the only engine here with independent human-preference evidence for Hindi code-mix: SpeechArena Elo 1021.9 overall and 1031.3 on code-mixed text, vs IndicF5 at 805.8 overall (arXiv 2604.21481, via `research/prior-art.md`, S).
   - Measured round-trip quality sits within noise of the best engine.
   - It's cheap: ≈₹146 per audio hour at list price.
   - The 0.3 setting needs a 10-clip naturalness listen first.
   - Use many voices; texts never repeat across voices (§3A counts every file).
2. **Second engine, conditional: Magpie 357M.** Measured best on intelligibility, entity and number fidelity (QA reject 11%). It's free on our GPUs, seeded, and has a clean licence for outputs. A second TTS family also diversifies the synthetic acoustics, which matters because the base checkpoint is already "trained on synthetic data" (§3B; INFERRED: largely Sarvam TTS). **Use it only after a listen confirms its English-speaker voices give acceptable Hindi.** Avoid it for spelled digit/letter strings, or keep it QA-gated (25% of acronym renders get an extra syllable).
3. **Drop Indic Parler-TTS:** 40% QA reject, the lowest entity accuracy, slowest per clip, and unreliable speaker control.
4. **Script rules for W2:**
   - English words in Devanagari, as they should be pronounced;
   - numbers verbalised;
   - commas allowed as pause hints;
   - **a speaker-gender field on every script**, so W3 can pick a voice whose verified gender matches the verb forms.
5. **Every render goes through `qa.py`** with the thresholds above; rejected takes are regenerated, archived, and never referenced.

## Spend log (Sarvam API, W3)

| Date | Requests | Billed chars (`len(text)`) | ₹ at list price | Note |
|---|---|---|---|---|
| 2026-10-14 | 102 Bulbul v3 TTS | 17,594 | ≈52.8 | Then **HTTP 402 `insufficient_quota_error` "No credits available"**. All 102 responses archived with their `request_id` |
| **Total** | 102 | 17,594 | **≈52.8 of the ₹2,000 cap** | |

## Status — 2026-10-14

**Done:**
- Probe set (100 items).
- Renders archived: 102 Bulbul, 140 Magpie, 140 Parler.
- Both QA ASRs run on all 382 renders.
- QA pipeline with thresholds tuned on 231 positive controls.
- Bake-off tables, the Devanagari-vs-Latin answer and the TTS-choice memo (above).
- 50-record pilot that passes `validate.py --partial`.
- A 41-item human spot-check list.

**Measured headline:**

| Engine | QA reject | CER p50 | Number-entity hit |
|---|---|---|---|
| Bulbul | 18.1% | 0.016 | 95.3% |
| Magpie | 11.0% | 0.010 | 98.2% |
| Parler | 40.0% | 0.020 | 92.7% |

- Devanagari input beats Latin on all three engines.
- Bulbul costs ≈₹146 per audio hour.

**Blocker: Bulbul credits.**
- The key returned 402 after ≈₹53. 182 Bulbul jobs are left (29,390 chars ≈ ₹88): the remaining Bulbul probe renders, 22 Latin pairs and the 10 long-form inputs.
- To resume: `uv run --no-project python w3/render.py bulbul`, which never re-requests. Then `qa.py asr w2v`, `asr ic`, `score`, `summary`.
- FAQ: the competition credits arrive "before the mid-term". They may land on a different key or need a top-up.

**Open (team-side, not organiser questions):**
- the billing unit (code points incl. spaces?), to confirm on the dashboard;
- naturalness of temperature 0.3, and Magpie's accent (ears);
- voice genders (ears).

No new organiser questions; W0's list covers the PS ambiguities this touches.

**Disk:**
- `data/w3` is 448 MB now.
- Deleted after use, all re-fetchable at the pinned revisions above:
  - Magpie GGUF + runtime (0.8 GB);
  - Parler weights (3.6 GB);
  - my duplicate IndicConformer copy (2.4 GB). `qa.py` now loads W4's `data/w4/ic600m`, verified file-for-file identical and used for the controls pass.
- Peak was ≈3.9 GB.
- uv cache: +0.2 GB for the Parler environment (transformers 4.46.1); torch was shared with W4.

**Handoffs:**
- **W2:**
  - The script rules in memo item 4, especially **speaker gender**.
  - Single-letter fillers (अ, ए) are unverifiable after TTS. Prefer उम्म/हम्म-type fillers, or rely on W5's duration check.
  - Spelled letter/digit strings are the riskiest content for every engine (number-entity hit 85–95% on digit strings, 86–88% on acronyms). Keep them, but budget for regeneration.
  - Replace `w3-probe-v0` text with W2 scripts for the bulk run.
- **W5:**
  - Align `data/w3/pilot/manifest.jsonl`: clean 24 kHz synthetic speech, with fillers and self-corrections.
  - Please report **filler word durations**. A filler aligned to under ~60 ms was probably not spoken, so the transcript would be wrong (§3A).
- **W6:** translate the same pilot. Self-corrections (p001, p005, p013 …) need the FAQ treatment.
- **W7:**
  - **Rebuild by copying, never by re-synthesising:** `tts/<id>.wav` = `_archive.w3` bytes; `audio/<id>.wav` = soxr resample with a canonical header (`w3/pilot.py:to24k`).
  - Strip `_w3`, `_qa`, `_archive` at release.
  - Pinned engine revisions are in the engine table above.
- **W1/W4:** vakyansh truncates whole clips longer than ~10 s. Decode in ≤ 7 s chunks cut at the quietest frame (`w3/qa.py:asr_w2v`). `w1/asr_check.py` decodes whole clips.
- **W8b:** Bulbul v3 at temperature 0.3 was more faithful than 0.6 on Hindi (12 pairs). Worth the same test on English `shubh` targets.

## Round 2, 2026-10-14: open engines at scale

**Context change (Harsh, 2026-10-14; CLAUDE.md):** personal learning project. Engines must be open-source or free; Bulbul is out (no credits, not open). Non-commercial (NC) licences are acceptable here and are labelled **NC** below. GPU work runs on SageMaker.

### Open Hindi TTS survey (2026-10-14)

Sources: HF cards and API metadata, GitHub, PyPI, read 2026-10-14 (S = the source's own claim). "Job-runnable" = ungated, because a SageMaker job gets no HF token (W4's IndicConformer needed an S3 mirror for that reason).

| Engine | Hindi voices | Weights / code licence | Gated | Size | Ran? |
|---|---|---|---|---|---|
| **OmniVoice** (k2-fsa, 2026-03) | zero-shot cloning, 3–10 s reference + transcript; 117 h Hindi in 581k h (S) | **CC BY-NC (NC)** (card; no HF tag) / Apache-2.0 | no | 3.3 GB | **yes, SageMaker** |
| **pocket-tts-hindi** (Saryps Labs, 2026-09) | cloning from a reference clip; 932 h IndicVoices + Kathbath (S) | CC BY 4.0 / MIT (pocket-tts runtime) | no | 0.22 GB | **yes, laptop CPU** |
| VoxCPM2 (OpenBMB) | cloning or text-described voices; 30 languages incl. Hindi (S); MiniMax Hindi WER 19.7 vs internal CER 0.79% (S, contradictory) | Apache-2.0 | no | 5.0 GB, bf16 | queued (no g6/g5 capacity) |
| Magpie 357M (NVIDIA) | 5 fixed English-speaker voices; no cloning (S) | NVIDIA Open Model License | no | 0.6 GB | round 1 |
| Indic Parler-TTS | 4 named Hindi speakers; unnamed descriptions give random voices (S) | Apache-2.0 | **yes** (not job-runnable) | 3.75 GB | round 1 (40% reject) |
| IndicF5 (AI4Bharat) | cloning; last of 7 in SpeechArena (Elo 805.8, S) | MIT weights, code has no licence | **yes, access not granted** (HTTP 403, 2026-10-14) | 1.4 GB | no |
| SPRING_F5 / F5-Hindi-24KHz (IIT Madras) | F5 cloning | Apache-2.0 tag, code no licence / CC BY 4.0 (fork) | no | 5.4 / 0.6 GB | no: no published Hindi numbers, F5 family ranks last in SpeechArena |
| Chatterbox Multilingual + `-hi` finetune (Resemble AI) | cloning | MIT; outputs carry Resemble's Perth watermark | no | 3.2 GB | no: package pins torch 2.6 (image has 2.9); the `-hi` loader exists only as modified code inside its demo Space |
| XTTS-v2 (coqui-tts) | cloning | CPML (**NC, outputs too**) / MPL-2.0 | no | 2.1 GB | no: older, superseded by OmniVoice in the same licence class |
| Indic-Speak, Veena, Kokoro, Piper, svara, Orpheus-hi | ≤ 5 fixed Hindi voices, or Llama-3.2 bases | various | mixed | 0.3–13 GB | no: "many voices" is the goal |
| Sooktam-2 (BharatGen), Higgs TTS 3, Fish S2 | cloning | research / NC licences | no | 6.7–11 GB | no: bigger, no Hindi numbers |
| Qwen3-TTS, CosyVoice 3, IndexTTS-2, kyutai pocket-tts (base) | — | — | — | — | no Hindi |

### Voices: 40 cloned IndicVoices speakers

- `w3/voices.py` picks one 6–11 s segment per speaker from the IndicVoices Hindi files W1 already holds: non-conversation (wideband) recordings with an excellent/good verification report, no noise, echo, chatter or volume flags, and a tag-free verbatim text. It keeps the cleanest segment per speaker (SNR as `pipeline/audio_qc.py` measures it), cut to 24 kHz with the pinned ffmpeg, then the 20 cleanest speakers per gender.
- 73 speakers qualified (41 m, 32 f); 40 kept (20 f, 20 m). Each clip is stored write-once at `raw/_voices/iv-<speaker>.wav` with a sidecar (source file and offsets, transcript, speaker metadata, sha256, licence CC BY 4.0, attribution AI4Bharat IndicVoices). `w3/voices.tsv` is the bank; a script gets a voice of its own `gender` by a stable hash of its `pid` (≈ 12 scripts per voice in the pilot).
- **Consent caveat (ethics):** these speakers consented to IndicVoices' collection for speech technology under CC BY 4.0, which permits reuse of the recordings. They did **not** explicitly consent to having their voices cloned. This is acceptable only for this private, non-commercial learning project. Never publish the synthetic audio, and never present it as the speaker. Before any release, replace these references with voices whose owners consented to cloning (e.g. Harsh's own voice, or TTS-corpus voice artists such as Rasa or SYSPIN, whose recordings were made for TTS).
- **Speaking rate (measured):** real IndicVoices *conversation* runs at 10.7 transcript chars per second of speech (p25–p75 9.3–12.4; 167 W1 pool clips). Extempore runs at 9.7 and read speech at 7.3. The 40 references are read/extempore speech at a median of 7.8 (range 4.0–13.0). OmniVoice derives output duration from the reference's pace, so its first smoke render of p001 took 34 s where other engines took 14–19 s. The stage therefore gives OmniVoice an explicit duration: chars / the voice's target rate + 0.4 s, where each voice gets a fixed target rate spread evenly over the conversational p25–p75 band (`tts.voice_rate`). Pocket ends on its own EOS and came out at 0.089 s/char (≈ 11 chars/s) without help.

### The `tts` stage (`w3/tts.py`; self-check `w3/test_tts.py`)

- **Contract:** `--in SCRIPTS.jsonl --audio-dir DIR --out OUT`, registered in `pipeline/run.py` STAGES as `tts` (and `tts_bench` = `w3/tts_bench.py`, its `--bench` mode, so a benchmark never moves the `tts` LATEST). Input = W2 scripts uploaded as a stage run (`tts.py upload SCRIPTS STAGE`: `stages/tts_in/<run>/`, no audio). It runs locally under the GPU flock and on SageMaker through `run.py --sm` (verified: smoke and benches below).
- **Ids:** `tts.py alloc` appends one id per script text (`txt:<sha256(tts_dev)[:16]>`) to `w3/ids.tsv` **before** any run; the job only reads them. W2 pilot-v1 (500 scripts, sha256 `e6b51167…`) holds `rec-400050`–`rec-400549`. A script that fails every take burns its id into `rejects.jsonl` (rule `qa`, every take's reasons).
- **Per script:** a voice of the script's gender (stable hash of `pid`); take *t* = seed *t*; with several `--engine`s the first take goes to one by a stable hash and each retake moves to the next engine. Every take goes through `pipeline/archive.archive_call` (service `tts-<engine>`, key = sha256 of engine, model, pinned revision, voice, reference key, text, seed, take, engine options), so every take, kept or rejected, is archived write-once at `archive/api/tts-<engine>/<key>.bin|.json`, and a re-run never re-renders (verified by the self-check's crash/resume case).
- **QA per take:** vakyansh CTC (GPU, ≤ 7 s quiet chunks) + IndicConformer RNNT (CPU ONNX; weights from W4's S3 mirror inside a job) → `qa.measure` + `qa.rules` (the round-1 thresholds) + "one agreed substitution → retake" (round 1 sent those to a human; §3.3 says drop, don't review). Hypotheses are cached per take key in `OUT/asr/`, so a resumed run never re-decodes.
- **Kept take → record:** `OUT/audio/<id>.wav` (24 kHz s16; soxr with the pinned ffmpeg if the engine isn't 24 kHz), and the same bytes write-once at `archive/tts/<id>.wav` + `.meta.json` (sha256, the raw take's archive key, request). `_archive.tts` points there, so `pipeline/rebuild.py` copies, never re-synthesises. `source = {corpus: "w2-scripts-v1", source_file: "tts/<id>.wav", start: 0, end: duration, synthetic: true, model, voice}`; `speaker_id = spk-tts-<model>-<voice>`; `sentences[].hi_text` = the script's transcript (what QA says was said); `_w2` = the script minus `_gen` (W6 needs `entity_spans` and `corrections`); `_w3` = engine, revision, take, raw-take key, rejected takes; `_qa.tts` = CER per ASR, spans, pace, pauses.
- `OUT/takes.jsonl`: one QA row per take (render and ASR seconds included). `tts.py report SCRIPTS DIR...` turns it into the tables below; `tts.py listen RUN OUT` builds the listening list.

### Bench on the probe set (100 items, first take, same QA as round 1)

| Engine / run | Takes | Audio h | CER ic p50/p90 | Entity hit | Number-entity hit | English-span hit | Agreed skip | Agreed insert | QA reject (r1 rules) | + 1 subst. | s/char p50 | Wall s per audio s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Bulbul v3 (round 1) | 72 | 0.26 | 0.016/0.048 | 96.6% | 95.3% | 98.9% | 4.2% | 0.0% | 18.1% | – | 0.091 | 0.28 (API) |
| Magpie 357M (round 1) | 100 | 0.33 | 0.010/0.027 | 98.6% | 98.2% | 99.2% | 1.0% | 7.0% | 11.0% | – | 0.095 | 0.78 |
| Indic Parler (round 1) | 100 | 0.34 | 0.020/0.077 | 94.1% | 92.7% | 97.1% | 7.0% | 5.0% | 40.0% | – | 0.095 | 1.29 |
| **pocket-tts-hindi**, 40 cloned voices (laptop CPU) | 100 | 0.32 | 0.017/0.049 | 96.4% | 96.4% | 97.1% | 5.0% | 1.0% | **21.0%** | 22.0% | 0.089 | 0.31 (render only, CPU) |
| pocket, sentence/comma chunks (rejected variant) | 100 | 0.39 | 0.017/0.052 | 97.2% | 95.9% | 97.7% | 8.0% | 0.0% | 19.0% | 21.0% | 0.111 | 0.27 |

- **Pocket** is on a par with round-1 Bulbul and clearly better than Parler, with 40 Indian voices instead of 4–12. By phenomenon (rejects/n): digit strings 6/31, spelled acronyms 2/16 (Magpie: 37.5%), lakh 4/17, fractions 4/16, dates 1/16, times 2/13, self-corrections 5/19, fillers 3/17, **names 11/32: its weakest**.
- **Chunking didn't help pocket.** Its own splitter only breaks at `. ! ?` and warns that a chunk over 50 tokens "may skip words"; with the punctuation stripped (card: trained without it), every script is one chunk. Rendering one sentence (long ones cut at commas) per call with 0.3 s gaps gave 19% vs 21% reject and 8 vs 5 agreed skips, i.e. no difference at n = 100, and a slower pace. Reverted; the simpler whole-text render stays.

### QA on W2 scripts: two corrections

- **ASR normalisation of colloquial forms looked like TTS substitutions.** On the first pilot takes, both ASRs agreed on these single-word "substitutions": ये→यह 5, पे→पर 5, मे→में 4, न→ना 1, ऐप→एप 1. A TTS reading पे does not say पर; both ASRs' language models standardise it. With "one agreed substitution → retake", such a script would fail every take and be lost, a bias *against* colloquial Hindi. `qa.SAME` now treats these pairs as one word, like same-value number spellings. On round 1's 382 renders it changes 0 verdicts.
- The remaining agreed substitutions look like real TTS errors (प्यूमा→पीमा, मृणाल→निराल, शेड→सेट), and agreed skips are mostly digits inside digit strings plus short function words (है, मैं, का), which listening to round 1 also suggested. Those rules stay.
