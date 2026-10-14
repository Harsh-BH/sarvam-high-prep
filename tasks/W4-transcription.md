# W4: Transcription of real audio

**Goal:** verbatim, spec-compliant Hindi transcripts for every real clip. Transcripts are a sampled item in the mid-term audit (12 pts, §7), and a format error fails the dataset gate (§8).

## PS rules (see W0 `SPEC.md` for the exact conventions)

- IndicVoices Level 1 ("as spoken"), disfluencies, repetitions and fillers included (§3A). In detail:
  - English words in Devanagari **without** bracketed Latin.
  - Numbers as spoken words.
  - Every sentence ends with sentence-final punctuation.
- Manual correction is allowed (§3A). Store corrections as archived patches so the rebuild stays reproducible.
- Real IndicVoices L1 transcripts add bracketed English (ओटीपी [OTP]) and use bracketed Latin filler tags ([umm]). **Both must be converted.**

## Research and measure

- **ASR benchmark** on 1–2 h of hand-checked pilot audio from W1. Candidates: Sarvam Saaras (₹30/h), IndicConformer, Whisper-large-v3 and its Hindi fine-tunes, MMS. Score on:
  - WER/CER;
  - does it write English words in Devanagari?
  - numbers;
  - **does it keep disfluencies, or clean them up?**
  - telephone vs wideband audio.
- **Reuse vs re-transcribe:** are corpus transcripts accurate enough to convert, or do we re-run ASR and merge?
- **Conversion pipeline:**
  - Strip bracketed Latin; map tags to Devanagari fillers or drop them.
  - Numbers → words exactly as spoken.
  - English words → Devanagari (IndicXlit or an LLM).
  - Sentence segmentation and punctuation.
- **Human correction:**
  - Tool: Label Studio or similar.
  - Which records get reviewed: entity-dense and low-confidence ones.
  - Measure throughput per annotator-hour, and run an agreement spot check.

## Done when

The pilot transcripts pass the validator, with measured error rates per source and a correction protocol written down.

## Findings, 2026-10-14 (W4)

### ASR ranking (read this first, W3)

Measured on this machine, 2026-10-14. Full method and tables in §2 below. Numbers and English are re-scored after a fix: tags are no longer counted as English words, and numbers are matched by value.

| Rank | Model (licence) | CER pool / held-out | Number words | Latin or digits | Use |
|---|---|---|---|---|---|
| 1 | **IndicConformer-600M multilingual**, `ai4bharat/indic-conformer-600m-multilingual`, RNNT or CTC (MIT, gated, access granted) | 4.6 / 6.5 | 96–97% | never | **W3 round-trip QA**: an AI4Bharat Conformer, a different family from Bulbul, IndicF5 and Parler. Also W4's P2 entity check and the audit draft. Use CTC for speed (0.23× real time on CPU), RNNT for accuracy (0.33×) |
| (1) | **Sarvam Saaras `saaras:v4` + `mode=verbatim`** (closed API, §10 R2; outputs never published, §10 R3) | 4.3 vs IndicConformer RNNT 4.7 on an 82-clip subset | 95% | never | As accurate as IndicConformer and slightly better on telephone audio, but it costs credits and the **credit balance ran out** during this run. Not needed while IndicConformer is free. `saaras:v4` *without* `verbatim` writes digits (54% of number words) |
| 2 | vakyansh wav2vec2 `Harveenchadha/vakyansh-wav2vec2-hindi-him-4200` (MIT) | 13.4 / 14.7 | 64% | never | Second opinion; independent of IndicVoices training. **Fails silently on ~4.5% of 8 kHz clips** (W1) |
| 3 | Whisper large-v3 (Apache-2.0) / large-v3-turbo (MIT) | v3 16.0, turbo 16.5 / turbo 25.0 | 32–33% | 1.4–1.9% of words; 7% on read speech | Only with a digit/Latin normaliser. Turbo is the fast independent check (0.08× on GPU) |
| — | MMS-1b-all `hin` (CC BY-NC 4.0) | 14.3 / 15.0 | 28% | 1.4% | Benchmark only. NC, so not a dataset tool (research T3) |

### 1. Conversion pipeline: IndicVoices L1 → SPEC §2 (`w4/convert.py`)

**Output:** the `transcribe` stage's `records.jsonl` (§5), one record per W1 clip with `id` + `sentences[].hi_text` + `_orig_text` + `_w4.flags`, in SPEC §6 key order; the gate (§4) then decides what reaches `data/w4/pool.jsonl`. On the v1 pool (711 records), `validate.py --partial` gave **0 violations**. Before conversion, the raw text gave hi_final 109, hi_chars 77, hi_nfc 36, hi_space 32 on the 50 pilot records alone (W1). Pool v2: 648 converted, 628 accepted, both with 0 violations.

**What the raw L1 text contains (measured over all 1,343 pool segments).** It is cleaner than the PS warns:
- **No digits and no Latin words** anywhere in `verbatim`. English and numbers are already Devanagari words as spoken (`फाइव थाउज़ेन रुपीज़`, `एक आठ नौ सात`). The `[OTP]`-style glosses live only in `normalized` (L2), which we don't use for text. So no number-to-words or transliteration step is needed for IndicVoices. **INFERRED:** SPRING-INX (Latin English) and other corpora will need one. Not built (YAGNI); the converter flags any Latin or digit as `chars:` instead of dropping it.
- **127 tag types in the corpus, 47 in the pool:** inhaling 780, uhh 454, noise 145, **stammers 107**, breathing 87, persistent-noise 139, horn 57, umm 55, …, ugh 14, hmm 12, uh-huh 8, ahh 5. Some are malformed: `[Inhaling>`, `rustling>`, a truncated `<Persistent-noise` at a segment end, and `और<inhaling>` with no space before the tag.
- **Other marks:** ZWJ ×54 (`अप्<ZWJ>लीकेसन`), hyphen ×42 (`आस-पास`, `कभी-कभी`), comma ×28, colon as visarga ×7 (`छ:`). Read-prompt clips also carry dotted acronyms (`पी.आर.ए.एन.`), `।` ×7 and `?` ×4. There are 834 precomposed nukta letters, 95 half-nasal clusters, and 49 word-final anusvaras after `ा`/`ू`.
- **Unspaced letter-by-letter acronyms:** 6 tokens, confirmed by an all-caps L2 gloss (`आरजीपीवी [RGPV]`, `बीजेपी [BJP]`, `आईडी [ID]`, `एआईएमआईएम [AIMIM]`).

**Rules applied** (SPEC §2.4, in order):

| Input | Output | Count in pool |
|---|---|---|
| `[uhh]` `[umm]` `[hmm]` `[ahh]` | `अ` `उम्म` `हम्म` `आ` (R9; `[uhh]` → `अ`; the nasal `अं` only by ear) | 454 / 55 / 12 / 5 |
| `[stammers]` `[uh-huh]` `[ugh]` | **Deleted from the provisional text and flagged** `tag@word-index`. They need ears (§2.4); never guessed | 107 / 8 / 14 → **102 clips** |
| second-voice / unintelligible tags | flag `drop:*` | 0 (W1 already excluded these clips) |
| every other tag | deleted (R11) | — |
| ZWJ/ZWNJ/ZWSP/BOM, NBSP | deleted / space; then NFC (decomposes nukta letters, R2) | 55 / 834 |
| `X.Y.Z.` (Devanagari letters) | `X Y Z` (R8) | read clips |
| glued acronym with an all-caps L2 gloss | spaced letter names (R8), only if it parses into exactly that many SPEC letter names | 6 |
| `-` | word break (§2.2: echo words are two words; `आराम-दायक` → `आराम दायक`) | 42 |
| `छ:` | `छः` (visarga) | 7 |
| `,` `;` quotes, newlines | deleted | — |
| `।` `|` `.` `?` `!` in the source | hard sentence boundary, keeping `?`/`!` | read clips |
| nasal + homorganic stop (and `न` + any stop in loanwords) | anusvara (R12): `हिन्दी`→`हिंदी`, `नम्बर`→`नंबर`, `पैन्ट`→`पैंट`. Not `उन्हें`, `अन्य`, `मम्मी`, `जन्म` | 95 |
| word-final `ां`/`ूं` | `ाँ`/`ूँ` (R12). Word-final only: mid-word `शांति` is a true nasal consonant | 49 |

**Sentence split (R3).** Inside each hard chunk, `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` places sentence boundaries and picks `।` or `?`:
- Apache-2.0, 47 languages; the card reports Hindi `।` F1 97.8 on written text. ONNX on CPU, about 50 s for the pool.
- Pinned to `punctuators==0.0.7` and model revision `d1769a5…`.
- Its commas are dropped. When it re-spaces a word (`इस्कूल` → `इ स्कूल`, 4 clips), its boundaries are mapped back by character offset. Words always come from the input.
- A `।` sentence with ≤ 1 non-filler word (`अ।`, `जिसका।`) is joined to the next sentence. Before this rule there were 50 one-word sentences; after it there are 4, all questions (`ठीक?`).
- **Result:** 2,635 sentences, 40,027 words. Words per sentence p10/50/90 = 5 / 15 / 26, max 43. Marks: 2,526 `।`, 109 `?`.
- **AMBIGUOUS / limits:** an intonation-only question can't be found from text (R3 needs ears; review rule). Splits in spontaneous conversation are sometimes fragmentary (`पूरे तरह से।`). Re-splitting never changes words, so W5 is unaffected and W6 may re-split.

**Reproducibility:** two full runs gave a byte-identical `pool.jsonl` (sha1 `794f1019…`). The rebuild needs only W1's manifest, the pinned model and `w4/patches.jsonl`.

**IV-specific decisions (OURS):**
- `[uhh]` → `अ` by default. 454 instances can't all be listened to; the R9 nasal variant `अं` is a reviewer correction.
- IV's L1 phonetic spellings (`बोहोत`, `हे`, `इशलिए`, `दौ`) are kept as they are (R5).
- Nukta is kept as IV wrote it (R12 "as heard"). We can't verify z/f without ears.

### 2. ASR benchmark (`w4/asr_run.py`, `w4/asr_score.py`)

**Set:** the first 174 pool clips (`rec-000001`…`174`) = the 50-clip pilot + 124 more, **1.00 h**: 70 conversation (8 kHz), 70 extempore, 34 read. That is 150 unflagged clips, 7,872 reference words (59 telephone, 91 wideband). Flagged clips are excluded because their reference has a known hole.

**Reference** = the W4-converted IV L1 text above.

**Scoring:**
- No Whisper normaliser (T6). Strict = NFC, punctuation out; digits and Latin stay and count as errors (§3A). WER/CER below are *lenient* (nukta, chandrabindu/anusvara and half-nasal folded on both sides: spelling noise, not content).
- **Fillers kept:** a reference `अ अं हम्म उम्म` aligned to an identical hypothesis word.
- **Repetitions kept:** both tokens of an immediate repeat aligned.
- **English:** reference words that carry a `[Latin]` gloss in IV L2.
- **Numbers:** Hindi and English-in-Devanagari number words (`asr_score.NUMVAL`, word→value).
- English words exclude IV tags written in square brackets: a bracket name that also occurs in the segment's L1 text is a tag, not a gloss. Fillers are excluded too.
- The set is frozen in `data/w4/bench/` (W1 pool v1, rec-000001…174), so these numbers don't move when the pool changes.

All of these are positional matches on a word-level Levenshtein alignment. Output: `data/w4/asr/scores.tsv`.

**Held-out control** (`w4/ivvalid_set.py` → `data/w4/ivvalid/`): 120 IndicVoices **valid** segments (0.32 h; 60 telephone, 60 wideband; one 5–20 s segment per file). The pool is IV *train*, so a model trained on IV could look better than it is.

| Model | WER | CER | CER 8 kHz | CER wide | CER read | Latin/digit % of hyp words | Fillers kept (65) | Repeats kept (59) | English words (734) | Number words (345) | IV-valid CER (8k / wide) | Real-time factor here |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| IndicConformer RNNT | **13.3** | **4.6** | 6.5 | 3.7 | 2.6 | 0 | 1.5% | **79.7%** | 83.8% | **96.8%** | — | 0.33 (CPU ONNX) |
| IndicConformer CTC | 13.9 | 4.7 | 6.7 | 3.8 | **2.0** | 0 | 1.5% | 74.6% | **84.1%** | 95.7% | **6.5** (9.4 / 3.9) | 0.23 (CPU ONNX) |
| Whisper large-v3 | 29.6 | 16.0 | 17.6 | 15.3 | 22.0 | 1.9 | 0 | 40.7% | 46.3% | 31.9% | — | 0.45 (GPU fp16) |
| vakyansh wav2vec2 | 31.4 | 13.4 | 22.1 | 9.4 | 6.3 | 0 | 0 | 39.0% | 51.6% | 63.5% | 14.7 (23.5 / 6.8) | (W1 run, CPU) |
| Whisper large-v3-turbo | 32.7 | 16.5 | 17.8 | 15.9 | 23.5 | 1.4 | 0 | 32.2% | 41.8% | 32.5% | 25.0 (35.6 / 15.4) | 0.08 (GPU fp16) |
| MMS-1b-all `hin` (NC) | 34.4 | 14.3 | 18.0 | 12.7 | 19.8 | 1.4 | 0 | 28.8% | 38.0% | 28.4% | 15.0 (22.2 / 8.6) | 0.015 (GPU fp16) |

Real-time factors were measured on a shared laptop (RTX 3050 6 GB, CPU contended by other jobs). On an H100 they're only good for ranking.

**Saaras** (`w4/saaras_run.py`, REST, `language_code=hi-IN`). Clips over 30 s are cut in an IndicVoices inter-segment gap; 215 requests per config.
- Every raw response is archived in `data/w4/saaras/<config>/<id>.json` (never re-requested). Outputs stay private (§10 R3).
- **The credit balance ran out mid-run.** HTTP 402 `insufficient_quota_error` ("No credits available") after 129–144 successful chunks per config. Failed requests return nothing and aren't billed.
- The scored subset is the **82 clips** where all three configs returned the whole clip (373 English words, 190 number words, 44 fillers, 28 repeats). Every other model is re-scored on the same subset (`data/w4/asr/scores-saaras.tsv`):

| Model | WER | CER | CER 8 kHz | CER wide | CER read | Latin/digit % | Fillers kept | Repeats kept | English | Numbers |
|---|---|---|---|---|---|---|---|---|---|---|
| **Saaras v4 + `mode=verbatim`** | **12.2** | **4.3** | **5.9** | 3.6 | 2.0 | 0 | 1/44 | **85.7%** | 84.5% | 94.7% |
| IndicConformer RNNT | 13.5 | 4.7 | 7.1 | 3.6 | 2.4 | 0 | 1/44 | 82.1% | 81.8% | **97.9%** |
| Saaras v3 + `mode=verbatim` | 13.6 | 4.7 | 6.5 | 3.9 | 2.1 | 0 | 1/44 | 82.1% | 84.7% | 94.2% |
| IndicConformer CTC | 13.8 | 4.8 | 7.4 | 3.7 | **1.8** | 0 | 1/44 | 78.6% | 83.1% | 95.3% |
| Saaras v4 (default `transcribe`) | 14.9 | 6.5 | 7.8 | 5.8 | 10.8 | 0.8 | 1/44 | 64.3% | 76.7% | 53.7% |
| vakyansh / MMS / Whisper-turbo / v3 | 30–33 | 12.8–16.7 | 18–22 | 8.6–15.9 | 5.4–23.1 | 0–1.7 | 0 | 43–46% | 37–48% | 38–71% |

- `mode` is documented for `saaras:v3` only, but `saaras:v4` accepted `mode=verbatim` (HTTP 200) and the output is clearly verbatim-style: no punctuation, no digits, repetitions kept.
- **"Verbatim … preserving filler words" (docs) doesn't hold for IndicVoices fillers:** 1 of 44 kept, the same as IndicConformer.
- Default v4 normalises numbers to digits and loses repetitions (64%). Never use it for §3A text.
- **Spend:** ₹18.19 (v4) + ₹19.95 (v4-verbatim) + ₹20.31 (v3-verbatim) = **≈ ₹58.5**, from the seconds of successful requests at ₹30/h billed per second (docs pricing page). The cap was ₹500.

**What the numbers say:**
- **Disfluencies (the PS's open question): no ASR keeps fillers.** 64 of 65 reference fillers vanish even in the best model, and all of them in every other model. Repetitions survive 75–80% of the time in IndicConformer and 29–41% in the rest. Example `rec-000042`: reference `हेलो अ जी … अ जी अ फ़्लॉवर`; every model drops all three `अ`. So **no ASR output is §3A-compliant L1 text without a human pass.**
- **English in Devanagari:** IndicConformer and vakyansh never emit Latin. Whisper and MMS do: `मेरा PRAN नंबर 189 765 124 177` (v3 on `rec-000020`, whose reference is `मेरा पी आर ए एन नंबर एक आठ नौ सात छः पाँच एक दौ चार एक सात सात`). Turbo writes `P R A N No. 189-76512-4177`.
- **Numbers:** Whisper and MMS get under a third of the number words, because they write digits and regroup digit strings. IndicConformer gets 96–97%, spoken form included.
- **Telephone vs wideband:** every model is worse on 8 kHz. IndicConformer CER ×1.8, vakyansh ×2.4, MMS ×1.4. Whisper is ×1.1 only because it's poor on both. On held-out telephone segments Whisper-turbo reaches 35.6% CER, with hallucinated text on short noisy segments.
- **WER is inflated by IndicVoices' own spellings** (`बोहोत`, `हे`, `इशलिए`, `दौ`): ASRs write the standard form. That's why CER is the better ranking metric here.
- **Contamination check: no clear signal.** IndicConformer is also best on held-out IV-valid (6.5% CER vs 14.7–25.0%). Its CER ratio to vakyansh is 0.35 on the pool vs 0.44 held out, which suggests some train-set advantage, but its ratio to Whisper-turbo is 0.28 vs 0.26, which doesn't. The two sets also differ in segment length (5–20 s segments vs 10–60 s clips), which confounds both ratios. **INFERRED:** IndicConformer's lead is real. To be safe, it is still kept out of the "independent" ASRs that decide the P1 review queue.
- **Saaras:** not run, no key. Research T5 says no mode meets §3A (digits/Latin), so it can't be a transcript source anyway. A benchmark row would cost about ₹30 for this 1 h set (₹30/h) once the key arrives.

### 3. Reuse vs re-transcribe: **reuse IndicVoices L1 (converted), re-transcribe nothing automatically**

| Evidence | Number |
|---|---|
| Reference vs a 4-model independent-ASR consensus (vakyansh, turbo, v3, MMS agree on one *different* word) | 2.0% of words, of which 74% are spelling-level (L1 phonetic spellings, matras). **Word-level: 0.53%** (0.41% with all 6 models). This is an upper bound on reference word errors: it includes alignment artefacts (`का`→`आपका`) and IV's deliberate L1 forms (`बोहोत`→`बहुत`) |
| Best ASR vs reference | WER 13.3%, CER 4.6% on the pool; 6.5% CER held out |
| Fillers an ASR transcript would lose | ≥ 98.5% |
| Repetitions an ASR transcript would lose | ≥ 20% (best model), 59–71% (the rest) |
| §3A violations in Whisper/MMS output | Latin/digits in 1.4–1.9% of words (7% on read speech); number words ≤ 33% correct |
| IV's own QA | `verification_report.decision = excellent` on 2,826/2,826 files (W1) |
| Best closed ASR (Saaras v4 verbatim) | CER 4.3%, fillers kept 1/44: the same disfluency loss |

**Decision:**
- Keep the human IndicVoices L1 transcript for every IV clip.
- ASR only (a) routes clips to review (P1), (b) gives the reviewer a draft (`asr_hint` = IndicConformer), and (c) serves W3's round trip.
- Re-transcribe by hand, with the IndicConformer draft, only where the reference is incomplete: `rec-000046`, where W1 found a missing menu list.
- **INFERRED, for future non-IV sources** (SPRING-INX, Vaani: no L1 disfluency marking, Latin English): use IndicConformer RNNT as the first pass. Every clip then needs the human filler/repetition pass, so budget roughly the P0 rate below for *all* of them, not 18%.

### 4. Gate and audit (policy 2026-10-14: no annotators, design doc §3.3)

The earlier human-review queue (P0 103 / P1 16 / P2 58 / P3 29 clips, ≈ 5 annotator-hours) was replaced by this policy, decided by Harsh. Suspect clips are **dropped, not reviewed**, and W1 recuts their audio. A human listens once, to one frozen audit sample.

**Pipeline** (since Task 10, two pipeline stages; see §5): `transcribe` (`w4/convert.py`; rejects P0) → `asr_check` (`w4/asr_check.py`: ASR drafts + P1/P2 via `review_list.check`). Every clip ends up in exactly one place:

- the `asr_check` stage's `records.jsonl` (accepted). Locally, `uv run w4/review_list.py build` copies it to `data/w4/pool.jsonl`, **W4's stage file for W5, W6 and W7**.
- one of the stages' `rejects.jsonl`: `{"id","stage","rule","detail"}`. The local copy is `data/w4/rejects.jsonl`.

`asr_check` decodes any clip that lacks a vakyansh, Whisper-turbo or IndicConformer-CTC hypothesis before gating it, so nothing is accepted without all three.

| Stage | Rule (first that fires) | Why |
|---|---|---|
| **P0** | Any converter flag: `[stammers]` `[uh-huh]` `[ugh]` (need ears), `drop:*`, `chars:*`, `stale_patch` | A heard fragment is missing from the text (§3A). W1 already retired the 103 such clips and recut around the tags (`w1/retired.tsv`) |
| **P1** | Both independent ASRs (vakyansh, Whisper-turbo; not IndicVoices-trained) have CER ≥ **0.35** vs the reference, or both hear ≥ **1.25×** its length. Also W1's six ear items | Candidate reference error or untranscribed speech. The thresholds were set on the v1 pool: they flag 16 of 711 clips (2.3%), and the flagged clips' independent CER is 0.35–0.80 |
| **P2** | Entity check against the IndicConformer-CTC draft. Entity words = number words (Hindi and English-in-Devanagari, by value) and English words IV glosses. **Reject if any number value conflicts** (the aligned draft word is a different number) **or if ≥ 2 entity words, and > 1/3 of them, have no match** within ±2 draft words. A match allows spelling variants (≤ max(1, len/3) code-point edits, 2 from 5 code points up: a conjunct is 2) and word splits (`स्टॉपवॉच` = `स्टॉप वॉच`) | Entities carry 18 KPI points (§7). Thresholds were set on the 291 clips with drafts at the time: 1,921 entity words, 1 value conflict, 16 single misses (1.3%), almost all IndicConformer errors (`कॉलोनी`→`सुकोलाने`) or splits. "Any conflict" catches the one real case; "≥ 2 and > 1/3" ignores IndicConformer's isolated slips |
| audit | The audit listener marked it `drop` | — |

An audit `ok` or `fix` (a human listened) overrides P1 and P2.

**Result on W1 pool v2** (648 clips, 3.526 h; v1 minus 103 retired, plus 40 replacements `rec-000712…751`):

| | Clips | Audio |
|---|---|---|
| **Accepted → `pool.jsonl`** | **628** | **3.443 h** (`validate.py --partial` with audio: 0 violations) |
| P0 | 0 | — (W1 retired them) |
| P1 | 16 | 4.2 min |
| P2 | 4 | 0.8 min |

The 4 P2 rejects:
- `rec-000498` (added in Task 10): the draft garbles the passage holding `लिप्टस` (Eucalyptus) and `एक`. The old 2-edit floor had falsely matched `एक` with `ये`; `test_asr_check.py` exposed it (`कलर` "matched" `और`), and short words now allow 1 edit.
- `rec-000207`: `फिफ्टी रुपीज` vs draft `एट्टी`, a real amount conflict.
- `rec-000643`: the reference `ट्रिवेंट एक्सेस हिन राइटिंग` looks wrong itself.
- `rec-000615`: `जर्मन शेफर्ड`, probably the draft's error. The conservative rule costs one 18 s clip.

All 40 replacements are accepted (no P1 or P2). Conversion of the 608 clips shared with v1 is byte-identical to the v1 run.

**Audit sample** (`w4/audit_ids.txt`, tracked, **frozen**: drawn once from the then 629 accepted clips and never redrawn; `rec-000498`, since rejected, is no longer listed, so 209 clips / 1.10 h remain):
- 210 clips, **1.11 h**: Conversation/8 kHz 25 min, Extempore/wideband 31 min, Read/wideband 10 min.
- Allocation is proportional to accepted hours with a 10-min floor per stratum; within a stratum, clips are taken in sha1(id) order.
- `data/w4/review.tsv` holds only this sample: `audio` (`data/w1/pool_pinned/audio/`), `current` (W4 text), `asr_hint` (IndicConformer draft), then `action` (`ok` / `fix` / `drop`), `corrected`, `minutes`, `note`.
- `uv run w4/review_list.py import` writes the batch write-once to `archive/human/transcribe/<date>-harsh.jsonl` in S3, appends the same lines to **`w4/patches.jsonl`** (the git copy; commit it), and prints the **audit WER** of the auto text, the measured reference accuracy the mid-term report needs.
- The `transcribe` stage applies patches with the sha1 staleness guard (a patch made against different text gets `stale_patch` and is never applied blind).
- **Estimate (not measured):** one listen at about 2.5–3× real time ≈ 3 h of Harsh's time.

**Stated / inferred / ambiguous:**
- STATED: §3A allows manual correction and requires L1 with disfluencies.
- INFERRED: dropping suspect clips is safe for hours, because supply exceeds the 200 h cap (W1).
- AMBIGUOUS: whether an audit-measured error rate is reportable as the dataset's transcript accuracy, since the sample is drawn after the gate. It measures the accepted pool, which is what the auditors sample.

### 5. Pipeline stages (S3 plan Task 10)

Both stages follow the plan's contract: `--in RECORDS.jsonl --audio-dir DIR --out OUT_DIR`, output `records.jsonl` + `rejects.jsonl` (`{"id","stage","rule","detail"}`), resumable by id, with a torn last line dropped and redone. Records accumulate fields like `audio_qc`'s do.

| Stage | Command | Runs on | Adds to the record | Reject rules |
|---|---|---|---|---|
| `transcribe` (C6) | `uv run w4/convert.py --in … --audio-dir … --out … [--no-sbd] [--patches w4/patches.jsonl]` | CPU; deps `punctuators==0.0.7`, `huggingface_hub` (PEP 723) | `sentences` (after `source`, SPEC §6 order), `_w4` (`flags`, `review` if patched, `sbd` if the split fell back), `_archive.transcribe` (the patch's S3 key) | `ear_tag` (`[stammers]` `[uh-huh]` `[ugh]`), `second_voice`, `chars`, `empty`, `stale_patch`, `audit_drop`, `audit_recut`; `detail.flags` gives tag@word-index for W1's recut |
| `asr_check` (C7) | `uv run w4/asr_check.py --in … --audio-dir … --out …` (locally: `flock data/.gpu.lock` first) | GPU for Whisper + vakyansh; IndicConformer on CPU onnxruntime; deps torch, transformers, accelerate, onnxruntime, numpy, huggingface_hub | `_qa.asr_check` = `{cer, len_ratio, entities, number_conflicts, entity_misses[, note]}` | `P1`, `P2` (`P0` as a guard). Same `detail` |

- `asr_check` writes `OUT_DIR/asr/<model>.jsonl` (`{id, hyp, secs}`, appended), and these files are uploaded with the run. A model is loaded only for ids without a hypothesis, one model at a time, and freed after. So the gate's input is archived and a resumed or re-run stage never re-decodes.
- `review_list.py import` archives each patch batch with `pipeline/s3.put_once` to `archive/human/transcribe/<date>-harsh.jsonl` (`-2`, `-3` … for later batches the same day):
  - Every line names its own key, so `w4/patches.jsonl` (the git copy) holds exactly the archived bytes.
  - A crash between the S3 write and the git append is safe: the re-run finds its identical batch already archived and reuses that key.
  - `transcribe` applies the git copy and records the key in `_archive.transcribe`.
- The old `converted.jsonl` → gate flow is gone, and that file has been deleted.

**Runnable checks** (all print `ok`; no GPU, no S3, no model downloads):

| Check | Covers |
|---|---|
| `uv run w4/test_convert.py` | Text rules (tags → fillers, ZWJ, hyphen, dotted and glued acronyms, nasal, visarga), every reject rule, patch fix / stale / drop, input fields carried through, `validate.py --partial` clean, resume without duplicates, torn last line |
| `uv run w4/test_asr_check.py` | Accept, P1, P2 number conflict, P2 English misses, an audit `ok` overriding P1, the P0 guard, `_qa.asr_check`, resume, torn line. It runs in an environment without torch, which proves archived hypotheses are never re-decoded |
| `uv run w4/test_review_list.py` | Import → write-once archive (`FakeS3` from `pipeline/test_s3.py`); a second same-day batch gets `-2` and leaves the first untouched; a no-op import; crash recovery with no duplicate key; `transcribe` applying fix / ok / drop with the archive key |

**Measured on the real pool:**
- `transcribe` over W1 pool v2: 648 records, 0 rejects, sentences identical to the pre-Task-10 conversion, 45 s on CPU. `validate.py --partial` with audio: 0 violations, 3.526 h.
- `asr_check`, seeded with the archived hypotheses: 628 accepted, P1 16, P2 4, in 9 s.
- Smoke test with fresh decoding on 3 pool clips (GPU, 1 min 34 s including three model loads). vakyansh on GPU is identical to W1's CPU decode on 3/3 clips, and the gate result is right (`rec-000001` → P1, a W1 ear item).

**For the master** (I did not edit `pipeline/run.py`):

```python
"transcribe": {"cmd": ["uv", "run", str(ROOT / "w4/convert.py")], "makes_audio": False, "gpu": False},
"asr_check": {"cmd": ["uv", "run", str(ROOT / "w4/asr_check.py")], "makes_audio": False, "gpu": True},
```

**SageMaker (`--sm ml.g5.xlarge`) caveats:**
1. **`ic-ctc` is a gated HF repo, and the job has no `HF_TOKEN`**, so `asr_check` will fail when it loads the draft model. Two options, both your call:
   - (a) Mirror the MIT weights (2.4 GB, real files) write-once to e.g. `raw/_models/ai4bharat/indic-conformer-600m-multilingual/<HF commit>/`, and I add an S3 fallback in `asr_run.indicconformer`.
   - (b) Put `HF_TOKEN` in the job's `Environment`. Not recommended: `DescribeTrainingJob` shows it in clear text.
2. IndicConformer runs on CPU onnxruntime: the CUDA execution provider needs an `onnxruntime-gpu` build that matches the image's CUDA 13. On the 4 vCPUs of a g5.xlarge, expect roughly real time for that one model. Fine for the pilot; switch to `onnxruntime-gpu` before 200 h.
3. `load16k` uses `ffmpeg` from PATH, else the pinned `pipeline/ffmpeg.binary()` (read from `raw/_tools/`).
4. The punctuation model (`transcribe`) and Whisper and vakyansh are ungated; they download into the job's `data/w4/hf/`.

**Audio paths:** everything W4 holds now points at `data/w1/pool_pinned/` (all 648 records identical in every field to pool v2, so no stage re-run was needed):
- `review.tsv`'s `audio` column.
- `asr_run.py`'s pool set (`records.jsonl`).
- `saaras_run.py`'s audio.
- W1's vakyansh TSVs were copied to `data/w4/asr/vakyansh-w2v2.jsonl`; the benchmark scores are unchanged.
- Nothing in `w4/` reads `data/w1/pool/` or `data/w1/pilot/` any more.
- The frozen benchmark (`data/w4/bench/`) needs text only. Re-decoding its 103 retired clips is no longer possible, because W1 deleted their audio; their hypotheses are archived in `data/w4/asr/`.

### Handoffs

- **W5 (alignment):**
  - Align `data/w4/pool.jsonl`. `hi_words` must equal its tokens (SPEC §3), fillers `अ` `उम्म` `हम्म` `आ` included.
  - **`data/w4/pool.jsonl` holds only gate-accepted clips** (628, no flags), so align all of it. In S3 it is `stages/asr_check/LATEST`'s `records.jsonl` once the master registers the stage.
  - `_w1.segments` (W1) gives coarse clip-relative segment times for free.
  - The IndicConformer HF wrapper (MIT) also exposes CTC word timestamps (`compute_timestamps="w"`, 80 ms frames) from the best model measured here, a candidate for alignment confidence. It runs on CPU ONNX; its weights are 2.4 GB and need real files (`snapshot_download(local_dir=…)`), because onnxruntime rejects external data reached through HF cache symlinks.
- **W6 (translation):**
  - Sentences are model-split. 109 are marked `?`; intonation-only questions are not marked yet (review rule).
  - Self-corrections and repetitions are in the text (R10) for you to translate per the FAQ. You may re-split, since words never change.
  - Read-task clips are the entity-dense ones (17.5 number words/min, W1).
- **W3 (TTS QA):** use **IndicConformer-600M CTC** for round trips (AI4Bharat Conformer, a different family from Bulbul, IndicF5 and Parler):
  - Devanagari only; numbers as words 96%; English words 84%; read-speech CER 2.0%.
  - MIT, gated (access granted); CPU ONNX at 0.23× real time (8 shared threads, measured).
  - Second opinion: vakyansh wav2vec2 (MIT, CTC). Avoid Whisper without a digit/Latin normaliser, and MMS (NC).
- **W7 (assembly / rebuild):**
  - Rebuild W4 text by running the `transcribe` stage (pinned `punctuators==0.0.7`, model rev `d1769a5`; patches from `w4/patches.jsonl` = `archive/human/transcribe/`) and the `asr_check` stage. The gate's ASR hypotheses are uploaded with each `asr_check` run (`stages/asr_check/<run>/asr/*.jsonl`); seed them into a re-run's `--out` for an identical accept/reject split without re-decoding.
  - Release only ids in `data/w4/pool.jsonl`; `data/w4/rejects.jsonl` lists the rest with reasons.
  - Strip `_w4`, `_orig_text` and `_archive` at release (SPEC §8).
- **W1:**
  - **Retire these 16 P1 ids:** rec-000001, 000021, 000022, 000086, 000142, 000192, 000252, 000257, 000370, 000434, 000442, 000514, 000567, 000574, 000576, 000580. **Also retire these 4 P2 ids:** rec-000207, 000498, 000615, 000643. None of the 40 replacements failed.
  - New clips need nothing from W1 beyond the cut records: the `asr_check` stage decodes vakyansh, Whisper-turbo and IndicConformer itself.
  - **For SPRING-INX or any non-IV source, W4 needs a Latin→Devanagari and digit→words step** that doesn't exist yet: IV's L1 has neither. The converter flags `chars:` instead of dropping anything.
- **W8c:** `data/w4/ivvalid/` has 120 IV-valid segments (0.32 h, 60 telephone + 60 wideband) with converted references and four ASR outputs. It's a ready mini dev set for ASR-style probes.

## Status — 2026-10-14

**Done:**

1. **`transcribe` stage** (`w4/convert.py`): IndicVoices L1 → SPEC §2 for W1 pool v2 (648 clips), 0 violations. It is byte-identical across runs, and identical to the v1 run on the 608 shared clips.
2. **`asr_check` stage** (`w4/asr_check.py` + `review_list.check`): **628 accepted, 3.443 h, `validate.py --partial` 0 violations**; P1 16, P2 4. Copied to `data/w4/pool.jsonl` and `rejects.jsonl` by `review_list.py build`.
3. **ASR benchmark** on the frozen 1 h set: six open model configs plus three Saaras configs (82-clip subset), and a held-out IV-valid control.
   - IndicConformer (free, MIT) and Saaras v4-verbatim (closed) tie at the top: CER 4.7 vs 4.3 on the common subset.
   - **No model keeps fillers** (≤ 1 of 65). Whisper and MMS write Latin and digits.
4. **Decision: reuse IndicVoices L1.**
   - ASR-consensus disagreement with the reference is 0.53% of words, an upper bound on reference word errors.
   - ASR text would lose ≥ 98.5% of fillers.
   - ASR is used only for the gate and the audit draft.
5. **Policy implemented** (no annotators): P0/P1/P2 dropped automatically, with documented thresholds. One frozen, stratified audit sample (`w4/audit_ids.txt`; 209 listed, 1.10 h, `data/w4/review.tsv`), with the patch archive and staleness guard tested end to end.
6. **S3 plan Task 10** (§5): stage contract for `transcribe` and `asr_check`; `review_list.py import` → `archive/human/transcribe/` with `put_once`; three runnable checks, all `ok`. **Not committed** (the master commits).

**For W1 (retire):**
- P1: rec-000001, 000021, 000022, 000086, 000142, 000192, 000252, 000257, 000370, 000434, 000442, 000514, 000567, 000574, 000576, 000580.
- P2: rec-000207, **000498** (new in Task 10), 000615, 000643.
- **0 of the 40 replacements failed P1 or P2.**

**Sarvam spend: ≈ ₹58.5** (v4 ₹18.19, v4-verbatim ₹19.95, v3-verbatim ₹20.31; the cap was ₹500). **The account's credits ran out mid-run** (HTTP 402 "No credits available"), so each config covers 99–112 of 174 clips. Raw responses are archived in `data/w4/saaras/`. **Whoever holds the Sarvam account should check the balance: W3's Bulbul work and Part B's English targets (§3B) depend on it.**

**Disk:** `data/w4` = 3.5 GB of the 6 GB budget. That includes the IndicConformer weights (2.4 GB, `data/w4/ic600m/`), kept because every new W1 clip needs a draft for P2; delete them when the pool is frozen. **GPU:** every job ran under the flock, and none is left running.

**Open questions for organisers** (add to W0's list):
1. §3A says "following IndicVoices Level 1". Does the audit accept IV's L1 *phonetic* spellings (`बोहोत`, `हे`, `इशलिए`, `दौ` for "बहुत / है / इसलिए / दो")? Or does it expect standard spelling where the pronunciation is close (SPEC R5, Q9)? This decides whether we keep about 2% of words as IV wrote them or normalise them.
2. Are Devanagari filler words (`अ`, `अं`, `उम्म`, `हम्म`) the expected way to "include" fillers (§3A)? Each must then be timed in `hi_words`. Or may filler tags simply be dropped, given that "noise tags are optional"?
3. Is there any expectation on sentence granularity for spontaneous speech (§3A)? For example, is a model-placed split acceptable, and is an intonation-only question required to carry `?`?

**Next:**
1. When W1's next pool lands: `transcribe` → `asr_check` (via `pipeline/run.py` once registered, or locally as in §5) → `review_list.py build`. The audit sample stays frozen.
2. Harsh listens to `data/w4/review.tsv` once, then runs `uv run w4/review_list.py import`; report the audit WER in the mid-term report.
3. The master decides the SageMaker access route for the gated IndicConformer weights (§5, caveat 1).
4. Latin→Devanagari and digit→words conversion is still needed before any SPRING-INX or Vaani clip.

**Owned files:**
- `w4/` (commit all): `convert.py` (`transcribe` stage), `asr_check.py` (`asr_check` stage), `review_list.py` (gate rules, audit, import), `asr_run.py`, `asr_score.py`, `saaras_run.py`, `ivvalid_set.py`, `hf_purge.sh`, `test_convert.py`, `test_asr_check.py`, `test_review_list.py`, **`audit_ids.txt`**. `patches.jsonl` appears after the first import; commit it too.
- `data/w4/` (gitignored):
  - `stages/{transcribe,asr_check}/` (local stage runs; `asr_check/asr/` = the gate's archived hypotheses), `pool.jsonl`, `rejects.jsonl`, `review.tsv`
  - `asr/`: hypotheses; the scores are `scores.tsv`, `scores-saaras.tsv` and `ivvalid/scores.tsv`
  - `saaras/`: raw API responses
  - `bench/`: the frozen 1 h set
  - `ivvalid/`, `hf/` (punctuation model), `ic600m/`
