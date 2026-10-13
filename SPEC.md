# SPEC: the Part A data contract

v0, 2026-10-14. Owner: W0. This file is binding for W1–W7. `validate.py` enforces every rule marked ✔ (the rule name follows the tick), and `uv run test_validate.py` proves that it does.

**Tags:**
- **[PS §x]**: stated in `ps.md`.
- **[INF]**: our inference from the PS.
- **[OURS]**: our own choice, not a PS rule. Chosen to be the safe reading.
- **[Qn]**: an open ambiguity; see question *n* in `tasks/W0-contracts.md`.

**Changing this spec:** edit this file, `validate.py` and `test_validate.py` together in one commit, then tell every stream. When Sarvam releases its sample records and format validator (§11 FAQ), diff them against this file the same day.

## 1. Audio

| Rule | Basis | Check |
|---|---|---|
| WAV, PCM, 16-bit, mono, 24,000 Hz | §3A, §4 | ✔ `audio_format` `audio_depth` `audio_channels` `audio_rate` |
| 0 < duration ≤ 60 s | §3A | ✔ `duration_max` |
| `duration` equals the WAV length within 2 ms | [OURS] (§5.2 gives no tolerance) | ✔ `duration_match` |
| File sits at `audio/<id>.wav` | §5.2 example; strict form [OURS] | ✔ `audio_path` |
| One speaker per clip | §3A | Manual: W1/W3 QA. If another person's speech is intelligible, recut or drop the clip [OURS, Q4] |
| Aim for 10–60 s clips | §4 describes the test sessions, not a dataset rule [INF] | not checked [Q10] |
| Telephone audio upsampled to 24 kHz is a real record, not augmentation | §4 "Lower-bandwidth recordings are upsampled"; §3A defines augmentation as *copies* [INF] | [Q11] |
| No augmented audio in the dataset; augment only during training | §3A | n/a |

## 2. Transcript (`hi_text`)

### 2.1 What IndicVoices Level 1 actually is

§3A requires IndicVoices Level 1. Sources read on 2026-10-14:

- **Paper, arXiv [2403.01926v1](https://arxiv.org/abs/2403.01926) §6 (PDF p. 16):**
  - Two levels. L1 is "transcription performed exactly as spoken", i.e. phonetic fidelity with non-standard spellings kept: "muje" stays "muje", and the Table 5 Hindi row is L1 `हमाए गाँव में` vs L2 `हमारे गाँव में`.
  - L1 transcribers also "add standard tags such as [baby_crying], [barking], [hmm], [horn]". There are 89 tags (Table 13, p. 45), including the filler-like `hmm umm uhh uh-huh ugh` and also `stammers` and `unintelligible`.
  - The Tamil row of Table 5 shows L1 English loanwords in native script followed by a bracketed Latin form (`ஓடிபி [OTP]`), plus inline tags (`[breathing]`).
- **Paper, Appendix D (pp. 46–52): the Hindi Level 2 guideline.** The PDF's Devanagari doesn't survive text extraction, but the romanised examples do. The rules that matter to us:
  - English words "in isolation" get the English spelling in brackets, `[amazon]` (p. 49).
  - Acronyms spoken letter by letter are written spaced: S B I (p. 49).
  - Acronyms spoken as one word are written as one word: ASAP, BhaJPa (p. 50).
  - Numbers uttered in English are written as spoken, in the target script (p. 50).
  - Echo words are hyphenated: chai-vai (p. 50).
  - English words with Hindi suffixes keep the suffix: calendar-on (p. 51).
  - The full per-language L1/L2 guidelines sit behind a login at ai4bharat.iitm.ac.in/indicvoices (footnote 8, p. 46). **We have not seen the full L1 text.**
- **AI4Bharat's own data prep, [github.com/AI4Bharat/IndicVoices](https://github.com/AI4Bharat/IndicVoices) `data_prep_scripts/`:**
  - Released text carries tags as `[tag]` or `<tag>`. The cleaner deletes them, deletes U+200B–U+200F (ZWSP, ZWNJ, ZWJ, LRM, RLM), applies NFKC, and **drops any segment containing `<unintelligible>`** (`utils_dataset_clean.py`).
  - The raw release has, per long-form recording, parallel `verbatim` and `normalized` segment lists. Each segment has `start`/`end` offsets into the long-form WAV and a `speaker_id` (`utils_dataset_create_iv.py`).
  - [INF] `verbatim` is L1 and `normalized` is L2.
  - Their Hindi character set (`artifacts/dictionaries/hi.dict.txt`) is the Devanagari block minus the digits ०–९.

### 2.2 Where we deviate from IndicVoices L1

| IndicVoices L1 | Ours | Why |
|---|---|---|
| English word followed by a Latin gloss: `ओटीपी [OTP]` | Devanagari only: `ओ टी पी` | §3A "without bracketed Latin forms" |
| Filler tags `[hmm] [umm] [uhh] [uh-huh]` | Devanagari words (§2.3 R9), each timed | §3A: fillers included, every word timed; no Latin |
| Other noise/event tags | Not written | §3A: tags are optional. They aren't words, so there's nothing to time, and they would need brackets/Latin [OURS] |
| `[unintelligible]` | Not allowed: recut around it or drop the clip | Every word must be transcribed and timed (§3A); AI4Bharat drops these segments too [OURS] |
| `[stammers]` | Write the heard fragment as a word | §3A "exactly what was said" [OURS] |
| Echo words hyphenated (L2 rule) | Two words, no hyphen: `चाय वाय` | Each word gets its own timing; hyphen is outside the character set [OURS] |
| Commas and other punctuation | Only the final mark | Strict subset of §3A [OURS] |
| NFKC | NFC | Task decision; identical on Devanagari for our character set [OURS] |

### 2.3 Rules

- **R1. Characters** ✔ `hi_chars`:
  - Allowed: Devanagari U+0900–U+0963 and U+0971–U+097F, the ASCII space, and the final mark (R3).
  - That excludes:
    - Latin letters;
    - ASCII digits and Devanagari digits ०–९ (U+0966–U+096F);
    - U+0964 `।` anywhere except the end;
    - `॥` and the abbreviation sign `॰`;
    - all symbols, including `₹ % / - , . : ( ) [ ] ' "`;
    - ZWJ, ZWNJ, ZWSP, NBSP and BOM.
  - Basis: §3A (numbers as words, no bracketed Latin); the exact set is [OURS].
- **R2. Unicode** ✔ `hi_nfc`:
  - Text is NFC.
  - Consequence: the nukta letters U+0958–U+095F never appear. NFC decomposes them, so क़ is stored as U+0915 U+093C (checked in Python).
  - ZWJ/ZWNJ are removed (R1). They only change rendering, never sound. [OURS]
- **R3. Sentences** ✔ `hi_final`:
  - One `sentences[]` entry holds one sentence.
  - `hi_text` ends with exactly one of `।` (U+0964, never ASCII `|` or `.`), `?` or `!`. The mark is attached to the last word, and no mark appears elsewhere. §3A requires the final mark; the set is [OURS].
  - Use `?` for questions, including intonation-only ones; `!` for exclamations; `।` otherwise.
  - Where to cut sentences in spontaneous speech is decided by W4 (and W2 for scripts), with W6. Re-splitting never changes words or timings: sentence times derive from the words (§3).
- **R4. Words** ✔ `hi_space`:
  - Words are separated by exactly one U+0020, with no leading or trailing space.
  - **A word is a space-separated token of `hi_text` after the final mark is stripped.** This is the unit `hi_words` times. [OURS, Q1]
- **R5. As spoken (L1):**
  - Spell what this speaker said (IndicVoices L1, Table 5).
  - Where the pronunciation fits the standard spelling, use the standard spelling. [OURS, Q9]
- **R6. English words:**
  - Write them in Devanagari as this speaker pronounced them, with no Latin anywhere (§3A). Use candra-o `ॉ` when it is audible: `ऑफिस`, `डॉक्टर`.
  - An English stem with a Hindi suffix is one word: `बैंकों`, `कार्डवा` (IndicVoices L2, p. 51).
- **R7. Numbers** are written as spoken words, in the language they were spoken in (§3A):
  - `पाँच लाख`, `ढाई सौ रुपये`, `साढ़े तीन बजे`.
  - Numbers spoken in English: `ट्वेंटी फाइव`.
  - Digit strings get one word per spoken item: `नौ आठ डबल सात`. [OURS]
  - Never symbols (₹, %, /).
- **R8. Acronyms** (IndicVoices L2 pp. 49–50):
  - Spoken letter by letter: spaced letter names, one word each: `एस बी आई`, `यू पी आई`, `ओ टी पी`.
  - Spoken as a word: one word, e.g. `नासा`.
  - Default letter names, unless this speaker says them differently: ए बी सी डी ई एफ जी एच आई जे के एल एम एन ओ पी क्यू आर एस टी यू वी डब्ल्यू एक्स वाई ज़ेड. [OURS]
- **R9. Fillers are words.**
  - Written in Devanagari, kept in `hi_text`, timed in `hi_words` (§3A).
  - Canonical spellings [OURS]: `अ` (uh), `अं` (nasal uh), `आ` (long aa), `ए` (eh), `उम्म` (umm), `हम्म` (hmm).
  - Lexical fillers (`मतलब`, `यानी`, `वो`, `लाइक`, `एक्चुअली`) are ordinary words.
- **R10. Disfluencies** (§3A):
  - Repetitions are written every time: `मैं मैं`.
  - A false start or cut-off word is written as the heard fragment, with no hyphen: `ब बैंक`.
  - Self-corrections are written in full: `पाँच नहीं नहीं छह लाख`. How the correction is translated is W6's job (§11 FAQ).
- **R11. Non-speech:**
  - No tags (see §2.2). Laughter, breath, cough and noise are not transcribed.
  - The clip contains no unintelligible speech and no intelligible speech from anyone but the speaker. [OURS, Q4]
- **R12. Orthography** [OURS; for consistency only, nothing about sound]:
  - Before a consonant, use the anusvara: `हिंदी`, `बंद`, `ठंडा`, not `हिन्दी`.
  - For a nasal vowel, use chandrabindu `ँ` when no matra rises above the headline (`हाँ`, `माँ`, `पाँच`), and the anusvara when one does (`में`, `नहीं`, `हैं`).
  - Nukta as heard: `ज़`/`फ़` only when the speaker says z/f; `ड़ ढ़` as in `पढ़ना`, `सड़क`.

### 2.4 Converting IndicVoices L1 text (W4)

| In the corpus text | Do |
|---|---|
| `देवनागरी [Latin]` | Drop the bracket, keep the Devanagari |
| `[hmm]` / `<hmm>` | `हम्म` (timed) |
| `[umm]` | `उम्म` |
| `[uhh]` | `अ` or `अं`, as heard |
| `[uh-huh]`, `[stammers]` | Listen and write what was said; drop it if the voice isn't this speaker's |
| `[unintelligible]` | Recut around it, or drop the clip |
| `[talking]`, `[children_talking]`, … | Listen: if a second voice is intelligible, recut or drop; otherwise delete the tag |
| Any other tag | Delete |
| Digits, Latin words | Rewrite as spoken words (R6–R8) |
| `,` `-` `.` `|` etc. | Delete; a `.` or `|` at a sentence end becomes `।` |
| ZWJ / ZWNJ, non-NFC text | Delete, then NFC |

## 3. Word timings (`hi_words`)

- **One entry per word, across all sentences, in order** ✔ `words_match`:
  - `[x["w"] for x in hi_words]` must equal the concatenation of every sentence's words (R4).
  - That includes English words, fillers and fragments (§3A).
- **Units:** seconds from the start of the record's audio, not of the source file (§5.2). Subtract `source.start`.
- **Order** ✔ `word_times`: `0 ≤ start_i < end_i ≤ start_{i+1}`, and the last `end ≤ duration`.
  - §5.2 says "ordered, non-negative and inside the clip". Forbidding overlap is our stricter reading [OURS].
- **Sentence span** ✔ `sentence_times`: `hi_start` equals the first word's `start`, and `hi_end` equals the last word's `end`. [OURS, Q8]
- **Precision** ✔ `time_precision`: every time in the record (`duration`, `source.start/end`, sentence and word times) has at most 3 decimals. [OURS; keeps the rebuild byte-identical, §3A]
- **Word span** = the word's acoustic extent; pauses belong to no word. [OURS; guidance for W5]

## 4. IDs and names

- **`id`** ✔ `id` ✔ `id_unique`:
  - Form: `rec-` + 6 digits (§5.2 example; the width is [OURS]).
  - Assigned once, when the candidate clip is created.
  - Never renumbered, never reused, not even after the candidate is dropped (§3B, §8 G3). Gaps are fine.
  - Ranges, so streams don't collide [OURS]:

    | Range | Owner |
    |---|---|
    | `rec-000001`–`rec-399999` | real (W1) |
    | `rec-400000`–`rec-799999` | synthetic (W3) |
    | `rec-800000`+ | reserved |

    Each range has exactly one allocator.
- **`source.source_file`**: one string per physical recording. It is globally unique and never renamed (§3A hours union, §8 G3; format [OURS]).
  - **Real:** `<corpus-key>/<path as the corpus distributes it>`.
    - `corpus-key` is a lowercase slug: `indicvoices`, `spring-inx`, `vaani`, `kathbath`, …
    - It must name the recording the offsets are measured in. Where the corpus ships long-form recordings plus segment offsets (IndicVoices does, §2.1), use the long-form file and recompute offsets in it, so that overlapping cuts dedupe in the hours count [Q6].
    - If two corpora contain the same physical recording, pick one key and use only that one.
  - **Synthetic:** `tts/<id>.wav`. This is the archived, kept take, exactly as the engine returned it (before any resampling). Rejected takes are archived elsewhere and never referenced (§3A).
- **`speaker_id`** (optional, §5.2):
  - If present, it must be a non-empty string ✔ `schema`.
  - Form [OURS]: `spk-<corpus-key>-<corpus speaker id>` for real speakers; `spk-tts-<model>-<voice>` for synthetic voices.

## 5. Manifest, field by field (§5.2)

The release manifest is JSONL, one record per line. No other keys are allowed at any level ✔ `schema`.

| Field | Type | Req. | Rule | Check |
|---|---|---|---|---|
| `id` | str | yes | §4 | `id`, `id_unique` |
| `audio` | str | yes | `audio/<id>.wav`, resolved against the dataset root | `audio_path` |
| `duration` | number | yes | Seconds; 0 < d ≤ 60; equals the WAV length | `duration_max`, `duration_match` |
| `speaker_id` | str | no | §4 | `schema` |
| `source.corpus` | str | yes | Real: corpus name. Synthetic: where the Hindi *text* came from, e.g. `w2-scripts-v1` (§5.2) | `source_empty` |
| `source.source_file` | str | yes | §4 | `source_empty`, `source_reuse` |
| `source.start`, `source.end` | number | yes | `0 ≤ start < end`; `end − start` equals `duration` within 2 ms (a contiguous cut, no speed change [OURS]). Synthetic: `start = 0` (§7) | `source_interval` |
| `source.synthetic` | bool | yes | — | `source_synthetic` |
| `source.model`, `source.voice` | str | iff synthetic | Present and non-empty for synthetic (§5.2). **Absent** for real: §5.2 lists them only for synthetic, and "no source field may be left empty" forbids empty placeholders [INF] | `source_synthetic`, `source_empty` |
| `sentences` | list | yes, ≥ 1 | Objects `{hi_text, hi_start, hi_end, en_text, en_alt?}` | `schema` |
| `sentences[].hi_text` | str | yes | §2 | `hi_*` |
| `sentences[].hi_start`, `hi_end` | number | yes | §3 | `sentence_times` |
| `sentences[].en_text` | str | yes | Non-empty, no Devanagari [INF]. Style is W6's | `en_text` |
| `sentences[].en_alt` | list[str] | no | Non-empty strings, no Devanagari (§5.2). Never concatenated into one target (§10 R8) | `en_text` |
| `hi_words` | list | yes | Objects with exactly `{w, start, end}` (§3) | `words_match`, `word_times` |
| `translation_method` | str | yes | Non-empty, per record (§5.2). W6 fixes the vocabulary | `schema` |

"No `source` field may be left empty" (§5.2): every source key that is present must be non-null and non-blank ✔ `source_empty`.

## 6. Serialisation [OURS; W7 relies on this for the byte-identical rebuild, §3A]

- UTF-8 without a BOM; LF line endings; one record per line; records sorted by `id`.
- Write each line with `json.dumps(rec, ensure_ascii=False)`, keys in the §5 order.
- Every time is passed through `round(x, 3)` before writing.

## 7. Hours ✔ `hours_cap`

- **Real:** for each `source_file`, take the union of its `[start, end]` intervals; sum over files (§3A, §5.2). Overlapping or repeated cuts count once.
- **Synthetic:** every distinct generated file counts in full, including the same text in another voice (§3A). We make "the file" and "the clip" identical:
  - one generated file = one record, `start = 0`, `end = duration` ✔ `source_interval` ✔ `source_reuse`;
  - **no trimming of TTS output.** Need it? Raise it with W0 first [OURS, Q5].
- **Never both:** a `source_file` is either real or synthetic, never both ✔ `source_reuse`.
- **Cap:** total ≤ 200 h (§3A). The validator prints real, synthetic and total hours on every run.
- After the mid-term, records and hours never grow (§3B, §8 G3).

## 8. Intermediate JSONL (stage files)

- **Record shape:** every stage reads and writes JSONL records with the §5 shape, joined on `id`. Serialisation as in §6.
- **Fields:** a stage writes only the fields it owns (table below). Fields not yet filled are simply absent.
- **Internal fields:**
  - Extra fields are allowed only with a leading `_`, at any level (record, `source`, sentence, word).
  - Any other unknown key is an error, which catches typos.
- **Validating:**
  - Stage files: `uv run validate.py stage.jsonl --audio-dir DIR --partial`. This skips absent fields and allows `_` keys; every field that *is* present gets the full check.
  - Release: drop every `_` key recursively, then run `validate.py` without `--partial`.
- **Location:** stage files live under `data/<stream>/` (gitignored). Archive handling belongs to W7.

| Field | Written by |
|---|---|
| `id`, `audio`, `duration`, `source.*`, `speaker_id` | W1 (real), W3 (synthetic) |
| `sentences[].hi_text` (sentence split included) | W4 (real); W2 script, corrected by W3 to what the TTS actually said (synthetic) |
| `hi_words`, `sentences[].hi_start/hi_end` | W5 |
| `sentences[].en_text/en_alt`, `translation_method` | W6 |
| Final assembly, `_` stripping, release | W7 |

Shared internal fields (optional, by convention):

| Field | Meaning |
|---|---|
| `_rev` | Pinned revision (HF commit or corpus version) of the source audio, so the rebuild fetches the same bytes |
| `_orig_text` | Corpus or ASR transcript before conversion to §2 |
| `_archive` | `{stage: path}` of the archived raw output each field came from, so the rebuild never re-calls an API (§3A) |
| `_qa` | `{stage: {score: value}}`: ASR CER, aligner confidence, TTS round-trip and so on |

## 9. Validator rules

| Rule | Meaning | Spec |
|---|---|---|
| `json` | Line is not a JSON object | §5 |
| `id` / `id_unique` | Malformed / duplicated id | §4 |
| `schema` | Missing, unknown or wrongly typed key; empty `translation_method`/`speaker_id` | §5 |
| `audio_path` | `audio` ≠ `audio/<id>.wav`, or file missing | §1 |
| `audio_format` | Not a readable PCM WAV, or truncated | §1 |
| `audio_channels` / `audio_rate` / `audio_depth` | Not mono / 24 kHz / 16-bit | §1 |
| `duration_max` | `duration` ≤ 0 or > 60 | §1 |
| `duration_match` | `duration` differs from the WAV length by > 2 ms | §1 |
| `hi_chars` / `hi_nfc` / `hi_final` / `hi_space` | R1 / R2 / R3 / R4 | §2.3 |
| `en_text` | Empty, or contains Devanagari (`en_text`/`en_alt`) | §5 |
| `words_match` | `hi_words` ≠ the transcript's words | §3 |
| `word_times` | Breaks `0 ≤ s < e ≤ next s`, or last end > duration | §3 |
| `sentence_times` | `hi_start`/`hi_end` ≠ its first/last word | §3 |
| `time_precision` | A time with more than 3 decimals | §3 |
| `source_empty` | A source key missing, null or blank | §5 |
| `source_synthetic` | `synthetic` not a bool, or model/voice presence wrong | §5 |
| `source_interval` | Bad interval, `end − start` ≠ `duration`, or synthetic `start` ≠ 0 | §5, §7 |
| `source_reuse` | Synthetic file in > 1 record, or a file both real and synthetic | §7 |
| `hours_cap` | Total > 200 h | §7 |

The validator does **not** check anything that needs ears or a model: whether the transcript is correct, one speaker per clip, translation quality, or alignment accuracy. Those are the QA gates of W3, W4, W5 and W6.
