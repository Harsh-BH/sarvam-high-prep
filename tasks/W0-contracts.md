# W0: Contracts and validator

**Goal:** freeze the rules every other stream codes against, by day 3. Afterwards the owner maintains them and becomes the W7 integrator.

## PS rules this stream encodes

- Audio: WAV, mono, 16-bit, 24 kHz; ≤60 s; one speaker per clip (§3A).
- Transcript: IndicVoices Level 1 ("as spoken"), disfluencies, repetitions and fillers included (§3A). In detail:
  - English words in Devanagari, no bracketed Latin.
  - Numbers as spoken words.
  - Every sentence ends with sentence-final punctuation.
  - Noise tags are optional.
- Word timings for every word: ordered, non-negative, inside the clip, in seconds from the start of the record (§3A, §5.2).
- Manifest format per §5.2. "No source field may be left empty". JSONL, one record per line.
- Hours (§3A):
  - Real audio: union of source-file intervals.
  - Synthetic audio: every distinct generated file counts.
  - Augmented copies: never records.
  - Cap: ≤200 h.

## Deliverables

- **`SPEC.md`:**
  - Transcription conventions:
    - IndicVoices L1 **minus all bracketed Latin**.
    - Fillers written in Devanagari (अं, उम्म, हम्म), each with word times.
    - Numbers exactly as spoken.
    - Acronyms as spaced Devanagari letters (एस बी आई).
    - Punctuation set `। ? !`, using U+0964, never ASCII `|`.
    - NFC normalisation; consistent nukta and ZWJ/ZWNJ handling.
  - ID scheme: `rec-XXXXXX`, stable forever.
  - `source_file` naming: globally unique per physical recording.
  - The intermediate JSONL format that every stage reads and writes. Extra internal fields are allowed there and stripped at release.
- **`validate.py`:**
  - Audio: format, sample rate, bit depth, channels; duration ≤60 s and equal to `duration`.
  - `hi_words`: the `w` sequence equals the tokens of `hi_text` with punctuation stripped; times satisfy `0 ≤ s_i < e_i ≤ s_{i+1}` with the last end ≤ duration.
  - `hi_text`: no `[A-Za-z]`, no ASCII digits, no `०-९`; ends with a final mark.
  - `source`: no empty fields; `model`/`voice` present for synthetic and absent for real.
  - Hours counter.
- **Organiser question list**: see the PS analysis, §10. Send the top questions on day 1.

## Research

- Get the full IndicVoices transcription guidelines. They're behind a login at ai4bharat.iitm.ac.in/indicvoices; the paper is arXiv 2403.01926.
- When Sarvam releases its sample records and official validator (§11 FAQ), diff them against our spec and fix ours.

## Done when

The validator rejects a deliberately broken record set (one bad record per rule) and passes a correct one.

## Organiser questions (ranked)

Each question gives what the PS states, the readings it allows, the safer one, and what we do until the answer arrives. The first four are marked **send day 1**: each one blocks a stream or could fail the §8 mid-term gate.

1. **Send day 1.** *Official validator and how `hi_words` pairs with `hi_text`.*
   - Stated: §11 FAQ promises sample records and a format validator; the release date is "to be finalised". §3A says every word in the transcript gets times.
   - Ambiguous: what counts as a "word". Is it whitespace tokens with punctuation stripped? Does the official validator require `hi_words[].w` to equal those tokens exactly, in order?
   - Ask: the release date, and the exact tokenisation rule.
   - Safer reading, and ours: exact token equality (SPEC §3). It's stricter, so it survives either answer.
2. **Send day 1.** *Fillers and noise tags.*
   - Stated: §3A says fillers are included, noise tags are optional, and English words go in Devanagari "without bracketed Latin forms".
   - Ambiguous, three ways:
     - Does "bracketed Latin" also ban IndicVoices-style tags such as `[umm]` and `[breathing]` (L1, paper Table 13), or only Latin glosses of English words (`ओटीपी [OTP]`)?
     - If tags are allowed, what is their format, and do they appear in `hi_words` with times?
     - Must fillers get word times?
   - Readings: (a) tags allowed and untimed; (b) no Latin in `hi_text` at all.
   - Safer, and ours: (b). No tags at all; fillers are Devanagari words with times (SPEC §2.2, R9).
3. **Send day 1.** *Own recordings.*
   - Stated: §3A says "real recordings only from datasets whose licence allows redistribution"; §10 R1 makes licence and consent our responsibility.
   - Ambiguous: can a team-recorded, consented set released under CC BY count as such a dataset?
   - Readings: (a) yes, with consent forms and a licence; (b) no, only pre-existing public corpora.
   - Safer: (b) until answered. W1 and W2 plan without own recordings.
4. **Send day 1.** *"One speaker per clip" (§3A).*
   - Ambiguous:
     - Does audible background speech (TV, a bystander) break it?
     - May we cut a single speaker's turns out of a two-party conversational recording (IndicVoices conversational and SPRING-INX are telephone dialogues)?
     - May such a clip contain short gaps where the other party spoke but is not audible?
   - Safer: no intelligible second voice anywhere in the clip; single-side cuts only where the channels are separate or the other side is silent (SPEC R11).
5. *Synthetic hours when the generated file ≠ the clip.*
   - Stated: §3A "every distinct generated file counts"; §5.2 says `source.start/end` locate the clip in the generated file.
   - Ambiguous: if we trim silence or cut two clips from one file, is the whole file counted, or only the used interval?
   - Safer: the whole file. We sidestep it with one file = one record, `start = 0`, no trimming (SPEC §7).
6. *Clip contiguity and the source file of pre-segmented corpora.*
   - Stated: §5.2 gives one `source_file` and one `[start, end]` per record.
   - Ambiguous:
     - May a record join non-contiguous audio (two utterances of the same speaker)?
     - When a corpus distributes segments of a longer recording, should `source_file` name the original long recording (with offsets in it) or the distributed segment file?
   - Readings: naming the segment file under-dedupes overlaps across segmentations (it over-counts hours, which is safe for the cap). Naming the long recording is exact.
   - Safer, and ours: contiguous only, and the long recording where the corpus gives offsets (SPEC §4).
7. *After the mid-term (§3B, §8 G3):* may we **delete** records, or **shrink** a record's interval (trim), while improving text? G3 only forbids adding records or hours. Safer: shrinking and deleting allowed, never growing. Ask before relying on it.
8. *`hi_start`/`hi_end` semantics (§5.2).*
   - Ambiguous: is it the exact span of the sentence's words, or may it include surrounding pause? Is untranscribed audio allowed at the clip edges (leading or trailing silence and noise)?
   - Ours: exact word span (SPEC §3); silence at the edges is untranscribed.
9. *L1 spelling tolerance.*
   - Stated: IndicVoices L1 keeps non-standard phonetic spellings (`हमाए` for `हमारे`, paper Table 5), and §3A demands L1.
   - Ambiguous: will the audit (§7) penalise standard spellings where the pronunciation deviates slightly, or penalise phonetic spellings as typos?
   - Ours: as heard; standard spelling where the sound fits it (SPEC R5).
10. *Clips shorter than 10 s.* §4 says test sessions run 10–60 s; §3A caps clips at 60 s and sets no minimum. Are short clips (e.g. 3–10 s read speech from Kathbath) acceptable records? Safer: yes, but keep most of the hours in 10–60 s.
11. *Upsampled telephone audio.* §4 says lower-bandwidth recordings are upsampled; §3A requires 24 kHz WAV. Confirm that an 8 kHz corpus recording resampled to 24 kHz is a valid real record and not an "augmented copy". Inferred yes (it's the only copy).

## Status — 2026-10-14

**Done:**
- `SPEC.md` v0:
  - transcription conventions R1–R12, grounded in §3A and IndicVoices sources, with a deviation table and a conversion table for W4;
  - the ID and `source_file` scheme;
  - §5.2 field by field;
  - the hours rules;
  - the intermediate stage-file format (`_` prefix, `--partial`, field ownership);
  - the validator rule table.
- `validate.py`:
  - stdlib only, plus a PEP 723 header so `uv run validate.py` works without a project;
  - 25 rules; one line per violation (`<id> <rule>: <detail>`); hours summary on stderr; exit code 1 on any violation;
  - `--partial` for stage files.
- `test_validate.py`:
  - a valid set of 4 records: 2 overlapping real cuts, and 2 synthetic files with the same text in two voices. It passes, with hours real = 3.0 s (the union) and synthetic = 3.0 s.
  - 31 broken records, plus a hours-cap run, cover all 25 rules. Each is caught by exactly its rule and nothing else.
  - It also covers partial vs release mode and the CLI exit codes.
- Organiser questions above.

**Measured:**
- `uv run test_validate.py` passes.
- Mutation check: 9 hand-made validator bugs (disabled rate check, loosened overlap check, widened character set, disabled NFC/interval/reuse/space/inner-mark checks, sentence-span check). The test catches all 9. Two of them initially slipped through, the overlap tolerance and the inner final mark; the test was tightened until both fail it.
- Validator speed: 200 × 30 s (1.67 h) in 0.12 s with the files in page cache. CPU is negligible; a full 200 h run is bound by reading about 35 GB of WAV from disk.

**Research findings (verified 2026-10-14; details in SPEC §2.1):**
- Paper §6 / Table 5 (p. 16): L1 is phonetic and keeps Latin glosses of English words plus 89 bracketed tags (Table 13, p. 45).
- Appendix D (pp. 46–52) is only the Hindi **L2** guideline. The full L1 guideline is behind a login (ai4bharat.iitm.ac.in/indicvoices). **Blocker, low:** someone on the team should log in, download the Hindi L1 guideline PDF, and drop it in `data/refs/` (not the repo, since its licence is unknown) so we can diff SPEC §2 against it.
- AI4Bharat's own cleaner drops `<unintelligible>` segments, strips the U+200B–U+200F joiners, and uses NFKC. On our character set NFKC equals NFC (checked over every allowed code point).

**Handoffs:**
- **W1:**
  - The IndicVoices raw release carries, per long-form recording, `verbatim` and `normalized` segment lists with `start`/`end` offsets and a per-segment `speaker_id`. AI4Bharat's own chunker skips segments with `speaker_id == 1` (`data_prep_scripts/utils_dataset_create_iv.py`).
  - [INF] So conversational files likely hold both parties in one channel. Clips must stay inside one speaker's segments (Q4).
  - Use the long-form file as `source_file`, with offsets in it (SPEC §4).
  - Their README downsamples the raw audio to 16 kHz before training. Measure the native rate of the raw files.
  - Allocate ids from `rec-000001`.
- **W2/W3:**
  - Synthetic ids from `rec-400000`; `source_file = tts/<id>.wav` (the kept take as returned).
  - One file = one record, and **no trimming** (SPEC §7).
  - `hi_text` must match what the TTS actually said (SPEC §8).
- **W4:** follow SPEC §2.3–2.4. The conversion table covers every IndicVoices tag form seen in AI4Bharat's cleaner.
- **W5:**
  - Word = the R4 token. Times are record-relative, 3 decimals, with no overlap.
  - Sentence spans are derived from the words.
- **W6:** `en_text` must not contain Devanagari, and `en_alt` is a list. The `translation_method` vocabulary is W6's to fix.
- **W7:**
  - Serialisation rules are in SPEC §6. Release = strip `_` keys, then validate without `--partial`.
  - Not built (YAGNI until the mid-term freeze): a `--against midterm.jsonl` G3 check (no new ids, no interval growth, hours not above the frozen total). It's about 20 lines in `validate.py`; ask W0 when it's needed.
  - Run the validator in CI on every build.

**Not done / deliberately skipped:**
- No check on WAV metadata chunks (W7 owns byte-identity).
- No `speaker_id` pattern check. It's optional in §5.2, so only "non-empty" is enforced.
