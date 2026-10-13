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
