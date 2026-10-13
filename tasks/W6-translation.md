# W6: Translation and target style

**Goal:** English translations that are faithful (audited at the mid-term) and that make good **streaming** targets (they become what the model learns to say, §3A).

## PS rules

- Sentence-level translations; you choose how literal they are and how sentence boundaries map (§3A).
- Numbers in words; digits are also accepted (§3A).
- **Amounts keep the speaker's units: lakh and crore** (§3A). LLMs tend to convert to millions; block that.
- A sentence may carry several translations via `en_alt`, which is optional (§5.2). In training, each alternative is a separate target; **never concatenate them** (that is scorer gaming, §10 R8).
- Self-corrections: the reference encodes the corrected meaning, and speaking the abandoned value as fact is wrong (§11 FAQ).
- English words spoken inside the Hindi must come out correctly (§11 FAQ).
- `translation_method` per record (§5.2).
- Translations can still be improved after the mid-term (§3B). Style v1 should be audit-grade; streaming variants can be refined once the checkpoint is known.

## Research

- **Literature:**
  - simultaneous interpretation vs offline translation;
  - monotonic translation ("interpretese");
  - wait-k policies;
  - Hibiki's contextual alignment;
  - how source and target sentence boundaries map in streaming.
- **Style guide v1:**
  - literalness and word order;
  - dropping disfluencies;
  - how self-corrections are rendered;
  - English passthrough;
  - name transliteration;
  - numbers in words with lakh/crore;
  - negations, conditions and questions made explicit.
- **`en_alt` policy:** store a natural and a monotonic variant? Check whether odd-looking variants could cost audit points.

## Engine benchmark (on the pilot)

- Engines: IndicTrans2, Sarvam Translate (credits), open LLMs, and closed LLMs if their terms allow (declare them, §10 R2).
- Metrics:
  - entity preservation (automatic);
  - negation and intent preservation (LLM judge + manual review);
  - human preference on 100 samples.

## Automatic QA

- Every Hindi number and amount appears in the English with the same value and unit.
- A negation-count heuristic.
- A length ratio check.
- Untranslated Devanagari left in the English.
- English loanwords correctly restored ("क्रेडिट कार्ड" → "credit card").

## Done when

Style guide v1, the translation pipeline with QA checks, and 200 manually reviewed examples.
