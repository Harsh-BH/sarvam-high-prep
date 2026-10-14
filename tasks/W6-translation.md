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

## Findings, 2026-10-14 (W6)

Code: `w6/` (stage `translate` registered in `pipeline/run.py`). Local data: `data/w6/` (gitignored; bench runs in
`bench2/` (IndicTrans2), `bench3/` (final prompt)). Every model call is archived under `archive/api/bedrock/` or
`archive/api/it2/` through `archive_call` (stream `w6`), so every run below re-runs for free and byte-identically.

### 1. Style guide v1: `w6/STYLE.md`

The "Rules" section is pasted verbatim into the translator and judge prompts, so the guide and the pipeline can't drift
(editing it changes every request key and re-translates). Basis of each rule:

| Rule | Basis |
|---|---|
| 1. One English segment per Hindi `sentences[]` entry, nothing moved across segments; fragments stay fragments | STATED: sentence-level translations, boundary mapping is ours to choose (§3A). INFERRED: 1:1 keeps W5's `hi_start/hi_end` usable as the alignment anchor for Part B pairs (Hibiki-Zero coarse alignment, prior-art §8d.5), and an English segment never says something its Hindi hasn't said yet (causal targets, §5.1 R2) |
| 2. Faithful; keep the speaker's order of information where English allows | STATED: §7 translation quality / intent / entity KPIs. INFERRED: Hindi-order targets lag less (§2 word order) |
| 3. Drop fillers, repetitions, false starts | INFERRED: the §7 human references are fluent English; spoken fillers in targets are what §10 R8 calls "filler speech" |
| 4. In-segment self-correction → corrected meaning only; a later segment that corrects an earlier one → explicit correction ("Sorry, not Thursday, Friday."); a contrast stays a contrast | STATED: FAQ "references encode the corrected meaning; speaking the abandoned value as fact is wrong". INFERRED: across segments, the only causal rendering is an explicit correction (moving the value backwards breaks rule 1). AMBIGUOUS: whether a scorer penalises the abandoned value when it is *negated* ("not zero"); safer reading = never assert it, mention it only negated |
| 5. `en_alt` only for in-segment self-corrections: the as-spoken repair ("three and a half lakh, no, no, three and three-quarter lakh") | STATED: `en_alt` optional (§5.2), never concatenated (§10 R8). OURS: see §2 below |
| 6. Numbers in words; thousand/lakh/crore, never million/billion; digit strings digit by digit with "double"/"triple" as said; letters as spaced capitals; acronyms as spelled | STATED: words (digits accepted), lakh/crore kept (§3A); equivalent forms accepted (FAQ). INFERRED: words give Bulbul one unambiguous reading (W8b) |
| 7. English words the speaker used come out as the same English word | STATED: FAQ "must come out correctly" |
| 8. Names romanised in usual spelling, never translated | INFERRED: §7 entity accuracy |
| 9. Negation, questions, conditions, requests explicit | STATED: §7 intent accuracy |
| 10. ASCII text only, no notes | `validate.py` `en_text` (no Devanagari) + OURS |

### 2. `en_alt` policy

- Only for segments where the abandoned value and its correction are both inside the segment; the alt is the
  as-spoken repair. It must pass strict checks or it is **dropped automatically** (it is optional): every Hindi number
  present, no extra number (`qa.check(spoken=True)`), and no critical judge error. Pilot: 5 alts kept of 312 segments.
- Why: no take-backs (§5.1 R2) means a streaming model that has already said "three and a half lakh" can only repair;
  the alt gives Part B a target for that, while `en_text` stays the FAQ's corrected-meaning reference. W8 decides
  whether to train on alts (each as its own target, §10 R8).
- AMBIGUOUS (audit): an auditor might read the abandoned value in an alt as "spoken as fact". Mitigations: explicit
  "no, no" repair, ≤ 2% of segments. Zero-risk option: W7 drops `en_alt` at release (one line); nothing else changes.
- Natural-vs-monotonic alternatives for every segment: deferred until the checkpoint is known (§3B allows text
  improvements after the mid-term).

### 3. `translation_method` vocabulary

`w6-style-v1/<Bedrock model id>`, plus `+repair` when the repair round's output was kept, plus `+human` when a
review patch (`w6/patches.jsonl`) replaced a segment.

### 4. Engine benchmark (pilot: 40 real + 50 synthetic records, 312 segments, final prompt)

Judge critical % = segments with ≥ 1 verified critical error (meaning, omission, addition, number, negation, intent,
correction, name). An engine is never judged by itself. Number flags = `qa.check` value/unit mismatches; I read every
flag to count the real ones.

| Engine (licence) | judge gpt-oss | judge Qwen3 | judge Mistral L3 | judge DeepSeek | number flags (real) | no_translation | loanwords restored |
|---|---|---|---|---|---|---|---|
| **Mistral Large 3** (Apache-2.0) | **16.3** | 27.9 | – | **24.0** | **3 (1)** | 0 | 87.5% |
| Qwen3-235B-A22B-2507 (Apache-2.0) | 17.2 | – | 29.1 | 24.6 | 15 (≈12) | 1 (split 3 segments into 4, 3 takes) | 85.6% |
| gpt-oss-120b (Apache-2.0) | – | 25.0 | 33.0 | 25.3 | 14 (≈12) | 0 | 88.8% |
| IndicTrans2 RoPE dist-200M (MIT) | 34.3 | 46.2 | 52.2 | 43.9 | 23 + digits in 4.5% of segments | 0 | 86.2% |

- **The judges can't separate the three LLMs (within ±1–3 points of each other); the number check can.** Qwen3 and
  gpt-oss misread Hindi number words: बयासी → "eighty-eight", तिरासी → "ninety-three", छियानवे → "seventy-nine",
  एक लाख सत्तानवे हज़ार → "one lakh seventy-nine thousand", सैंतीस → "seventy-three", पचास लाख → "fifty thousand",
  सवा छह बजे → "one-fifteen". Mistral's only real one: चार हज़ार नौ सौ निन्यानवे → "three hundred and ninety-nine". Entity
  accuracy is 18 of the 70 final points (§9.1) and targets teach the model to speak (prior-art W6: models copy their
  teacher), so **Mistral Large 3 is the engine**.
- IndicTrans2 doesn't follow the style at all: digits and "Rs." ("Rs. 125" for बारह सौ पचास), million ("two million"
  for बीस लाख, "125 million shares" for सवा करोड़), English number words in Devanagari read as homophones ("nine at
  seven six five for three to one"), the abandoned value stated as fact. Kept only as the benchmark baseline.
- Real clips are harder than synthetic for every engine (Mistral: 21% vs 12% critical, gpt-oss judge): verbatim L1
  spellings, fillers and mid-phrase sentence splits from W4's SBD (e.g. "… पटियाला। सलवार सूट के लगेंगे ही।").
- Sarvam-Translate: not run (API 402; weights GPL, T1). Llama/Gemma: not used (T1). IndicVoices-ST English: not used.

### 5. Judge benchmark

The judge must quote `en_span` (from the English), `hi_span` (from the Hindi) and a `fix`. Errors whose quotes don't
exist or whose fix equals the span are discarded in code (`translate.verified`). Before this, the DeepSeek judge
flagged 40% of segments with errors like "'पचासी' is eighty-five, not eighty-five". Then, as a synthetic control,
`bench.py inject` corrupts one clean segment per record, re-judges the record and reports recall on the corrupted
segment (Mistral output, 25 records per kind; unit and correction apply to fewer):

| Corruption | gpt-oss recall | Qwen3 recall | DeepSeek recall | automatic QA recall |
|---|---|---|---|---|
| number changed | **96** | 96 | 84 | 100 |
| negation removed | **100** | 100 | 96 | 80 |
| lakh → million (n=6) | 100 | 100 | 100 | 100 |
| name replaced | **96** | 80 | 92 | 0 |
| last clause dropped | 92 | 80 | 100 | 12 |
| abandoned value as fact (n=2) | 100 | 100 | 100 | 50 |
| critical flags on untouched segments | **0–15%** | 0–30% | 0–27% | – |

- **gpt-oss-120b is the production judge**: best recall overall and the fewest false flags, plus it's the cheapest and
  a different family from the engine. Qwen3 is the second judge for the review list. DeepSeek V3.2 (MIT) was a
  benchmark judge only.
- Judges agree poorly on *natural* errors (Cohen's κ 0.27–0.45 on the critical flag, `bench.py agree`) while
  catching ~95% of *injected* ones: natural flags are mostly borderline or wrong. Two consequences, both in the stage:
  1. **The number parser overrules the judge on values** (`translate.arbitrate`): a judge `number` error on a segment
     where every Hindi number appears in the English and none is extra is discarded. Example: gpt-oss claimed
     एक लाख सत्तानवे हज़ार should be "seventy-seven thousand". On the synthetic run this cut judge-critical
     segments from 9 to 2 of 165.
  2. The judge prompt lists what is *not* an error: an implied currency made explicit, discourse words, tense or word
     order, explicit cross-segment corrections.
- Repair round: on any critical flag (judge or `qa`), the engine gets its previous output plus the quoted issues
  ("either may be wrong") and re-translates. The result is kept only if the re-judged critical count drops. Known
  failure: the judge can prefer a worse repair (p043 "टमाटर पचास किलो था नहीं साठ किलो" came back with the
  abandoned value as fact). Residual flags go to the review pool.

### 6. Automatic QA (`w6/qa.py`, self-test `python3 w6/qa.py`)

| Check | How | Gate |
|---|---|---|
| Numbers and amounts, value + unit | Hindi number words → values (0–99 table incl. L1 spellings, सौ/हज़ार/लाख/करोड़, साढ़े/सवा/पौने/डेढ़/ढाई, English-in-Devanagari digits, डबल/ट्रिपल, digit strings as strings); English words/digits → values ("three and a half lakh", "a quarter to nine", "eight forty-five"); multiset compare. Missing values are soft inside a self-correction; extra values never | critical → repair |
| lakh/crore kept, no million/billion | regex | critical |
| Negation | negator counts; flag when only one side has any (correction sentences exempt) | soft (judge) |
| Length | English/Hindi word ratio outside 0.4–2.2 (Hindi ≥ 4 words) | soft |
| Leftover Devanagari, meta-text, empty | regex | hard → retry (3 takes), else `no_translation` reject |
| Digits in English | regex | soft (style) |
| English loanwords restored | IndicVoices L2 `normalized` glosses (`फ़ेमस [Famous]`; tags = brackets also in L1, so excluded) must appear in the English, real clips only | soft |

On the pilot (Mistral) the checks fire on 1.6% (numbers/units), 1.3% (negation), 0.6% (length) of segments;
loanword restoration 87.5% of 160 glossed words.
