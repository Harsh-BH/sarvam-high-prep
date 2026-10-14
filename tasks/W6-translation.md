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

### 7. Production runs (stage `translate`, Mistral Large 3 + gpt-oss judge + one repair round)

| Run | Input | Records | Segments | Rejects | `validate.py --partial` |
|---|---|---|---|---|---|
| `stages/translate/20261014T110654Z-0246ad0` (**LATEST**) | `stages/transcribe` LATEST (real) | 642 | 2,330 | 0 | 0 violations, 3.498 h |
| `stages/translate/20261014T110622Z-0246ad0` | `stages/tts` LATEST (synthetic pilot) | 50 | 165 | 0 | 0 violations, 0.179 h |

Copies in `data/w6/final/{real,synthetic}/`. Older runs under `stages/translate/` are superseded (checker fixes only;
same archived translations).

| Measure | Real | Synthetic |
|---|---|---|
| Segments with a verified critical judge error (after repair) | 181 (7.8%) | 2 (1.2%) |
| Segments with any verified judge error | 275 (11.8%) | 5 (3.0%) |
| Segments with an automatic critical flag (number/unit/hard) | 64 (2.7%) | 1 (0.6%) |
| Records repaired (`+repair` kept) | 203 / 642 | 11 / 50 |
| Judge errors discarded (quotes didn't verify) / number claims overruled by the parser | 64 / 8 | 6 / 3 |
| `en_alt` kept / dropped by the strict check | 13 / 13 | 3 / 6 |
| English loanwords restored (IndicVoices L2 glosses) | 89.0% of 2,225 | – |
| Segments flagged critical by **both** gpt-oss and Qwen3 (review pool) | 99 of 2,495 segments in total (4.0%) | |

- Real speech carries most residual flags. Reading a random 14 of them: about 40% are real errors and the rest are
  judge misreadings (देख तो लुंगी → "lungi"). INFERRED from my reading only; part A of the review list measures it.
  The real ones are mostly Mistral **completing a trailing fragment** ("I like chicken … there are many varieties of
  chicken" for a sentence W4's split cut off), a kept "uh", and an omitted noun. Real-speech number flags are mostly
  checker misses of L1 spellings; the real number errors found were "पान सौ" (L1 पाँच सौ) → "one hundred and fifty" and
  "एक सौ बीस तीस" → "one hundred and twenty-three".
- Self-corrections (W3 handoff: p001, p005, p013 …): in-segment ones render the corrected value only, with the
  as-spoken repair in `en_alt` where it passes the strict check (p001 "three and three-quarter lakh"); cross-segment ones
  are explicit (p005 "Sorry, the last three digits are two one nine, not zero."; p008 "Sorry, not Thursday, Friday.");
  contrasts stay contrasts (p040 "thirty kilos, not twenty-five"). Known bad one: p043 (टमाटर पचास किलो था नहीं साठ किलो),
  which the repair round made worse; it is in the review pool.
- Bugs found in the first full run and fixed before these runs: the `meta` check rejected legitimate "… in English"
  (2 records); Mistral split one long segment in all 3 takes (the retry now states the segment count); the parser
  merged "दो हज़ार बीस दो हज़ार इक्कीस", misread spoken years and सो/थाऊजंड spellings.

### 8. The one review list for Harsh (optional, ≈ 20–25 min, 45 items, ≈ 2,300 words)

`data/w6/review.tsv` (open in any spreadsheet; Hindi context in `context_before`). Fill `answer`, optionally `fix`:

- **Items 1–20 (part A):** the judges say this English has a critical error (their claims are in `judges_say`).
  `y` = real error, `n` = the English is fine. If `y`, write the corrected English in `fix`. Measures judge precision.
- **Items 21–35 (part B):** nothing was flagged. `y` = there is an error anyway (+ `fix`), `n` = fine. Measures the
  residual error rate the automatic QA misses.
- **Items 36–45 (part C):** two engines' English for the same Hindi, sides randomised. `a`, `b` or `=`. Human
  preference check of the engine choice.

Then: `uv run w6/review.py import /home/harsh1/github-repos/sarvam-high-prep/data/w6/review.tsv` prints the three
numbers for the mid-term report and writes the fixes to `w6/patches.jsonl` (commit it) and
`archive/human/translate/` (write-once). Re-run the stage to apply them (archive hits, ≈ 2 min): fixed segments get
`translation_method` `+human`, and a fix is refused if that English changed since the review.
This replaces the task file's "200 manually reviewed examples" and "human preference on 100". The automatic proxies
are the injected-error recall (§5), two-judge agreement, the number parser and loanword glosses on 100% of segments.

### 9. Declarations (§10 R1, R2) and licences

| Used for | Model / tool | Licence | Access |
|---|---|---|---|
| **Released `en_text` / `en_alt`** | Mistral Large 3 675B Instruct (2512), Bedrock id `mistral.mistral-large-3-675b-instruct` | Apache-2.0 (open weights) | Amazon Bedrock ap-south-1, paid API (declare under §10 R2) |
| Judge + repair feedback (QA metadata only, `_qa`, stripped at release) | gpt-oss-120b, `openai.gpt-oss-120b-1:0` | Apache-2.0 | Bedrock |
| Benchmark engine; second judge for the review list | Qwen3-235B-A22B-Instruct-2507 | Apache-2.0 | Bedrock |
| Benchmark judge only | DeepSeek-V3.2 | MIT | Bedrock |
| Benchmark baseline only | IndicTrans2 indic-en dist-200M, RoPE variant `prajdabre/rotary-indictrans2-indic-en-dist-200M` @ `00213ee8` (linked from the official IndicTrans2 README; the gated `ai4bharat/*` repos aren't granted to us) + IndicTransToolkit 1.1.1 | MIT / MIT | local GPU, weights deleted after use |

Traps T1–T3 (prior-art): none hit. No Sarvam API (402), no Llama 3.x or Gemma 3 output, no NC model, no
IndicVoices-ST English, nothing GPL in code or data. Bedrock passes outputs to us with no ownership claim; the models'
own licences are permissive.

### 10. Spend (all W6; Bedrock on-demand, ap-south-1; `uv run w6/bench.py spend`)

| Model | Calls | USD at list price |
|---|---|---|
| DeepSeek V3.2 (benchmark judge) | 1,000 | 1.92 |
| Mistral Large 3 (engine) | 1,702 | 2.14 |
| gpt-oss-120b (judge, benchmark) | 1,818 | 1.89 |
| Qwen3-235B (benchmark, second judge) | 1,453 | 1.30 |
| **Total** | **5,973** | **7.26** |

SageMaker: $0. IndicTrans2 ran on the laptop GPU. The archive's pre-call estimate (the cap mechanism) has charged ₹962
of the ₹1,250 W6 cap; it over-counts by about 1.5×. **Production cost (measured):** the 2,096 archived calls the final
692 records reference cost $2.35, i.e. $0.0034 per record or $0.64 per audio hour (translation + judge + repairs).
**Scaling (INFERRED):** about $130 for a 200 h dataset, which needs a cap raise when the pool grows. Re-runs after a
hi_text change only pay for the changed records.

## Status — 2026-10-14

**Done:**
- Style guide v1 (`w6/STYLE.md`) with its basis table, `en_alt` policy and `translation_method` vocabulary (§1–3 above).
- `translate` stage (`w6/translate.py`, registered in `pipeline/run.py`): stage contract, resumable,
  `validate.py --partial` 0 violations on both production runs. Every call goes through `archive_call`, so a
  re-run or rebuild reproduces it byte for byte with no new spend.
- Engine benchmark on the 90-record pilot: 4 engines × up to 4 judges; **Mistral Large 3** chosen on number
  accuracy (1 real number error in 312 segments vs ≈12 for Qwen3 and gpt-oss).
- Judge benchmark with injected errors: **gpt-oss-120b** chosen (recall 92–100% per error type).
- Automatic QA (`w6/qa.py`): numbers with value and unit, lakh/crore, negation, length, Devanagari/meta, digits,
  loanword restoration. Checks: `python3 w6/qa.py`, `uv run w6/test_translate.py` (fake LLM: repair, alt guard,
  arbitration, count retry, patches, resume, review import).
- Production: 692/692 records translated (642 real, 50 synthetic), 0 rejects. Residual: 7.8% (real) and 1.2%
  (synthetic) of segments carry a verified critical judge error, and 99 segments (4.0%) are flagged by both judges.
- One review list for Harsh (`data/w6/review.tsv`, 45 items) plus an import that turns his answers into report
  numbers and patches.

**Spend:** $7.26 Bedrock at list price (5,973 calls, all archived); SageMaker $0; Sarvam $0. Production share
$2.35 ($0.64 per audio hour).

**Disk:** `data/w6` 18 MB. IndicTrans2 weights (0.85 GB) and its uv environment were deleted after the benchmark.
Its outputs are archived (`archive/api/it2/`).

**For Harsh (one list, optional, ≈ 20–25 min):** fill `answer`/`fix` in `data/w6/review.tsv` (instructions in §8),
then run `uv run w6/review.py import /home/harsh1/github-repos/sarvam-high-prep/data/w6/review.tsv` and re-run the
stage. Nothing else needs a human: no gated-repo click, no listening.

**Handoffs:**
- **Master (shared pipeline code; I didn't edit it):**
  1. `pipeline/s3.py` `_code()` raises `AttributeError: 'NoneType' object has no attribute 'get'` when a botocore
     error carries `response=None` (`ReadTimeoutError`, `EndpointConnectionError`), so a transient network error
     inside `get`/`put_once` crashes with the wrong exception. Seen 3 times today. Fix:
     `(getattr(e, "response", None) or {})`. The archive spool kept every paid response; the stage resumed fine.
  2. `archive.spent()` GETs every archive object serially (≈ 6,000 now): several minutes per process start, ×N
     parallel processes. W6 works around it with `translate.prefill_spent()` (same sum, 32 threads, ≈ 40 s). A spend
     index, or this parallel scan inside `archive.py`, would fix it for every stream.
  3. Two inputs, one `LATEST`: `translate` ran on real and synthetic separately. LATEST = the real run
     (`20261014T110654Z-0246ad0`); the synthetic run is `20261014T110622Z-0246ad0`. W5 hits the same issue. W7 needs
     both run ids, or a merge convention.
  4. Superseded runs under `stages/translate/` (`…102623Z-7e7dc14`, `…102855Z-f9abb27`, `…102955Z-071ab3c`,
     `…105656Z-08de901`) can be ignored; the noncurrent-version lifecycle doesn't remove them (they're current objects).
- **W7:** take both runs. Strip `_qa` and `_archive` (W6 writes `_archive.translate`, the list of archive keys) at
  release; keep `translation_method`. `en_alt` is optional: dropping it removes the only audit ambiguity (§2). After
  any `hi_text` change (W4 audit patches) or any W5 output, re-run `translate` on the new input: unchanged records
  hit the archive, changed ones cost ≈ $0.003 each. Apply `w6/patches.jsonl` by re-running the stage.
- **W4:** mid-phrase sentence splits ("… पटियाला। सलवार सूट के लगेंगे ही।") make fragments that the translator sometimes
  completes (the main residual error on real speech). Fewer trailing fragments in the SBD would help W6 and Part B.
  `translate` also ran on the 20 clips `asr_check` retires: harmless, W7 drops them by id.
- **W8b / W8d:** targets are fluent English with numbers in words and Indian units ("three and three-quarter lakh",
  "nine eight double seven"), worth a Bulbul `shubh` probe for these forms. 16 `en_alt` give an as-spoken repair
  variant for self-corrections, the only streaming-faithful target when a correction arrives after the value
  (§5.1 R2); train each as its own target or not at all (§10 R8). Segments map 1:1 to `sentences[]`, so W5's
  `hi_start`/`hi_end` are the alignment anchors.
- **W2/W3:** cross-segment self-corrections render as explicit corrections, in-segment ones as the corrected value
  (+ `en_alt`). Both are fine. Put both values in one sentence when a script wants the repair variant.

**Open (organisers, add to W0's list):** does the dataset audit accept an `en_alt` that voices the abandoned value
inside an explicit repair ("three and a half lakh, no, no, three and three-quarter lakh")? FAQ only says speaking it
*as fact* is wrong.
