# W2: Content design and script generation

**Goal:** natural, spoken-style Hinglish monologue scripts of 10–60 s, dense in the entities and intent phenomena the hidden test checks. They feed W3 (TTS), and own recordings if those are allowed. Intent and entity accuracy are worth 36 of the 70 KPI points (§9.1).

## What the test targets (§2, §4, §7, §11 FAQ)

- **Entities:**
  - names, places, organisations, products;
  - amounts in rupees, lakh and crore;
  - numbers, dates, times;
  - phone and account numbers.
- **Intent:**
  - asked, refused, confirmed, corrected;
  - negations and conditions;
  - who does what to whom.
- **Self-corrections:** the reference keeps only the corrected value.
- **Code-mixing:** English words, phrases or clauses mixed into the Hindi.

## Coverage matrix to build

- **Domains:** banking, UPI and payments, telecom, e-commerce and returns, travel, health and insurance, government services, everyday conversation.
- **Number forms:**
  - lakh/crore;
  - fractional words: डेढ़, ढाई, सवा, साढ़े, पौने ("पौने चार लाख", "साढ़े तीन बजे");
  - the irregular numerals 21–99;
  - digit strings with डबल/ट्रिपल and phone-number grouping;
  - Aadhaar, PAN, IFSC, OTP and PIN codes;
  - digits spoken in English.
- **Intent phenomena:**
  - negation placement;
  - **the tag "ना" that is not a negation** ("आप आएंगे ना?");
  - नहीं तो ("otherwise") and मत (negative imperative);
  - conditionals (अगर…तो, वरना);
  - intonation-only questions;
  - refusals and confirmations;
  - causatives (किया vs करवाया);
  - tense-ambiguous कल/परसों;
  - pro-drop subjects; ने/को roles.
- **Self-corrections:** with markers (नहीं नहीं, मतलब, सॉरी, "I mean", एक मिनट), and especially corrections *of entities*.
- Set a target frequency for every cell.

## Research

- **Entity banks:** regionally diverse Indian names, cities and localities, banks, organisations, products and plans.
- **Generator LLM:** open-weight on Modal vs Sarvam's LLM (credits) vs a closed API. Check each one's licence or terms of service for training use; §10 R2 requires us to declare it.
- **Prompting for real spoken style:** fillers, restarts and natural code-mixing. Output = spoken-form Devanagari (numbers verbalised the way they'd be said) + metadata (entities, phenomena, correction spans).
- **Naturalness:** blind comparison against real IndicVoices transcripts; n-gram dedup for diversity.

## Done when

500 pilot scripts, the generator code, and a coverage report (cell counts vs targets).

## Coverage matrix v0 (2026-10-14, W2)

Every script is one **cell**: a value on each dimension below. `w2/gen.py:design` fills the targets exactly (largest-remainder quotas per dimension, crossed by independent shuffles), so at N = 500 each count below is met by construction before generation; the coverage report then measures what the kept scripts *realise*. Targets are [OURS]; the rationale is §4/§7 (content list, intent and entity KPIs = 36 of 70 points, §9.1).

| Dimension | Values and target share | Why |
|---|---|---|
| Domain | banking 16, UPI/payments 14, telecom 11, e-commerce/returns 13, travel 12, health/insurance 12, government 10, everyday 12 | §2's caller scenario; money domains carry the lakh/crore and account-number entities |
| Primary number form | per domain, table below | §4 "amounts (rupees, lakh, crore), numbers, dates, times, phone and account numbers" |
| Intent phenomenon | negation placement 10, tag ना (not negation) 10, नहीं तो 7, मत 8, अगर…तो/वरना 10, intonation-only question 9, refusal 8, confirmation 8, causative किया/करवाया 9, tense-ambiguous कल/परसों 8, pro-drop 6, ने/को roles 7 | §7 intent: "asked, refused, confirmed or corrected, negations, conditions, who does what to whom" |
| Self-correction | none 40, of an entity 45, of an ordinary word 15; markers नहीं नहीं / मतलब / सॉरी / आई मीन / एक मिनट, 12 each | FAQ: the reference keeps only the corrected value |
| Code-mix | light (1–3 English words) 30, medium (4–8) 50, heavy (≥ 9 incl. an English clause) 20 | §4 "English words, phrases or clauses" |
| Length | short 30–50 words (≈10–18 s) 30, medium 55–95 (≈20–35 s) 50, long 105–140 (≈38–52 s) 20 | §4 sessions are 10–60 s; W3 measured 0.37 s/word, 0.099 s/char |
| Speaker gender | f 50, m 50 | W3 handoff: Hindi verbs agree with the speaker |

Primary number form by domain (% of that domain's scripts; `·` = 0). Medium and long scripts also get a second number entity drawn from the same row.

| Domain | lakh/crore | fraction (डेढ़ ढाई सवा साढ़े पौने) | 21–99 | digit string (डबल/ट्रिपल) | ID code (Aadhaar PAN IFSC OTP PIN PNR) | digits in English | date | time | plain amount (सौ/हज़ार) |
|---|---|---|---|---|---|---|---|---|---|
| banking | 25 | 15 | 10 | 15 | 15 | 5 | 5 | · | 10 |
| UPI | 5 | 15 | 15 | 15 | 15 | 10 | 5 | 5 | 15 |
| telecom | · | 10 | 20 | 20 | 10 | 15 | 10 | 5 | 10 |
| e-commerce | 5 | 10 | 20 | 10 | 10 | 10 | 15 | 5 | 15 |
| travel | · | 15 | 10 | 10 | 15 | 10 | 15 | 25 | · |
| health | 20 | 15 | 15 | 5 | 10 | 5 | 10 | 15 | 5 |
| government | 15 | 10 | 10 | 10 | 25 | 5 | 15 | 5 | 5 |
| everyday | 10 | 25 | 15 | 10 | 5 | 10 | 5 | 15 | 5 |

Named entities: 1 (short), 2 (medium) or 3 (long) of person name (speaker's own, gender-matched, or someone else's), organisation, product/plan, place, from hand-written banks in `w2/banks.py`: 103 first names and 66 surnames in 7 regional/community pools, 86 cities and localities, 5–14 organisations and 5–10 products/plans per domain.

**Entities are made by code, not by the LLM.** `w2/banks.py` verbalises every given entity (Indian grouping, the irregular 21–99, डेढ़/ढाई/सवा/साढ़े/पौने with सौ/हज़ार/लाख/करोड़/बजे, colloquial बारह सौ पचास, digit strings with डबल/ट्रिपल and phone grouping, PAN/IFSC/PNR letter names per SPEC R8, English digits and clock times) and keeps its canonical value. The LLM must copy each one verbatim; a script that drops or alters one is rejected. So every given entity's value is known for W6/W8c entity checks, and number words are never the LLM's spelling.

## Script schema: `data/w2/scripts.jsonl` (W3 renders it directly)

One JSON object per line, sorted by `pid`. The first block is **exactly W3's probe format** (`w3/probe.py:parse` produces it), so `w3/render.py` can read this file in place of `data/w3/probe.jsonl`.

| Field | Meaning |
|---|---|
| `pid` | `w2-NNNNN`, stable for this design (`pilot-v1`) |
| `domain` | coverage-matrix domain |
| `tts_dev` | TTS input, English in Devanagari, commas as pause hints. **Render this** (W3 memo item 1) |
| `tts_lat` | same with English in Latin (W3's input-script test only) |
| `hi_text`, `sentences` | SPEC §2 transcript (commas dropped), split after । ? !. Passes `validate.check_record` (via `probe.parse`) |
| `entities` | spoken form of every `[..]` span, in order, including retracted values (they are spoken, so W3 QA must hear them) |
| `en_words` | spoken form of every `{..|..}` span |
| `chars` | non-space characters of `hi_text` |
| `marked` | the script in probe.tsv markup: `{देवनागरी|latin}` English, `[..]` entity |
| `tags` | realised content tags in W3's vocabulary (`lakh crore frac num2199 digits enum acronym date time name place brand selfcorr filler en`) plus `i:<phenomenon>` for intent phenomena detected in the text |
| `gender` | `f`/`m`: **pick a voice of this verified gender** (first-person verbs agree; checked by `gen.gender_marks`) |
| `gender_marks` | `[masc, fem]` first-person markers found; `[0, 0]` = the text itself doesn't force a gender |
| `entity_spans` | per entity, in order: `text` (spoken), `type` (name place org product amount number date time phone account id), `value` (canonical, e.g. `375000`, `8:45`, `27 September`, `98765 43210`, `SBIN0004567`, `Sunita Raghavan`; `null` if the LLM added it), `kind` (number form), `given`, `retracted` |
| `corrections` | `[{wrong, right, marker, type, value}]` for entity corrections (`value` = the corrected, true value), `[{marker}]` for ordinary-word corrections |
| `detected` | intent phenomena found in the text (`gen.detect`) |
| `cell` | the requested coverage cell |
| `est_sec`, `words` | estimated audio seconds (0.099 s/char), word count |
| `scenario`, `setting` | prompt context |
| `model`, `model_id`, `take` | generator and the take that passed |
| `_gen` | prompt version, the exact user prompt, every try's archive key and failures, USD, raw output |

For W6: translate `hi_text`; entity values and `corrections[].value` are the references for the entity check, and a `retracted` span must not appear in the English as fact (FAQ). For W3: `source.corpus = "w2-scripts-v1"` (SPEC §5).
