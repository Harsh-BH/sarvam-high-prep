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
| `model`, `model_id`, `take` | generator and the take that passed; `take` ≥ `--takes` (5) is a repair turn |
| `_gen` | prompt version, the exact user prompt, every try's archive key, failures and `repair` flag, USD, raw output |

For W6: translate `hi_text`; entity values and `corrections[].value` are the references for the entity check, and a `retracted` span must not appear in the English as fact (FAQ). For W3: `source.corpus = "w2-scripts-v1"` (SPEC §5).

## Generator bake-off (2026-10-14, W2)

**Candidates.** Open-weight models served on Bedrock in ap-south-1 (`aws bedrock list-foundation-models`, 2026-10-14) whose licence puts no terms on outputs; licence read from each Hugging Face card on 2026-10-14:

| Model (Bedrock id) | Licence | USD / 1K tokens in, out (Pricing API, ap-south-1 on-demand) |
|---|---|---|
| Qwen3 235B A22B 2507 (`qwen.qwen3-235b-a22b-2507-v1:0`) | Apache-2.0 | 0.00026, 0.00104 |
| gpt-oss-120b (`openai.gpt-oss-120b-1:0`) | Apache-2.0 | 0.00018, 0.00071 |
| Mistral Large 3 675B (`mistral.mistral-large-3-675b-instruct`) | Apache-2.0 | 0.00059, 0.00176 |
| DeepSeek-V3.2 (`deepseek.v3.2`) | MIT | 0.00074, 0.00222 |

Not run:
- Gemma 3 and Llama 3.x: BASE output terms (prior-art Trap T1).
- Closed models on Bedrock (Claude, GPT-5.x/6): §10 R2 allows a declared closed API, but their terms restrict using outputs to train competing models. Whether that covers our use is AMBIGUOUS, and open weights are as good here.
- Kimi K2.5 and MiniMax M2.5: modified MIT; fine, just not needed.
- Sarvam's LLM: HTTP 402.

**Method.** The same 30 cells (`design(30, "bake-v1")`), the same prompt (v1: rules + 2 hand-written few-shot examples), temperature 0.8, top-p 0.95, first take only. Every call went through `archive_call` (service `bedrock`, stream `w2`).

A take passes only if all of these hold:
- `spec_text`: `probe.parse` → `validate.check_record` (SPEC §2) passes.
- `entity_missing`: every given entity is present verbatim, as a `[..]` span.
- `entity_type`: every type is known.
- `correction`: the self-correction is realised in order: wrong value, then marker, then right value.
- `gender`: no first-person verb of the other gender.
- `filler`: none of the banned fillers अ अं ए.
- `length`: the estimate is 9–57 s.
- `intent`: a detectable intent phenomenon is present.

### Constraint adherence, first take (prompt v1)

| model | licence | n | pass | spec_text | entity_missing | entity_type | correction | gender | filler | intent | words in band | USD per 1k scripts | latency s p50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **deepseek-v3.2** | MIT | 30 | **77%** | 4 | 1 | 0 | 0 | 0 | 1 | 1 | 92% | 1.76 | 4.8 |
| gpt-oss-120b | Apache-2.0 | 30 | 53% | 5 | 8 | 0 | 0 | 1 | 1 | 0 | 88% | 1.70 | 9.5 |
| mistral-large-3 | Apache-2.0 | 30 | 50% | 4 | 7 | 2 | 1 | 1 | 0 | 3 | 73% | 1.15 | 1.6 |
| qwen3-235b | Apache-2.0 | 30 | 33% | 14 | 4 | 1 | 1 | 0 | 0 | 0 | 94% | 0.86 | 3.4 |

The `spec_text` failures are real model errors, not checker noise:
- Latin acronyms outside the markup (`UPI`, `EMI`, `home broadband`);
- `-`, `—`, `…` and `.`;
- broken markup (`[..|id}`, a type word with a space, an entity bracket opened but never closed).

`entity_missing` covers two cases:
- a given entity spoken but left without its `[..]` wrapper;
- an entity re-verbalised. Mistral rewrote the PIN code's digits into a different grouping, which changes what W3 must hear.

### Naturalness proxies vs real IndicVoices (per 100 words unless noted)

The reference is W4's 628-clip pool (`data/w4/pool.jsonl`, IV Level 1 → SPEC §2). The same code is applied to both sides (`w2/report.py:style`).

| text | texts | words/text | words/sent | fillers | discourse words | repeats | written-register words | distinct-2 | unmarked English % |
|---|---|---|---|---|---|---|---|---|---|
| IndicVoices L1 | 628 | 54.7 | 15.1 | 1.34 | 1.13 | 0.53 | 0.18 | 0.64 | — |
| deepseek-v3.2 | 26 | 66.3 | 12.2 | 0.17 | 2.38 | 0.46 | 0.12 | 0.86 | 16–18 |
| gpt-oss-120b | 25 | 76.7 | 15.3 | 0.16 | 1.46 | 0.26 | **0.63** | 0.78 | 28–31 |
| mistral-large-3 | 26 | 64.0 | 13.5 | 0.12 | 2.88 | 0.54 | 0 | 0.83 | **8** |
| qwen3-235b | 16 | 63.6 | 12.0 | 0.10 | 2.26 | 0.39 | 0.20 | 0.86 | 36–38 |

Definitions:
- **fillers:** अ अं ए उम्म हम्म.
- **discourse words:** मतलब यानी अच्छा देखिए देखो बस एक्चुअली लाइक वैसे अरे हाँ यार.
- **repeats:** adjacent identical words, digit-string words excluded.
- **written register:** एवं तथा हेतु अतः उक्त किंतु परंतु यदि अथवा कृपया द्वारा अवश्य प्रदान सूचित निवेदन अनुरोध संपर्क.
- **distinct-2:** unique word bigrams / all bigrams. It falls as a corpus grows (628 vs ~26 texts), so compare it across models only, not with IndicVoices.
- **unmarked English:** words the generators marked English in at least 2 scripts that appear *outside* `{..|..}` markup. It's a proxy for markup recall, and the range reflects the lexicon growing between runs.
- **Diversity:** no script pair in any model shares more than 7% of its word trigrams. The highest overlap with a few-shot example is 3%. The most common two-word opening covers 15% of DeepSeek's scripts and 38% of Qwen's.

**Decision: DeepSeek-V3.2.**
- It has the highest first-take pass rate (77% vs ≤ 53%; paired cells, ~2 binomial SE over the runner-up).
- It has the fewest dropped entities, the highest length adherence after Qwen, and the most varied openings.
- Its naturalness proxies are no worse than any other candidate's.
- Cost is irrelevant at this scale (≈ $1.8 per 1,000 scripts).
- Licence: MIT weights. INFERRED: using its outputs as training data carries no output terms; Bedrock adds only AWS's service terms.
- Mistral Large 3 is the fallback: best markup recall, fastest, Apache-2.0.

### Prompt iteration on the winner (same 30 cells, first take)

| prompt | change | pass | fillers | discourse | repeats | written | words in band |
|---|---|---|---|---|---|---|---|
| v1 | — | 77% | 0.17 | 2.38 | 0.46 | 0.12 | 92% |
| v2 | intent hints without copyable sentences (DeepSeek had pasted "मुझे वो कार्ड नहीं चाहिए" into a school-fees script); "make the story fit the entities" (it had written a ₹16 crore school bill); "usually one filler"; stronger English-markup rule; crore amounts capped at 10 | 60% | **0.06** | 2.08 | 0.24 | 0 | 79% |
| **v3** | v2 + filler (50%) and repetition (30%) as **per-script requirements, checked** like entities; given entities spoken verbatim but left unwrapped get wrapped in code (exact string, so the value stays known) | 63% | **1.05** | 1.79 | **0.49** | 0.19 | 81% |

- Asking for disfluency in the system prompt does nothing: 0 of 24 v2 scripts had a filler. Making it a checked cell gets v3 to IndicVoices' rates (1.05 vs 1.34 fillers and 0.49 vs 0.53 repeats per 100 words).
- v3's lower first-take pass rate comes from the new `disfluency` check (4 of 30). The pilot runs up to 5 takes.
- The v1-v3 differences in pass rate are within noise at n = 30. Its prompt changes are kept for the naturalness proxies, not for the pass rate.
- Discourse words remain ~1.6× IndicVoices.
- English share is ~12% of words (lexicon) vs ~1% in IndicVoices extempore (same lexicon, a lower bound there). This is by design: the scripts are customer-service Hinglish (§4 "English words, phrases or clauses"), not IV's general-topic speech. It is the one proxy deliberately far from IV.

## Pilot (2026-10-14, W2)

`uv run w2/gen.py pilot --model deepseek-v3.2 --n 500 --takes 5 --repairs 4`. It writes `data/w2/scripts.jsonl` (500) and `rejects.jsonl` (0). A rerun from this commit replays every call from the archive.

**Prompt v4** (on top of v3):
- The heavy code-mix instruction asks for an English clause "of your own", with no copyable example sentence.
- Entity self-corrections are near misses (`gen.near`: a count keeps its unit, an amount stays within 3×).
- The banned-filler check skips English spans: ए is also the letter A in {ए टी एम|ATM}.

**Repair turns (new).** After 5 fresh takes, up to 4 turns show the model its own last answer plus the rules it broke, quoting the offending text where code can find it (`gen.feedback`).
- All 58 cells still failing after 5 takes were **heavy** code-mix: the model writes its English clause in Latin outside the markup ("आपने वादा किया था कि I will get it by yesterday").
- Repair turns rescued all 58; two of them needed the check fixes below.
- The repaired heavy scripts keep their English: marked English p50 is 30% of words, vs 27% for the 42 heavy scripts that passed on a fresh take.
- Heavy cells pass the first take only 12% of the time (all cells: 65%).
- First-take pass rate: 323/500 = 65%. Takes needed: 0 → 323, 1 → 72, 2 → 29, 3 → 11, 4 → 7, repair 1 → 47, repair 2 → 8, repair 3 → 3.

**Fixes found on the pilot:**
- `gender` false positive: in "मैं [ग्रेस डिसूज़ा|name] हूँ" the name's final ा read as a masculine verb. Entity spans are now masked before the gender check.
- Repair feedback missed Latin on the Devanagari side of the markup (`{KYC|KYC}`).
- Verbaliser bug: an English clock time with minutes 1–9 said "एट वन" for 8:01. It now says "एट ओ वन" (eight oh one). 4 scripts were regenerated; no other cell's prompt changed (checked).
- Report: the English lexicon now requires a word to be marked in ≥ 50% of its occurrences. The LLM sometimes wraps a Hindi word with its translation ({पैसे|money}), which had made मेरा and पैसे "English". With the new lexicon the bake-off ranking on unmarked English is unchanged; the bake tables above used the old definition.

### Coverage report (500 kept of 500 designed)

`uv run --no-project python w2/report.py coverage` prints the full report.

**Designed cells:** every dimension is at target exactly, because no cell was lost:
- domain: 80/70/55/65/60/60/50/60;
- intent: 12 phenomena, 30–50 each;
- self-correction: entity 225, other 75, none 200, each marker 60;
- code-mix: 150/250/100;
- length: 150/250/100;
- gender: 250/250;
- filler: 250; repetition: 150.

Domain × primary number form (kept / target):

| domain | lakh/crore | fraction | 21–99 | digits | ID code | English digits | date | time | plain amount |
|---|---|---|---|---|---|---|---|---|---|
| banking | 20/20 | 12/12 | 8/8 | 12/12 | 12/12 | 4/4 | 4/4 | · | 8/8 |
| upi | 4/4 | 11/11 | 11/11 | 11/11 | 10/10 | 7/7 | 3/3 | 3/3 | 10/10 |
| telecom | · | 6/6 | 11/11 | 11/11 | 6/6 | 8/8 | 5/5 | 3/3 | 5/5 |
| ecommerce | 3/3 | 7/7 | 13/13 | 7/7 | 6/6 | 6/6 | 10/10 | 3/3 | 10/10 |
| travel | · | 9/9 | 6/6 | 6/6 | 9/9 | 6/6 | 9/9 | 15/15 | · |
| health | 12/12 | 9/9 | 9/9 | 3/3 | 6/6 | 3/3 | 6/6 | 9/9 | 3/3 |
| gov | 8/8 | 5/5 | 5/5 | 5/5 | 13/13 | 3/3 | 7/7 | 2/2 | 2/2 |
| everyday | 6/6 | 15/15 | 9/9 | 6/6 | 3/3 | 6/6 | 3/3 | 9/9 | 3/3 |
| **total** | 53/53 | 74/74 | 72/72 | 61/61 | 65/65 | 43/43 | 47/47 | 44/44 | 41/41 |

**Realised content** (scripts containing it, measured on the text):

| content | scripts |
|---|---|
| lakh | 99 |
| crore | 51 |
| fraction | 131 (डेढ़ 28, ढाई 26, सवा 32, साढ़े 34, पौने 28) |
| 21–99 | 256 |
| digit string (≥ 4 digits) | 258 |
| डबल/ट्रिपल | 218 |
| English digits | 168 |
| acronym | 185 |
| date | 103 |
| time | 104 |
| name | 290 |
| place | 129 |
| brand | 423 |
| self-correction | 300 |
| filler | 251 |
| English | 500 |

**Intent:** every requested phenomenon that code can detect is present in all of its scripts:
- neg_place 50/50, tag_na 50/50, nahi_to 35/35, mat 40/40, cond 50/50, intonation_q 45/45, causative 45/45, kal_parson 40/40.
- They also occur unrequested, e.g. tag ना in 95 scripts and causatives in 121.
- refusal (40), confirmation (40), prodrop (30) and ne_ko (35) need a reader and are requested only, not verified.

**Entities:**
- 2,084 spans, 4.2 per script.
- 2,043 were pre-verbalised by `banks.py`, so the value is known; the LLM added 41.
- 225 are retracted (corrected away).
- By type: org 336, amount 327, name 311, product 248, id 205, date 137, place 137, time 124, number 100, phone 95, account 64.

**Entity values vs an independent parser.** W6's `w6/qa.py:hi_numbers` was written separately from `banks.py`. Read back against each value:

| type | W6 parse = value |
|---|---|
| amount | 322/322 |
| number | 100/100 |
| phone | 94/94 |
| account | 64/64 |
| id (digits) | 203/203 |
| time | 115/118 |
| date | 104/115 |

All 14 disagreements are parser conventions, not verbaliser errors:
- a bare एक is ignored as an article (एक बजे, एक जुलाई);
- English years are not joined (ट्वेंटी ट्वेंटी सेवन);
- ओ ("oh") is not read as zero.

So the number spelling and the values need no listening.

**Length and code-mix, realised:**

| length | asked (words) | est. seconds p10/p50/p90 | words in band |
|---|---|---|---|
| short | 30–50 | 13.5/16.8/20.7 | 73% |
| medium | 55–95 | 20.5/24.6/28.7 | 93% |
| long | 105–140 | 29.4/33.9/40.3 | **18%** |

| code-mix | marked English % of words p10/p50/p90 |
|---|---|
| light | 7/14/24 |
| medium | 11/18/29 |
| heavy | 20/28/38 |

- Long scripts come out at ≈ 90 words, not 105–140. The check only enforces 9–57 s, so nothing goes past ≈ 45 s.
- "Light" carries more English than asked, because everyday loanwords ({बैंक|bank}, {फ़ोन|phone}) count.
- Estimated audio: **3.38 h** at one render per script.
- 331/500 scripts carry a first-person gender marker that the voice must match.

**Naturalness vs IndicVoices L1** (628 W4 pool transcripts, same code; per 100 words):

| text | words/sent | fillers | discourse | repeats | written register | English (lexicon) % | distinct-2 |
|---|---|---|---|---|---|---|---|
| IndicVoices L1 | 15.1 | 1.34 | 1.13 | 0.53 | 0.18 | 3.6 | 0.64 |
| W2 pilot | 13.7 | 0.91 | 1.59 | 0.71 | 0.07 | 13.3 | **0.49** |

**Diversity:**
- The maximum word-trigram Jaccard between any two scripts is 0.07; no pair is above 0.2.
- The maximum overlap with a few-shot example is 0.04.
- The most common two-word opening covers 8% of scripts.
- distinct-2 is lower than IV's at a similar corpus size. INFERRED: the domain-bound customer-care register repeats bigrams (मेरा अकाउंट, कर दीजिए); IV is free-topic speech.

**Known content limits:**
- 26/500 scripts (5%) in e-commerce, telecom or everyday put a lakh-scale amount on a small purchase, e.g. ₹19.5 lakh for DTH. The design draws the amount and the scenario independently.
- English-markup recall is ≈ 84%: 15.8% of lexicon words stand outside `{..|..}`. This affects only `tts_lat` and `en_words`, not `tts_dev`, `hi_text` or entities.

## Status — 2026-10-14

**Purpose changed (master, 2026-10-14):** this is now a personal learning project, not a competition entry. Models must be open-source or free; the open-weight Bedrock models used here qualify. Nothing below depends on the submission rules.

**Done (done-when met):**
- **500 pilot scripts** in `data/w2/scripts.jsonl`, 0 rejects. All 500 re-parse through `w3/probe.py:parse` → `validate.check_record` with 0 SPEC §2 failures, checked independently after generation.
- **Generator code:** `w2/banks.py` (entity banks + verbaliser), `w2/gen.py` (design, prompts, archived Bedrock calls, checks, repair turns), `w2/report.py` (bake-off, coverage, review list). Self-checks: `uv run --no-project python w2/banks.py`; `gen.py` and `report.py` run asserts before their main.
- **Coverage report:** above.

**Measured:**
- First-take pass rate: 65%.
- Pass rate after 5 takes: 88%. After the repair turns: 100%.
- 995 calls for the 500 kept scripts, $1.83.

**Declared models and data (§10 R2):**
- Generator: DeepSeek-V3.2 (MIT weights) on Amazon Bedrock ap-south-1 (`deepseek.v3.2`).
- Bake-off only: Qwen3 235B A22B 2507, gpt-oss-120b and Mistral Large 3, all Apache-2.0.
- No closed model and no Sarvam API.
- IndicVoices transcripts (W4 pool) were used only as a comparison reference; none of their text is in the scripts.
- W6's `qa.py` was used only as a checker.

**Spend (real, from the archived token usage at the Pricing-API rates):**
- W2 total: **$5.19** for 2,887 Bedrock calls. This session: $1.89 (1,021 calls).
- Included in the total are **886 orphaned calls ($1.59)** from the dead session's last run. Their prompts are not ones the committed generator produces: the cells differ (checked against `c8a840b`). INFERRED: that run used uncommitted code. They are archived but unused.
- No SageMaker.
- AWS credits (`data/master/credits.sh`): usage $1.98, NET $0.00, so credits cover everything.
- The W2 spend ledger (`archive/spend/w2/`) lacked 1,328 markers for calls archived before the ledger existed. I wrote them from the archived metadata, which is `archive.backfill` restricted to stream w2.
  - `spent('w2')` now reads ₹1,726 recorded. That figure is an upper bound or estimate; the real cost is ₹457.
  - `gen.CAP_INR` subtracts the measured overcount, so the cap still means ≈ $15 real.
  - Master: other streams may still need `archive.backfill()`.

**Blockers:** none.

**Handoffs:**
- **W3:**
  - Render `data/w2/scripts.jsonl` in place of `data/w3/probe.jsonl`: same fields, `pid` = `w2-NNNNN`.
  - Render `tts_dev`. Pick a voice of the script's `gender`.
  - Set `source.corpus = "w2-scripts-v1"`.
  - Expect ≈ 3.4 h of audio.
  - 258 scripts have digit strings and 185 have spelled acronyms: budget re-renders for those (your 85–95% hit rate).
- **W6:**
  - Translate `hi_text`. The entity references are `entity_spans[].value` and `corrections[].value`; a `retracted` span must not survive as fact.
  - `hi_numbers` could add three conventions to match these scripts: ओ = 0 in English clock and digit runs (एट ओ वन = 8:01), English year pairs (ट्वेंटी ट्वेंटी सेवन = 2027), and एक before बजे or a month as a number.
- **W2 bulk run (next):**
  - Use a new design seed (e.g. `bulk-v1`). Never change the `pilot-v1` cells: record IDs are frozen once rendered.
  - Before it, three prompt or design fixes are worth the cost:
    1. Put "English clause in Devanagari inside one {..|..}" into the heavy instruction (heavy first-take pass was 12%; 42% within 5 takes).
    2. Enforce the long band, or relax it to what the model writes.
    3. Cap lakh/crore amounts in small-purchase scenarios.

**The one list for Harsh (optional, ≈ 10 min):** `data/w2/review.tsv`, 20 rows.
- It has 10 pilot scripts and 10 IndicVoices transcripts of similar length, shuffled. Mark each real/generated and natural 1–3.
- The answers are in `review_key.tsv`; don't open it first.
- No automatic proxy covers this. Number spelling and values are covered by the W6 round trip above, so the earlier 118-row spelling section is gone.

**Disk:** `data/w2` is 9.3 MB. No model weights or caches downloaded.
