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
