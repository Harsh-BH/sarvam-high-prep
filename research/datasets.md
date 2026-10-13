# Open-source dataset scout

Feeds W1 (real audio), W2 (scripts), W4 (transcription), W6 (translation), W8c (dev set). Models/papers/repos are in `prior-art.md`; this file covers datasets only. W1's corpora (IndicVoices, SPRING-INX, Vaani, Kathbath, MUCS SLR103) are **not re-measured here**. Rows for them only add facts W1 lacks.

**Method.** Everything below comes from pages fetched on 2026-10-14: HF Hub API `cardData`/`gated`, dataset-card text, the HF datasets-server `/statistics` and `/rows` endpoints (metadata only, no audio), OpenSLR resource pages, AIKosh listings and arXiv papers. No HF token was used. Gated cards are readable, but gated files are not, so anything that needs the files is marked **UNVERIFIED**. Disk used: under 1 MB, against a 1 GB budget.

## Binding rules → roles

- §3A + §10 R1: real recordings only "from datasets whose licence allows redistribution", and "licence and consent compliance is your responsibility". Our dataset goes privately to Sarvam (§9.2), and Sarvam holds the IP of our code (§10 R5). **INFERRED:** NC and "academic-only" licences are unsafe for Part A, because the recipient is a company.
- §3B + FAQ: "Only your Part A dataset may be used for training." Third-party data may still be used to *build* Part A (scripts, references, open-weight models, §10 R1), and to evaluate.
- §10 R3: Sarvam outputs are never published. **INFERRED:** a ShareAlike source becomes awkward once its records carry Sarvam-generated translations (see Q2).
- §10 R1: every corpus below that we touch, even only as a text seed, goes into `data_card.md`.

| Role | Meaning |
|---|---|
| **A-real** | Real Hindi audio we may cut into Part A records |
| **A-text** | Text used to *build* W2 scripts, W6 style references or W4 lexicons. Never itself a training input |
| **Aug** | Noise, RIR or codec material for training-time augmentation (§3B ambiguity below) |
| **Eval** | Dev/test only (W4 ASR choice, W8c proxies). Never trained on |
| **Trap** | Looks useful, but a licence or rule forbids it or makes it risky |

## Headline findings

1. **Licence is not what limits real hours.** About 2,000 h of licence-clean real Hindi exist (§E): Vaani alone has **1,003.78 h transcribed Hindi** ([Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part), update of 2026-10-08). That figure is far larger than W1's "only partly transcribed" suggests. What binds us is session shape (contiguous single-speaker 10–60 s windows), how fast W4 can convert transcripts, the telephone/noise share and entity density.
2. **New A-real sources W1 does not list:**
   - Rural Women ASR v2 (~87 h, open, phone-recorded, IV schema);
   - SYSPIN S1.0 (~110 h, studio, 2 voices);
   - Rasa (50.8 h, studio, 2 voices);
   - Sarvam's Indic DiarBench (10.7 h of conversational code-mixed speech; best kept as Eval).
3. **Labels launder licences.**
   - On HF, aggregators re-label CC BY-SA, CC0-with-terms and "see source" material as CC BY 4.0 (agarwalayushi/hinglish, zara-hinglish, dianavdavidson/MUCS-Hinglish).
   - AI4Bharat puts CC BY 4.0 over sources whose own terms are SA (Spoken Tutorial), reportedly NC (NPTEL), permission-only (Shrutilipi's AIR audio) or unknown (SeamlessAlign web audio, WordProject).
   - **Always cite the original publisher's licence. Where two disagree, take the stricter one.**
4. **Mirrors disagree.**

   | Corpus | Licence on one mirror | Licence on another |
   |---|---|---|
   | Kathbath | CC BY 4.0 (HF tag) | CC BY-SA 4.0 ([AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/hindi_asr_benchmark_dataset_kathbath_hindi_test_known.html)) |
   | Gram Vaani | "academic purposes" ([OpenSLR 118](https://www.openslr.org/118/)) | CC BY-NC 4.0 ([AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/gram_vaani_hindi_asr_dataset.html)) |
   | Lahaja | MIT (HF metadata tag) | CC BY 4.0 (its own card text) |

   **Handoff to W1:** Kathbath's licence needs resolving before any Kathbath clip goes into Part A.
5. **Translation-side trap.** IndicVoices-ST's English side is Llama-3.1-405B output. The Llama 3.1 licence §1.b.i says: *"If you use … any outputs or results of the Llama Materials to create, train, fine tune, or otherwise improve an AI model, which is distributed or made available, you shall also include "Llama" at the beginning of any such AI model name"* ([LICENSE](https://raw.githubusercontent.com/meta-llama/llama-models/main/models/llama3_1/LICENSE)). Use those translations only as W6 references, never as targets (Q6).
6. **Prefer real noisy records to augmentation.** Real noisy clips count as hours, are unambiguously Part A data, and match §4 exactly. Vaani's noise-event timestamps let W1 *select* such clips.

## A. Real Hindi / code-mixed speech

Key to columns: SR = sample rate; style R = read, E = extempore, C = conversational, T = telephone. Hours are Hindi only.

### A1. New candidates (not in W1's list)

| Dataset | Licence (where stated) | Role | Hindi h | Style / SR | Transcript / code-mix | Access | Why it matters |
|---|---|---|---|---|---|---|---|
| [Rural Women ASR v2](https://huggingface.co/datasets/ai4bharat/Rural_Women_ASR_v2) (AI4Bharat) | CC BY 4.0 in the README "Licensing" section; no metadata tag | **A-real** | ≈87 h of chunks (30,827 rows × 10.18 s mean chunk, from datasets-server partial stats on 5,880 rows). Unique-source hours are lower, because rows share raw files | E 62% / R 38% (partial stats); mobile phones via the Kathbath app; rural MP/UP/Bihar; 99.7% female; 48 kHz | IV-style `verbatim` + `normalized` with speaker-timed segments and `<Persistent-noise-start>` tags. **Light code-mix:** 35/80 sampled rows carry bracketed English (`[mobile]`, `[Online]`, `[discount]`) | **Open**, no gate (31.4 GB Hindi) | Same JSON schema and speaker-ID format as IndicVoices, so W1's loader nearly applies. Phone mic + rural noise is close to §4. Measured: **0 of 16 sampled speakers overlap the 1,390 IV Hindi speakers in W1's local sample** (weak evidence of a separate collection). Consent statement: UNVERIFIED |
| [IISc SYSPIN S1.0](https://aikosh.indiaai.gov.in/home/datasets/details/iisc_syspin_s1_0_corpus.html) | "Attribution 4.0 International (CC BY- 4.0)" (AIKosh) | A-real (low rank) | 54h58m male + 54h55m female human-checked, plus 9.4 h unchecked | Studio, voice artists, 2 speakers. SR UNVERIFIED | Text domains: agriculture, finance, health-care, technology, sports, books… | AIKosh registration | Real and entity-bearing (finance/health text), but 2 studio voices are far from the test acoustics. Use only if we need read finance text |
| [Rasa](https://huggingface.co/datasets/ai4bharat/Rasa) (AI4Bharat, Interspeech 2024) | CC BY 4.0 | A-real (low rank) | 27.05 h female + 23.78 h male = 50.83 h | Studio, 48 kHz, 2 speakers. Neutral, 6 Ekman emotions, **command-style (Alexa, BigBasket, UMANG, DigiPay)**, conversation, news | Prompt-text format UNVERIFIED (files gated) | Gated (auto) | Command-style utterances are entity-dense (payments, e-commerce, gov services). But the mean utterance is ≈6.4 s (27.05 h / 15,107), and §5.2 wants contiguous `[start,end]` of one file, so they cannot be concatenated into 10–60 s sessions |
| [Indic DiarBench](https://huggingface.co/datasets/sarvamai/indic-diarbench) (**Sarvam**, Interspeech 2026) | CC BY 4.0 | **Eval** (A-real possible for NF/FF) | 10.7 h = 4.2 h near-field (online-meeting audio) + 4.0 h far-field + 2.5 h in-the-wild (YouTube) | C, multi-speaker, 16.6% overlap; 16 kHz | Human speaker-attributed, time-aligned. Native-script and normalised (English in Roman) versions; explicitly code-mixed | **Open** | The only human-labelled, code-mixed, conversational Hindi with real far-field and meeting-codec audio we found. Best W8c dev set for §4 conditions. The ITW part is YouTube (consent → Trap for Part A) |
| [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r) | CC BY 4.0 | Trap-ish (AMBIGUOUS) | 74.60 h (4.31 R + 70.28 E; [paper](https://arxiv.org/html/2409.05356) Table 3) | 48 kHz, **enhanced** with HTDemucs → VoiceFixer → DeepFilterNet3 | IV transcripts | Gated | Same recordings as IndicVoices, made cleaner and so *further* from the noisy test. Its files are different from IV's files for the same speech, which invites double-counting hours (§3A). Use IV originals (W1) |
| [IndicVoices-ST](https://huggingface.co/datasets/ai4bharat/IndicVoices-ST) (BhasaAnuvaad) | CC BY 4.0 | A-real via IV (W1) + W6 reference | 282.9 h / 21.6K utts, so a **mean of ≈47 s** ([paper](https://arxiv.org/html/2411.04699) Table A2) | IV audio resampled to 16 kHz | Hindi = IV transcripts; English text from **Llama-3.1-405B-Instruct**; English speech from Indic Parler-TTS | Gated. The Hindi parquet is only 1.89 GB, so the column content is UNVERIFIED | Paragraph-length IV segments with an English side: a ready-made 10–60 s cut list for W1/W6. Take audio from IV originals, not the 16 kHz copy. Its English is neither Bulbul audio (§3B) nor licence-clean text (Llama naming clause) |
| [YODAS](https://huggingface.co/datasets/espnet/yodas) / [YODAS2](https://huggingface.co/datasets/espnet/yodas2) `hi000` | CC BY 3.0 (YouTube CC-BY uploads) | **Trap** (AMBIGUOUS) | 480.79 h with "manual" captions (`hi000`) + 2.75 h automatic (`hi100`) | Mixed YouTube; 16 kHz (yodas), 24 kHz long-form (yodas2) | Card: manual "only indicates that it is uploaded by users, but not necessarily transcribed by a human" | Open | The licence allows redistribution, but only the uploader granted it, not the speakers. Content is multi-speaker or music, and captions are unreliable. Fails "consent compliance" (§3A) on any strict reading (Q5) |

### A2. Speech-translation corpora (Hindi→English direction)

| Dataset | Licence | Role | Hindi h | Notes |
|---|---|---|---|---|
| [FLEURS](https://huggingface.co/datasets/google/fleurs) `hi_in` | CC BY 4.0 | **Eval** | Measured from W1's local TSVs: **train 6.66 h** (2,120 utts / 1,311 sentences), dev 0.71 h, test 1.34 h | 16 kHz read FLoRes sentences. "2009 n-way parallel sentences" (card), so we get Hindi speech + English text (FLoRes) + English speech (`en_us`) for the same sentence. The only open Hi→En S2ST set. Transcripts contain digits and Latin script, so normalise before scoring |
| [FLEURS-R](https://huggingface.co/datasets/google/fleurs-r) | CC BY 4.0 | Skip | as FLEURS, minus ~10% | 24 kHz, restored ([arXiv 2408.06227](https://arxiv.org/abs/2408.06227v1)). Cleaner than the test conditions; no use to us |
| [BhasaAnuvaad](https://arxiv.org/html/2411.04699) subsets (AI4Bharat HF repos) | CC BY 4.0 *as labelled by AI4Bharat*. The paper names no licence and says nothing about source rights | **Trap**, except the IV part above | [SeamlessAlign](https://huggingface.co/datasets/ai4bharat/SeamlessAlign) 2,355.4 h · [NPTEL](https://huggingface.co/datasets/ai4bharat/NPTEL) 1,433.9 h · WordProject 88.5 h · [Spoken-Tutorial](https://huggingface.co/datasets/ai4bharat/Spoken-Tutorial) 63.6 h · [Mann-ki-Baat](https://huggingface.co/datasets/ai4bharat/Mann-ki-Baat) 40.9 h | Each source carries its own restriction (detailed after this table). All audio is resampled to 16 kHz |
| [IndicST](https://huggingface.co/datasets/krutrim-ai-labs/IndicST) (Krutrim) | "other", Krutrim Community License (per [AIKosh](https://aikosh.indiaai.gov.in/home/datasets/details/indicst_indian_multilingual_speech_translation_corpus.html)) | Trap / redundant | (10.8k h over 9 languages) | 14 open ASR sets repackaged with IndicTrans2 translations. A custom licence on top of inherited ones. Go to the originals instead |
| [CVSS](https://huggingface.co/datasets/google/cvss) | CC BY 4.0 | n/a | 0 | Its 21 X→En languages do not include Hindi (verified in the card's `language:` list) |

The BhasaAnuvaad sources, one by one:
- **SeamlessAlign:** Meta released only "metadata information for reconstructing the dataset", which points to web audio URLs ([README](https://github.com/facebookresearch/seamless_communication/blob/main/docs/m4t/seamless_align_README.md)). Meta grants nothing for the audio itself, and AI4Bharat's rehosted copy states no rights chain.
- **NPTEL:** Hindi lectures re-recorded from English originals. The licence is reported as CC BY-NC-SA ([SpicyIP 2014](https://spicyip.com/2014/02/national-program-on-technology-enhanced-learning-indias-contribution-to-the-open-education-movement.html), a secondary source).
- **Spoken Tutorial:** "CC BY-SA 4.0" at [spoken-tutorial.org](https://spoken-tutorial.org/), owned by IIT Bombay.
- **Mann Ki Baat:** one speaker (the PM's radio address). Government copyright terms are UNVERIFIED.
- **WordProject:** Bible audiobooks; site rights UNVERIFIED.

### A3. Eval-only speech sets

| Dataset | Licence | Hindi size | Use |
|---|---|---|---|
| [Lahaja](https://huggingface.co/datasets/ai4bharat/Lahaja) | Card text "CC BY 4.0" (metadata tag says MIT) | 12.5 h, 132 speakers, 83 districts, R + E | W4 ASR benchmark. Card: performance declines "especially with content heavy in named entities and specialized terminology" |
| [Vaani-Benchmark-V1.0](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Benchmark-V1.0) | CC BY 4.0, gated | 10.9 h, 5,050 segments, 1,103 speakers, **3 independent human transcripts each** | W4 ASR choice on spontaneous, image-prompted speech (Vaani protocol). Held out from Vaani by design, so it is not in Vaani's public release |
| [sarvamai/contextual_asr_benchmark](https://huggingface.co/datasets/sarvamai/contextual_asr_benchmark) `hi-IN` | **No licence** | 611 turns, ≤30 s, ~15 MB at 16 kHz (≈8 min) | User turns in banking / e-commerce / healthcare voice-bot calls (debt collection, loans) with the bot's context. Card: "synthesized or recorded". A W8c entity/intent probe, and a W2 *style* reference (do not copy text; no licence) |
| [Svarah](https://huggingface.co/datasets/ai4bharat/Svarah) | Card text "CC BY 4.0" | 9.6 h **Indian-accented English**, 117 speakers, R + C | Not Hindi. W8c uses it to pick the output-side English ASR, because our output voice `shubh` is en-IN |
| [MUCS SLR104](https://www.openslr.org/104/) Hi-En | CC BY-SA 4.0 | 89.86 h train + **5.18 h test**, 16 kHz | Spoken-tutorial lectures with technical code-switching. The test split is a code-mixed dev set. Part A use is blocked by Q2 (SA) |
| [Gram Vaani SLR118](https://www.openslr.org/118/) | "used freely for academic purposes, but permission for any commercial use… contact@gramvaani.org"; CC BY-NC 4.0 on AIKosh | 100 h train, **5 h dev, 3 h eval** (labelled) + ~1,000 h unlabelled; mp3, 8–48 kHz | The closest open match to §4 telephone + noise. Part A: Trap. Eval-only use is AMBIGUOUS (Q3) |
| BhasaAnuvaad-test | as BhasaAnuvaad | Table A3 lists 2.50 h Hi→En (the paper's text says "~20 minutes") | Mined, so check its sources before trusting it |

### A4. Traps and non-starters (verified)

- **Shrutilipi** ([HF](https://huggingface.co/datasets/ai4bharat/Shrutilipi), card says CC BY 4.0). The audio is All India Radio news, and newsonair's [copyright policy](https://www.newsonair.gov.in/copyright-policy) requires the NSD's "prior written permission" for anything beyond personal non-commercial use. It is also read broadcast news. **Trap.** W1 already lists it.
- **Common Voice Hindi.** From Oct 2025 releases ship only via Mozilla Data Collective, whose [terms](https://mozilladatacollective.com/terms) §2(b)(ix) bar "making the Dataset available on any platform… other than the Platform" unless the licence expressly permits it. Older releases are CC0. W1 already lists it as avoid.
- **Snow Mountain** (Hindi Bible readings, one speaker per language). CC BY-SA 4.0 ([arXiv 2206.01205](https://arxiv.org/abs/2206.01205v1)). SA, read, single speaker.
- **HF aggregates that launder licences:**
  - [agarwalayushi/hinglish](https://huggingface.co/datasets/agarwalayushi/hinglish) claims 2,264 h under a "CC BY 4.0" label, but bundles Common Voice, IndicTTS and several "See source" sets.
  - [zaid-zara/zara-hinglish-master-v1](https://huggingface.co/datasets/zaid-zara/zara-hinglish-master-v1) re-packs that set together with [dianavdavidson/MUCS-Hinglish](https://huggingface.co/datasets/dianavdavidson/MUCS-Hinglish), which re-labels CC BY-SA SLR104 as CC BY 4.0.
  - [SPRINGLab/Hindi-1482Hrs](https://huggingface.co/datasets/SPRINGLab/Hindi-1482Hrs) has 907k rows, no licence and no provenance.
  - **Never source from an aggregate.**
- **Vendor samples:**
  - [liva-ai/hindi-english-asr](https://huggingface.co/datasets/liva-ai/hindi-english-asr) is CC BY 4.0, 48 kHz, channel-separated and human-verified, but only ~10 MB.
  - [HumynLabs e-commerce](https://huggingface.co/datasets/HumynLabs/e-commerce-customersupport-hinglish-audio) / [airline](https://huggingface.co/datasets/HumynLabs/airline-customersupport-Hinglish-audio) Hinglish sets are CC BY 4.0 but <1,000 clips of **one fixed sentence**.
  - Too small to matter.
- **[shivamgupta/hinglish-s2s-synthetic-calls](https://huggingface.co/datasets/shivamgupta/hinglish-s2s-synthetic-calls).** Licence "other". 29.8 h of synthetic Hinglish support calls: Gemma-4-31B dialogue, Kokoro / IndicF5 TTS, domains food / e-commerce / cab / bank / telecom. Not usable, but it is a close prior-art pipeline (→ `prior-art.md`).
- **Not applicable:** [People's Speech](https://huggingface.co/datasets/MLCommons/peoples_speech) is English only; [Speech-MASSIVE](https://huggingface.co/datasets/FBK-MT/Speech-MASSIVE) has no Hindi and is CC BY-NC-SA; [Omnilingual ASR corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus) has no Hindi config; [Vistaar](https://github.com/AI4Bharat/vistaar) is a benchmark wrapper (Kathbath, CV, FLEURS, IndicTTS, MUCS, Gram Vaani) that inherits each one's licence.
- **Bhashini / ULCA / AIKosh** are re-hosting portals ([ULCA](https://github.com/bhashini-dibd/ulca), [AIKosh](https://aikosh.indiaai.gov.in)), not sources. Their licences can differ from the originals (finding 4). Cite the original publisher.

### A5. Facts for W1's own corpora (no re-measurement)

- **Vaani:** the [main card](https://huggingface.co/datasets/ARTPARK-IISc/Vaani) gives ~31,278 h of spontaneous image-prompted speech, 156K speakers, 165 districts, 2,122 h transcribed. [Vaani-transcription-part](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part) has **Hindi 1,003.78 h transcribed**, plus Bhojpuri 26.60 h, Maithili 20.84 h, Chhattisgarhi 17.69 h and English 22.46 h.
- **Vaani-Noise-Event-Dataset** ([HF](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset), CC BY 4.0):
  - 90,637 Vaani clips (~154.6 h), 21.8 h of them with verified timestamps;
  - 7 noise categories: animal, vehicle_traffic (horns, sirens), baby_child, singing_music, phone_signal_alarm (ringtones), appliance_machine (fans, TV), human_non_speech;
  - per-event start/end times.

  **Use: pick noisy real Hindi clips for Part A** (§4 noise, counted as real hours).
- **IndicVoices:** IV-ST shows ≥282.9 h of Hindi IV already cut into ~47 s paragraphs with English. IV-R shows that at least 74.6 h of Hindi IV is 44.1 kHz read/extempore (what survived IV-R's quality filter).

## B. Text (A-text / Eval)

The helper's report was spot-checked against the HF API licence fields (MASSIVE, COMI-LINGUA, IN22-Conv, Naamapadam, MultiCoNER v2, HinGE), the PMIndia README, the IndicTrans2 README licence table, the Hinglish-TOP README and the L3Cube README. IndicCMix and SpeechArenaBench are my own additions.

**How licences bind text.**
- Text is used only to *build* scripts and references. It never feeds training directly (§3B).
- **NC.** The recipient of our dataset is a company that also takes code IP (§10 R5). Anything NC, or built on NC data, is out (INFERRED).
- **SA.** Copying or closely paraphrasing SA text makes our scripts adapted material that must be shared BY-SA. Using it only as inspiration does not adapt it: harvesting entity names (facts), or writing abstract templates such as `alarm <TIME> <DAY>` (INFERRED).
- Every source we consult still goes into `data_card.md` (§10 R1).

Abbreviations: Dev = Devanagari, Rom = Roman.

### B1. Usable seeds and references (A-text)

| Dataset | Licence (exact) | Size (Hindi) | Script / code-mix | Use for W2 / W4 / W6 |
|---|---|---|---|---|
| [BPCC](https://huggingface.co/datasets/ai4bharat/BPCC) **H-Daily** ([IndicTrans2 README](https://github.com/AI4Bharat/IndicTrans2)) | "BPCC-H-Wiki & BPCC-H-Daily: CC-BY-4.0". The mined parts are "CC0", but AI4Bharat states "We do not own any of the text from which this data has been extracted" | H-Daily 139K pairs over 22 languages (Hindi share UNVERIFIED); gated | Dev / none | Human-translated everyday sentences. W6 style reference. Avoid the mined parts |
| [MASSIVE](https://huggingface.co/datasets/AmazonScience/massive) hi-IN | cc-by-4.0 | 16,521 utterances (11,514 / 2,033 / 2,974) | Dev, numbers as words ("सुबह पांच बजे") / light | Command templates for time, date, alarm, contacts and payments. Slot labels give an entity inventory |
| [IndicCMix](https://huggingface.co/datasets/ai4bharat/IndicCMix) (AI4Bharat, 2026-04) | "License: MIT." | 104,800 hi rows; gated | Three renderings per English sentence: Dev with **English kept in Latin**; `full_native_script` with those English tokens **transliterated into Devanagari** (word-aligned); and romanised | `full_native_script` already follows our transcript convention (English words in Devanagari), so it gives W4 a ~100K-sentence loanword-spelling source and W2 code-mix patterns. The translation method is unstated (machine-generated, INFERRED), and the register is written. Its English audio is F5-TTS, so it is never a target (§3B) |
| [COMI-LINGUA](https://huggingface.co/datasets/LingoIITGN/COMI-LINGUA) (2025) | cc-by-4.0 | 125K+ instances (MT 29,692; NER 24,913; LID 35,487) | Dev with Latin English ("Google के CEO सुंदर पिचाई") / medium | The closest open match to Dev-script Hinglish. MT drafts were made with Llama 3.3 then human-corrected; text sources UNVERIFIED. Keep its MT test held out |
| [PMIndia](https://data.statmt.org/pmindia/README) | "released under the CC-BY-4.0" | hi-en 56,831–62,451 pairs, depending on the aligner | Dev / none | Formal amounts, schemes and names. Good for government-scheme entity banks |
| [MultiCoNER v2](https://huggingface.co/datasets/MultiCoNER/multiconer_v2) hi | CC BY 4.0 | 28,545 rows | Dev / none | Fine-grained entity bank (products, medicines, places, people) |
| [Naamapadam](https://huggingface.co/datasets/ai4bharat/naamapadam) | cc0-1.0, plus "We do not own any of the text" | ~1.0M hi rows | Dev | **Entity lists only** (names, places, orgs). Its sentences are projected from Samanantar (NC on HF), so don't copy them |
| [Aksharantar](https://huggingface.co/datasets/ai4bharat/Aksharantar) | "Manually collected data: Released under CC-BY license. Mined dataset…: CC0" | hin-en 1.3M word pairs | Dev ↔ Rom words | W4: spell English loanwords in Devanagari. Some pairs come from Dakshina (SA), so use it for lookup, not republication |
| [samvaad-hi-v1](https://huggingface.co/datasets/sarvamai/samvaad-hi-v1) (Sarvam) | apache-2.0 | 101,476 conversations | Dev + English + Hinglish | Written assistant chat; generation method UNVERIFIED. Low value |
| [indic-numtowords](https://github.com/AI4Bharat/indic-numtowords) (tool) | MIT | — | — | Digits → Hindi words for W2 and W4. **Whether it handles lakh/crore and डेढ़/ढाई is UNVERIFIED; test it before relying on it** |

### B2. Inspiration only (SA: patterns and entity facts, never verbatim)

- **[Hinglish-TOP](https://github.com/google-research-datasets/Hinglish-TOP-Dataset)** (Apache-2.0). "Queries are derived from TOPv2", which is CC BY-SA 4.0. 10K human + 170K synthetic Rom queries. The best open date/time/name slot patterns; no lakh/crore.
- **mTOP Hindi** (CC BY-SA 4.0). Phone numbers and contacts.
- **PHINC** (cc-by-4.0 on the translations, but the text is from Twitter/Facebook). 13,738 Rom pairs in a natural Hinglish voice.
- **Dakshina** (CC BY-SA 4.0). A lexicon.

### B3. Eval-only text

| Set | Licence | Size | Use |
|---|---|---|---|
| [IN22-Conv](https://huggingface.co/datasets/ai4bharat/IN22-Conv) | cc-by-4.0 | 1,503 sentences, incl. banking 57, insurance 82, healthcare 67, daily_dialogue 117 | **W6 engine benchmark and W8c text-MT dev**. Conversational and domain-matched |
| [IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen) | cc-by-4.0 | 1,024 | General MT dev |
| FLORES+ / NTREX-128 | CC BY-SA 4.0. FLORES+ card: "It should not be used as training data." | 997 + 1,012 / 1,997 | Standard MT test. Note that FLEURS-hi uses FLoRes sentences |
| [Sarvam TTS robustness](https://huggingface.co/datasets/sarvamai/tts-robustness-benchmark) / [general](https://huggingface.co/datasets/sarvamai/tts-general-benchmark) | "license: other". The general set says "evaluation-only… not for model training" | hi 103 sentences (Numerics, Indian named entities, code-mixed, abbreviations, URLs) / 1,815 prompts | **W3 TTS QA probes.** They show exactly what Sarvam stress-tests. Ask before use (Q9) |
| [SpeechArenaBench](https://huggingface.co/datasets/ai4bharat/SpeechArenaBench) (AI4Bharat, Interspeech 2026) | mit | 5,357 sentences over 10 languages, hi included; 120K+ human pairwise judgements of 7 TTS systems (incl. **Bulbul v3 Beta** and IndicF5) | W3 evidence for choosing a Hindi TTS engine, plus its code-mixed and symbolic stress sentences. Its audio comes from commercial TTS, so it is never Part A audio |

### B4. Text traps

- **NC / NC-SA:**
  - HinGE (built on IITB, which is CC BY-NC 4.0);
  - MixMT 2022 (HinGE + PHINC);
  - L3Cube HingCorpus / HingLID ("CC BY-NC-SA 4.0", Twitter);
  - X-RiSAWOZ (from RiSAWOZ, CC BY-NC 4.0);
  - hinglishNorm (NC-SA);
  - IITB En-Hi ("CC BY-NC 4.0", plus OpenSubtitles);
  - HindEnCorp (CC BY-NC-SA 3.0);
  - Samanantar (cc-by-nc-4.0 on HF, but CC0 in the IndicTrans2 README: a conflict, so take the stricter).
- **Social-media / unclear:**
  - LinCE (licence UNVERIFIED, tweets);
  - GLUECoS (data re-fetched through the Twitter API);
  - CMU Hinglish DoG (cc-by-sa-3.0 / gfdl, "[More Information Needed]").
- **Paid:** MultiATIS++ (LDC, which also lists no Hindi).
- **Generated by a closed model:** Hinglish-Everyday-Conversations-1M ("generated using GPT4o-mini").

## C. Noise, RIR and codec (Aug)

The helper's report was spot-checked against the OpenSLR 17/20/26/28 pages, the ESC-50 README and the DNS-Challenge README "Dataset licenses" section. **We found no open Indian noise or telephone corpus with a clean licence.**

### C1. Reverb and noise banks

| Material | Licence (exact) | Size / SR | Role | Notes |
|---|---|---|---|---|
| [SLR26 simulated RIRs](https://www.openslr.org/26/) | "Apache 2.0" | 60,000 RIRs; `sim_rir_8k.zip` 128 MB / 16k 178 MB | Aug (Reading 2) | Image method. The 8 kHz set suits the phone path |
| [SLR28 RIRS_NOISES](https://www.openslr.org/28/) | "Apache 2.0" | 1.3 GB, 16 kHz | Aug (Reading 2) | 325 real RIRs (RWCP / REVERB / AIR) + simulated + point-source noise. Upstream terms of the real RIRs: UNVERIFIED |
| [BUT ReverbDB](https://speech.fit.vut.cz/software/but-speech-fit-reverb-database) `RIR-Only` | "CC-BY 4.0" | 8.7 GB, 9 rooms | Aug (Reading 2) | Skip the LibriSpeech tarball (third-party speech) |
| [AIR v1.4](https://www.openslr.org/20/) (Aachen) | OpenSLR: "Not stated in the download"; RWTH page points to an included MIT file (not read) | 175–203 MB | Aug (Reading 2), once the MIT file is read | **The only open IRs measured at a mock-up phone in hand-held / hands-free positions** |
| [MUSAN](https://www.openslr.org/17/) noise part | Whole: "CC BY 4.0". Noise files: Freesound "Public Domain", SoundBible "Attribution 3.0" or "Public Domain" | 6 h 7 min, 932 files, 16 kHz | Aug (Reading 2) | Has DTMF, dial tones, phone vibration and horns. **Never use MUSAN speech or music** (LibriVox speech; music with per-file BY-SA / BY-ND) |
| [DNS Challenge](https://github.com/microsoft/DNS-Challenge) `freesound_*` blobs only | "Only files with CC0 licenses were selected" | ~4.5 GB | Aug (Reading 2) | The AudioSet blobs are YouTube audio: Google's CC BY covers the ID lists, not the audio. **Trap** |
| [FSD50K](https://zenodo.org/records/4060432) | Per clip: CC0 19,873 / CC-BY 23,506 / BY-NC 6,041 / Sampling+ 1,777 | 108.3 h, 44.1 kHz | Aug (Reading 2), CC0 + CC-BY clips only | Filter by `*_clips_info`, and drop the "Human voice" classes |
| Freesound "india traffic / street / market / train / mumbai / delhi" | Per file. CC0 counts on 2026-10-14: 148 / 127 / 38 / 34 / 45 / 64 (queries overlap) | A few hundred clips | Aug (Reading 2), CC0 only, after a speech filter | The most realistic open Indian street audio. Expect intelligible speech in the field recordings |
| [Vaani-Noise-Event](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset) | CC BY 4.0 | 154.6 h of *speech* clips, Hindi among them | **A-real selector**, not Aug | It is speech, so it counts as third-party training data under both readings. Use it to choose noisy Part A clips (A5) |
| **Trap: NC** | ESC-50 (CC BY-NC 3.0; only ESC-10 is CC BY), UrbanSound8K (cc-by-nc-4.0), WHAM! (CC BY-NC 4.0), NISQA corpus ("non-commercial research"), IDMT-Traffic (cc-by-nc-nd) | | Trap | The recipient is a company (§10 R5) |
| **Trap: SA or unclear** | DEMAND (Zenodo field says cc-by-4.0, its PDF says BY-SA 3.0), QUT-NOISE (CC BY-SA), MicIRP (CC BY-SA 4.0), MIT IR Survey (no licence stated) | | Trap | SA risk exists if trained weights count as adapted material |
| **Trap: paid / no licence** | CHiME-3/4 backgrounds (LDC only); iNoise (TCS, IEEE DataPort paywall, no licence); [ajajali09/indian-noise-dataset](https://huggingface.co/datasets/ajajali09/indian-noise-dataset) (no licence, partly YouTube, largest class "People Talking") | | Trap | |

### C2. Telephone channel (procedural, no third-party data)

The local Arch ffmpeg is built with `--enable-gpl --enable-version3 --enable-libgsm --enable-libopencore_amrnb --enable-libopus`.

- **Codecs available:**
  - G.711 µ-law / A-law (`pcm_mulaw`, `pcm_alaw`);
  - GSM-FR (`libgsm`, a permissive TU Berlin notice);
  - AMR-NB (`libopencore_amrnb`, Apache-2.0);
  - Opus (`libopus`, BSD-3);
  - also g722, g726 and speex.
- **No AMR-WB encoder** in this build.
- **Call ffmpeg as a subprocess and don't vendor it.** ffmpeg here is GPL-3.0, and Sarvam takes the IP of our training code (§10 R5).
- **ITU-T STL (G.191).** Has the IRS / P.341 filters and rpeltp. Its licence is a copyleft "ITU-T Software Tools' General Public License", so treat it as an external binary at most. Better: write our own IRS-style band-pass.
- **RIR generators:** pyroomacoustics (MIT) and Habets RIR-Generator (MIT). Avoid gpuRIR (AGPL-3.0).
- **Augmentation library:** audiomentations (MIT).
- AMR patent status: UNVERIFIED.

### C3. Does third-party noise count as "training data" under §3B? (AMBIGUOUS)

**STATED.**
- §3B: "Only your Part A dataset may be used for training."
- FAQ: "Training data must still come from your Part A dataset."
- §3A: augmentation "(noise, speed, reverb and so on)… applied during training adds no hours."

**Reading 1 (strict).** Noise and RIR waveforms enter the loss and are not Part A, so they are forbidden. In support: the FAQ opens the door to external *models* only. Against it: §3A names noise and reverb augmentation without restricting the source.

**Reading 2 (a transform).** The record, its transcript and its Bulbul target all stay Part A. The noise is a transform parameter like a speed factor, and the cap that "only Part A" protects is speech hours. Even under this reading, **noise containing intelligible speech** (babble, cafe, crowd, Freesound street recordings) imports third-party speech. INFERRED: that is out under both readings.

**Safer, and valid under both readings:**
1. Seeded synthetic RIRs (pyroomacoustics) at training time.
2. A telephone chain on the fly:
   - our own band-limit to 8 kHz;
   - G.711, GSM-FR, AMR-NB or Opus-NB via ffmpeg;
   - packet loss, clipping and AGC;
   - upsample to 24 kHz (§4).
3. Procedural noise: coloured noise, hum, synthesised DTMF and beeps.
4. Non-speech spans cut from **inside** a record's `[source.start, source.end]`.
   - Spans outside record intervals belong to no record.
   - Using them arguably adds hours, because §3A counts "the union of the source-file intervals you use".
5. Most important: put real noisy, telephone Hindi into Part A as records (IndicVoices conversational 8 kHz, SPRING-INX, Vaani noisy clips, Rural Women).

Use the third-party banks in C1 only after a written Reading-2 answer (Q1).

**Declare it either way** in `data_card.md`, `model_card.md` and `TRAINING.md`:
- the transforms, ranges and seeds;
- each tool and its licence;
- for any third-party audio: name, URL, version, exact licence, a file-hash manifest and a fetch script.

R7 means re-runs must work, and §9.2 means the noise files themselves are never shipped.

## D. Ranked shortlists

**A-real**, ranked by closeness to §4 (spontaneous, one speaker, phone/noise, code-mixed) × licence cleanliness:

1. **IndicVoices Hindi** (W1). Extempore + 8 kHz conversational. IV-ST's ~47 s paragraph cuts are a ready segmentation reference.
2. **SPRING-INX Hindi** (W1). Telephone, conversational, ~10% code-mixed.
3. **Vaani Hindi** (W1). 1,003.78 h transcribed. Pick noisy clips with the Vaani-Noise-Event tags.
4. **Rural Women ASR v2** (new). ≈87 h of chunks, open, phone mics, IV schema, light code-mix.
5. **MUCS SLR103** (W1). Read stories, 8 kHz telephone band.
6. **Rasa / SYSPIN** (new). Studio, 2 voices each. Only if we want real read speech of payments, e-commerce or finance text.

Held back: Kathbath (licence conflict, finding 4) and Indic DiarBench (better as Eval).

**A-text:**
1. MASSIVE hi-IN (templates)
2. BPCC-H-Daily (everyday parallel)
3. IndicCMix `full_native_script` (English-in-Devanagari convention)
4. COMI-LINGUA (Dev-script code-mix style)
5. PMIndia (formal amounts and schemes)
6. Entity banks: MultiCoNER v2, Naamapadam lists, Aksharantar lookup
7. SA sets as inspiration only

**Aug:**
1. Procedural telephone chain (G.711 / GSM-FR / AMR-NB / Opus-NB via ffmpeg, our own 8 kHz band-limit, packet loss)
2. Seeded pyroomacoustics RIRs
3. Coloured noise, DTMF and beeps
4. Non-speech spans from inside Part A records

Only after a written yes on Q1: SLR26/28, MUSAN noise, BUT RIR-Only, AIR phone IRs, DNS `freesound_*`, FSD50K CC0/CC-BY, Freesound CC0 India clips.

**Eval** (W8c dev set first):
1. FLEURS hi_in dev + test, with `en_us` speech and FLoRes English. The only open Hi→En S2ST.
2. Indic DiarBench Hindi NF + FF (8.2 h, conversational, code-mixed).
3. IN22-Conv (banking, insurance, healthcare MT).
4. contextual_asr hi-IN (entity/intent turns, Q9).
5. SLR104 test (5.18 h, code-mixed).
6. Gram Vaani dev/eval (telephone), only if Q3 clears it.
7. Vaani-Benchmark + Lahaja (W4 ASR choice).
8. Svarah (picking the output ASR).

**Trap**, short list:
- Shrutilipi (AIR copyright)
- YODAS (no speaker consent)
- BhasaAnuvaad non-IV parts (SeamlessAlign / NPTEL / Spoken Tutorial / WordProject / Mann Ki Baat)
- IndicVoices-R (enhanced duplicate of IV)
- HF aggregates
- Gram Vaani in Part A
- Every NC text, noise or RIR set listed above

## E. Redistributable real Hindi under the 200 h cap

The licence-clean pool is CC BY 4.0 / MS O-UDA, consented collections. Hours are verified unless marked W1.

| Source | Hindi hours | Note |
|---|---|---|
| Vaani | 1,003.78 h transcribed | |
| SPRING-INX | 351 h | W1's number |
| IndicVoices | ≥282.9 h | Only the part visible through IV-ST. W1 has the full number |
| SYSPIN | ≈110 h | Studio |
| MUCS SLR103 | 100.6 h | |
| Rural Women | ≈87 h | Chunk sum; unique-source hours are lower |
| Rasa | 50.8 h | Studio |
| **Total** | **≈1,990 h** | Ten times the cap |

Excluded from that total:
- Kathbath (licence conflict).
- Lahaja, FLEURS and DiarBench (kept for eval).

**INFERRED estimate, for W1 to replace with measured yield:**
- Of that pool, the part that fits §4 is roughly 120–180 h: contiguous single-speaker 10–60 s windows of *spontaneous* speech, with transcripts W4 can convert.
- It comes from IV extempore + conversational, SPRING-INX, Vaani's noisier clips and Rural Women.
- Read and studio sources would add tens of hours more, but are a poor match for the test.
- **So the 200 h cap can be reached with real audio alone.** What limits real hours is W4's QA throughput and the real/synthetic mix that W2/W3's entity coverage needs, not licences.
- Any hours left for synthetic audio should go to the entity- and correction-dense content that prompt-driven real corpora are unlikely to supply. In the Rural Women sample, the task topics are hobbies, village life, culture, education and finance. Measuring entity density is W1's job.

## F. Questions for organisers (doubt form, §10 R9)

1. **§3B and augmentation.** May third-party noise/RIR corpora be mixed into Part A audio at training time, or does "only your Part A dataset" exclude them? (Both readings are in C3.)
2. **ShareAlike sources.** May Part A include CC BY-SA recordings (MUCS SLR104, Spoken Tutorial, Kathbath if its AIKosh licence governs), given that those records carry Sarvam-API translations that §10 R3 says we must not redistribute, and that SA requires adaptations to be licensed BY-SA?
3. **NC / academic-only sets.** May we use them for **internal evaluation only**, never training? Gram Vaani dev/eval is "academic purposes" on OpenSLR and CC BY-NC on AIKosh; NISQA is "non-commercial research".
4. **Mirror conflicts.** Which licence governs when the publisher's mirrors disagree? Kathbath: CC BY 4.0 on HF vs CC BY-SA 4.0 on AIKosh. Lahaja: MIT tag vs CC BY 4.0 text. Gram Vaani: academic vs CC BY-NC.
5. **YouTube CC BY audio** (YODAS, DiarBench in-the-wild). Does an uploader's CC BY licence satisfy "licence and consent compliance" (§3A) without speaker consent?
6. **Open-weight models with output clauses.** Llama 3.x requires "Llama" at the start of the name of any model trained on its outputs that is "distributed or made available". Does private submission of our weights to Sarvam count? Or should translation targets avoid such models? (Applies to IV-ST and COMI-LINGUA drafts, and to W6 engine choice.)
7. **Restored recordings.** Are enhanced versions (IndicVoices-R, FLEURS-R) "real recordings"? Are their hours counted against the original recording, or per file?
8. **Contamination.** Is any hidden-test audio drawn from public corpora (IndicVoices, Vaani, SPRING-INX, Indic DiarBench)? If so, should we exclude it from Part A?
9. **Sarvam's own HF benchmarks** with `license: other` or no licence (tts-robustness, tts-general, contextual_asr_benchmark). May we use them as internal dev or QA probes?

## G. Handoffs

- **W1:**
  - Add Rural Women ASR v2 (open, no gate, IV schema; check unique-source hours and consent).
  - Vaani transcribed Hindi is 1,003.78 h. Use Vaani-Noise-Event tags to pick noisy clips.
  - The IV-ST ~47 s segments are a cut-list reference.
  - Resolve the Kathbath licence conflict.
  - Rasa and SYSPIN are low-priority studio options.
  - Never cut the same speech from both IV and IV-R.
- **W2:**
  - MASSIVE hi-IN templates, IndicCMix code-mix patterns, COMI-LINGUA style, PMIndia/MultiCoNER/Naamapadam entity banks.
  - contextual_asr_benchmark bot contexts give domain realism (debt collection, loans); style only, no copying.
  - Hinglish-TOP and mTOP are SA, so patterns only.
- **W3:**
  - SpeechArenaBench human preferences (Bulbul v3 Beta, IndicF5 and others, Hindi + code-mixed subsets).
  - Sarvam tts-robustness hi (103 sentences) as a QA probe, after Q9.
- **W4:**
  - Lahaja and Vaani-Benchmark (3 human references per segment) for the ASR benchmark.
  - Aksharantar + IndicCMix `full_native_script` for English→Devanagari spelling.
  - indic-numtowords, once its lakh/crore handling is checked.
  - Rural Women uses the same bracketed-English convention as IV (`[mobile]`), so the same converter applies.
- **W6:**
  - IN22-Conv as the engine benchmark.
  - IV-ST English (Llama-405B) as a reference only (Q6).
  - Note the BPCC mined-text ownership caveat.
- **W8c:** the dev set in the Eval shortlist (FLEURS hi/en, DiarBench NF+FF, IN22-Conv, SLR104 test; Gram Vaani if cleared), and Svarah for choosing the output ASR.
- **Training / augmentation:** C2 is a ready licence-checked tool list (ffmpeg as a subprocess, not vendored; avoid gpuRIR, which is AGPL).
- **research (`prior-art.md`):** shivamgupta/hinglish-s2s-synthetic-calls is a close prior-art pipeline (Gemma dialogue → Kokoro/IndicF5 → LoRA S2S).

## Status — 2026-10-14

**Done:**
- About 90 datasets classified by role, every licence and size taken from a page fetched today.
- The helpers' text and noise findings were spot-checked against primary pages (17 checks, all consistent).

**Measured, from metadata only (<1 MB on disk):**
- FLEURS hi_in hours per split: 6.66 / 0.71 / 1.34 h.
- Rural Women Hindi:
  - ≈87 h of chunks;
  - E 62% / R 38%, 99.7% female;
  - 44% of rows contain bracketed English;
  - 0 of 16 sampled speakers overlap W1's 1,390 IV speakers.
- Vaani transcribed hours per language parsed from the card: Hindi 1,003.78 h.

**Open:**
- Q1–Q9 for organisers.
- Gated files not inspected, because there is no HF token:
  - IV-ST parquet columns (the 1.89 GB Hindi file is too small for 283 h of PCM);
  - Rasa and SYSPIN transcript formats;
  - SYSPIN sample rate.
- Not verified: the LinCE licence, the Hindi share of BPCC-H-Daily, and AMR patent status.

**Blockers:** none for this scout. Deeper checks need an HF token in `.env` (`HF_TOKEN=…`) plus accepted terms on each gated repo:
- https://huggingface.co/datasets/ai4bharat/IndicVoices-ST
- https://huggingface.co/datasets/ai4bharat/Rasa
- https://huggingface.co/datasets/ARTPARK-IISc/Vaani-Noise-Event-Dataset
