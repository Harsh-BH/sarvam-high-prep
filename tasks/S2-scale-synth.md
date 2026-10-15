# S2: scale synthetic Hindi audio (~+10 h)

**Goal:** about 10 more hours of complete synthetic records (≈ 1,500 scripts), entity- and phenomenon-dense like W2's pilot, with the pilot's fixes. Same pipeline: `w2/gen.py` scripts → `tts` (OmniVoice, cloned IndicVoices voices, QA retakes) → `align` → `translate`.

Owner files: `scale/synth/`, `data/scale-synth/`, this file. Models are the ones Harsh approved for W2/W3: DeepSeek-V3.2 (MIT) on Bedrock for scripts; OmniVoice (**weights CC BY-NC: NC**) for audio; vakyansh + IndicConformer (MIT) for QA; W5's vakyansh aligner; W6's Mistral Large 3 + gpt-oss-120b (Apache-2.0) translation.

**NC label:** every record from this stream has NC audio (OmniVoice) cloned from IndicVoices speakers who never consented to cloning (W3 consent caveat). Private learning use only: never publish the audio or anything derived from it.

## 1. Scripts: `w2/gen.py bulk` (prompt v5)

`uv run w2/gen.py bulk --model deepseek-v3.2 --n 1600 --seed bulk-v1 --takes 5 --repairs 4` → `data/scale-synth/bulk-v1/{scripts,rejects}.jsonl`, pids `w2b-00001`…

Same coverage matrix as the pilot (`design(1600, "bulk-v1")`: a new seed, so pilot-v1's cells are never touched). The `bulk()` profile changes, on top of prompt v4:

| Fix | Change | Evidence |
|---|---|---|
| W2 1: heavy code-mix | The heavy instruction now says the English clause goes in Devanagari inside one `{..|..}` span, never Latin outside the braces | Pilot heavy first-take pass 12%; bulk smoke (30 cells, seed `bulk-smoke`) 4/6 |
| W2 2: long band | Long asks 90–110 words (was 105–140; the model wrote ~93). New hard check: hi_text ≤ 400 non-space chars, with its own repair message | W3 lost 2 pilot scripts of 430–440 chars past 60 s; pilot long p50 342 chars, 11 over 400 |
| W2 3: amounts vs scenario | Amounts are redrawn above a per-domain cap: telecom < 1 lakh, UPI ≤ 2 lakh, e-commerce ≤ 3 lakh, travel ≤ 5 lakh, everyday ≤ 10 lakh (wrong values of a self-correction too) | Pilot: 26/500 scripts put a lakh-scale amount on a small purchase |
| W3: rare proper nouns | Bank values never drawn: Medanta, Lhakpa, Fortis, Uber, Thyronorm, David. Rule: ≥ 2 entity misses and ≥ 25% of the pilot takes that contained them (Medanta 13/17, Lhakpa 12/16, Fortis 5/13, Uber 4/8, Thyronorm 2/8, David 2/6) | `stages/tts/20261014T163810Z-ddf6f86/takes.jsonl` |
| W3: spelling | Whole words आइ → आई and भई → भाई, in code before parsing | Both ASRs rewrote them on 38 and 8 pilot takes |

- Spend goes to its own ledger, stream `scale-synth` (W2's ledger was near its cap), capped at ₹1,320 of archive estimates.

**Result (2026-10-15):** **1,597 kept of 1,600** (3 rejects after 5 takes + 4 repairs: one ordinary-word correction, one gender, one spec_text). 2,414 Bedrock calls, **$4.35** (real, from archived token usage). 25 min with 8 workers.

| | pilot-v1 (prompt v4) | bulk-v1 (v5) |
|---|---|---|
| first-take pass, all | 65% | **73%** |
| first-take pass, heavy | 12% | **40%** |
| heavy kept within 5 fresh takes | 42% | **93%** |
| kept after repairs | 100% | 99.8% |
| long scripts: words p50 / chars max | 93 / 460 | 79 / **400** |
| scripts over 400 chars | 11 | 0 |
| estimated audio (0.099 s/char) | 3.38 h | **10.16 h** |

Checked on the output: no आइ or भई left (आई 803, भाई 116); no avoided name in any given entity or anywhere in the text; no given amount over its domain cap and no करोड़ in any small-purchase domain; all 1,597 pass `validate.check_record` (SPEC §2) again; pids and texts unique, none equal to a pilot text. Gender 798 f / 799 m. Est. seconds p50: short 16.2, medium 24.3, long 28.6.

Ids: `uv run w3/tts.py alloc` → **`rec-400550`–`rec-402146`** in `w3/ids.tsv` (1,597, committed before any render).
- Self-checks (`gen.py`'s `__main__`): all 500 pilot-v1 cells and user prompts replay unchanged (so pilot-v1 still replays from the archive); on all 1,600 bulk cells every entity passes the filters; respelling, the 400-char rule and the heavy wording.

## 2. Held-out split

w8c's rule (`w8c/split.py`, 2026-10-15): synthetic hold-out is whole voices, 4 of the 31 OmniVoice voices (2 f, 2 m), plus held-out real speakers and pilot script pids. A bulk record rendered on a held-out voice would be neither train nor in w8c's frozen dev/test lists, i.e. wasted. So the bulk renders use only the voices `is_heldout()` passes: `scale/synth/voices.py` → `scale/synth/voices-train.tsv` (27 voices: 11 f, 16 m), passed to `tts.py --voices`. Nothing is held out by hand; every final record is re-checked with `is_heldout()`. (After `git merge main` the bank built from main's `w8c/splits/heldout.json` is byte-identical.)

## 3. Render: `tts` stage, 4 shards on SageMaker GPU

The stage has no `--shard`, so the scripts are split round-robin into 4 files (`data/scale-synth/bulk-v1/shard-{0..3}.jsonl`), each uploaded as its own `tts_in` run, and each rendered by its own job on a different GPU type (quota is 1 per type, shared with other sessions; g5.xlarge/g5.2xlarge were busy with w8c/w8d). On-demand only: W3 never got g4dn spot capacity. One job per shard, never two (the kept-take archive is write-once).

`uv run pipeline/run.py tts --in stages/tts_in/<shard run> --run <run> --sm <instance> -- --engine omnivoice --max-takes 4 --corpus w2-scripts-bulk-v1 --voices scale/synth/voices-train.tsv`

| Shard | Scripts | Input | Run | Instance |
|---|---|---|---|---|
| s0 | 400 | `stages/tts_in/20261014T210414Z-f936755` | `stages/tts/20261014T210433Z-f936755-s0` | ml.g4dn.2xlarge |
| s1 | 399 | `stages/tts_in/20261014T210417Z-f936755` | `stages/tts/20261014T210433Z-f936755-s1` | ml.g6.2xlarge |
| s2 | 399 | `stages/tts_in/20261014T210419Z-f936755` | `stages/tts/20261014T210433Z-f936755-s2` | ml.g4dn.xlarge |
| s3 | 399 | `stages/tts_in/20261014T210421Z-f936755` | ~~`stages/tts/20261014T210433Z-f936755-s3`~~ → **`stages/tts/20261014T213550Z-c83f577-s3`** | ml.g6.xlarge waited 31 min for capacity and was stopped unstarted ($0) → ml.g4dn.4xlarge |

All four ran on-demand at once (s3 first waited ~2 min for the account's 15-instance training limit, held mostly by scale-real's CPU cut jobs). Each run: exit 0, `validate.py --partial` 0 violations.

| Run | Records | Burned | Hours | Billed s | Instance $/h | $ |
|---|---|---|---|---|---|---|
| `stages/tts/20261014T210433Z-f936755-s0` | 387 | 13 | 2.635 | 6,065 | 1.035 | 1.74 |
| `stages/tts/20261014T210433Z-f936755-s1` | 380 | 19 | 2.601 | 3,623 | 1.467 | 1.48 |
| `stages/tts/20261014T210433Z-f936755-s2` | 383 | 16 | 2.628 | 7,041 | 0.811 | 1.59 |
| `stages/tts/20261014T213550Z-c83f577-s3` | 389 | 10 | 2.639 | 5,568 | 1.656 | 2.56 |
| **total** | **1,539** | **58** | **10.504** | 22,297 | | **7.37** |

**Result:** **1,539 / 1,597 scripts kept (96.4%), 10.504 h** (pilot: 96.8%, 3.439 h). 2,011 takes rendered (13.95 h of audio), 1.31 takes per kept record; kept at take 1/2/3/4: 1,348/156/21/14. Clip length p10/p50/p90/max 15.8/24.2/33.5/50.1 s; **no take over 55 s** (the pilot lost 2 scripts past 60 s; the 400-char cap fixed it). 27 voices, all used; voice gender = script gender 1,539/1,539; **`is_heldout()` flags 0 of 1,539**. Pace 0.102 s/char p50. g6.2xlarge (L4) was the fastest: 3,623 s for its shard vs 6,065 s on g4dn.2xlarge (T4); overall 0.31 s render + 0.11 s ASR per audio second.

First takes (1,597), same QA as the W3 pilot:

| Run | CER ic p50/p90 | Entity hit | English-span hit | Agreed skip | Agreed insert | QA reject (r1 rules) | + 1 subst. |
|---|---|---|---|---|---|---|---|
| W3 pilot (500) | 0.018/0.036 | 98.8% | 99.3% | 2.0% | 0.6% | 10.6% | 13.0% |
| **S2 bulk (1,597)** | 0.019/0.039 | 99.0% | 99.3% | 4.0% | 1.4% | 13.1% | **15.6%** |

- First-take reasons (a take can have several): English span missed 84, one agreed substitution 71, agreed skip 64, entity missed 63, insert 22, CER 14, clipping 9.
- By phenomenon (first-take reject; scripts lost after 4 takes): digit strings 16% (20/742 lost), spelled acronyms 16% (19/571), lakh 18% (19/301), crore 15% (7/103), fractions 16% (23/397), dates 15% (8/354), times 15% (11/342), names 17% (39/942), self-corrections 16% (41/957), fillers 16% (26/792).
- The avoided names are gone from the misses. The top misses are now **LLM loanword spellings** both ASRs hear otherwise: कम्प्लेन 23 (+ कंप्लेन 5), कांटैक्ट 20, डैमेज्ड 8, इन्शुरन्स 8, plus the product **वनप्लस नॉर्ड** (26 misses), कोच्चि 8, देबाशीष बरुआ 6. INFERRED: non-standard spellings (half-nasal instead of anusvara, SPEC R12), which a TTS reads differently or an ASR doesn't know. Next run: add OnePlus Nord/Kochi to the avoid list and name these spellings in the prompt.

## 4. Align (W5's stage, SageMaker ml.c5.2xlarge on-demand)

`uv run pipeline/run.py align --in stages/tts/<shard run> --sm ml.c5.2xlarge`. Each: exit 0, 0 violations.

| Run | In | Records | Rejects | Billed s |
|---|---|---|---|---|
| `stages/align/20261014T223750Z-c83f577-s1` | tts s1 | 372 | 8 | 618 |
| `stages/align/20261014T230837Z-43a7e08-s0` | tts s0 | 381 | 6 | 650 |
| `stages/align/20261014T230839Z-43a7e08-s2` | tts s2 | 376 | 7 | 690 |
| `stages/align/20261014T231121Z-43a7e08-s3` | tts s3 | 380 | 9 | 700 |
| **total** | | **1,509** | **30 (1.9%)** | 2,658 = **$0.30** |

Rejects: 29 `unspoken` (the aligner hears no frame for a word), 1 `align_vocab` (ऍ, the LLM's candra E). The unspoken words are mostly **ओ** (9: "oh" in English clock times and digit runs, e.g. एट ओ वन, the same form W3 lost), then ई/इ/आई letter names and short function words. W3 pilot: 8/484 (1.7%).

## 5. Translate (W6's stage, on the laptop: the SageMaker role has no Bedrock access)

`w6/translate.py` charged every call to W6's ledger, which other sessions were filling at the same time (₹1,232 → ₹1,364 of ₹2,500 in an hour). It now takes `--stream`/`--cap-inr` (default unchanged; `uv run w6/test_translate.py` passes), and S2 translates on its own ledger:

`uv run pipeline/run.py translate --in stages/align/<run> --local -- --engine mistral3 --judge gptoss --stream scale-synth --cap-inr 2000`

`--local` downloads the shard's audio (~435 MB) for `validate.py`; each shard's cache was deleted after its run, at most 2 at a time (1 GB budget).

| Run | In | Records | Rejects | Hours |
|---|---|---|---|---|
| `stages/translate/20261014T230903Z-43a7e08-s1` | align s1 | 371 | 1 (`no_translation`: 5 segments returned for 4, every take) | 2.538 |
| `stages/translate/20261014T232244Z-43a7e08-s0` | align s0 | 381 | 0 | 2.600 |
| `stages/translate/20261014T232346Z-43a7e08-s2` | align s2 | 376 | 0 | 2.595 |
| `stages/translate/20261014T233249Z-43a7e08-s3` | align s3 | **stopped at 154/380** | – | – |

**s3 is unfinished.** Claude Code stopped its process when the laptop ran critically low on memory (1.2 of 15 GB free, other sessions' jobs). Nothing was uploaded; the 154 done records sit in `data/cache/translate/20261014T233249Z-43a7e08-s3/out/` with s3's audio cache (436 MB), and every call it made is archived. Resume (same `--run`, so it continues; archived calls cost nothing):

```
uv run pipeline/run.py translate --in stages/align/20261014T231121Z-43a7e08-s3 --run 20261014T233249Z-43a7e08-s3 --local -- --engine mistral3 --judge gptoss --stream scale-synth --cap-inr 2000
rm -rf /home/harsh1/github-repos/sarvam-high-prep/data/cache/audio/stages_tts_20261014T213550Z-c83f577-s3_audio /home/harsh1/github-repos/sarvam-high-prep/data/cache/translate/20261014T233249Z-43a7e08-s3
```

Every finished record: `is_heldout()` = False (0 of 1,509).

## Spend log (S2 cap $30)

Bedrock: real cost from each archived call's token usage at the ap-south-1 on-demand prices (W2/W6 tables). Ledger `archive/spend/scale-synth/` holds ₹1,398 of pre-call estimates (1.4× the real cost).

| Item | Where | Calls / billed s | $ |
|---|---|---|---|
| Scripts: DeepSeek-V3.2 (incl. the 30-script smoke) | Bedrock | 2,459 calls | 4.43 |
| Translate: Mistral Large 3 engine + gpt-oss-120b judge (s0, s1, s2 + 154 of s3) | Bedrock | 4,186 calls | 5.40 |
| TTS s0, g4dn.2xlarge | SageMaker | 6,065 s | 1.74 |
| TTS s1, g6.2xlarge | SageMaker | 3,623 s | 1.48 |
| TTS s2, g4dn.xlarge | SageMaker | 7,041 s | 1.59 |
| TTS s3, g4dn.4xlarge | SageMaker | 5,568 s | 2.56 |
| TTS s3, g6.xlarge (never got capacity, stopped) | SageMaker | 0 s | 0 |
| Align ×4, c5.2xlarge | SageMaker | 2,658 s | 0.30 |
| **Total** | | | **17.50** |

- Translation costs $0.0042 per record; s3's remaining 226 records ≈ $0.95 → **≈ $18.45 when done**.
- **Cost per kept hour:** $17.50 / 10.304 h aligned = **$1.70**; with s3 translated, $18.45 / 10.30 h ≈ **$1.79** (scripts $0.43, TTS $0.72, align $0.03, translate $0.61). TTS alone: $7.37 / 10.504 h = $0.70 (W3 pilot $0.74).
- `credits.sh` at status: usage $16.45, NET $0.00 (credits cover everything; Cost Explorer lags ~24 h).
- S3: +~2 GB (takes, kept audio, stage runs), ≈ $0.05/month.

## Status — 2026-10-15

**Done:**
1. **Scripts:** `w2/gen.py bulk` (prompt v5 with W2's three fixes and W3's handoff), seed `bulk-v1`: **1,597/1,600 kept**, $4.43. Pilot-v1 cells and prompts unchanged (self-check replays all 500).
2. **Held-out:** renders use only voices w8c's `is_heldout()` leaves for training (27 of 31). **0 of 1,509 final records are held out**; nothing held out by hand.
3. **Ids + render:** `rec-400550`–`rec-402146` in `w3/ids.tsv`; 4 shards on 4 GPU types in parallel; **1,539 kept (96.4%), 10.504 h**, 0 violations; first-take QA reject 15.6%; 58 ids burned (`rejects.jsonl` of each tts run).
4. **Align:** 1,509 records, 10.304 h, 30 rejects. **Translate:** 3 of 4 shards, 1,128 records, 7.733 h, 1 reject.

**Final run ids for W7 assembly** (whole records, no id overlap, `source.corpus = w2-scripts-bulk-v1`):
- `stages/translate/20261014T230903Z-43a7e08-s1`
- `stages/translate/20261014T232244Z-43a7e08-s0`
- `stages/translate/20261014T232346Z-43a7e08-s2`
- `stages/translate/20261014T233249Z-43a7e08-s3` (**after the resume above**; until then its input is `stages/align/20261014T231121Z-43a7e08-s3`, 380 records, 2.572 h, no translations)

**Blocker:** translate s3 was stopped by the memory-pressure reaper (not a code failure). Not restarted, per the reaper's rule. One command resumes it (section 5); ~10 min, ~$0.95.

**Handoffs:**
- **Master:** three small, backward-compatible changes outside `scale/synth/`. `w2/gen.py`: `bulk` mode and prompt v5, pilot path byte-identical. `w3/tts.py`: `--voices` (default `w3/voices.tsv`). `w6/translate.py`: `--stream`/`--cap-inr` (default `w6`). Each one's self-check passes.
- **W7:** these records are **NC** (OmniVoice weights) and clone IndicVoices speakers who never consented to cloning. Private use only. Synthetic ids bind to `_archive.tts` as for the pilot. The `tts` LATEST points at the last shard to finish; use the run ids above.
- **W8 (Part B):** all 1,509 are train by w8c's rule (0 held out).
- **Next synthetic run:** add OnePlus Nord and Kochi to `bulk()`'s AVOID. Name the loanword spellings the ASRs hear differently in the prompt (कंप्लेंट not कम्प्लेन, कॉन्टैक्ट not कांटैक्ट, anusvara per SPEC R12). Drop the "ओ" (oh) clock form, which both QA and the aligner lose. The voice bank is the main diversity limit: 11 female voices for 798 female scripts.
- **Monitoring gotcha (zsh):** `for j in $J` does not split `$J` in zsh, and `grep` without `--line-buffered` holds Monitor events. Two of my watches were silent for this reason.

**The one list for Harsh: empty.** No new voices or engines; W3's voice/filler list still covers what needs ears. Entity spelling and values are checked automatically (QA entity hit 99.0%, number values pre-verbalised by `banks.py`).

**Disk:** `data/scale-synth` 25 MB (scripts, shards, logs). Plus s3's resume state in the shared cache: 436 MB audio + 5 MB; delete it after the resume (command above). No model weights downloaded locally.
