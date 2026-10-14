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
| s3 | 399 | `stages/tts_in/20261014T210421Z-f936755` | `stages/tts/20261014T210433Z-f936755-s3` | ml.g6.xlarge |
