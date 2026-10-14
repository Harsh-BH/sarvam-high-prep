# W8d: fine-tune SeamlessStreaming on our data (first Part B experiment)

**Goal:** fine-tune an open streaming S2ST model on release/pilot-v2 *train* ids, then measure it before and after with the W8c harness. Learning project: a small or null effect is a valid result, as long as it is measured honestly.

Origin: W8d in [W8-partb-prep.md](W8-partb-prep.md) and research/prior-art.md §8d.3 and §8d.6, adapted. The base is **SeamlessStreaming**, Harsh's decision.

**PS rules that still apply as engineering targets:**
- §3B: training data comes from Part A only.
- §5.1: 80 ms frames, 24 kHz, real time.
- §7: quality, intent, entity, latency and audio KPIs.
- §10 R3: Sarvam (Bulbul) audio in pilot-v2 is used for private training only and never published.

**Licences:** the seamless_communication code is MIT. The weights (`facebook/seamless-streaming`, rev 82746da) are **CC BY-NC 4.0**. That is acceptable for private learning. Anything derived from them (our fine-tuned decoder, outputs) is **NC: never publish it**.

## 1. Recipe research (verified in the code, not from the paper)

Read at seamless_communication `9a081e9` (2026-09-08) and fairseq2 `v0.2.0`.

### What the streaming model is made of

The model loads from two cards plus a vocoder:

| Part | Card / module | Size | Used in streaming as |
|---|---|---|---|
| Speech encoder | `seamless_streaming_unity`: w2v-BERT 2.0 conformer + length adaptor (`UnitYModel.encode_speech`) | in a 3.6 GB file together with T2U | `OfflineWav2VecBertEncoderAgent`: re-encodes the **whole received prefix** at every step. Non-causal, no encoder state is carried over |
| Text decoder + policy | `seamless_streaming_monotonic_decoder`, arch `dense_1b`: 24 layers, d = 1024, NLLB vocab 256,102, tied embeddings. **Each layer has a `PChooseLayer`** (EMMA): 4-layer energy MLPs on the decoder state and on 2×-pooled encoder states | 4.3 GB fp32, ~1.07 B params | `MMATextDecoderAgent`: greedy. For each token, write while `p_choose(last token, last source frame) ≥ 0.5`, otherwise read |
| T2U | `seamless_streaming_unity.t2u_model`: NAR T2U | (in the unity file) | `NARUnitYUnitDecoderAgent`. **Its input is the monotonic decoder's hidden states** (`decoder_features`), not just the text |
| Vocoder | `vocoder_v2`: unit HiFi-GAN | 168 MB | `VocoderAgent`, 16 kHz output (we resample to 24 kHz) |

The unity card's own offline text decoder is disabled in streaming (`use_text_decoder = False`).

### What can be fine-tuned with released code

| Part | Released training code? | Notes |
|---|---|---|
| UnitY S2T: encoder + offline text decoder | **Yes**: `m4t_finetune --mode SPEECH_TO_TEXT` | Trains the *offline* decoder, which streaming never uses. Tuning only the encoder this way would shift the input of a monotonic decoder that was trained on the frozen encoder. Not useful on its own |
| T2U | **Partly**: `m4t_finetune --mode TEXT_TO_SPEECH` | Raises `NotImplementedError("T2U finetuning implemented only for UnitYT2UModel")` for the NAR T2U that v2 and streaming use (`cli/m4t/finetune/trainer.py`). It would also need English target speech units, and we have no English audio (W8b) |
| **Monotonic decoder: translation** | **No trainer, but the forward is trainable** | `MonotonicDecoderModel.decode()` returns `(states, padding, p_choose)`. Cross-attention is ordinary soft attention over all encoder states, and `p_choose` is a side output. Teacher-forced cross-entropy needs ~150 lines of our own: loss = fairseq2 `SequenceModelOutput.compute_loss`, data = M4T dataloader conventions |
| Monotonic decoder: EMMA policy (`p_choose`) | **No** | There is no expected-alignment / latency loss anywhere in the repo (grep for `expected_alignment`, `latency`, `cumprod`: nothing). EMMA training (2312.05187 §4) is not released. fairseq v1's MMA code has a different p_choose and a different framework |
| Vocoder | **No** | |
| LoRA | **Yes, in fairseq2 0.2** | `fairseq2.nn.lora.wrap_lora` / `freeze_non_lora` / `unwrap_lora(merge=True)` wrap fairseq2 `Projection`s by name regex. Its `lora_state_dict()` looks up module names as state keys and returns nothing useful, so we save `lora_*` keys ourselves |

### Gotchas found in the code

- **Feature path.**
  - Streaming inference computes fbank with `waveform_scale = 1.0` (16 kHz float in [-1, 1]) and **`standardize = False`** (`online_feature_extractor.py`, no `--denormalize`).
  - The M4T fine-tune dataloader uses `waveform_scale = 32768` and `standardize = True`.
  - Training features must copy the streaming path, not the M4T dataloader.
- **Length cap.**
  - `streaming_evaluate` defaults to `max_len_a = 0, max_len_b = 100`: at most 100 target tokens per session.
  - pilot-v2 records average ~65 English tokens and reach ~200.
  - The runner must raise the cap, identically before and after (told W8c).
- **Long input.**
  - Without the VAD agent, one session is one growing decoder sequence; an early EOS just triggers a read (`no_early_stop`).
  - With `SileroVADAgent`, the input is cut at pauses and the states are reset.
  - We evaluate without VAD and train on whole records, so training matches inference.
- **Environment.**
  - `fairseq2 0.2.*` needs `fairseq2n` built against one exact torch: PyPI 0.2.1 pins torch 2.2.2, and Meta's index has `fairseq2n 0.2.1+cu121` for pt2.2.0.
  - The repo's SageMaker image (torch 2.9, py312) cannot run it. We use DLC `pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker` + `fairseq2n==0.2.1+cu121` from `fair.pkg.atmeta.com/fairseq2/whl/pt2.2.0/cu121`.

### The choice: LoRA on the monotonic decoder, cross-entropy only

**What we train:**
- Train only the **monotonic text decoder**.
- LoRA r = 16, α = 32, dropout 0.05 on all 24 layers' self-attn and cross-attn q/k/v/out and FFN in/out.
- Teacher-forced cross-entropy (label smoothing 0.1) on whole records: Hindi audio → the record's en_text sentences joined with spaces.

**What stays frozen:**
- the speech encoder (its outputs are precomputed once);
- embeddings, layer norms and every `p_choose` layer;
- T2U and the vocoder.

**Why:**
- It is the smallest change that changes what the streaming model *says*: both text and speech come out of this decoder.
- It is the only component whose released forward is trainable *and* used in streaming.
- LoRA plus a low-ish learning rate limits forgetting on 6 h. CMU IWSLT 2025 used LoRA r = 32, lr 1e-4; prior-art §8d.6.
- A frozen encoder means encoder outputs can be cached. Training is then a 1 B decoder on ~6 h of 160 ms frames: minutes on one GPU.
- The merged decoder drops straight into the base pipeline (`{"model": state_dict}` in fairseq2 key format is accepted as is by `convert_monotonic_checkpoint`).

**Known side effects (measured, not avoided):**
1. **Policy drift.**
   - `p_choose` is frozen, but its input is the decoder state, which LoRA changes. Latency can move either way.
   - The training job logs the mean p_choose and the mean |Δp_choose| on dev (teacher-forced) as a cheap early signal.
   - The real measure is W8c's latency metrics.
2. **T2U drift.** T2U consumes the decoder's final states, so speech can degrade even if text improves. The speech track (ASR-BLEU, UTMOS) catches it.
3. **Train/inference mismatch.**
   - The decoder is trained on full-utterance encoder states, but at inference it sees the encoding of a prefix.
   - This is the same mismatch Meta's EMMA training had (offline encoder states), so it is not new.

**Not done (next experiments, in order):**
- add sentence-level clips (they match VAD inference);
- a p_choose distillation term (BCE to the frozen policy) if latency drifts;
- a full-decoder fine-tune at lr 1e-6 as a second arm;
- the encoder as a LoRA target, which needs the encoder in the loop and no caching.

## 2. Training pairs

- **Source:**
  - `release/pilot-v2` manifest + audio (24 kHz);
  - ids from W8c's `data/w8c/splits/train.txt`: 1,010 records, 6.09 h;
  - dev/test are never touched for training.
  - Dev for early stopping = dev ids that already have references (33 held-out-voice synthetic records at first; the real rec-8xxxxx clips once W8c's text stages finish).
- **Pair** = one record:
  - audio resampled 24 → 16 kHz in the job (torchaudio);
  - target = `" ".join(s["en_text"] for s in sentences)`, tokenised `[eos, __eng__, …, eos]`;
  - loss skips the language token, as `m4t_finetune` does.
- **Never augmented**, never a new dataset record (SPEC §1).

## 3. Runs

**Code:**
- `w8d/submit.py`: builds the pairs, guards held-out ids/speakers/voices, uploads to `s3://…/w8d/<run>/`, and submits one job per GPU type. The first to start wins; the rest are stopped.
- `w8d/train.py`: the in-job trainer. It writes `log.jsonl`, `last.pt`, `best_lora.pt`, `decoder.pt` and `done.json` to `w8d/<run>/ckpt/`.

**Checks:**
- `uv run w8d/test_submit.py`: the pair builder and the held-out guard.
- `<venv>/bin/python w8d/test_train.py`: overfit on a tiny random decoder, spot-resume, and the merged-checkpoint round trip. It needs the fairseq2 env:
  ```
  uv venv -p 3.10 venv
  uv pip install -p venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.2.0 torchaudio==2.2.0
  uv pip install -p venv/bin/python --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.2.0/cpu --index-strategy unsafe-best-match "fairseq2n==0.2.1+cpu" fairseq2==0.2.1 simuleval==1.1.4 soundfile overrides "numpy<2"
  uv pip install -p venv/bin/python --no-deps <seamless_communication checkout>
  ```
  That is ~0.9 GB. Put it in tmpfs, not the laptop disk.

| Run | Instance | Data | Hyperparameters | Result | Spend |
|---|---|---|---|---|---|
| `20261014T205030Z-4b7a960` | ml.g5.2xlarge on-demand, 886 s billed. g6e.xlarge, g6.2xlarge and g4dn.2xlarge were also submitted; they were still pending when g5 started and were stopped at $0 | train 1,010 records (6.09 h); dev = the 33 held-out-voice pilot-v2 ids (synthetic only; real dev references not yet ready) | LoRA r16, α32, dropout 0.05, 13.4 M trainable; batch 8, 800 steps (6.3 epochs), lr 1e-4, 50 warm-up then linear to 0; label smoothing 0.1; bf16 autocast | **Dev NLL 0.963 → 0.650** nats/token (best at step 750; still creeping down, no overfitting yet). Train loss (smoothed) 3.38 → 2.21. Teacher-forced p_choose: mean 0.447 → 0.488, share ≥ 0.5 0.401 → 0.445, mean \|Δ\| 0.048, so the policy should write earlier. Timing: encode 1,043 records 275 s; 800 steps 300 s | $0.45 ($1.82/h) |

**Run 1 dev-curve:**

| Step | 0 | 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|---|
| Dev NLL | 0.963 | 0.866 | 0.796 | 0.723 | 0.693 | 0.673 | 0.660 | 0.653 | 0.652 | 0.650 |

**Caveat on the offline number:**
- The references are W6's Mistral-Large translations in the w6-style-v1 house style.
- Part of the NLL drop is the decoder learning that style (punctuation, number formats, register), not better understanding of the Hindi.
- COMET and the intent/entity proxies in the streaming eval are the better read on meaning.

## 4. Evaluation

_W8c harness (`w8c/run_seamless.py` + `w8c/eval.py score`), base vs fine-tuned, dev and test, text and speech tracks, latency._
