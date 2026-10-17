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
| `20261015T001450Z-6aaaaed` | ml.g5.2xlarge on-demand, 1,026 s billed. g6.2xlarge, g6e.xlarge and g4dn.2xlarge stopped at $0 | train as run 1; **dev = W8c's full dev set** (`stages/w8c_sets/20261014T221953Z-e398964/dev.jsonl`): 91 real IV-valid + 33 synthetic, all with references | As run 1, but the schedule decays over 1,600 steps, patience 6 | **Best at step 500** (≈ 4 epochs), then dev rises; early stop at 800. Full-dev NLL 1.222 → 0.951. **Real dev 1.347 → 1.096**; synthetic dev 0.963 → 0.653. p_choose share ≥ 0.5 0.382 → 0.436, mean \|Δ\| 0.061 | $0.52 |

**Dev curves** (NLL nats/token; run 1 = 33 synthetic dev, run 2 = full 124-record dev):

| Step | 0 | 50 | 100 | 200 | 300 | 400 | 500 | 600 | 700 | 800 |
|---|---|---|---|---|---|---|---|---|---|---|
| Run 1 | 0.963 | 0.866 | 0.796 | 0.723 | 0.693 | 0.673 | 0.660 | 0.653 | 0.652 | 0.650 |
| Run 2 | 1.222 | 1.090 | 1.041 | 0.992 | 0.970 | 0.960 | **0.951** | 0.951 | 0.960 | 0.968 |

**Overfitting:**
- On real speech the adapter starts overfitting after ~4 epochs at lr 1e-4.
- Run 1 avoided it only because its LR had decayed to near 0 by then.
- The 33 synthetic dev records alone were too easy to show it: same TTS engines and W2 script style as train, only the voices held out.

**Caveat on the offline number:**
- The references are W6's Mistral-Large translations in the w6-style-v1 house style.
- Part of the NLL drop is the decoder learning that style (punctuation, number formats, register), not better understanding of the Hindi.
- COMET and the intent/entity proxies in the streaming eval are the better read on meaning.

## 4. Evaluation (run 1 vs base)

**How it was run:**
- W8c ran its harness for base and FT with identical settings:
  - simulated real time: 80 ms frames, 320 ms agent blocks;
  - no VAD, `max_len_a = 1`, `max_len_b = 200`, fp16, on an A10G;
  - Whisper large-v3-turbo on the output speech, `wmt22-comet-da`, W6 entity parser.
- 0 failed sessions anywhere.
- Raw numbers:
  - `stages/w8c_eval/20261014T222707Z-e398964` (base) vs `…/20261014T222712Z-e398964` (FT) for dev + test;
  - `…/20261014T210742Z-46901b9` vs `…/20261014T210815Z-46901b9` for FLEURS.
  - I re-read every `scores.json` from S3; the tables below are those files.
- Full write-up and bootstrap CIs: tasks/W8c-eval.md on stream/w8c (ab4c1f2).

**Sets:**
- dev 124 records / 0.75 h: 91 real IndicVoices-valid + 33 synthetic held-out voices;
- test 199 / 1.21 h: 154 + 45;
- FLEURS-hi 418 / 1.34 h, human references.

**Speech track** (Whisper on output audio), base → FT:

| Set | ASR-BLEU | ASR-chrF++ | ASR-COMET | Numbers acc | Entity acc |
|---|---|---|---|---|---|
| dev real (91) | 22.6 → 31.3 | 46.4 → 55.9 | 0.739 → 0.751 | 0.64 → 0.62 | 0.80 → 0.82 |
| dev synthetic (33) | 13.8 → 35.6 | 34.6 → 57.8 | 0.681 → 0.772 | 0.30 → 0.45 | 0.44 → 0.58 |
| **test real (154)** | 18.1 → 23.5 | 41.6 → 50.2 | **0.716 → 0.742** | 0.44 → 0.71 | 0.75 → 0.79 |
| test synthetic (45) | 12.9 → 35.1 | 35.5 → 58.5 | 0.691 → 0.773 | 0.30 → 0.42 | 0.37 → 0.44 |
| **FLEURS (418)** | 27.8 → 24.9 | 53.7 → 53.7 | **0.821 → 0.824** | 0.72 → 0.81 | 0.81 → 0.84 |

On test real, W8c's paired bootstrap gives ASR-COMET Δ +0.026 [95% CI +0.014, +0.038], and FT is better on 107/154 records. On FLEURS: Δ +0.003 [−0.004, +0.011].

**Text track** (S2TT, the decoder's own text), base → FT:

| Set | BLEU | COMET | Numbers acc |
|---|---|---|---|
| test real | 16.9 → 25.7 | 0.729 → 0.762 | 0.64 → 0.80 |
| test synthetic | 17.5 → 45.8 | 0.730 → 0.829 | 0.46 → 0.73 |
| FLEURS | 25.3 → 22.1 | 0.830 → 0.823 (Δ −0.007 [−0.012, −0.002]) | 0.78 → 0.80 |

**Latency** (`t0` = zero-compute timeline, so independent of the GPU; seconds):

| Set | StartOffset | EndOffset | LAAL | Text LAAL |
|---|---|---|---|---|
| test real | 3.96 → 3.51 | 2.90 → 3.60 | 4.97 → 4.58 | 3.80 → 3.13 |
| test synthetic | 3.59 → 3.37 | 1.09 → 4.93 | 5.58 → 5.64 | 3.68 → 3.50 |
| FLEURS | — | 2.57 → 2.47 | 4.56 → 4.26 | 2.74 → 2.38 |

With this A10G's compute included (`t`), test LAAL is 5.60 → 5.32.

**Output length / reference length (words):**
- test real: text 0.83 → 1.00, ASR 0.80 → 0.97;
- test synthetic: text 0.66 → 1.07, ASR 0.60 → 0.94.
- The base **stops early on multi-sentence clips**. Example `rec-400033`: the base drops "Is my KYC incomplete?"; FT says it.

**Reading:**
1. **Most of the gain is coverage plus house style, not better listening.**
   - The base under-translates our 10–60 s multi-sentence sessions. FT learned to keep going to the end.
   - That alone explains most of the BLEU jump and the EndOffset rise: the base's EndOffset looked good only because it went quiet early.
   - FT also adopted W6's style ("fourteenth of August", "lakh"), which lifts BLEU against W6 references and helps the number/entity scorers.
2. **Meaning improves modestly but significantly on unseen real speakers.** Test real ASR-COMET +0.026 (CI excludes 0). Dev real is only +0.012, and dev-real number accuracy dips slightly (0.64 → 0.62), so the real-speech gain is small.
3. **No general gain, mild text-style forgetting on FLEURS.**
   - FLEURS has human references and news-style single sentences.
   - Speech-track COMET is flat (+0.003, CI spans 0). Text COMET is −0.007 (CI excludes 0). BLEU −3.
   - Number accuracy still rises (+0.09).
   - Expected from 6 h of one style: the adapter moves the output towards W6's register.
4. **Latency improved where it is policy-driven.**
   - StartOffset −0.2 to −0.45 s, LAAL −0.3 to −0.4 s on real and FLEURS, text LAAL −0.4 to −0.7 s.
   - This matches the teacher-forced p_choose shift (share ≥ 0.5: 0.401 → 0.445). Even with `p_choose` frozen, LoRA on the decoder moved its inputs towards "write".
   - Not a free lunch: it is uncontrolled. The next lever is p_choose distillation, or a decision-threshold sweep at inference.
5. **Speech (T2U drift):**
   - No measured damage: ASR-based scores track the text-track gains, so the T2U still voices the new decoder states intelligibly.
   - **Not measured:** UTMOS / naturalness. W8c's harness has no audio-quality metric yet.

## 5. What 50 h would change

These are inferences from the runs above, not measurements.

- **This recipe scales cheaply.**
  - The encoder is frozen and cached: encoding was 275 s for 6.6 h, so ~35 min for 50 h on an A10G.
  - Decoder steps cost ~0.37 s at batch 8.
  - 4 epochs of 50 h (~8,300 records) at batch 16 ≈ 2,100 steps ≈ 25 min. A full run is under $3 on ml.g5.2xlarge.
- **Overfitting moves out.** At 6 h the real-dev optimum was ~4 epochs (500 steps at lr 1e-4). With 8× the data, the same budget is half an epoch per 500 steps. That leaves room for:
  - higher LoRA rank, or a full-decoder fine-tune at lr ~1e-6 (Hibiki-Zero's < 200 h stage used lr 1e-6);
  - training longer before dev turns.
- **Real speech is where the headroom is.**
  - Synthetic dev gains are large but easy: same engines and script style as train.
  - Real dev NLL 1.35 → 1.10 and test-real ASR-COMET +0.026 are the honest numbers.
  - 50 h with a larger real share would show whether the gain grows. If it plateaus, the bottleneck is the frozen encoder on spontaneous, telephone-ish Hindi.
  - Then the next step is LoRA on the w2v-BERT encoder. That means no caching, the encoder in the loop, ~10× compute: still a few dollars at 50 h.
- **Style vs meaning.**
  - More data in one W6 house style deepens the FLEURS style drift (text COMET −0.007 already).
  - If FLEURS-style general quality matters, mix in 10–20% public parallel speech (FLEURS-train hi→en, CC BY) or lower the rank.
- **What data does not fix:**
  - **Latency control.** The EMMA policy has no released training code. Our latency changes are side effects of decoder drift. A latency target needs either p_choose distillation/regularisation (cheap, our code) or reimplementing EMMA's expected-alignment loss (2312.05187 §4; days of work).
  - **The voice.** T2U and vocoder are not trainable with released code, and the output is Seamless's own 16 kHz unit voice, not Bulbul `shubh`.

## 6. Kaggle run: run 1's recipe on 46 h

AWS credits ran out on 2026-10-14, so this ran on Kaggle's free T4s (2 per session, 12 h cap).
- **Code:**
  - `w8d/kaggle/ctl.py` runs on the laptop: data, push, status, pull, upload, report.
  - `w8d/kaggle/kernel.py` runs inside the kernel.
  - Audio reaches Kaggle as 7-day presigned S3 URLs inside a private dataset. AWS keys never enter a kernel, and the kernels are private.

**Data:**
- **Train:** every pilot-v3 record that `is_heldout()` passes, minus the 9 burned ids. That is 7,901 records, 46.39 h, of which 1,946 records (13.31 h) are synthetic. FT-6h trained on 6.6 h.
- **Dev/test:** W8c's frozen sets, `stages/w8c_sets/20261014T221953Z-e398964`.

**Train** (`20261016T212711Z-243a636`, 1.72 h on 2×T4):
- Same recipe as run 1: LoRA r16 on the monotonic decoder, lr 1e-4, batch 8.
- The batch runs as `--accum 4` micro-batches. The gradient is the same; it's needed because a batch of 8 records of 60 s runs out of memory on a 16 GB T4.
- Two arms, both with patience 5 × 100 steps:

  | Arm | Length | Dev NLL (base 1.222) | Stopped early? |
  |---|---|---|---|
  | A | 4 epochs, 3,952 steps | 0.875 | No; still falling 0.0007 over its last 500 steps |
  | B | 2 epochs, 1,976 steps | 0.897 | No |

- **Selection:** the metric was declared in advance: dev ASR-COMET, speech track, both arms measured on T4. A scored 0.753 and B 0.744, so **A is FT-47h**.

**Eval** (`20261017T064502Z-1c92b1c`, 2.46 h): A on dev and test, B on dev, and the base on dev as a T4 control.

**T4 caveat:**
- On a T4 the agent streams at about 1.05× real time (compute_s / src_s over finished sessions).
- Every session with ≥ 43 s of source ends more than 30 s behind, so it scores as failed: empty hypothesis and worst-case latency (§5.1 R3). That is 4 of 124 dev sessions and 14 of 199 test sessions. The same 4 dev ids fail for the base model on T4.
- So the raw `ctl.py report` tables mix hardware with model. For example, test ASR-COMET is 0.726 for FT-47h vs 0.749 for FT-6h, and that gap is the T4.
- On sessions that finished in every system, the base model gives the same results on T4 as on A10G: ASR-COMET −0.000 [−0.002, +0.002] real and +0.003 synthetic, with identical t0 latency. **The T4 changes speed, not output.**
- The comparison below uses those sessions: 120 dev, 185 test.
- It leaves out the longest records, but it leaves out the same ones for every system. Compute-aware latency (`lat_t`) from a T4 is not comparable to A10G numbers.

ASR-COMET on sessions that finished in every system, paired bootstrap 95% CI:

| Set | Kind (n) | Base | FT-6h | FT-47h | FT-47h − base | FT-47h − FT-6h |
|---|---|---|---|---|---|---|
| dev | real (87) | 0.740 | 0.751 | 0.762 | +0.022 [+0.008, +0.036] | +0.011 [−0.001, +0.027] |
| dev | synthetic (33) | 0.682 | 0.772 | 0.782 | +0.100 [+0.073, +0.130] | +0.010 [−0.003, +0.026] |
| test | real (140) | 0.717 | 0.744 | 0.749 | **+0.032 [+0.020, +0.043]** | +0.004 [−0.002, +0.011] |
| test | synthetic (45) | 0.691 | 0.773 | 0.783 | +0.092 [+0.072, +0.112] | +0.010 [+0.000, +0.019] |

- **Test latency t0** on the same sessions (LAAL / StartOffset / EndOffset, in seconds): base 4.91 / 3.90 / 2.31, FT-6h 4.64 / 3.51 / 3.74, FT-47h 4.77 / 3.36 / 3.90.
- **Read:**
  - The gain over the base holds: +0.032 on real test.
  - But 7× the data under the same recipe adds almost nothing over FT-6h: +0.004 on real test with a CI that spans 0, and +0.010 on synthetic.
  - Inference, not measured: decoder-only LoRA has plateaued, as §5 predicted for a frozen encoder. Next would be encoder LoRA, or a higher rank and a longer schedule; arm A was still improving when it stopped.
- **Artifacts:**
  - LoRA weights: `w8d/kaggle/20261016T212711Z-243a636/{A,B}/best_lora.pt`.
  - The merged `decoder.pt` exists only in the output of Kaggle kernel `sarvam-w8d-train`, and the next train push overwrites it.
  - Evals: `stages/w8c_eval/20261017T064502Z-1c92b1c-{base,ft47h-A,ft47h-B}`.
- **Spend:**
  - Kaggle GPU: $0.
  - AWS: S3 downloads of about 8 GB, inside the free 100 GB per month, plus PUTs of about $0.01.

## Status — 2026-10-15

**Update 2026-10-17:**
- FT-47h trained and evaluated on Kaggle (§6).
- Real test is +0.032 over the base, but only +0.004 (CI spans 0) over FT-6h.
- AWS credits are exhausted: AWS is now paid, and GPU work runs on Kaggle.

**Done:**
- Recipe researched in the code (§1).
- Pairs from train.txt only (§2), guarded against held-out ids, speakers and voices.
- Two LoRA runs on SageMaker (§3). Streaming eval of base vs FT with W8c's harness on dev, test and FLEURS (§4).

**Choosing the checkpoint:**
- Picked on dev, never on test, by the metric declared beforehand: dev ASR-COMET, speech track.
- Run 1 0.757 vs run 2 0.753. Real 0.752 vs 0.747; synthetic 0.772 vs 0.770. Inside the dev CI.
- **Run 1 is the result:** `s3://sarvam-parta-114872709824/w8d/20261014T205030Z-4b7a960/ckpt/decoder.pt` (NC-derived, private).
- Run 2's lower real-dev NLL did not turn into better streaming COMET. Offline NLL is a weak selector here.
- Run 2 eval: `stages/w8c_eval/20261015T003603Z-ab4c1f2/dev`.

**Result** (test, unseen real speakers, run 1 vs base):
- ASR-COMET 0.716 → 0.742 (+0.026, 95% CI [+0.014, +0.038]); ASR-BLEU 18.1 → 23.5; number accuracy 0.44 → 0.71.
- LAAL 4.97 → 4.58 s; StartOffset 3.96 → 3.51 s.
- EndOffset 2.90 → 3.60 s. It rises because the base stopped early: output/ref length 0.83 → 1.00.
- FLEURS (human references, out of domain): speech COMET flat (+0.003, CI spans 0), text COMET −0.007, BLEU −3, numbers +0.09.
- Collapsed outputs (< 40% of reference words) on test: base 2 real + 2 synthetic, FT 1 real (`rec-800116`: "I'll take care of it myself.") + 0.
- **Honest read:** 6 h moved the model, mainly through coverage (it no longer stops early on multi-sentence sessions), W6 house style and number rendering. The meaning gain on real speech is real but small, and there is no general-domain gain.

**Spend:**

| Item | Cost |
|---|---|
| Run 1: ml.g5.2xlarge, 886 s | $0.45 |
| Run 2: ml.g5.2xlarge, 1,026 s | $0.52 |
| Losing capacity-race jobs: 6, never started | $0 |
| **W8d's own jobs** | **$0.97** of the $40 cap |
| Evaluation jobs (run-1 FT dev+test $3.10 and FLEURS $2.09; run-2 dev ≈ $1.3) | On W8c's ledger: see tasks/W8c-eval.md, not double-counted here |

- credits.sh: NET $0.00. Cost Explorer lags ~24 h, so today's jobs aren't in it yet.

**Licences:** Seamless weights CC BY-NC 4.0. `decoder.pt` and all outputs are NC-derived and trained partly on Bulbul (Sarvam) audio: private, never publish.

**Handoffs:**
- **W8c / master:**
  - The fine-tuned decoder loads with `load_state_dict(strict=True)` over the base `seamless_streaming_monotonic_decoder`.
  - Env: DLC `pytorch-training:2.2.0-gpu-py310-cu121` + `fairseq2n==0.2.1+cu121` from Meta's pt2.2.0 index (no torch reinstall).
- **Next experiments,** in the order I'd run them:
  1. LoRA rank/LR sweep with early stopping on the full dev set (cheap, ~$0.5 per run).
  2. Add sentence-level clips.
  3. p_choose distillation, to hold latency fixed while quality moves.
  4. Encoder LoRA, once real hours grow (§5).
  5. A UTMOS metric in W8c's harness, to measure T2U drift.
- **Scale streams:** rerunning `w8d/submit.py` on a larger release is one command (`--source release/<tag>/manifest.jsonl=release/<tag>/audio/`). Re-tune the steps: the optimum was ~4 epochs at 6 h.

**Blockers:** none.

**Optional for Harsh** (≤ 10 min, ears; naturalness is the one thing not measured):
- Listen to 3 base/FT pairs:
  - `rec-400033`: coverage, the base drops the last question;
  - `rec-800201`: the biggest real gain;
  - `rec-800116`: FT collapses.
- Fetch them:
  ```
  for i in rec-400033 rec-800201 rec-800116; do for r in 20261014T222707Z-e398964 20261014T222712Z-e398964; do aws s3 cp s3://sarvam-parta-114872709824/stages/w8c_eval/$r/test/$i.wav /tmp/$i-$r.wav --profile harsh-admin; done; done
  ```
  `…222707Z` is the base, `…222712Z` is FT. The wavs are the emitted frames, not timeline-placed.
- Question: does FT sound as natural as the base? T2U and vocoder were not trained, but they now read different decoder states.
