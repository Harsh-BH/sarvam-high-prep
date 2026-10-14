"""W8d: LoRA fine-tune of SeamlessStreaming's monotonic text decoder (tasks/W8d-finetune.md §1).

SageMaker entry (w8d/submit.py runs it as `python w8d/train.py --run RUN [--steps N ...]`).
Reads s3://<bucket>/w8d/<run>/{train,dev}.jsonl ({"id", "audio_key", "en"}) and the wavs they name.
Speech encoder frozen (outputs computed once, exactly as the streaming agent does at end of input); p_choose,
embeddings, norms, T2U and vocoder frozen; teacher-forced CE on whole records.
Writes /opt/ml/checkpoints (SageMaker syncs it to s3://<bucket>/w8d/<run>/ckpt/): log.jsonl, last.pt (spot resume),
best_lora.pt, decoder.pt = merged best decoder as {"model": fp16 state_dict} (w8c loads it with strict=True), done.json.
"""
import argparse, io, json, random, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SC_REV = "9a081e935c29c6b1e0e05bcbf3fcb623ad306524"  # seamless_communication, read for tasks/W8d-finetune.md §1
PIP = [["--extra-index-url", "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.2.0/cu121",  # fairseq2n built for the image's torch 2.2.0
        "fairseq2n==0.2.1+cu121", "fairseq2==0.2.1", "simuleval==1.1.4", "soundfile", "numpy<2"],
       ["--no-deps", f"git+https://github.com/facebookresearch/seamless_communication@{SC_REV}"]]
UNITY, DECODER = "seamless_streaming_unity", "seamless_streaming_monotonic_decoder"
# every attention + FFN projection of the 24 decoder layers; p_choose_layer's energy MLPs never match
LORA_KEYS = [r".*text_decoder\.layers\.\d+\.(self_attn|encoder_decoder_attn)\.(q_proj|k_proj|v_proj|output_proj)$",
             r".*text_decoder\.layers\.\d+\.ffn\.(inner_proj|output_proj)$"]


def args_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32.0)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2026)
    return ap


def batch(items, tok_enc, eos, pad, device):
    """items: [(enc (S, M), English text)] -> padded decoder batch. Targets are [eos, __eng__, tokens.., eos] split
    into prev/target exactly as m4t_finetune's dataloader does; the loss skips the language token (prefix 1)."""
    import torch
    from fairseq2.nn.padding import PaddingMask
    toks = [torch.cat([tok_enc(en), torch.tensor([eos])]) for _, en in items]
    pad_to = lambda xs, v: torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=v).to(device)
    enc_lens = torch.tensor([e.size(0) for e, _ in items])
    tok_lens = torch.tensor([t.size(0) - 1 for t in toks])
    return {"enc": pad_to([e for e, _ in items], 0.0), "enc_mask": PaddingMask(enc_lens.to(device), int(enc_lens.max())),
            "prev": pad_to([t[:-1] for t in toks], pad), "prev_mask": PaddingMask(tok_lens.to(device), int(tok_lens.max())),
            "target": pad_to([t[1:] for t in toks], pad), "ntok": int((tok_lens - 1).sum())}


def loss_sum(model, b, vocab_info, label_smoothing):
    """-> (summed NLL over target tokens, token count, p_choose), language token excluded."""
    from fairseq2.models.sequence import SequenceModelOutput
    out, _, p_choose = model.decode(b["prev"], b["prev_mask"], b["enc"].to(next(model.parameters()).dtype), b["enc_mask"])
    loss = SequenceModelOutput(model.project(out), vocab_info).compute_loss(b["target"], ignore_prefix_size=1, label_smoothing=label_smoothing)
    return loss, b["ntok"], p_choose


def evaluate(model, items, ctx):
    """-> (dev NLL per token without label smoothing, [per-record p_choose (L*H, T, S/2) fp16 on CPU])."""
    import torch
    model.eval()
    tot, n, pch = 0.0, 0, []
    with torch.no_grad(), ctx["amp"]():
        for it in items:  # one record per forward: p_choose has no padding to strip
            b = batch([it], ctx["tok_enc"], ctx["eos"], ctx["pad"], ctx["device"])
            l, k, p = loss_sum(model, b, ctx["vocab_info"], 0.0)
            tot, n = tot + float(l), n + k
            pch.append(p.half().cpu())
    model.train()
    return tot / n, pch


def lora_weights(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if ".lora_" in k}


def train(model, train_items, dev_items, a, ckpt, ctx):
    """LoRA-train `model` in place; resumes from ckpt/last.pt. -> summary dict (also written to ckpt/done.json)."""
    import torch
    from fairseq2.nn.lora import LoRAConfig, freeze_non_lora, unwrap_lora, wrap_lora
    ckpt.mkdir(parents=True, exist_ok=True)
    log = (ckpt / "log.jsonl").open("a")
    emit = lambda **kw: (log.write(json.dumps(kw) + "\n"), log.flush(), print(json.dumps(kw), flush=True))
    base_loss, base_p = evaluate(model, dev_items, ctx)  # before any LoRA: the "before" number for the offline track
    wrap_lora(model, LoRAConfig(r=a.rank, alpha=a.alpha, dropout_p=a.lora_dropout, keys=LORA_KEYS))
    freeze_non_lora(model)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min((s + 1) / a.warmup, max(0.0, (a.steps - s) / max(1, a.steps - a.warmup))))
    scaler = torch.cuda.amp.GradScaler(enabled=ctx["scale"])
    step, best, bad = 0, base_loss, 0
    if (ckpt / "last.pt").exists():  # spot restart
        st = torch.load(ckpt / "last.pt", map_location="cpu")
        assert not model.load_state_dict(st["lora"], strict=False).unexpected_keys
        opt.load_state_dict(st["opt"]), sched.load_state_dict(st["sched"])
        step, best, bad = st["step"], st["best"], st["bad"]
        random.setstate(st["rng"])
        emit(event="resumed", step=step, best=best)
    else:
        emit(event="start", step=0, dev_loss=base_loss, trainable=sum(p.numel() for p in params), train=len(train_items), dev=len(dev_items))
        torch.save(lora_weights(model), ckpt / "best_lora.pt")  # B = 0 at init: "best so far" is the base model
    order, t0 = [], time.time()
    while step < a.steps and bad < a.patience:
        if len(order) < a.batch:
            order += random.sample(range(len(train_items)), len(train_items))  # next epoch, reshuffled
        idx, order = order[:a.batch], order[a.batch:]
        b = batch([train_items[i] for i in idx], ctx["tok_enc"], ctx["eos"], ctx["pad"], ctx["device"])
        with ctx["amp"]():
            l, k, _ = loss_sum(model, b, ctx["vocab_info"], a.label_smoothing)
        scaler.scale(l / k).backward()
        scaler.unscale_(opt)
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        scaler.step(opt), scaler.update(), opt.zero_grad(set_to_none=True), sched.step()
        step += 1
        if step % 10 == 0:
            emit(step=step, train_loss=float(l) / k, lr=sched.get_last_lr()[0], grad_norm=float(gn), sec=round(time.time() - t0))
        if step % a.eval_every == 0 or step == a.steps:
            dl, _ = evaluate(model, dev_items, ctx)
            if dl < best:
                best, bad = dl, 0
                torch.save(lora_weights(model), ckpt / "best_lora.pt")
            else:
                bad += 1
            emit(step=step, dev_loss=dl, best=best, bad=bad)
            torch.save({"lora": lora_weights(model), "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step,
                        "best": best, "bad": bad, "rng": random.getstate()}, ckpt / "last.pt")
    assert not model.load_state_dict(torch.load(ckpt / "best_lora.pt"), strict=False).unexpected_keys
    unwrap_lora(model, merge=True)
    tuned_loss, tuned_p = evaluate(model, dev_items, ctx)
    torch.save({"model": {k: v.half() for k, v in model.state_dict().items()}}, ckpt / "decoder.pt")
    cat = lambda ps: torch.cat([p.float().flatten() for p in ps])
    summary = {"steps": step, "stopped_early": bad >= a.patience, "dev_loss_base": base_loss, "dev_loss_best": best,
               "dev_loss_merged": tuned_loss, "p_choose_mean_base": float(cat(base_p).mean()),
               "p_choose_mean_tuned": float(cat(tuned_p).mean()), "p_choose_mean_abs_delta": float((cat(tuned_p) - cat(base_p)).abs().mean()),
               "p_choose_ge_0.5_base": float((cat(base_p) >= 0.5).float().mean()), "p_choose_ge_0.5_tuned": float((cat(tuned_p) >= 0.5).float().mean())}
    emit(event="done", **summary)
    (ckpt / "done.json").write_text(json.dumps(summary, indent=1))
    return summary


def encode(pairs, get_audio, device):
    """-> {id: speech encoder output (S, 1024) fp16 on CPU}, the agent's computation once all input has arrived:
    torchaudio 24->16 kHz, fbank (scale 1.0, no standardisation), Collater(pad 0, multiple of 2), encode_speech."""
    import torch, torchaudio
    from fairseq2.assets import asset_store
    from fairseq2.data.audio import WaveformToFbankConverter
    from fairseq2.data.data_pipeline import Collater
    from fairseq2.nn.padding import get_seqs_and_padding_mask
    from seamless_communication.models.unity import load_unity_config, load_unity_model
    cfg = load_unity_config(UNITY)
    cfg.use_text_decoder, cfg.use_text_encoder, cfg.t2u_config = False, False, None
    card = asset_store.retrieve_card(UNITY)
    card.field("model_config").set(cfg)
    unity = load_unity_model(card, device=device, dtype=torch.float16).eval()
    fbank = WaveformToFbankConverter(num_mel_bins=80, waveform_scale=1.0, standardize=False, device=device, dtype=torch.float16)
    collate = Collater(pad_value=0, pad_to_multiple=2)
    out = {}
    with torch.inference_mode():
        for p in pairs:
            wav, sr = torchaudio.load(io.BytesIO(get_audio(p["audio_key"])))
            wav = torchaudio.functional.resample(wav, sr, 16000)
            seqs, mask = get_seqs_and_padding_mask(collate(fbank({"waveform": wav, "sample_rate": 16000})["fbank"]))
            out[p["id"]] = unity.encode_speech(seqs, mask)[0][0].cpu()
    del unity
    torch.cuda.empty_cache()
    return out


def main():
    a = args_parser().parse_args()
    for args in PIP:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)
    import torch
    sys.path.insert(0, str(ROOT / "pipeline"))
    import s3 as store
    from seamless_communication.models.monotonic_decoder import load_monotonic_decoder_model
    from seamless_communication.models.unity import load_unity_text_tokenizer
    random.seed(a.seed), torch.manual_seed(a.seed)
    device = torch.device("cuda")
    pairs = {s: [json.loads(l) for l in store.get(f"w8d/{a.run}/{s}.jsonl").decode().splitlines() if l.strip()] for s in ("train", "dev")}
    t0 = time.time()
    enc = encode(pairs["train"] + pairs["dev"], store.get, device)
    print(f"encoded {len(enc)} records in {time.time() - t0:.0f}s", flush=True)
    tok = load_unity_text_tokenizer(UNITY)
    model = load_monotonic_decoder_model(DECODER, device=device, dtype=torch.float32)
    model.train()
    bf16 = torch.cuda.is_bf16_supported()  # A10G/L4/L40S yes; T4 no -> fp16 + GradScaler
    ctx = {"device": device, "tok_enc": tok.create_encoder(lang="eng", mode="target"), "eos": tok.vocab_info.eos_idx,
           "pad": tok.vocab_info.pad_idx, "vocab_info": tok.vocab_info, "scale": not bf16,
           "amp": lambda: torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16)}
    items = {s: [(enc[p["id"]], p["en"]) for p in pairs[s]] for s in pairs}
    train(model, items["train"], items["dev"], a, Path("/opt/ml/checkpoints"), ctx)


if __name__ == "__main__":
    main()
