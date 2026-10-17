"""CPU check of w8d/train.py on a tiny random monotonic decoder: loss falls, a spot restart resumes, and decoder.pt
loads strict into a fresh model and reproduces the trained one. Needs the fairseq2 0.2 env (tasks/W8d-finetune.md §3):
    <venv>/bin/python w8d/test_train.py
"""
import contextlib, json, random, sys, tempfile
from pathlib import Path
import torch
from fairseq2.data import VocabularyInfo
from seamless_communication.models.monotonic_decoder.builder import MonotonicDecoderConfig, create_monotonic_decoder_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train as T

V = VocabularyInfo(size=64, unk_idx=1, bos_idx=2, eos_idx=3, pad_idx=0)
CFG = MonotonicDecoderConfig(model_dim=32, max_seq_len=128, vocab_info=V, num_decoder_layers=2, num_decoder_attn_heads=4,
                             ffn_inner_dim=64, dropout_p=0.0, energy_bias_value=-0.5, monotonic_temperature=0.2,
                             num_monotonic_energy_layers=2, pre_decision_ratio=2)
tok_enc = lambda text: torch.tensor([3, 4] + [10 + ord(c) % 50 for c in text])  # [eos, lang, chars], like the NLLB target encoder


def model():
    torch.manual_seed(0)
    return create_monotonic_decoder_model(CFG).train()


def main():
    random.seed(0), torch.manual_seed(0)
    items = [(torch.randn(random.randint(5, 12), 32).half(), t) for t in ("hello there", "a cat sat", "go now", "yes sir ok")]
    ctx = {"device": torch.device("cpu"), "tok_enc": tok_enc, "eos": 3, "pad": 0, "vocab_info": V, "scale": False,
           "amp": contextlib.nullcontext}
    a = T.args_parser().parse_args(["--run", "t", "--steps", "40", "--batch", "2", "--lr", "3e-2", "--warmup", "5",
                                    "--rank", "4", "--alpha", "8", "--eval_every", "20", "--patience", "9"])
    with tempfile.TemporaryDirectory() as d:
        ck = Path(d)
        a.steps = 20  # first "spot instance" dies after 20 steps: simulate by finishing a 20-step run, then resume to 40
        m1 = model()
        T.train(m1, items, items, a, ck, ctx)
        a.steps = 40
        (ck / "done.json").unlink(), (ck / "decoder.pt").unlink()
        m2 = model()
        s = T.train(m2, items, items, a, ck, ctx)
        log = [json.loads(l) for l in (ck / "log.jsonl").read_text().splitlines()]
        assert [e for e in log if e.get("event") == "resumed"][0]["step"] == 20, log
        assert s["dev_loss_best"] < 0.5 * s["dev_loss_base"], s  # dev == train here: it must overfit
        assert abs(s["dev_loss_merged"] - s["dev_loss_best"]) < 1e-3, s  # merge + unwrap kept the best adapter
        m3 = model()
        m3.load_state_dict(torch.load(ck / "decoder.pt")["model"], strict=True)
        m3.float()
        loss3, _, per = T.evaluate(m3, items, ctx)
        assert len(per) == len(s["dev_per_record"]) == len(items)
        assert abs(loss3 - s["dev_loss_merged"]) < 2e-2, (loss3, s)  # fp16-saved weights reproduce the trained model
        assert not any(".lora_" in k for k in torch.load(ck / "decoder.pt")["model"])
    lora = []  # --accum: one step of batch 4 as 2 micro-batches == one batch of 4 (dropout off)
    for acc in ("1", "2"):
        b = T.args_parser().parse_args(["--run", "t", "--steps", "1", "--batch", "4", "--accum", acc, "--lr", "3e-2", "--warmup", "1",
                                        "--rank", "4", "--alpha", "8", "--lora_dropout", "0", "--eval_every", "1"])
        with tempfile.TemporaryDirectory() as d:
            random.seed(0)
            T.train(model(), items, items, b, Path(d), ctx)
            lora.append(torch.load(Path(d) / "last.pt")["lora"])
    assert all(torch.allclose(lora[0][k], lora[1][k], atol=1e-6) for k in lora[0]) and any(v.abs().sum() > 0 for k, v in lora[0].items() if "lora_B" in k)
    print("ok", json.dumps(s))


if __name__ == "__main__":
    main()
