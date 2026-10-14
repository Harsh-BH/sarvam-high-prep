# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["torch==2.14.1", "transformers==4.47.1", "sentencepiece", "einops", "IndicTransToolkit==1.1.1", "boto3"]
# ///
"""IndicTrans2 baseline for the W6 engine benchmark: translate every record's segments with the RoPE variant of
IndicTrans2 indic-en dist-200M (MIT; linked from the official IndicTrans2 README, ungated) and archive each record's
output through archive_call (service `it2`), so `w6/translate.py --engine it2` can QA and judge it without torch.

    flock /home/harsh1/github-repos/sarvam-high-prep/data/.gpu.lock uv run w6/it2.py --in RECORDS.jsonl [--hf-home DIR]
"""
import argparse, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "w6")]
import archive, s3 as store  # noqa: E402
from translate import IT2, CAP_INR, it2_request, prefill_spent  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True)
    ap.add_argument("--hf-home", default="/home/harsh1/github-repos/sarvam-high-prep/data/w6/hf")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    os.environ["HF_HOME"] = a.hf_home
    recs = [json.loads(l) for l in open(a.inp, encoding="utf-8")]
    todo = [[s["hi_text"] for s in r["sentences"]] for r in recs]
    todo = [h for h in todo if not store.exists(f"archive/api/it2/{archive.request_key(it2_request(h))}.json")]
    print(f"{len(recs)} records, {len(todo)} not archived", file=sys.stderr)
    if not todo:
        return
    prefill_spent()  # scans the whole archive once; do it before the GPU work, not after
    import torch
    from IndicTransToolkit.processor import IndicProcessor
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(IT2["model"], revision=IT2["rev"], trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(IT2["model"], revision=IT2["rev"], trust_remote_code=True,
                                                  torch_dtype=torch.float16).to("cuda").eval()
    ip = IndicProcessor(inference=True)
    flat = [s for h in todo for s in h]
    out = []
    for i in range(0, len(flat), a.batch):
        b = ip.preprocess_batch(flat[i:i + a.batch], src_lang="hin_Deva", tgt_lang="eng_Latn")
        enc = tok(b, padding="longest", truncation=True, max_length=512, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**enc, num_beams=it2_request([])["num_beams"], max_new_tokens=256, early_stopping=True)
        out += ip.postprocess_batch(tok.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True),
                                    lang="eng_Latn")
        print(f"{len(out)}/{len(flat)}", file=sys.stderr)
    k = 0
    for h in todo:
        res = {"segments": out[k:k + len(h)]}
        k += len(h)
        archive.archive_call("it2", it2_request(h), lambda req, res=res: (res, None), stream="w6", inr=0, cap_inr=CAP_INR)


if __name__ == "__main__":
    main()
