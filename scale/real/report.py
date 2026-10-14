# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""S1 bookkeeping, from the bucket alone (records only, never audio).

    uv run scale/real/report.py heldout stages/cut/<run>      # exit 1 if any record is held out (w8c) or IV-valid
    uv run scale/real/report.py ids stages/cut/<run> ...      # append the runs' ids_new.tsv rows to w1/ids.tsv
    uv run scale/real/report.py stats                         # every chain in data/scale-real/chains/: hours per stage,
                                                              # rejects by rule, SageMaker + Bedrock cost
"""
import collections as C, json, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "w8c"), str(ROOT / "w1")]
import s3 as store  # noqa: E402
from split import is_heldout  # noqa: E402

IDS = ROOT / "w1/ids.tsv"
CHAINS = Path("/home/harsh1/github-repos/sarvam-high-prep/data/scale-real/chains")
STAGES = ["cut", "audio_qc", "dedupe", "transcribe", "align", "asr_check", "translate"]
# USD per instance-hour, SageMaker training on-demand, ap-south-1 (W7's figures; c5.xlarge = half of c5.2xlarge).
# Spot cost = BillableTimeInSeconds x this price.
SM_USD_H = {"ml.m5.large": 0.121, "ml.c5.xlarge": 0.204, "ml.c5.2xlarge": 0.408}
BEDROCK_USD_M = {"mistral.mistral-large-3-675b-instruct": (0.59, 1.76), "openai.gpt-oss-120b-1:0": (0.18, 0.71)}  # w6


def records(run):
    return [json.loads(l) for l in store.get(f"{run}/records.jsonl").decode().splitlines() if l.strip()]


def heldout(run):
    valid = {l.strip() for l in open(Path(__file__).with_name("valid_speakers.txt")) if l.strip() and not l.startswith("#")}
    recs = records(run)
    bad = [r["id"] for r in recs if is_heldout(r) or r["speaker_id"].removeprefix("spk-indicvoices-") in valid
           or "/train/" not in r["source"]["source_file"]]
    print(f"{run}: {len(recs)} records, {len(bad)} held out / IV-valid / not train: {bad[:10]}")
    return not bad


def append_ids(runs):
    """Add the shards' new rows to the register: every id and span must be new, ids sorted."""
    have = [l.rstrip("\n").split("\t") for l in open(IDS)][1:]
    ids, spans = {r[0] for r in have}, {tuple(r[1:]) for r in have}
    new = []
    for run in runs:
        b = store.get(f"{run}/ids_new.tsv")
        new += [l.split("\t") for l in (b or b"").decode().splitlines() if l]
    assert len({r[0] for r in new}) == len(new) and not ids & {r[0] for r in new}, "an id is allocated twice"
    assert not spans & {tuple(r[1:]) for r in new}, "a span is registered twice"
    assert all(r[0].startswith("rec-") and int(r[0][4:]) < 400000 for r in new), "outside the real id range"
    with open(IDS, "a") as f:
        f.writelines("\t".join(r) + "\n" for r in sorted(new))
    print(f"w1/ids.tsv: +{len(new)} rows from {len(runs)} runs")


def chains():
    """-> {tag: {stage: run}} for the ok steps; the cut run is the chain's input (tsv name: <tag>.tsv, cut in <tag>.cut)."""
    out = {}
    for p in sorted(CHAINS.glob("*.tsv")):
        runs = {"cut": (CHAINS / f"{p.stem}.cut").read_text().strip()}
        runs.update({st: f"stages/{st}/{r}" for st, r, ok in (l.split() for l in open(p)) if ok == "ok"})
        out[p.stem] = runs
    return out


def stats():
    ch, sm = chains(), store.session().client("sagemaker")
    hours, recs, rules, usd = C.Counter(), C.Counter(), C.Counter(), C.Counter()
    complete = []
    for tag, runs in ch.items():
        for st in STAGES:
            if st not in runs:
                break
            rs = records(runs[st])
            hours[st] += sum(r["duration"] for r in rs) / 3600
            recs[st] += len(rs)
            for l in (store.get(f"{runs[st]}/rejects.jsonl") or b"").decode().splitlines():
                if l.strip():
                    rules[(st, json.loads(l).get("rule"))] += 1
            job = "sarvam-" + st.replace("_", "-") + "-" + runs[st].rsplit("/", 1)[1]
            try:
                d = sm.describe_training_job(TrainingJobName=job[:63].rstrip("-"))
                usd["sagemaker"] += (d.get("BillableTimeInSeconds") or 0) / 3600 * SM_USD_H[d["ResourceConfig"]["InstanceType"]]
            except sm.exceptions.ClientError:
                pass  # a local run
            if st == "translate":
                complete.append(runs[st])
                keys = [k + ".json" for r in rs for k in r.get("_archive", {}).get("translate", [])]
                for m in ThreadPoolExecutor(32).map(lambda k: json.loads(store.get(k)), keys):
                    pi, po = BEDROCK_USD_M[m["request"]["model"]]
                    u = m["response"]["usage"]
                    usd["bedrock"] += (u["inputTokens"] * pi + u["outputTokens"] * po) / 1e6
    print("| Stage | Records | Hours |\n|---|---|---|")
    for st in STAGES:
        print(f"| {st} | {recs[st]} | {hours[st]:.3f} |")
    print("\nRejects by rule: " + ", ".join(f"{st}/{r} {n}" for (st, r), n in sorted(rules.items())))
    total = sum(usd.values())
    print(f"\nCost (list price): SageMaker ${usd['sagemaker']:.2f}, Bedrock ${usd['bedrock']:.2f}, total ${total:.2f}; "
          f"per kept hour ${total / max(hours['translate'], 1e-9):.2f}")
    print("\nComplete runs (assembly input):\n" + "\n".join(complete))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "heldout":
        sys.exit(0 if all([heldout(r) for r in args]) else 1)
    elif cmd == "ids":
        append_ids(args)
    elif cmd == "stats":
        stats()
    else:
        sys.exit(__doc__)
