# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""W8d: build training pairs from a release + W8c's split, upload them, run w8d/train.py as a SageMaker job.

    uv run w8d/submit.py --instances ml.g5.2xlarge ml.g6e.xlarge [--spot] [--run R] [-- --steps 800 ...]

Pairs (tasks/W8d-finetune.md §2): train = W8c's train.txt ids found in the sources, refused if any is held out
(heldout.json); dev = dev.txt ids whose every sentence already has en_text. One job per instance type, because GPU
capacity in ap-south-1 is scarce: the first to start wins, the rest are stopped, so only one writes w8d/<run>/ckpt/.
"""
import argparse, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
import run as pipeline_run, s3 as store, sm  # noqa: E402

SPLITS = Path("/home/harsh1/github-repos/sarvam-high-prep/data/w8c/splits")
OUT = Path("/home/harsh1/github-repos/sarvam-high-prep/data/w8d")
IMAGE = "763104351884.dkr.ecr.ap-south-1.amazonaws.com/pytorch-training:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker"  # fairseq2 0.2 needs torch 2.2 + py ≤ 3.11
SOURCES = ["release/pilot-v2/manifest.jsonl=release/pilot-v2/audio/"]


def ids(name, splits=SPLITS):
    return [l.strip() for l in (splits / f"{name}.txt").read_text().splitlines() if l.strip()]


def heldout(r, held, dev_test):
    """W8c's rule (w8c/split.py is_heldout) on what a release record still carries; scripts are only in stage files,
    so that part rests on W8c's train.txt."""
    s, spk = r["source"], set(held["speakers"])
    return (r["id"] in dev_test or (r.get("speaker_id") or "").removeprefix("spk-indicvoices-") in spk
            or (s.get("voice") or "").removeprefix("iv-") in spk or [s.get("model"), s.get("voice")] in held["voices"])


def pairs(records, splits=SPLITS):
    """records: {id: (record, audio_prefix)} -> (train pairs, dev pairs) as {"id", "audio_key", "en"}."""
    held, dev_test = json.loads((splits / "heldout.json").read_text()), set(ids("dev", splits) + ids("test", splits))
    pair = lambda i: {"id": i, "audio_key": records[i][1] + f"{i}.wav", "en": " ".join(s["en_text"] for s in records[i][0]["sentences"])}
    train = [i for i in ids("train", splits) if i in records]
    if bad := [i for i in train if heldout(records[i][0], held, dev_test)]:
        raise SystemExit(f"{len(bad)} train ids are held out (dev/test id, speaker or voice): {bad[:5]}")
    dev = [i for i in ids("dev", splits) if i in records and all(s.get("en_text") for s in records[i][0].get("sentences", [{}]))]
    return [pair(i) for i in train], [pair(i) for i in dev]


def watch(client, names):
    """First job to leave the capacity queue wins; stop the rest. -> (winner, final description)."""
    win, last = None, {}
    while True:
        for n in names:
            d = client.describe_training_job(TrainingJobName=n)
            st = (d["TrainingJobStatus"], d["SecondaryStatus"])
            if st != last.get(n):
                print(f"{datetime.now():%H:%M:%S} {n}: {st[0]}/{st[1]} {d.get('FailureReason', '')}", file=sys.stderr, flush=True)
                last[n] = st
            if win is None and st[1] in ("Downloading", "Training", "Uploading", "Completed"):
                win = n
                for o in names:
                    if o != n and last.get(o, ("InProgress",))[0] == "InProgress":
                        client.stop_training_job(TrainingJobName=o)
            if n == win and st[0] in ("Completed", "Failed", "Stopped"):
                return win, d
        if win is None and all(s[0] in ("Failed", "Stopped") for s in last.values()) and len(last) == len(names):
            return None, None
        time.sleep(30)


def main():
    argv = sys.argv[1:]
    train_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instances", nargs="+", required=True)
    ap.add_argument("--spot", action="store_true")
    ap.add_argument("--run")
    ap.add_argument("--source", nargs="*", default=SOURCES, help="MANIFEST_KEY=AUDIO_PREFIX (later sources override)")
    a = ap.parse_args(argv)
    if sm._dirty():
        raise SystemExit("refusing: commit first, a job must map to one git sha")
    sha = pipeline_run.git_sha()
    run = a.run or pipeline_run.run_id(sha, datetime.now(timezone.utc))
    records = {}
    for src in a.source:
        key, audio = src.split("=")
        for l in store.get(key).decode().splitlines():
            if l.strip():
                r = json.loads(l)
                records[r["id"]] = (r, audio)
    train, dev = pairs(records)
    print(f"{run}: {len(train)} train pairs ({sum(records[p['id']][0]['duration'] for p in train) / 3600:.2f} h), {len(dev)} dev pairs", file=sys.stderr)
    for name, ps in (("train", train), ("dev", dev)):
        store.put(f"w8d/{run}/{name}.jsonl", "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in ps).encode())
    store.put(f"w8d/{run}/job.json", json.dumps({"sha": sha, "sources": a.source, "train_args": train_args, "instances": a.instances,
                                                  "spot": a.spot, "splits": {f: (SPLITS / f).read_text() for f in ("train.txt", "dev.txt", "test.txt")}}).encode())
    tar = subprocess.run(["git", "-C", str(ROOT), "archive", "--format=tar.gz", "HEAD"], capture_output=True, check=True).stdout
    store.put_once(f"code/{sha}.tar.gz", tar)
    hp = dict(zip([x.lstrip("-") for x in train_args[::2]], train_args[1::2]))
    client, names = store.session().client("sagemaker"), []
    for inst in a.instances:
        req = sm.job_request("w8d", "", "", [], run=run, sha=sha, instance=inst, spot=a.spot)
        req["TrainingJobName"] = f"sarvam-w8d-{run}-{inst.removeprefix('ml.').replace('.', '-')}"[:63].rstrip("-")
        req["AlgorithmSpecification"]["TrainingImage"] = IMAGE
        req["HyperParameters"] = {"sagemaker_program": "w8d/train.py", "sagemaker_submit_directory": f"s3://{store.BUCKET}/code/{sha}.tar.gz",
                                  "run": run, **hp}
        req["CheckpointConfig"]["S3Uri"] = f"s3://{store.BUCKET}/w8d/{run}/ckpt/"
        req["OutputDataConfig"]["S3OutputPath"] = f"s3://{store.BUCKET}/w8d/_sm_output/"
        client.create_training_job(**req)
        names.append(req["TrainingJobName"])
        print(f"submitted {names[-1]}", file=sys.stderr)
    win, d = watch(client, names)
    if win is None:
        raise SystemExit("every job failed or was stopped before starting")
    local = OUT / run
    for f in ("log.jsonl", "done.json"):  # not decoder.pt: 2 GB, it stays in S3 for the eval job
        if (b := store.get(f"w8d/{run}/ckpt/{f}")) is not None:
            local.mkdir(parents=True, exist_ok=True)
            (local / f).write_bytes(b)
    print(json.dumps({"job": win, "status": d["TrainingJobStatus"], "billable_s": d.get("BillableTimeInSeconds"),
                      "instance": d["ResourceConfig"]["InstanceType"], "local": str(local)}), flush=True)
    sys.exit(0 if d["TrainingJobStatus"] == "Completed" else 1)


if __name__ == "__main__":
    main()
