# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""S1 bookkeeping, from the bucket alone (records only, never audio).

    uv run scale/real/report.py heldout stages/cut/<run>      # exit 1 if any record is held out (w8c) or IV-valid
    uv run scale/real/report.py ids stages/cut/<run> ...      # append the runs' ids_new.tsv rows to w1/ids.tsv
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT / "pipeline"), str(ROOT / "w8c"), str(ROOT / "w1")]
import s3 as store  # noqa: E402
from split import is_heldout  # noqa: E402

IDS = ROOT / "w1/ids.tsv"


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


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "heldout":
        sys.exit(0 if all([heldout(r) for r in args]) else 1)
    elif cmd == "ids":
        append_ids(args)
    else:
        sys.exit(__doc__)
