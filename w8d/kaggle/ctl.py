# /// script
# requires-python = ">=3.12"
# dependencies = ["boto3"]
# ///
"""W8d on Kaggle, the laptop side (tasks/W8d-finetune.md "Kaggle run"): W8d run 1's recipe on all of pilot-v3's
train-eligible records, trained and evaluated on Kaggle's free T4s. The in-kernel side is w8d/kaggle/kernel.py.

    uv run w8d/kaggle/ctl.py data            # manifests + 7-day presigned GET URLs -> private dataset sarvam-w8d-data
    uv run w8d/kaggle/ctl.py push train|eval [--limit N] [--resume]   # kernel from git HEAD (clean worktree)
    uv run w8d/kaggle/ctl.py status train|eval
    uv run w8d/kaggle/ctl.py pull train|eval # outputs -> data/w8d-kaggle/out/<kind>/ (train: small files only)
    uv run w8d/kaggle/ctl.py upload train|eval   # write-once: w8d/kaggle/<run>/, stages/w8c_eval/<run>-<system>/
    uv run w8d/kaggle/ctl.py report RUN          # 3-way table base / FT-6h / FT-47h from S3 scores (RUN: eval job.json)

Train = every record of data/master/v3_runs.txt that is_heldout() passes, minus the 9 burned ids (tasks/W7-assembly.md).
Dev/test = W8c's frozen sets. Nothing secret goes into a kernel: AWS keys stay here, the URLs sit in the private
dataset only, and the kernel never prints them. Kernels and dataset are private (NC weights, Sarvam audio).
"""
import argparse, base64, io, json, os, random, subprocess, sys, tarfile, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "pipeline"))
sys.path.insert(0, str(ROOT / "w8c"))
import run as pipeline_run, s3 as store, sm  # noqa: E402
from split import is_heldout  # noqa: E402

DATA = Path("/home/harsh1/github-repos/sarvam-high-prep/data")
HOME = DATA / "w8d-kaggle"
DS = HOME / "dataset"
RUNS = DATA / "master/v3_runs.txt"
SETS = "stages/w8c_sets/20261014T221953Z-e398964"  # W8c's frozen dev/test, the sets every W8d eval used
BURNED = {f"rec-4000{x}" for x in ("51", "52", "54", "55", "56", "58", "59", "60", "62")}  # tasks/W7-assembly.md
BASE, FT6 = "stages/w8c_eval/20261014T222707Z-e398964", "stages/w8c_eval/20261014T222712Z-e398964"  # A10G, W8c
EXPIRES = 7 * 24 * 3600 - 600  # SigV4 presigned URLs live at most 7 days
SLUG = {"data": "sarvam-w8d-data", "train": "sarvam-w8d-train", "eval": "sarvam-w8d-eval"}
CODE = ["w8d/train.py", "w8d/kaggle/kernel.py", "w8c/run_seamless.py", "w8c/eval.py", "w6/qa.py"]
BATCH = 8  # train.py's default, W8d run 1


def kaggle(*args, **kw):
    return subprocess.run(["kaggle", *map(str, args)], check=True, text=True, **kw)


def user():
    if u := os.environ.get("KAGGLE_USERNAME"):
        return u
    p = Path.home() / ".kaggle/kaggle.json"
    if not p.exists():
        raise SystemExit("no Kaggle credentials: kaggle.com -> Settings -> API -> Create New Token -> ~/.kaggle/kaggle.json")
    return json.loads(p.read_text())["username"]


def select(runs):
    """runs: [(audio_from, [records])] -> train rows {"id", "en", "duration", "synthetic", "key"}.
    Out: dev/test ids and anything else is_heldout() flags (speaker, cloned voice, voice, script), the burned ids."""
    rows = []
    for audio_from, recs in runs:
        for r in recs:
            if is_heldout(r) or r["id"] in BURNED:
                continue
            if not all(s.get("en_text") for s in r["sentences"]):
                raise SystemExit(f"{r['id']}: a sentence without en_text")
            rows.append({"id": r["id"], "en": " ".join(s["en_text"] for s in r["sentences"]), "duration": r["duration"],
                         "synthetic": r["source"]["synthetic"], "key": audio_from.rstrip("/") + "/" + r["audio"].removeprefix("audio/")})
    if len({r["id"] for r in rows}) != len(rows):
        raise SystemExit("an id occurs in two runs")
    return rows


def data(_):
    from botocore.config import Config
    runs = [l.strip() for l in RUNS.read_text().splitlines() if l.strip()]
    loaded = [(json.loads(store.get(f"{r}/run.json"))["audio_from"],
               [json.loads(l) for l in store.get(f"{r}/records.jsonl").decode().splitlines() if l.strip()]) for r in runs]
    rows = select(loaded)
    sets = {s: store.get(f"{SETS}/{s}.jsonl").decode() for s in ("dev", "test")}
    keys = {r["id"]: r["key"] for r in rows} | {json.loads(l)["id"]: f"{SETS}/{json.loads(l)['audio']}"
                                                for t in sets.values() for l in t.splitlines() if l.strip()}
    c = store.session().client("s3", endpoint_url=f"https://s3.{store.REGION}.amazonaws.com",
                               config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}))
    now = time.time()
    urls = {i: c.generate_presigned_url("get_object", Params={"Bucket": store.BUCKET, "Key": k}, ExpiresIn=EXPIRES) for i, k in keys.items()}
    DS.mkdir(parents=True, exist_ok=True)
    (DS / "urls.json").write_text(json.dumps(urls))
    (DS / "train.jsonl").write_text("".join(json.dumps({k: r[k] for k in ("id", "en", "duration", "synthetic")}, ensure_ascii=False) + "\n" for r in rows))
    for s, t in sets.items():
        (DS / f"{s}.jsonl").write_text(t)
    syn = [r for r in rows if r["synthetic"]]
    meta = {"runs": runs, "sets": SETS, "burned": sorted(BURNED), "train": len(rows), "train_h": round(sum(r["duration"] for r in rows) / 3600, 3),
            "synthetic": len(syn), "synthetic_h": round(sum(r["duration"] for r in syn) / 3600, 3), "created": int(now), "expires": int(now + EXPIRES)}
    (DS / "manifest.json").write_text(json.dumps(meta, indent=1))
    (DS / "dataset-metadata.json").write_text(json.dumps({"title": SLUG["data"], "id": f"{user()}/{SLUG['data']}", "licenses": [{"name": "other"}]}))
    print(json.dumps({k: v for k, v in meta.items() if k != "runs"}))
    exists = subprocess.run(["kaggle", "datasets", "status", f"{user()}/{SLUG['data']}"], capture_output=True, text=True).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", DS, "-m", f"presigned URLs to {datetime.fromtimestamp(now + EXPIRES):%Y-%m-%d %H:%M}")
    else:
        kaggle("datasets", "create", "-p", DS)  # private unless --public
    for _ in range(60):
        st = subprocess.run(["kaggle", "datasets", "status", f"{user()}/{SLUG['data']}"], capture_output=True, text=True).stdout.strip()
        if st == "ready":
            break
        time.sleep(10)
    print(f"dataset {user()}/{SLUG['data']}: {st}")


def arms(n_train, limit):
    """Two LoRA arms of W8d run 1's recipe (train.py defaults), one per T4. A: a 4-epoch horizon that early-stops
    where dev NLL turns (W8d run 2's role); B: run 1's shape, linear decay to 0 at 2 epochs (the expected optimum)."""
    if limit:
        return {"A": ["--steps", "20", "--eval_every", "10"], "B": ["--steps", "10", "--eval_every", "10"]}
    ep = -(-n_train // BATCH)
    return {"A": ["--steps", str(4 * ep), "--eval_every", "100", "--patience", "5"],
            "B": ["--steps", str(2 * ep), "--eval_every", "100", "--patience", "5"]}


def boot(cfg, tar):
    """The pushed kernel: unpack the code (a git archive of HEAD) and run kernel.py main."""
    return (f"# W8d on Kaggle: run {cfg['run']}, git {cfg['sha']}. Generated by w8d/kaggle/ctl.py; the code is kernel.py.\n"
            "import base64, io, json, subprocess, sys, tarfile\n"
            f"CODE = {base64.b64encode(tar).decode()!r}\n"
            f"CFG = {cfg!r}\n"
            'tarfile.open(fileobj=io.BytesIO(base64.b64decode(CODE))).extractall("/tmp/w8d/code")\n'
            'sys.exit(subprocess.run([sys.executable, "/tmp/w8d/code/w8d/kaggle/kernel.py", "main", json.dumps(CFG)]).returncode)\n')


def push(a):
    if sm._dirty():
        raise SystemExit("refusing: commit first, a kernel version must map to one git sha")
    meta = json.loads((DS / "manifest.json").read_text())
    if meta["expires"] - time.time() < 13 * 3600:
        raise SystemExit("presigned URLs expire within a session: run `ctl.py data` first")
    sha = pipeline_run.git_sha()
    cfg = {"mode": a.kind, "run": pipeline_run.run_id(sha, datetime.now(timezone.utc)) + ("-smoke" if a.limit else ""),
           "sha": sha, "limit": a.limit, "arms": arms(meta["train"], a.limit), "base_dev": True}
    tar = subprocess.run(["git", "-C", ROOT, "archive", "--format=tar.gz", "HEAD", *CODE], capture_output=True, check=True).stdout
    d = HOME / "push" / a.kind
    d.mkdir(parents=True, exist_ok=True)
    (d / "boot.py").write_text(boot(cfg, tar))
    u = user()
    sources = [f"{u}/{SLUG['train']}"] * (a.kind == "eval") + [f"{u}/{SLUG['eval']}"] * a.resume  # resume: last eval output
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{u}/{SLUG[a.kind]}", "title": SLUG[a.kind], "code_file": "boot.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [f"{u}/{SLUG['data']}"], "kernel_sources": sources}, indent=1))
    kaggle("kernels", "push", "-p", d)
    with open(HOME / "pushes.jsonl", "a") as f:
        f.write(json.dumps({**cfg, "kernel": f"{u}/{SLUG[a.kind]}", "sources": sources, "at": datetime.now(timezone.utc).isoformat()}) + "\n")
    print(cfg["run"])


def status(a):
    kaggle("kernels", "status", f"{user()}/{SLUG[a.kind]}")


def pull(a):
    d = HOME / "out" / a.kind
    d.mkdir(parents=True, exist_ok=True)
    pat = [] if a.kind == "eval" else ["--file-pattern", r".*(job\.json|status\.json|log\.jsonl|done\.json|best_lora\.pt|\.log)$"]
    kaggle("kernels", "output", f"{user()}/{SLUG[a.kind]}", "-p", d, "-o", *pat)
    print(d, (d / "status.json").read_text() if (d / "status.json").exists() else "no status.json")


def upload(a):
    d = HOME / "out" / a.kind
    job = json.loads((d / "job.json").read_text())
    if not json.loads((d / "status.json").read_text())["ok"]:
        raise SystemExit(f"{d}/status.json: not ok; nothing uploaded")
    put = lambda k, b: print(("wrote " if store.put_once(k, b) else "exists ") + k)  # noqa: E731
    if a.kind == "train":
        for p in sorted(x for x in d.rglob("*") if x.is_file()):
            put(f"w8d/kaggle/{job['run']}/{p.relative_to(d).as_posix()}", p.read_bytes())
        return
    sel = json.loads((d / "selection.json").read_text())
    systems = {}
    for t in sorted((d / "eval").glob("*.tar")):
        system, s = t.stem.split(".")
        rid = f"{job['run']}-{'base' if system == 'base' else 'ft47h-' + system}"
        with tarfile.open(t) as f:
            for m in f.getmembers():
                if m.isfile():
                    put(f"stages/w8c_eval/{rid}/{s}/{Path(m.name).name}", f.extractfile(m).read())
                    if Path(m.name).name == "scores.json":
                        systems.setdefault((system, rid), {})[s] = json.loads(f.extractfile(m).read())
    for (system, rid), scores in systems.items():
        put(f"stages/w8c_eval/{rid}/run.json", json.dumps({
            "stage": "w8c_eval", "run": rid, "sets": SETS, "only": ",".join(sorted(scores)), "name": system,
            "ckpt": None if system == "base" else f"s3://{store.BUCKET}/w8d/kaggle/{job['train_run']}/{system}/best_lora.pt (LoRA; merged decoder.pt stayed on Kaggle)",
            "limit": job["limit"], "instance": "kaggle NvidiaTeslaT4 (one T4 per process)", "image": "kaggle + w8d/kaggle/kernel.py venv",
            "seamless": "9a081e935c29c6b1e0e05bcbf3fcb623ad306524", "selection": sel, "sha": job["sha"],
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"), "scores": scores}, indent=1).encode())


def bootstrap(a, b, n=10000, seed=0):
    """Paired bootstrap of mean(b - a) -> (mean, 2.5%, 97.5%)."""
    d, rng = [y - x for x, y in zip(a, b)], random.Random(seed)
    m = sorted(sum(rng.choices(d, k=len(d))) / len(d) for _ in range(n))
    return sum(d) / len(d), m[int(0.025 * n)], m[int(0.975 * n)]


ROWS = [("ASR-COMET", lambda x: x["asr_comet"], 3), ("ASR-BLEU", lambda x: x["asr_bleu"], 1),
        ("numbers acc (speech)", lambda x: x["asr_num_acc"], 2), ("LAAL t0 (s)", lambda x: x["lat_t0"]["laal"], 2),
        ("StartOffset t0 (s)", lambda x: x["lat_t0"]["start_offset"], 2), ("EndOffset t0 (s)", lambda x: x["lat_t0"]["end_offset"], 2),
        ("failed sessions", lambda x: x["failed"], 0)]


def table(cols, sc, kind):
    """cols: [(label, key)]; sc: {key: scores.json} -> markdown rows for one record kind."""
    out = [f"| {kind} | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    return out + [f"| {n} | " + " | ".join(f"{f(sc[k][kind]):.{nd}f}" for _, k in cols) + " |" for n, f, nd in ROWS]


def report(a):
    """3-way speech-track table (dev + test) and paired bootstraps; then the T4-vs-A10G drift of the base on dev."""
    win = json.loads(store.get(f"stages/w8c_eval/{a.run}-base/run.json"))["selection"]["winner"]
    runs = {"base": BASE, "FT-6h": FT6, "FT-47h": f"stages/w8c_eval/{a.run}-ft47h-{win}", "base T4": f"stages/w8c_eval/{a.run}-base"}
    sets = {n: ("dev",) if n == "base T4" else ("dev", "test") for n in runs}
    sc = {(n, s): json.loads(store.get(f"{r}/{s}/scores.json")) for n, r in runs.items() for s in sets[n]}
    per = {(n, s): {p["id"]: p for p in map(json.loads, store.get(f"{r}/{s}/per_record.jsonl").decode().splitlines())}
           for n, r in runs.items() for s in sets[n]}
    cols = [(f"{s} {n}", (n, s)) for s in ("dev", "test") for n in ("base", "FT-6h", "FT-47h")]
    for kind in ("all", "real", "synthetic"):
        print("\n".join(["", *table(cols, sc, kind)]))
    print()
    pairs = [(s, kind, x, "FT-47h") for s in ("dev", "test") for kind in ("real", "synthetic") for x in ("base", "FT-6h")]
    for s, kind, x, y in pairs + [("dev", k, "base", "base T4") for k in ("real", "synthetic")]:
        ids = sorted(i for i, p in per["base", s].items() if p["kind"] == kind)
        m, lo, hi = bootstrap([per[x, s][i]["asr_comet"] for i in ids], [per[y, s][i]["asr_comet"] for i in ids])
        print(f"{s} {kind} ({len(ids)}): ASR-COMET {y} - {x} = {m:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    print("\n".join(["", *table([("dev base (A10G)", ("base", "dev")), ("dev base (T4)", ("base T4", "dev"))], sc, "all")]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("data")
    p = sub.add_parser("push")
    p.add_argument("kind", choices=["train", "eval"])
    p.add_argument("--limit", type=int, help="smoke test: N train/dev/test records, a few steps")
    p.add_argument("--resume", action="store_true", help="eval: mount the last eval output and continue it")
    for c in ("status", "pull", "upload"):
        sub.add_parser(c).add_argument("kind", choices=["train", "eval"])
    sub.add_parser("report").add_argument("run", help="the eval kernel's run id (out/eval/job.json)")
    a = ap.parse_args()
    {"data": data, "push": push, "status": status, "pull": pull, "upload": upload, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
