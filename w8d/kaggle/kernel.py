"""W8d on Kaggle, the in-kernel side (tasks/W8d-finetune.md "Kaggle run"). Kaggle's python runs `main`; the model
steps run on a py3.10 venv with W8d's env (torch 2.2.0 cu121 + fairseq2n 0.2.1, tasks/W8d-finetune.md §1):

    python kernel.py main CFG_JSON              # env, audio, then train or eval (CFG from w8d/kaggle/ctl.py push)
    venv/bin/python kernel.py prefetch          # Seamless weights once, so parallel processes don't race for them
    venv/bin/python kernel.py encode JOB.json   # frozen speech encoder on one GPU -> {id: states} .pt
    venv/bin/python kernel.py fit JOB.json      # w8d/train.py train() on the cached states -> /kaggle/working/<arm>/

Train: each GPU encodes a shard, then one LoRA arm per GPU (W8d run-1 recipe; arms differ only in steps/patience).
Eval: w8c/run_seamless.py + w8c/eval.py unchanged, every job's records split across the GPUs. Every arm runs dev;
the best dev ASR-COMET (declared in advance, as W8d) runs test; the base reruns dev here to measure T4-vs-A10G drift.
Audio comes from presigned S3 URLs (urls.json in the private dataset): never printed, never in /kaggle/working.
"""
import json, os, subprocess, sys, tarfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parent.parent
W, OUT, INPUT = Path("/tmp/w8d"), Path("/kaggle/working"), Path("/kaggle/input")
VPY = W / "venv/bin/python"
SEAMLESS = "https://github.com/facebookresearch/seamless_communication/archive/9a081e935c29c6b1e0e05bcbf3fcb623ad306524.tar.gz"
TORCH = ["torch==2.2.0+cu121", "torchaudio==2.2.0+cu121"]
PKGS = ["fairseq2n==0.2.1+cu121", "fairseq2==0.2.1", "simuleval==1.1.4", "soundfile", "numpy<2",  # = w8c/sm_eval.py's
        "sacrebleu==2.4.3", "unbabel-comet==2.2.4", "transformers==4.46.3"]                    # FAIRSEQ2 + SCORING
INDEXES = ["https://download.pytorch.org/whl/cu121", "https://fair.pkg.atmeta.com/fairseq2/whl/pt2.2.0/cu121"]
CARDS = {"seamless_streaming_unity": ("checkpoint", "tokenizer", "char_tokenizer"),
         "seamless_streaming_monotonic_decoder": ("checkpoint",), "vocoder_v2": ("checkpoint",)}
BUDGET_S = 11 * 3600  # Kaggle stops a session at 12 h; stop early so /kaggle/working is saved and the next one resumes
T0 = time.time()


def sh(*cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    return subprocess.run(list(map(str, cmd)), check=True, **kw)


def setup():
    if VPY.exists():
        return
    sh(sys.executable, "-m", "pip", "install", "-q", "uv")
    if b"libsndfile.so.1" not in subprocess.run(["ldconfig", "-p"], capture_output=True).stdout:
        sh("bash", "-c", "apt-get update -qq && apt-get install -y -qq libsndfile1")  # fairseq2n's audio reader links it
    uv = [sys.executable, "-m", "uv"]
    sh(*uv, "venv", "-q", "-p", "3.10", W / "venv")
    (W / "pins.txt").write_text("\n".join(TORCH + ["numpy<2"]) + "\n")
    idx = [x for u in INDEXES for x in ("--extra-index-url", u)]
    sh(*uv, "pip", "install", "-q", "-p", VPY, "--index-strategy", "unsafe-best-match", *idx, "-c", W / "pins.txt", *TORCH, *PKGS)
    sh(*uv, "pip", "install", "-q", "-p", VPY, "--no-deps", SEAMLESS)
    sh("nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader")


def gpus():
    return len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=True).stdout.splitlines())


def dataset():
    """-> (dataset dir, {id: presigned url}). Refuses URLs that expire within the hour."""
    d = next(INPUT.rglob("urls.json")).parent
    meta = json.loads((d / "manifest.json").read_text())
    if meta["expires"] - time.time() < 3600:
        raise SystemExit("the dataset's presigned URLs have expired: rerun `ctl.py data` on the laptop")
    return d, json.loads((d / "urls.json").read_text())


def fetch(ids, urls, dest, workers=32, tries=4):
    """Download <id>.wav for every id. Errors name the id and the HTTP status only: a URL is a 7-day read grant."""
    dest.mkdir(parents=True, exist_ok=True)

    def one(i):
        p = dest / f"{i}.wav"
        err = ""
        for k in range(tries):
            if p.exists():
                return None
            try:
                b = urllib.request.urlopen(urls[i], timeout=120).read()
                if b[:4] != b"RIFF" or b[8:12] != b"WAVE":
                    raise ValueError("not a WAV")
                (dest / f"{i}.part").write_bytes(b)
                (dest / f"{i}.part").rename(p)
            except Exception as e:  # noqa: BLE001  (str(e) of some urllib errors quotes the URL)
                err = f"{type(e).__name__} {getattr(e, 'code', '')}".strip()
                time.sleep(2 ** k)
        return None if p.exists() else f"{i}: {err}"
    tic = time.time()
    with ThreadPoolExecutor(workers) as ex:
        bad = [r for r in ex.map(one, ids) if r]
    if bad:
        raise SystemExit(f"{len(bad)} of {len(ids)} downloads failed, e.g. {bad[:5]}")
    print(f"audio: {len(ids)} files in {dest} ({time.time() - tic:.0f} s)", flush=True)


def queue(per_gpu, logs):
    """per_gpu: [[cmd, ...] per GPU]; each GPU runs its commands in order, all GPUs at once, until the budget.
    -> True if every command finished with exit 0. Logs: one file per command, its tail printed on failure."""
    ok = []

    def worker(k, cmds):
        for j, cmd in enumerate(cmds):
            log = logs / f"gpu{k}.{j}.log"
            left = BUDGET_S - (time.time() - T0)
            print(f"gpu{k}: {' '.join(map(str, cmd[1:4]))} ... -> {log.name}", flush=True)
            try:
                with open(log, "w") as f:
                    rc = subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT, timeout=max(1, left),
                                        env={**os.environ, "CUDA_VISIBLE_DEVICES": str(k), "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).returncode
            except subprocess.TimeoutExpired:
                rc = "timeout"
            if rc != 0:
                print(f"gpu{k}: exit {rc}; tail of {log.name}:\n" + "".join(open(log).readlines()[-40:]), flush=True)
                return ok.append(False)
        ok.append(True)
    logs.mkdir(parents=True, exist_ok=True)
    ts = [threading.Thread(target=worker, args=(k, c)) for k, c in enumerate(per_gpu) if c]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    return all(ok)


def train(cfg):
    ds, urls = dataset()
    tr = [[r["id"], r["en"]] for r in map(json.loads, open(ds / "train.jsonl"))]
    if cfg.get("limit"):  # smoke test: the longest targets, so it also proves batch 8 fits a 16 GB T4
        tr = sorted(tr, key=lambda p: -len(p[1]))[:cfg["limit"]]
    dv = [[r["id"], " ".join(s["en_text"] for s in r["sentences"])] for r in map(json.loads, open(ds / "dev.jsonl"))][:cfg.get("limit")]
    ids = [i for i, _ in tr + dv]
    fetch(ids, urls, W / "audio")
    sh(VPY, HERE / "kernel.py", "prefetch")
    n, enc = gpus(), []
    jobs = []
    for k in range(n):
        enc.append(W / f"enc{k}.pt")
        (W / f"enc{k}.json").write_text(json.dumps({"ids": ids[k::n], "audio": str(W / "audio"), "out": str(enc[-1])}))
        jobs.append([[VPY, HERE / "kernel.py", "encode", W / f"enc{k}.json"]])
    if not queue(jobs, OUT / "logs/encode"):
        raise SystemExit("encode failed")
    arms = list(cfg["arms"].items())
    for g in range(0, len(arms), n):  # one arm per GPU at a time
        jobs = [[] for _ in range(n)]
        for k, (name, args) in enumerate(arms[g:g + n]):
            (OUT / name).mkdir(parents=True, exist_ok=True)
            spec = {"args": ["--run", cfg["run"], *args], "enc": list(map(str, enc)), "train": tr, "dev": dv, "out": str(OUT / name)}
            (W / f"fit.{name}.json").write_text(json.dumps(spec))
            jobs[k].append([VPY, HERE / "kernel.py", "fit", W / f"fit.{name}.json"])
        if not queue(jobs, OUT / "logs" / f"fit{g}"):
            raise SystemExit("fit failed")
    for name, _ in arms:
        print(name, (OUT / name / "done.json").read_text()[:600], flush=True)


def evaluate(cfg):
    ds, urls = dataset()
    sets, ev = W / "sets", W / "eval"
    recs = {s: [json.loads(l) for l in open(ds / f"{s}.jsonl") if l.strip()][:cfg.get("limit")] for s in ("dev", "test")}
    fetch([r["id"] for rs in recs.values() for r in rs], urls, sets / "audio")
    for s, rs in recs.items():
        (sets / f"{s}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rs))
    sh(VPY, HERE / "kernel.py", "prefetch")
    for t in INPUT.rglob("eval/*.tar"):  # an earlier session's outputs: run_seamless skips records already done
        with tarfile.open(t) as f:
            f.extractall(ev, filter="data")
        print(f"restored {t.name}", flush=True)
    ckpt = {a: str(next(p for p in INPUT.rglob("decoder.pt") if p.parent.name == a)) for a in cfg["arms"]}
    ckpt["base"] = None
    n = gpus()

    def done(system, s):
        return all((ev / system / s / f"{r['id']}.json").exists() for r in recs[s])

    def infer(jobs):
        per = [[] for _ in range(n)]
        for system, s in jobs:
            (W / "parts").mkdir(exist_ok=True)
            for k in range(n):
                part = W / "parts" / f"{system}.{s}.{k}.jsonl"
                part.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs[s][k::n]))
                per[k].append([VPY, CODE / "w8c/run_seamless.py", "--set", part, "--audio-dir", sets, "--out", ev / system / s,
                               *(["--ckpt", ckpt[system]] if ckpt[system] else [])])
        return queue(per, OUT / "logs" / f"infer.{'.'.join(a + s for a, s in jobs)}")

    def score(jobs):
        per = [[] for _ in range(n)]
        for j, (system, s) in enumerate(j for j in jobs if done(*j) and not (ev / j[0] / j[1] / "scores.json").exists()):
            per[j % n].append([VPY, CODE / "w8c/eval.py", "score", "--set", sets / f"{s}.jsonl", "--sys", ev / system / s])
        return queue(per, OUT / "logs" / f"score.{'.'.join(a + s for a, s in jobs)}")

    def pack():
        (OUT / "eval").mkdir(exist_ok=True)
        for d in sorted(ev.glob("*/*")):
            with tarfile.open(OUT / "eval" / f"{d.parent.name}.{d.name}.tar", "w") as f:
                f.add(d, arcname=f"{d.parent.name}/{d.name}")

    dev = [(a, "dev") for a in cfg["arms"]]
    ok = infer([j for j in dev if not done(*j)]) and score(dev)
    if ok:
        comet = {a: json.loads((ev / a / "dev/scores.json").read_text())["all"]["asr_comet"] for a in cfg["arms"]}
        win = max(comet, key=comet.get)
        (OUT / "selection.json").write_text(json.dumps({"rule": "max dev ASR-COMET (speech track, all dev records)",
                                                        "dev_asr_comet": comet, "winner": win}, indent=1))
        print(f"selection: {comet} -> {win}", flush=True)
        more = [(win, "test")] + ([("base", "dev")] if cfg.get("base_dev") else [])
        ok = infer([j for j in more if not done(*j)]) and score(more)
    pack()
    if not ok:
        raise RuntimeError("eval incomplete (failure or time budget): outputs packed; push again to resume")


# --- venv side -------------------------------------------------------------------------------------------------

def prefetch():
    import seamless_communication  # noqa: F401  (registers the cards)
    from fairseq2.assets import asset_store, download_manager
    for name, fields in CARDS.items():
        card = asset_store.retrieve_card(name)
        for f in fields:
            get = download_manager.download_checkpoint if f == "checkpoint" else download_manager.download_tokenizer
            get(card.field(f).as_uri(), name)


def encode(job):
    import torch
    sys.path.insert(0, str(CODE / "w8d"))
    import train as T
    audio, tic = Path(job["audio"]), time.time()
    enc = T.encode([{"id": i, "audio_key": i} for i in job["ids"]], lambda i: (audio / f"{i}.wav").read_bytes(), torch.device("cuda"))
    torch.save(enc, job["out"])
    print(f"encoded {len(enc)} records in {time.time() - tic:.0f} s", flush=True)


def fit(job):
    """w8d/train.py main() without SageMaker: same model, tokenizer, precision and seed; local encoder states."""
    import random, torch
    sys.path.insert(0, str(CODE / "w8d"))
    import train as T
    from seamless_communication.models.monotonic_decoder import load_monotonic_decoder_model
    from seamless_communication.models.unity import load_unity_text_tokenizer
    a = T.args_parser().parse_args(job["args"])
    random.seed(a.seed), torch.manual_seed(a.seed)
    enc = {}
    for p in job["enc"]:
        enc.update(torch.load(p))
    device = torch.device("cuda")
    tok = load_unity_text_tokenizer(T.UNITY)
    model = load_monotonic_decoder_model(T.DECODER, device=device, dtype=torch.float32)
    model.train()
    bf16 = torch.cuda.is_bf16_supported()  # T4/P100: no -> fp16 autocast + GradScaler (train.py's path)
    ctx = {"device": device, "tok_enc": tok.create_encoder(lang="eng", mode="target"), "eos": tok.vocab_info.eos_idx,
           "pad": tok.vocab_info.pad_idx, "vocab_info": tok.vocab_info, "scale": not bf16,
           "amp": lambda: torch.autocast("cuda", dtype=torch.bfloat16 if bf16 else torch.float16)}
    items = lambda ps: [(enc[i], en) for i, en in ps]  # noqa: E731
    T.train(model, items(job["train"]), items(job["dev"]), a, Path(job["out"]), ctx)


def main(cfg):
    """Always exits 0 so Kaggle keeps /kaggle/working (an eval resumes from it); status.json says what happened."""
    W.mkdir(parents=True, exist_ok=True)
    if cfg["mode"] == "eval":  # which training run the decoders came from
        cfg["train_run"] = next(j["run"] for j in map(json.loads, map(Path.read_text, INPUT.rglob("job.json"))) if j["mode"] == "train")
    (OUT / "job.json").write_text(json.dumps(cfg, indent=1))
    status = {"ok": False}
    try:
        setup()
        {"train": train, "eval": evaluate}[cfg["mode"]](cfg)
        status["ok"] = True
    except BaseException as e:  # noqa: BLE001
        status["error"] = f"{type(e).__name__}: {e}"
        print(status["error"], flush=True)
    status["hours"] = round((time.time() - T0) / 3600, 3)
    (OUT / "status.json").write_text(json.dumps(status, indent=1))
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    cmd, *rest = sys.argv[1:]
    if cmd == "main":
        main(json.loads(rest[0]))
    elif cmd == "prefetch":
        prefetch()
    else:
        {"encode": encode, "fit": fit}[cmd](json.loads(Path(rest[0]).read_text()))
