"""Sarvam Saaras (REST speech-to-text) over the W4 1 h benchmark set. Every raw response is archived first.

    python3 -I w4/saaras_run.py CONFIG [N]      # CONFIG: v4 | v3-verbatim | v3-transcribe | v4-verbatim

Archive: data/w4/saaras/<CONFIG>/<id>.json = {id, config, chunks: [{start, end, status, response}]}. A clip whose
archive exists is never re-requested (no seed, PS §3A; Sarvam outputs stay private, §10 R3). Then the joined
transcripts go to data/w4/asr/saaras-<CONFIG>.jsonl for w4/asr_score.py.
REST takes <= 30 s of audio per request (docs.sarvam.ai/api-reference/speech-to-text/transcribe, read 2026-10-14):
longer clips are cut in the middle of an IndicVoices inter-segment gap (`_w1.segments`), never inside speech.
Billing ₹30/h, per second (docs pricing page): spend is printed from the seconds actually sent.
"""
import json, subprocess, sys, time, urllib.error, urllib.request, uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "data/w4/bench/manifest.jsonl"  # the frozen 1 h set (W1 pool v1 rec-000001..174)
AUDIO = ROOT / "data/w1/pool_pinned"  # retired clips have no audio left; their archives are complete
URL = "https://api.sarvam.ai/speech-to-text"
CONFIGS = {"v4": {"model": "saaras:v4"}, "v4-verbatim": {"model": "saaras:v4", "mode": "verbatim"},
           "v3-verbatim": {"model": "saaras:v3", "mode": "verbatim"}, "v3-transcribe": {"model": "saaras:v3", "mode": "transcribe"}}
MAX = 29.5  # s per request, under the 30 s REST limit
RATE = 30 / 3600  # ₹ per second


def key():
    for line in open(ROOT / ".env"):
        if line.startswith("SARVAM_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("SARVAM_API_KEY missing from .env")


def chunks(rec):
    """Clip-relative [start, end] pieces <= MAX s, cut mid-gap between IV segments."""
    segs, dur = rec["_w1"]["segments"], rec["duration"]
    if dur <= MAX:
        return [(0.0, dur)]
    out, s0 = [], 0.0
    for a, b in zip(segs, segs[1:] + [None]):
        if b is None:
            break
        if b["end"] - s0 > MAX:  # next segment would overflow: cut in the gap after `a`
            cut = round(min((a["end"] + b["start"]) / 2, s0 + MAX), 3)  # mid-gap, or as late as the limit allows
            assert a["end"] <= cut <= s0 + MAX, f"{rec['id']}: no gap cut under {MAX} s"
            out.append((s0, cut))
            s0 = cut
    out.append((s0, dur))
    assert all(e - s <= 30 for s, e in out), f"{rec['id']}: chunk over the 30 s REST limit {out}"  # tail: + edge pad
    return out


def wav16k(path, s, e):
    return subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(path), "-ss", f"{s:.3f}", "-to", f"{e:.3f}",
                           "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "-"], check=True, capture_output=True).stdout


def post(audio, fields, api_key):
    b = uuid.uuid4().hex
    body = b"".join(f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items())
    body += f'--{b}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    body += audio + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(URL, data=body, method="POST", headers={
        "api-subscription-key": api_key, "Content-Type": f"multipart/form-data; boundary={b}"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as err:
            if err.code in (429, 500, 503) and attempt < 4:  # nothing was billed/returned: retry is not a re-request
                time.sleep(2 ** attempt)
                continue
            raw = err.read().decode(errors="replace")
            try:
                return err.code, json.loads(raw)
            except ValueError:
                return err.code, {"raw": raw}


def main():
    name, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 174
    fields = {**CONFIGS[name], "language_code": "hi-IN"}
    arch = ROOT / f"data/w4/saaras/{name}"
    arch.mkdir(parents=True, exist_ok=True)
    api_key, sent = key(), 0.0
    for line in list(open(BENCH))[:n]:
        r = json.loads(line)
        out = arch / f"{r['id']}.json"
        if out.exists() and all(c["status"] == 200 for c in json.loads(out.read_text())["chunks"]):
            continue  # have it: never re-request (a failed request returned nothing, so it may be retried)
        if not (AUDIO / r["audio"]).exists():
            print(r["id"], "audio gone (retired by W1), skipped", file=sys.stderr)
            continue
        parts = []
        for s, e in chunks(r):
            status, resp = post(wav16k(AUDIO / r["audio"], s, e), fields, api_key)
            sent += e - s
            parts.append({"start": s, "end": e, "status": status, "response": resp})
            if status != 200:
                print(r["id"], status, json.dumps(resp, ensure_ascii=False)[:300], file=sys.stderr)
        out.write_text(json.dumps({"id": r["id"], "config": fields, "chunks": parts}, ensure_ascii=False, indent=1))
    hyps = ROOT / f"data/w4/asr/saaras-{name}.jsonl"
    ok = total = 0
    with open(hyps, "w") as f:
        for line in list(open(BENCH))[:n]:
            if not (arch / f"{json.loads(line)['id']}.json").exists():
                continue
            a = json.loads((arch / f"{json.loads(line)['id']}.json").read_text())
            total += 1
            if all(c["status"] == 200 for c in a["chunks"]):
                ok += 1
                f.write(json.dumps({"id": a["id"], "hyp": " ".join(c["response"]["transcript"] for c in a["chunks"])},
                                   ensure_ascii=False) + "\n")
    print(f"{name}: {ok}/{total} clips ok -> {hyps}; sent this run {sent:.1f} s ≈ ₹{sent * RATE:.2f}")


if __name__ == "__main__":
    fake = {"id": "t", "duration": 40.0, "_w1": {"segments": [{"start": 0, "end": 12}, {"start": 13, "end": 28},
                                                              {"start": 30, "end": 39}]}}
    assert chunks(fake) == [(0.0, 29.0), (29.0, 40.0)], chunks(fake)
    fake["_w1"]["segments"] = [{"start": 0, "end": 29.4}, {"start": 30.4, "end": 39}]
    assert chunks(fake) == [(0.0, 29.5), (29.5, 40.0)], chunks(fake)
    main()
