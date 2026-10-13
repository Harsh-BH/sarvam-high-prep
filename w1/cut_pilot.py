"""Cut single-speaker 10-60 s real clips from IndicVoices long-form recordings + their existing transcripts.

    uv run --no-project --with numpy python -I w1/cut_pilot.py data/w1/iv_train data/w1/pool --pool-hours 4 --n-pilot 50

A clip is the contiguous long-form span of consecutive transcribed segments of the file's one speaker, padded
into the neighbouring silence, resampled to 24 kHz mono s16 (soxr) and trimmed sample-exactly.
One-speaker checks (PS §3A, SPEC R11), see runs():
  - segments tagged with another voice or <unintelligible> break the clip and are left out;
  - a gap (untranscribed time) that may hold a voice breaks the clip there;
  - conversations: IndicVoices ships each call as <uuid>_0 / <uuid>_1, one party per file. The partner file's
    segment times say where the other party talks: segments it talks over are left out, and gaps it talks in
    must be at the noise floor. No partner JSON on disk -> file skipped.
IDs come from w1/ids.tsv (SPEC §4: assigned once, never reused); new clips get the next free id.
Writes OUT/manifest.jsonl (validate.py --partial stage file, SPEC §8) and OUT/../pilot/ (first --n-pilot).
"""
import argparse, collections as C, json, os, re, subprocess, sys, wave, zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from inventory import load_iv, windows  # noqa: E402

RATE, PAD, MIN_PAD, QUIET_DB, ABOVE_FLOOR, INAUDIBLE_DB = 24000, 0.2, 0.05, -30.0, 6.0, -45.0
REJECT_TAG = re.compile(r"talking|whisper|yelling|singing|unintelligible", re.I)
REJECT_FLAGS = ("chatter_intermittent", "chatter_persistent")
IDS = Path(__file__).parent / "ids.tsv"
# pinned archive revisions (HEAD on 2026-10-14), so a rebuild can tell if the source bytes changed
IV_REV = {
    "v1": '"59a22bd11d8361717e1d879bf29eaecb-854"', "v2": '"a4635d32abd203f0804863dfe55b393b-1068"',
    "v3": '"c1e0167a2290851b0a434a5261f612d6-3264"', "v4": '"16c88b2ff34e7ef59a3a93d09c441f4e-2479"',
    "v5": '"405e91bc7577e619399928c7fb1b347a-1285"',
}
IV_URL = "https://iv-release.objectstore.e2enetworks.net/dmu_release/{v}_Hindi_{split}.tgz"


def level(x, sr, a, b, q=90):
    """q-th percentile of 50 ms frame energies (dB) in [a, b) seconds; None if shorter than one frame."""
    g, f = x[int(a * sr):int(b * sr)].astype(np.float64), sr // 20
    if len(g) < f:
        return None
    return float(np.percentile(10 * np.log10((g[: len(g) // f * f].reshape(-1, f) ** 2).mean(1) + 1e-9), q))


def overlaps(a, b, segs, slack=0.0):
    return any(s < b + slack and e > a - slack for s, e in segs)


def runs(rec, x, sr, ref, floor, partner):
    """Split the file's segments into runs that may share a clip. -> (segs, runs, gap_ok, bad, stats)

    A segment is bad if tagged with another voice / unintelligible, or if the partner talks over it.
    A gap is a break if it holds a possible voice: louder than QUIET_DB re. the speaker and >= ABOVE_FLOOR dB over
    the file's noise floor (stationary field noise is not a voice). A gap the partner talks in must stay within
    ABOVE_FLOOR of the floor or below INAUDIBLE_DB re. the speaker: the partner may leak into this channel
    (measured: 19% of partner-only regions above -30 dB), and leaked audio can be speech the ASR cannot decode.
    """
    segs = sorted(rec["segs"])
    st = C.Counter()
    bad = []
    for s, e, t in segs:
        b = bool(REJECT_TAG.search(t)) or (partner is not None and overlaps(s, e, partner, 0.1))
        st["seg: second voice / unintelligible / partner overlap"] += b
        bad.append(b)
    gap_ok = []
    for (s0, e0, _), (s1, e1, _) in zip(segs, segs[1:]):
        lv = level(x, sr, e0 + MIN_PAD, s1 - MIN_PAD) if s1 - e0 > 2 * MIN_PAD + 0.05 else None
        if lv is None:
            ok = True
        elif partner is not None and overlaps(e0, s1, partner):
            ok = lv - floor < ABOVE_FLOOR or lv - ref < INAUDIBLE_DB
        else:
            ok = not (lv - ref > QUIET_DB and lv - floor >= ABOVE_FLOOR)
        st["gap: possible voice in untranscribed audio"] += not ok
        gap_ok.append(ok)
    out, cur = [], []
    for i, seg in enumerate(segs):
        if not bad[i]:
            cur.append(i)
        if bad[i] or i == len(segs) - 1 or not gap_ok[i]:
            if cur:
                out.append(cur)
            cur = []
    return segs, out, gap_ok, bad, st


def clip_candidates(rec, x, sr, partner, max_gap):
    own = [x[int(s * sr):int(e * sr)] for s, e, _ in rec["segs"] if e - s > 0.3]
    if not own:
        return [], C.Counter()
    ref = 10 * np.log10(np.mean(np.concatenate(own).astype(np.float64) ** 2) + 1e-9)
    floor = level(x, sr, 0, rec["raw_dur"], q=10)
    segs, rs, gap_ok, bad, st = runs(rec, x, sr, ref, floor, partner)
    out = []
    for r in rs:
        sub = [segs[i] for i in r]
        for s, e, _ in windows(sub, rec["raw_dur"], max_gap=max_gap, hi=60 - 2 * PAD):
            i0 = next(i for i in r if segs[i][0] == s)
            i1 = max(i for i in r if segs[i][1] == e)
            # pad into the neighbouring gap: up to PAD if it is quiet and no other segment is near, else MIN_PAD
            pre_room = s - (segs[i0 - 1][1] if i0 else 0.0)
            post_room = (segs[i1 + 1][0] if i1 + 1 < len(segs) else rec["raw_dur"]) - e
            pre = min(PAD if (i0 == 0 or gap_ok[i0 - 1]) and not (i0 and bad[i0 - 1]) else MIN_PAD, pre_room / 2 if i0 else pre_room)
            post = min(PAD if (i1 + 1 == len(segs) or gap_ok[i1]) and not (i1 + 1 < len(segs) and bad[i1 + 1]) else MIN_PAD,
                       post_room / 2 if i1 + 1 < len(segs) else post_room)
            a_ms, b_ms = max(0, round((s - pre) * 1000)), min(int(rec["raw_dur"] * 1000), round((e + post) * 1000))
            out.append(dict(a_ms=a_ms, b_ms=b_ms, inside=segs[i0:i1 + 1]))
    return out, st


def cut(src, a_ms, b_ms, out):
    """Resample the whole file (no filter edge at the cut), then keep samples [a, b) at 24 kHz."""
    a, b = a_ms * RATE // 1000, b_ms * RATE // 1000
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-af",
                    f"aresample={RATE}:resampler=soxr,atrim=start_sample={a}:end_sample={b}",
                    "-ac", "1", "-c:a", "pcm_s16le", "-fflags", "+bitexact", "-flags:a", "+bitexact", out], check=True)
    with wave.open(out) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (RATE, 1, 2, b - a), out


def order(cands, weights):
    """Round-robin over kinds (by weight), within a kind round-robin over speakers; deterministic."""
    by_kind = C.defaultdict(lambda: C.defaultdict(list))
    for c in sorted(cands, key=lambda c: zlib.crc32(f"{c['source_file']}:{c['a_ms']}".encode())):
        by_kind[c["kind"]][c["speaker"]].append(c)
    queues = {}
    for k, spk in by_kind.items():
        q, lists = [], [spk[s] for s in sorted(spk, key=lambda s: zlib.crc32(s.encode()))]
        while any(lists):
            q += [l.pop(0) for l in lists if l]
        queues[k] = q
    out = []
    while any(queues.values()):
        for k, w in weights:
            for _ in range(w):
                if queues.get(k):
                    out.append(queues[k].pop(0))
    return out


def load_ids():
    if not IDS.exists():
        return {}
    rows = [l.rstrip("\n").split("\t") for l in open(IDS)][1:]
    return {(f, int(a), int(b)): rid for rid, f, a, b in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("out", type=Path)
    ap.add_argument("--pool-hours", type=float, default=4.0)
    ap.add_argument("--n-pilot", type=int, default=50)
    ap.add_argument("--max-gap", type=float, default=2.0)
    a = ap.parse_args()

    jsons = {Path(r["file"]).stem: r for r in load_iv(a.root)}
    cands, st = [], C.Counter()
    for stem, rec in jsons.items():
        if not Path(rec["file"]).exists():
            continue
        vr = rec["raw"]["verification_report"]
        if rec["n_spk"] > 1 or any(vr.get(k) for k in REJECT_FLAGS):
            st["file: >1 speaker_id or chatter flag"] += 1
            continue
        partner = None
        if rec["kind"] == "Conversation":
            p = jsons.get(stem[:-1] + {"0": "1", "1": "0"}[stem[-1]]) if stem[-2:] in ("_0", "_1") else None
            if p is None:
                st["file: conversation without partner JSON"] += 1
                continue
            partner = [(s, e) for s, e, _ in p["segs"]]
        with wave.open(rec["file"]) as w:
            sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        clips, s2 = clip_candidates(rec, x, sr, partner, a.max_gap)
        st.update(s2)
        rel = Path(rec["file"]).relative_to(a.root)  # Hindi/<v>/<split>/<name>.wav, as the archive distributes it
        for c in clips:
            cands.append(dict(c, rec=rec, kind=rec["kind"], speaker=rec["speaker"], sr=sr, rel=rel,
                              source_file=f"indicvoices/{rel}", partner_checked=partner is not None))
    by = C.Counter(c["kind"] for c in cands)
    hrs = C.Counter()
    for c in cands:
        hrs[c["kind"]] += (c["b_ms"] - c["a_ms"]) / 3.6e6
    print(f"{len(cands)} candidates {dict(by)}, hours { {k: round(v, 2) for k, v in hrs.items()} }; {dict(st)}", file=sys.stderr)

    picked, total = [], 0
    for c in order(cands, [("Conversation", 2), ("Extempore", 2), ("Read", 1)]):
        if total >= a.pool_hours * 3.6e6:
            break
        picked.append(c)
        total += c["b_ms"] - c["a_ms"]

    ids = load_ids()
    nxt = max((int(v[4:]) for v in ids.values()), default=0) + 1
    new = []
    for c in picked:
        key = (c["source_file"], c["a_ms"], c["b_ms"])
        if key not in ids:
            ids[key] = f"rec-{nxt:06d}"
            new.append((ids[key], *key))
            nxt += 1
        c["id"] = ids[key]
    assert nxt <= 400000, "real id range exhausted (SPEC §4)"
    with open(IDS, "a") as f:
        if f.tell() == 0:
            f.write("id\tsource_file\tstart_ms\tend_ms\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in new)

    pilot = a.out.parent / "pilot"
    for d in (a.out / "audio", pilot / "audio"):
        d.mkdir(parents=True, exist_ok=True)
    pilot_ids = {c["id"] for c in picked[: a.n_pilot]}
    with open(a.out / "manifest.jsonl", "w") as pool_f, open(pilot / "manifest.jsonl", "w") as pilot_f:
        for c in sorted(picked, key=lambda c: c["id"]):
            rid, raw, rel = c["id"], c["rec"]["raw"], c["rel"]
            version, split = rel.parts[1], rel.parts[2]
            out_wav = a.out / "audio" / f"{rid}.wav"
            cut(c["rec"]["file"], c["a_ms"], c["b_ms"], str(out_wav))
            off = c["a_ms"] / 1000
            norm = {(s["start"], s["end"]): s["text"] for s in raw["normalized"]}
            r = {
                "id": rid,
                "audio": f"audio/{rid}.wav",
                "duration": round((c["b_ms"] - c["a_ms"]) / 1000, 3),
                "speaker_id": f"spk-indicvoices-{c['speaker']}",
                "source": {"corpus": "IndicVoices", "source_file": c["source_file"],
                           "start": c["a_ms"] / 1000, "end": c["b_ms"] / 1000, "synthetic": False},
                "_rev": f"{IV_URL.format(v=version, split=split)} etag={IV_REV.get(version, '?')}",
                # IndicVoices `verbatim` (L1-like, with tags) joined; W4 converts per SPEC §2.4
                "_orig_text": " ".join(t for *_, t in c["inside"]),
                "_w1": {"segments": [{"start": round(s - off, 3), "end": round(e - off, 3), "verbatim": t,
                                      "normalized": norm.get((s, e))} for s, e, t in c["inside"]],
                        "orig_sample_rate": c["sr"], "scenario": raw["scenario"], "task_name": raw["task_name"],
                        "gender": raw["gender"], "age_group": raw["age_group"], "state": raw["state"],
                        "district": raw["district"], "partner_checked": c["partner_checked"],
                        "verification_flags": sorted(k for k, v in raw["verification_report"].items() if v is True)},
            }
            line = json.dumps(r, ensure_ascii=False) + "\n"
            pool_f.write(line)
            if rid in pilot_ids:
                pilot_f.write(line)
                if not (pilot / r["audio"]).exists():
                    os.link(out_wav, pilot / r["audio"])
    kinds = C.Counter(c["kind"] for c in picked)
    pk = C.Counter(c["kind"] for c in picked[: a.n_pilot])
    print(f"pool: {len(picked)} clips, {total / 3.6e6:.2f} h, {dict(kinds)}; pilot {sum(pk.values())} {dict(pk)}; "
          f"{len(new)} new ids", file=sys.stderr)


if __name__ == "__main__":
    q = order([dict(source_file=f"f{i}", a_ms=i, kind=k, speaker=s) for i, (k, s) in enumerate(
        [("A", "x"), ("A", "x"), ("A", "y"), ("B", "z")])], [("A", 2), ("B", 1)])
    assert [c["kind"] for c in q] == ["A", "A", "B", "A"] and q[0]["speaker"] != q[1]["speaker"], q
    # a loud gap must split, a second-voice segment must be dropped from every clip
    sr = 1000
    x = np.zeros(70 * sr, dtype=np.int16)
    for s, e in [(0, 6), (6.5, 12), (13, 24), (24.5, 27), (40, 52)]:
        x[int(s * sr):int(e * sr)] = 3000
    x[int(12.3 * sr):int(12.7 * sr)] = 3000  # loud audio in the 12-13 s gap
    rec = dict(segs=[(0, 6, "a"), (6.5, 12, "b"), (13, 24, "c"), (24.5, 27, "<talking> d"), (40, 52, "e")], raw_dur=70)
    got = [(c["a_ms"], c["b_ms"]) for c in clip_candidates(rec, x, sr, None, 2.0)[0]]
    assert got == [(0, 12050), (12950, 24050), (39950, 52200)], got
    main()
