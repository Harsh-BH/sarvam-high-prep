# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy"]
# ///
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
  - segments tagged [stammers] / [uh-huh] / [ugh] (either bracket form, any case) are dropped the same way,
    because a clip that needs a listener is dropped, not reviewed (S3 cleaning spec §3.3); --keep-disfluency-tags
    turns this off.
IDs come from w1/ids.tsv (SPEC §4: assigned once, never reused); new clips get the next free id. Retired ids are
burned in w1/retired.tsv and never issued again.
Writes OUT/manifest.jsonl (validate.py --partial stage file, SPEC §8) and OUT/../pilot/ (first --n-pilot).

--recut: keep every OUT/manifest.v1.jsonl clip not in w1/retired.tsv byte-for-byte, move retired ones to
OUT/rejects.jsonl (with the reason), and cut replacement clips with fresh ids around the dropped segments.
Retired clips with no droppable tag inside (e.g. an incomplete transcript) are burned whole: no replacement.

Stage contract (S3 plan, Global Constraints): `uv run w1/cut_pilot.py --in RECORDS.jsonl --audio-dir DIR --out OUT`
re-cuts every record's existing [start, end] from DIR/<source.source_file> (the raw/ layout), ids and intervals
unchanged; OUT/records.jsonl (input lines verbatim), OUT/audio/<id>.wav, OUT/rejects.jsonl; resumable.
All audio goes through pipeline/ffmpeg.py (the pinned build): rebuilds only match cuts made with it.
"""
import argparse, collections as C, json, os, re, sys, wave, zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pipeline"))
import ffmpeg  # noqa: E402
from inventory import load_iv, windows  # noqa: E402

RATE, PAD, MIN_PAD, QUIET_DB, ABOVE_FLOOR, INAUDIBLE_DB = 24000, 0.2, 0.05, -30.0, 6.0, -45.0
REJECT_TAG = re.compile(r"talking|whisper|yelling|singing|unintelligible", re.I)
DISFLUENCY_TAG = re.compile(r"[\[<]\s*(?:stammers|uh-huh|ugh)\s*[\]>]", re.I)
REJECT_FLAGS = ("chatter_intermittent", "chatter_persistent")
IDS = Path(__file__).parent / "ids.tsv"
RETIRED = Path(__file__).parent / "retired.tsv"
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


def runs(rec, x, sr, ref, floor, partner, drop_disfluency=True, used=()):
    """Split the file's segments into runs that may share a clip. -> (segs, runs, gap_ok, bad, stats)

    A segment is bad if tagged with another voice / unintelligible / (drop_disfluency) stammers, uh-huh, ugh, if the
    partner talks over it, or if it overlaps a `used` span (a clip that already exists or an id that is burned).
    A gap is a break if it holds a possible voice: louder than QUIET_DB re. the speaker and >= ABOVE_FLOOR dB over
    the file's noise floor (stationary field noise is not a voice). A gap the partner talks in must stay within
    ABOVE_FLOOR of the floor or below INAUDIBLE_DB re. the speaker: the partner may leak into this channel
    (measured: 19% of partner-only regions above -30 dB), and leaked audio can be speech the ASR cannot decode.
    """
    segs = sorted(rec["segs"])
    st = C.Counter()
    bad = []
    for s, e, t in segs:
        dis = drop_disfluency and bool(DISFLUENCY_TAG.search(t))
        b = bool(REJECT_TAG.search(t)) or (partner is not None and overlaps(s, e, partner, 0.1))
        st["seg: second voice / unintelligible / partner overlap"] += b
        st["seg: stammers / uh-huh / ugh"] += dis and not b
        bad.append(b or dis or overlaps(s, e, used))
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


def clip_candidates(rec, x, sr, partner, max_gap, drop_disfluency=True, used=()):
    own = [x[int(s * sr):int(e * sr)] for s, e, _ in rec["segs"] if e - s > 0.3]
    if not own:
        return [], C.Counter()
    ref = 10 * np.log10(np.mean(np.concatenate(own).astype(np.float64) ** 2) + 1e-9)
    floor = level(x, sr, 0, rec["raw_dur"], q=10)
    segs, rs, gap_ok, bad, st = runs(rec, x, sr, ref, floor, partner, drop_disfluency, used)
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
            for ua, ub in used:  # a pad must not reach into an existing clip
                if a_ms < ub * 1000 and b_ms > ua * 1000:
                    a_ms, b_ms = (round(ub * 1000), b_ms) if ua * 1000 <= a_ms else (a_ms, round(ua * 1000))
            if a_ms <= s * 1000 and b_ms >= e * 1000 and 10000 <= b_ms - a_ms <= 60000:
                out.append(dict(a_ms=a_ms, b_ms=b_ms, inside=segs[i0:i1 + 1]))
    return out, st


def cut(src, a_ms, b_ms, out):
    """Pinned-ffmpeg cut of [a_ms, b_ms): whole file resampled to 24 kHz (no filter edge at the cut), then trimmed.
    The system ffmpeg (n9.0.1) differs from the pinned n9.0.2 by 1 LSB, so never call it directly."""
    ffmpeg.cut(src, a_ms / 1000, b_ms / 1000, out)


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


def load_retired():
    return dict(l.rstrip("\n").split("\t", 1) for l in list(open(RETIRED))[1:]) if RETIRED.exists() else {}


def assign_ids(cands, retired):
    """Reuse the registered id of a known span, else the next free one; spans of burned ids are skipped."""
    ids = load_ids()
    nxt = max((int(v[4:]) for v in ids.values()), default=0) + 1
    new, out = [], []
    for c in cands:
        key = (c["source_file"], c["a_ms"], c["b_ms"])
        if ids.get(key) in retired:
            continue
        if key not in ids:
            ids[key] = f"rec-{nxt:06d}"
            new.append((ids[key], *key))
            nxt += 1
        c["id"] = ids[key]
        out.append(c)
    assert nxt <= 400000, "real id range exhausted (SPEC §4)"
    with open(IDS, "a") as f:
        if f.tell() == 0:
            f.write("id\tsource_file\tstart_ms\tend_ms\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in new)
    return out, len(new)


def file_candidates(stem, rec, jsons, root, max_gap, drop_disfluency, used=()):
    """All clip candidates of one long-form file, with the record fields make_record needs. -> (cands, stats)"""
    st = C.Counter()
    vr = rec["raw"]["verification_report"]
    if rec["n_spk"] > 1 or any(vr.get(k) for k in REJECT_FLAGS):
        st["file: >1 speaker_id or chatter flag"] += 1
        return [], st
    partner = None
    if rec["kind"] == "Conversation":
        p = jsons.get(stem[:-1] + {"0": "1", "1": "0"}[stem[-1]]) if stem[-2:] in ("_0", "_1") else None
        if p is None:
            st["file: conversation without partner JSON"] += 1
            return [], st
        partner = [(s, e) for s, e, _ in p["segs"]]
    with wave.open(rec["file"]) as w:
        sr, x = w.getframerate(), np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    clips, s2 = clip_candidates(rec, x, sr, partner, max_gap, drop_disfluency, used)
    st.update(s2)
    rel = Path(rec["file"]).relative_to(root)  # Hindi/<v>/<split>/<name>.wav, as the archive distributes it
    return [dict(c, rec=rec, kind=rec["kind"], speaker=rec["speaker"], sr=sr, rel=rel,
                 source_file=f"indicvoices/{rel}", partner_checked=partner is not None) for c in clips], st


def make_record(c, out_dir):
    """Cut the clip's audio into out_dir/audio and return its stage-file line."""
    rid, raw, rel = c["id"], c["rec"]["raw"], c["rel"]
    version, split = rel.parts[1], rel.parts[2]
    cut(c["rec"]["file"], c["a_ms"], c["b_ms"], str(out_dir / "audio" / f"{rid}.wav"))
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
    return json.dumps(r, ensure_ascii=False) + "\n"


def hours(lines):
    return sum(json.loads(l)["duration"] for l in lines) / 3600


def recut(a, jsons):
    """Retire burned clips, keep the rest byte-for-byte, cut replacements around the dropped segments."""
    v1_path, pilot = a.out / "manifest.v1.jsonl", a.out.parent / "pilot"
    if not v1_path.exists():
        os.replace(a.out / "manifest.jsonl", v1_path)
        os.replace(pilot / "manifest.jsonl", pilot / "manifest.v1.jsonl")
    v1 = {json.loads(l)["id"]: l for l in open(v1_path)}
    retired = load_retired()
    keep = {k: l for k, l in v1.items() if k not in retired}
    gone = {k: json.loads(l) for k, l in v1.items() if k in retired}
    spans = C.defaultdict(lambda: {"used": [], "retired": []})
    for l in keep.values():
        r = json.loads(l)
        spans[r["source"]["source_file"]]["used"].append((r["source"]["start"], r["source"]["end"]))
    for r in gone.values():
        droppable = any(DISFLUENCY_TAG.search(x["verbatim"]) for x in r["_w1"]["segments"])
        # nothing droppable inside (e.g. incomplete transcript): burn the whole span, cut nothing from it
        spans[r["source"]["source_file"]]["retired" if droppable else "used"].append((r["source"]["start"], r["source"]["end"]))
    cands, st = [], C.Counter()
    for f, sp in spans.items():
        if not sp["retired"]:
            continue
        stem = Path(f).stem
        cs, s2 = file_candidates(stem, jsons[stem], jsons, a.root, a.max_gap, not a.keep_disfluency_tags, sp["used"])
        st.update(s2)
        cands += [c for c in cs if overlaps(c["a_ms"] / 1000, c["b_ms"] / 1000, sp["retired"])]
    cands, n_new = assign_ids(sorted(cands, key=lambda c: (c["source_file"], c["a_ms"])), retired)
    lines = dict(keep)
    lines.update({c["id"]: make_record(c, a.out) for c in cands})
    with open(a.out / "manifest.jsonl", "w") as f:
        f.writelines(lines[k] for k in sorted(lines))
    with open(a.out / "rejects.jsonl", "w") as f:
        for k in sorted(gone):
            (a.out / gone[k]["audio"]).unlink(missing_ok=True)
            (pilot / gone[k]["audio"]).unlink(missing_ok=True)
            f.write(json.dumps(dict(gone[k], _reject_reason=retired[k]), ensure_ascii=False) + "\n")
    p1 = [json.loads(l)["id"] for l in open(pilot / "manifest.v1.jsonl")]
    with open(pilot / "manifest.jsonl", "w") as f:
        f.writelines(keep[k] for k in p1 if k in keep)
    print(f"v1: {len(v1)} clips {hours(v1.values()):.3f} h | retired {len(gone)} ({hours(v1[k] for k in gone):.3f} h) | "
          f"kept {len(keep)} | replacements {len(cands)} ({n_new} new ids, {hours(lines[c['id']] for c in cands):.3f} h) | "
          f"now {len(lines)} clips {hours(lines.values()):.3f} h | pilot {sum(k in keep for k in p1)}/{len(p1)} | {dict(st)}", file=sys.stderr)


def stage(inp, audio_dir, out):
    """Re-cut every input record at its existing interval with the pinned ffmpeg (see module doc)."""
    retired = load_retired()
    (out / "audio").mkdir(parents=True, exist_ok=True)
    for f in ("records.jsonl", "rejects.jsonl"):
        p = out / f
        if p.exists() and (b := p.read_bytes()) and not b.endswith(b"\n"):
            p.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: drop it, that id is redone
    done = {json.loads(l)["id"] for f in ("records.jsonl", "rejects.jsonl") if (out / f).exists() for l in open(out / f)}
    n = C.Counter()
    with open(out / "records.jsonl", "a", encoding="utf-8") as ok, open(out / "rejects.jsonl", "a", encoding="utf-8") as bad:
        def reject(rid, rule, detail):
            bad.write(json.dumps({"id": rid, "stage": "cut", "rule": rule, "detail": detail}, ensure_ascii=False) + "\n")
            n[rule] += 1
            done.add(rid)

        for l in open(inp, encoding="utf-8"):
            r = json.loads(l)
            src = audio_dir / r["source"]["source_file"]
            if r["id"] in done:
                continue
            if r["id"] in retired:
                reject(r["id"], "retired", retired[r["id"]])
            elif not src.exists():
                reject(r["id"], "source_missing", str(src))
            else:
                cut(src, round(r["source"]["start"] * 1000), round(r["source"]["end"] * 1000), out / "audio" / f"{r['id']}.wav")
                ok.write(l.rstrip("\n") + "\n")
                n["records"] += 1
                done.add(r["id"])
        for rid, why in sorted(retired.items()):  # burned ids not in the input are still accounted for
            if rid not in done:
                reject(rid, "retired", why)
    print(f"cut stage: {dict(n)} (already done: {len(done) - sum(n.values())})", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--in", dest="inp", type=Path, help="stage mode: records to re-cut at their intervals")
    ap.add_argument("--audio-dir", type=Path, help="stage mode: root that source.source_file resolves against")
    ap.add_argument("--out", dest="stage_out", type=Path, help="stage mode: output dir")
    ap.add_argument("--pool-hours", type=float, default=4.0)
    ap.add_argument("--n-pilot", type=int, default=50)
    ap.add_argument("--max-gap", type=float, default=2.0)
    ap.add_argument("--keep-disfluency-tags", action="store_true", help="do not drop [stammers]/[uh-huh]/[ugh] segments")
    ap.add_argument("--recut", action="store_true", help="retire w1/retired.tsv ids from OUT and cut replacements")
    a = ap.parse_args()
    if a.inp:
        return stage(a.inp, a.audio_dir, a.stage_out)
    if not (a.root and a.out):
        ap.error("need ROOT OUT, or --in/--audio-dir/--out")

    jsons = {Path(r["file"]).stem: r for r in load_iv(a.root)}
    if a.recut:
        return recut(a, jsons)
    cands, st = [], C.Counter()
    for stem, rec in jsons.items():
        if Path(rec["file"]).exists():
            cs, s2 = file_candidates(stem, rec, jsons, a.root, a.max_gap, not a.keep_disfluency_tags)
            cands += cs
            st.update(s2)
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

    picked, n_new = assign_ids(picked, load_retired())

    pilot = a.out.parent / "pilot"
    for d in (a.out / "audio", pilot / "audio"):
        d.mkdir(parents=True, exist_ok=True)
    pilot_ids = {c["id"] for c in picked[: a.n_pilot]}
    with open(a.out / "manifest.jsonl", "w") as pool_f, open(pilot / "manifest.jsonl", "w") as pilot_f:
        for c in sorted(picked, key=lambda c: c["id"]):
            line = make_record(c, a.out)
            pool_f.write(line)
            if c["id"] in pilot_ids:
                pilot_f.write(line)
                if not (pilot / "audio" / f"{c['id']}.wav").exists():
                    os.link(a.out / "audio" / f"{c['id']}.wav", pilot / "audio" / f"{c['id']}.wav")
    kinds = C.Counter(c["kind"] for c in picked)
    pk = C.Counter(c["kind"] for c in picked[: a.n_pilot])
    print(f"pool: {len(picked)} clips, {total / 3.6e6:.2f} h, {dict(kinds)}; pilot {sum(pk.values())} {dict(pk)}; "
          f"{n_new} new ids", file=sys.stderr)


if __name__ == "__main__":
    assert all(round(m / 1000 * 1000) == m for m in range(0, 4_000_000, 7))  # cut()'s ms -> s -> ms round trip is exact
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
    # [stammers] / <Ugh> / [UH-HUH] drop their segment and break the clip by default; --keep-disfluency-tags keeps it
    x = np.zeros(40 * sr, dtype=np.int16)
    x[0:int(33 * sr)] = 3000
    for tag in ("[stammers]", "<Ugh>", "[UH-HUH]"):
        rec = dict(segs=[(0, 11, "a"), (11, 22, f"b {tag} b"), (22, 33, "c")], raw_dur=40)
        got = [(c["a_ms"], c["b_ms"]) for c in clip_candidates(rec, x, sr, None, 2.0)[0]]
        assert got == [(0, 11000), (22000, 33200)], (tag, got)
        assert [(c["a_ms"], c["b_ms"]) for c in clip_candidates(rec, x, sr, None, 2.0, drop_disfluency=False)[0]] == [(0, 33200)]
    assert not DISFLUENCY_TAG.search("[laugh] [ughh] stammersx [uh-uhh]")
    # a `used` span is never cut again, and pads stop at its edge
    rec = dict(segs=[(0, 11, "a"), (11.1, 22, "b"), (22.1, 33, "c")], raw_dur=40)
    got = [(c["a_ms"], c["b_ms"]) for c in clip_candidates(rec, x, sr, None, 2.0, used=[(10.9, 22.05)])[0]]
    assert got == [(22050, 33200)], got
    main()
