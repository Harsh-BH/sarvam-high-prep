# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "boto3"]
# ///
"""Held-out split for Part B evaluation (rule: tasks/W8c-eval.md "Split rule").

    uv run w8c/split.py     # -> data/w8c/cut/{records.jsonl,audio/}, w8c/splits/ (git) and data/w8c/splits/ (copy)

Scale sessions import only is_heldout (stdlib):
    sys.path.insert(0, "<repo>/w8c"); from split import is_heldout

Synthetic: whole voices held out, per engine, crc32 order, until >= 10% of that engine's pilot-v2 hours (each script
has one take, so whole scripts go with them). A cloned voice is eligible only if the person it imitates isn't a
pilot-v2 train speaker. Real: IndicVoices *valid* is NOT speaker-disjoint from train (same speaker_id and
demographics in both), so eligible = valid speakers absent from pilot-v2 real and from every w3/voices.tsv
reference; speaker -> dev if crc32 % 3 == 0 else test; clips cut with W1's cutter + drop
rule; ids rec-800000+ from w8c/ids.tsv (SPEC §4 reserved range; this file is its one allocator).
"""
import json, zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SPLITS = HERE / "splits"
DATA = Path("/home/harsh1/github-repos/sarvam-high-prep/data")
_held = None


def _load():
    global _held
    if _held is None:
        h = json.loads((SPLITS / "heldout.json").read_text())
        ids = {l.strip() for f in ("dev.txt", "test.txt") for l in open(SPLITS / f) if l.strip()}
        _held = (set(h["speakers"]), {tuple(v) for v in h["voices"]}, set(h["scripts"]), ids)
    return _held


def is_heldout(r):
    """True if a record must never be trained on: a dev/test id, a held-out person (a real speaker, or the person a
    cloned voice imitates), a held-out (model, voice), or a held-out script."""
    spk, voices, scripts, ids = _load()
    s = r.get("source") or {}
    voice = s.get("voice") or ""
    script = (r.get("_w2") or r.get("_w3") or {}).get("pid")
    return (r.get("id") in ids or (r.get("speaker_id") or "").removeprefix("spk-indicvoices-") in spk
            or voice.removeprefix("iv-") in spk or (s.get("model"), voice) in voices or script in scripts)


SYNTH_SHARE, DEV_SHARE = 0.10, 1 / 3
REAL_CUT_H = {"dev": 1.0, "test": 2.05}  # ~15% over the 1 h / 2 h targets: transcribe/align/asr_check reject some
ID0 = 800000
VALID_ETAG = {"v1": "d8a1ad1b95dbc802ba8e74b7fae70c57-13", "v2": "9a8b4037d74192072ea913b2fe1474f8-11",
              "v3": "9b6f72eef6428f8f985726a074be5a94-25", "v4": "920319d0885cb7545b03e3362e892d88-23",
              "v5": "98cd36b1325070a5a19dccdc96a79694-16"}  # HEAD of {v}_Hindi_valid.tgz on 2026-10-15
crc = lambda s: zlib.crc32(s.encode())


def pick_voices(synth):
    """synth: [(model, voice, hours, eligible)] per record -> {(model, voice): split}."""
    by = {}
    for m, v, h, ok in synth:
        by.setdefault(m, {}).setdefault(v, [0.0, ok])[0] += h
    held = []
    for m, vs in sorted(by.items()):
        total, got = sum(h for h, _ in vs.values()), 0.0
        for v in sorted((v for v in vs if vs[v][1]), key=lambda v: crc(f"{m}/{v}")):
            if got >= SYNTH_SHARE * total:
                break
            held.append(((m, v), vs[v][0]))
            got += vs[v][0]
    out, dev, total = {}, 0.0, sum(h for _, h in held)
    for mv, h in held:
        out[mv] = "dev" if dev < DEV_SHARE * total else "test"
        dev += h * (out[mv] == "dev")
    return out


def main():
    import collections as C, shutil, sys
    sys.path.insert(0, str(ROOT / "pipeline"))
    sys.path.insert(0, str(ROOT / "w1"))
    import s3 as store, cut_pilot as W1

    pv2 = DATA / "w8c/pv2"
    for k in ["manifest.jsonl", "inputs/1.jsonl", "inputs/2.jsonl"]:
        if not (pv2 / k).exists():
            (pv2 / k).parent.mkdir(parents=True, exist_ok=True)
            (pv2 / k).write_bytes(store.get(f"release/pilot-v2/{k}"))
    recs = [json.loads(l) for l in open(pv2 / "manifest.jsonl")]
    pid = {r["id"]: (r.get("_w2") or r["_w3"])["pid"] for k in ("1", "2") for r in map(json.loads, open(pv2 / f"inputs/{k}.jsonl"))}
    train_spk = {r["speaker_id"].removeprefix("spk-indicvoices-") for r in recs if not r["source"]["synthetic"]}

    synth = [r for r in recs if r["source"]["synthetic"]]
    voices = pick_voices([(r["source"]["model"], r["source"]["voice"], r["duration"] / 3600,
                           r["source"]["voice"].removeprefix("iv-") not in train_spk) for r in synth])
    split = {r["id"]: voices[(r["source"]["model"], r["source"]["voice"])] for r in synth
             if (r["source"]["model"], r["source"]["voice"]) in voices}
    held_spk = {v.removeprefix("iv-") for _, v in voices if v.startswith("iv-")}

    # real: eligible IndicVoices valid speakers
    iv = DATA / "w1/iv_full"
    vt = [l.rstrip("\n").split("\t") for l in open(ROOT / "w3/voices.tsv") if not l.startswith("#")]
    voice_spk = {r[0].removeprefix("iv-") for r in vt[1:] if r[0].startswith("iv-")}
    # the IV-train sample only lends conversation partners (a valid call's other side may sit in train)
    jsons = {Path(r["file"]).stem: r for d in (DATA / "w1/iv_train", iv) for r in W1.load_iv(d)}
    bad = train_spk | voice_spk
    cands, st = [], C.Counter()
    for stem, rec in sorted(jsons.items()):
        if "/valid/" in rec["file"] and rec["speaker"] not in bad and Path(rec["file"]).exists():
            cs, s2 = W1.file_candidates(stem, rec, jsons, iv, 2.0, True)
            cands += cs
            st.update(s2)
    print(f"{len(cands)} candidates, {sum(c['b_ms'] - c['a_ms'] for c in cands) / 3.6e6:.2f} h; {dict(st)}", file=sys.stderr)
    picked, hrs = [], C.Counter()
    for c in W1.order(cands, [("Conversation", 2), ("Extempore", 2), ("Read", 1)]):
        sp = "dev" if crc(c["speaker"]) % 3 == 0 else "test"
        if hrs[sp] < REAL_CUT_H[sp] * 3.6e6:
            picked.append((c, sp))
            hrs[sp] += c["b_ms"] - c["a_ms"]

    # ids: registered spans keep theirs, new spans get the next free one (never reused, SPEC §4)
    reg = HERE / "ids.tsv"
    rows = [l.rstrip("\n").split("\t") for l in open(reg)][1:] if reg.exists() else []
    ids = {(f, int(a), int(b)): rid for rid, f, a, b in rows}
    nxt = max((int(r[4:]) for r in ids.values()), default=ID0 - 1) + 1
    with open(reg, "a") as f:
        if not rows:
            f.write("id\tsource_file\tstart_ms\tend_ms\n")
        for c, _ in sorted(picked, key=lambda x: (x[0]["source_file"], x[0]["a_ms"])):
            key = (c["source_file"], c["a_ms"], c["b_ms"])
            if key not in ids:
                ids[key] = f"rec-{nxt:06d}"
                f.write("\t".join(map(str, (ids[key], *key))) + "\n")
                nxt += 1
            c["id"] = ids[key]
    assert nxt <= 900000, "w8c id range exhausted"

    out = DATA / "w8c/cut"
    (out / "audio").mkdir(parents=True, exist_ok=True)
    lines, cut = [], W1.cut
    W1.cut = lambda *a: None  # make_record would re-cut every clip on a re-run
    for c, sp in sorted(picked, key=lambda x: x[0]["id"]):
        if not (out / "audio" / f"{c['id']}.wav").exists():
            cut(c["rec"]["file"], c["a_ms"], c["b_ms"], str(out / "audio" / f"{c['id']}.wav"))
        r = json.loads(W1.make_record(c, out))
        v = c["rel"].parts[1]
        r["_rev"] = f"{W1.IV_URL.format(v=v, split='valid')} etag=\"{VALID_ETAG[v]}\""
        r["_w8c"] = {"split": sp}
        lines.append(json.dumps(r, ensure_ascii=False) + "\n")
        split[c["id"]] = sp
        held_spk.add(c["speaker"])
    (out / "records.jsonl").write_text("".join(lines))

    SPLITS.mkdir(exist_ok=True)
    for sp in ("dev", "test"):
        (SPLITS / f"{sp}.txt").write_text("".join(f"{i}\n" for i in sorted(i for i in split if split[i] == sp)))
    held = {"speakers": sorted(held_spk), "voices": sorted(map(list, voices)),
            "scripts": sorted(pid[i] for i in split if i in pid)}
    (SPLITS / "heldout.json").write_text(json.dumps(held, indent=1) + "\n")
    global _held
    _held = None
    train = [r["id"] for r in recs if not is_heldout(r)]
    (SPLITS / "train.txt").write_text("".join(f"{i}\n" for i in train))
    shutil.copytree(SPLITS, DATA / "w8c/splits", dirs_exist_ok=True)

    hours = lambda ids_: sum(r["duration"] for r in recs if r["id"] in ids_) / 3600
    for sp in ("dev", "test"):
        syn = {i for i in split if split[i] == sp and i in pid}
        print(f"{sp}: synthetic {len(syn)} ({hours(syn):.3f} h), real {sum(s == sp for _, s in picked)} "
              f"({hrs[sp] / 3.6e6:.3f} h, {len({c['speaker'] for c, s in picked if s == sp})} speakers)")
    print(f"train: {len(train)} of {len(recs)} pilot-v2 ({hours(set(train)):.3f} h); held-out voices {sorted(voices.items())}")
    # the rule is consistent: no train record is held out, every dev/test pilot-v2 record is
    assert not any(is_heldout(r) for r in recs if r["id"] in set(train))
    assert all(is_heldout(r) for r in recs if r["id"] in split)
    assert not held_spk & train_spk, "a held-out speaker is in pilot-v2 train"


if __name__ == "__main__":
    main()
