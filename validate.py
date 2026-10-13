# /// script
# requires-python = ">=3.12"
# ///
"""Validate a Part A manifest against SPEC.md (PS §3A, §5.2).

    uv run validate.py manifest.jsonl [--audio-dir DIR] [--partial]

`audio` paths resolve against --audio-dir (default: the manifest's folder).
--partial checks a stage file: absent fields are skipped and `_`-prefixed keys are allowed.
Prints one line per violation (`<id> <rule>: <detail>`) and an hours summary on stderr;
exits 1 on any violation. Rule names are listed in SPEC.md §9.
"""
import argparse
import json
import math
import re
import sys
import unicodedata
import wave
from collections import defaultdict
from pathlib import Path

RATE, WIDTH, MAX_DUR, MAX_HOURS = 24000, 2, 60.0, 200.0
TOL = 0.002  # seconds; covers 3-decimal rounding of duration/start/end
ID_RE = re.compile(r"rec-\d{6}")
FINAL = "।?!"
# Devanagari block minus danda/double danda (final mark only), digits ०-९ and abbreviation sign ॰
HI_CHAR = re.compile(r"[\u0900-\u0963\u0971-\u097f]")
DEVANAGARI = re.compile(r"[\u0900-\u097f]")

TOP = {"id", "audio", "duration", "speaker_id", "source", "sentences", "hi_words", "translation_method"}
TOP_REQ = TOP - {"speaker_id"}
SRC = {"corpus", "source_file", "start", "end", "synthetic", "model", "voice"}
SRC_REQ = {"corpus", "source_file", "start", "end", "synthetic"}
SENT = {"hi_text", "hi_start", "hi_end", "en_text", "en_alt"}
SENT_REQ = SENT - {"en_alt"}
WORD = {"w", "start", "end"}


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def text(x):
    return isinstance(x, str) and x.strip() != ""


def tokens(t):
    """hi_text -> words: final mark stripped, split on single spaces."""
    return (t[:-1] if t and t[-1] in FINAL else t).split(" ")


def check_record(rec, audio_dir, partial):
    errs = []

    def err(rule, msg):
        errs.append((rule, msg))

    def keys(obj, allowed, required, where):
        for k in obj:
            if k not in allowed and not (partial and k.startswith("_")):
                err("schema", f"unknown key {where}{k}")
        if not partial:
            for k in sorted(required - obj.keys()):
                err("schema", f"missing {where}{k}")

    times = []  # every time value, for the precision check
    rid = rec.get("id")
    if not (isinstance(rid, str) and ID_RE.fullmatch(rid)):
        err("id", f"{rid!r} is not rec-XXXXXX")
    keys(rec, TOP, TOP_REQ, "")
    for k in ("speaker_id", "translation_method"):
        if k in rec and not text(rec[k]):
            err("schema", f"{k} must be a non-empty string")

    dur = rec.get("duration")
    if "duration" in rec:
        if not num(dur) or not 0 < dur <= MAX_DUR:
            err("duration_max", f"duration {dur!r} not in (0, {MAX_DUR:g}]")
            dur = None
        else:
            times.append(dur)

    if "audio" in rec:
        if rec["audio"] != f"audio/{rid}.wav":
            err("audio_path", f"{rec['audio']!r} != 'audio/{rid}.wav'")
        else:
            path = Path(audio_dir) / rec["audio"]
            try:
                with wave.open(str(path), "rb") as w:
                    ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
                    short = len(w.readframes(n)) != n * ch * width
            except FileNotFoundError:
                err("audio_path", f"{path} not found")
            except (wave.Error, EOFError, OSError) as e:
                err("audio_format", f"not a PCM WAV: {e}")
            else:
                if short:
                    err("audio_format", "data chunk shorter than header says")
                if ch != 1:
                    err("audio_channels", f"{ch} channels")
                if rate != RATE:
                    err("audio_rate", f"{rate} Hz")
                if width != WIDTH:
                    err("audio_depth", f"{8 * width}-bit")
                if dur is not None and abs(n / rate - dur) > TOL:
                    err("duration_match", f"audio is {n / rate:.4f} s, duration says {dur}")

    src = rec.get("source")
    if "source" in rec and not isinstance(src, dict):
        err("schema", "source must be an object")
    elif src is not None:
        for k in src:
            if k not in SRC and not (partial and k.startswith("_")):
                err("schema", f"unknown key source.{k}")
        for k in sorted(SRC_REQ - src.keys()) if not partial else []:
            err("source_empty", f"source.{k} missing")
        for k in ("corpus", "source_file", "model", "voice"):
            if k in src and not text(src[k]):
                err("source_empty", f"source.{k} is {src[k]!r}")
        syn = src.get("synthetic")
        if "synthetic" in src and not isinstance(syn, bool):
            err("source_synthetic", f"source.synthetic is {syn!r}, not a bool")
        elif syn is True and not {"model", "voice"} <= src.keys():
            err("source_synthetic", "synthetic record without source.model/voice")
        elif syn is False and src.keys() & {"model", "voice"}:
            err("source_synthetic", "real record with source.model/voice")
        s, e = src.get("start"), src.get("end")
        if "start" in src or "end" in src:
            if not (num(s) and num(e) and 0 <= s < e):
                err("source_interval", f"need 0 <= start < end, got {s!r}, {e!r}")
            else:
                times += [s, e]
                if dur is not None and abs(e - s - dur) > TOL:
                    err("source_interval", f"end - start = {e - s:.3f} != duration {dur}")
                if syn is True and s != 0:
                    err("source_interval", "synthetic clip must be the whole generated file (start 0)")

    sents = rec.get("sentences")
    toks, counts = [], []  # all hi_text tokens; tokens per sentence
    if "sentences" in rec and not (isinstance(sents, list) and sents and all(isinstance(x, dict) for x in sents)):
        err("schema", "sentences must be a non-empty list of objects")
        sents = None
    for i, sen in enumerate(sents or []):
        keys(sen, SENT, SENT_REQ, f"sentences[{i}].")
        t = sen.get("hi_text")
        if "hi_text" in sen and not isinstance(t, str):
            err("schema", f"sentences[{i}].hi_text is not a string")
        elif t is not None:
            if unicodedata.normalize("NFC", t) != t:
                err("hi_nfc", f"sentences[{i}].hi_text is not NFC")
            if not t or t[-1] not in FINAL or any(c in FINAL for c in t[:-1]):
                err("hi_final", f"sentences[{i}].hi_text must end with exactly one of । ? ! (and have none inside)")
            bad = sorted({c for c in t[:-1] if c != " " and c not in FINAL and not HI_CHAR.match(c)})
            if bad:
                err("hi_chars", f"sentences[{i}].hi_text has " + " ".join(f"{c!r}(U+{ord(c):04X})" for c in bad))
            tk = tokens(t)
            if "" in tk:
                err("hi_space", f"sentences[{i}].hi_text: words must be separated by single spaces, mark attached")
            tk = [x for x in tk if x]
            toks += tk
            counts.append(len(tk))
        for k in ("hi_start", "hi_end"):
            if k in sen:
                if num(sen[k]):
                    times.append(sen[k])
                else:
                    err("schema", f"sentences[{i}].{k} is not a number")
        if "en_text" in sen and not (text(sen["en_text"]) and not DEVANAGARI.search(sen["en_text"])):
            err("en_text", f"sentences[{i}].en_text must be non-empty English")
        alt = sen.get("en_alt")
        if "en_alt" in sen and not (isinstance(alt, list) and all(text(a) and not DEVANAGARI.search(a) for a in alt)):
            err("en_text", f"sentences[{i}].en_alt must be a list of non-empty English strings")

    words = rec.get("hi_words")
    if "hi_words" in rec:
        if not (isinstance(words, list) and all(isinstance(x, dict) for x in words)):
            err("schema", "hi_words must be a list of objects")
            words = None
        elif any(not WORD <= x.keys() or any(k not in WORD and not (partial and k.startswith("_")) for k in x) for x in words):
            err("schema", "every hi_words entry needs exactly w, start, end")
            words = None
        elif not all(num(x["start"]) and num(x["end"]) for x in words):
            err("schema", "hi_words times must be numbers")
            words = None
    if words is not None:
        for x in words:
            times += [x["start"], x["end"]]
        ws = [x["w"] for x in words]
        matched = sents is not None and len(counts) == len(sents) and ws == toks
        if sents is not None and len(counts) == len(sents) and not matched:
            i = next((j for j, (a, b) in enumerate(zip(ws, toks)) if a != b), min(len(ws), len(toks)))
            err("words_match", f"hi_words[{i}] {ws[i] if i < len(ws) else None!r} != hi_text token {toks[i] if i < len(toks) else None!r} "
                               f"({len(ws)} words vs {len(toks)} tokens)")
        for i, x in enumerate(words):
            nxt = words[i + 1]["start"] if i + 1 < len(words) else dur
            if not 0 <= x["start"] < x["end"] or (nxt is not None and x["end"] > nxt):
                err("word_times", f"hi_words[{i}] {x['start']}–{x['end']} breaks 0 <= s < e <= next s (or duration {dur})")
                break
        if matched:
            j = 0
            for i, (sen, c) in enumerate(zip(sents, counts)):
                if c and "hi_start" in sen and "hi_end" in sen and num(sen["hi_start"]) and num(sen["hi_end"]):
                    if sen["hi_start"] != words[j]["start"] or sen["hi_end"] != words[j + c - 1]["end"]:
                        err("sentence_times", f"sentences[{i}] {sen['hi_start']}–{sen['hi_end']} != its words "
                                              f"{words[j]['start']}–{words[j + c - 1]['end']}")
                j += c

    if any(round(x, 3) != x for x in times):
        err("time_precision", "times must have at most 3 decimals")
    return errs


def union(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            total += 0 if cur_e is None else cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    return total + (0 if cur_e is None else cur_e - cur_s)


def validate(lines, audio_dir, partial=False):
    """-> (violations [(line_no, id, rule, msg)], hours {real, synthetic, total, real_files, synthetic_files})"""
    out, seen = [], {}
    real, syn = defaultdict(list), defaultdict(list)
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            if not isinstance(rec, dict):
                raise ValueError("not an object")
        except ValueError as e:
            out.append((n, f"line:{n}", "json", str(e)))
            continue
        rid = rec.get("id") if isinstance(rec.get("id"), str) else f"line:{n}"
        out += [(n, rid, rule, msg) for rule, msg in check_record(rec, audio_dir, partial)]
        if rid in seen:
            out.append((n, rid, "id_unique", f"also on line {seen[rid]}"))
        seen.setdefault(rid, n)
        src = rec.get("source")
        if isinstance(src, dict) and isinstance(src.get("source_file"), str) and num(src.get("start")) and num(src.get("end")):
            f = src["source_file"]
            if src.get("synthetic") is True:
                if f in syn:
                    out.append((n, rid, "source_reuse", f"synthetic source_file {f!r} already used on line {syn[f][0][0]}"))
                syn[f].append((n, src["end"] - src["start"]))
            elif src.get("synthetic") is False:
                real[f].append((src["start"], src["end"]))
    for f in real.keys() & syn.keys():
        out.append((syn[f][0][0], "-", "source_reuse", f"{f!r} used as both real and synthetic"))
    h_real = sum(union(v) for v in real.values()) / 3600
    h_syn = sum(v[0][1] for v in syn.values()) / 3600  # one generated file = one record (SPEC §8)
    hours = {"real": h_real, "synthetic": h_syn, "total": h_real + h_syn,
             "real_files": len(real), "synthetic_files": len(syn)}
    if hours["total"] > MAX_HOURS:
        out.append((0, "-", "hours_cap", f"{hours['total']:.3f} h > {MAX_HOURS:g} h"))
    return out, hours


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--audio-dir", type=Path, help="root that `audio` paths resolve against (default: manifest's folder)")
    ap.add_argument("--partial", action="store_true", help="stage file: skip absent fields, allow `_` keys")
    a = ap.parse_args()
    with open(a.manifest, encoding="utf-8") as f:
        lines = f.read().split("\n")
    out, h = validate(lines, a.audio_dir or a.manifest.parent, a.partial)
    for n, rid, rule, msg in out:
        print(f"{rid} {rule}: {msg}")
    recs = sum(1 for x in lines if x.strip())
    print(f"{recs} records, {len(out)} violations | real {h['real']:.3f} h over {h['real_files']} source files, "
          f"synthetic {h['synthetic']:.3f} h over {h['synthetic_files']} files, total {h['total']:.3f} / {MAX_HOURS:g} h",
          file=sys.stderr)
    sys.exit(1 if out else 0)


if __name__ == "__main__":
    main()
