# /// script
# requires-python = ">=3.12"
# ///
"""W0 done-when: the validator passes a correct record set and catches one broken record per rule.

    uv run test_validate.py
"""
import copy
import json
import subprocess
import sys
import tempfile
import wave
from pathlib import Path

import validate

HERE = Path(__file__).parent


def wav(path, seconds, rate=24000, ch=1, width=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(ch), w.setsampwidth(width), w.setframerate(rate)
        w.writeframes(b"\0" * (round(seconds * rate) * ch * width))


def record(rid, sentences, dur, source):
    """sentences: [(hi_text, en_text)]. Words get evenly spaced times inside (0.1, dur - 0.1)."""
    toks = [validate.tokens(hi) for hi, _ in sentences]
    n = sum(map(len, toks))
    step = (dur - 0.2) / n
    words, sents, k = [], [], 0
    for (hi, en), tk in zip(sentences, toks):
        first = k
        for t in tk:
            words.append({"w": t, "start": round(0.1 + k * step, 3), "end": round(0.1 + (k + 0.8) * step, 3)})
            k += 1
        sents.append({"hi_text": hi, "hi_start": words[first]["start"], "hi_end": words[k - 1]["end"], "en_text": en})
    return {"id": rid, "audio": f"audio/{rid}.wav", "duration": dur, "speaker_id": "spk-test",
            "source": source, "sentences": sents, "hi_words": words, "translation_method": "test fixture"}


S1 = ("मैंने कल अपना पुराना बैंक खाता बंद करवा दिया।", "Yesterday I got my old bank account closed.")
S2 = ("अं एस बी आई का ओ टी पी क्या है?", "Um, what is the SBI OTP?")
REAL = lambda s, e: {"corpus": "indicvoices", "source_file": "indicvoices/hindi/x.wav", "start": s, "end": e, "synthetic": False}
SYN = lambda f, d: {"corpus": "w2-scripts-v0", "source_file": f, "start": 0, "end": d, "synthetic": True,
                    "model": "bulbul:v3", "voice": "v1"}

VALID = [
    record("rec-000001", [S1, S2], 2.0, REAL(10.0, 12.0)),
    record("rec-000002", [S1], 2.0, REAL(11.0, 13.0)),  # overlaps rec-000001: real union = 3 s
    record("rec-400001", [S2], 1.5, SYN("tts/rec-400001.wav", 1.5)),
    record("rec-400002", [S2], 1.5, SYN("tts/rec-400002.wav", 1.5)),  # same text, other voice: counts again
]


def broken(root):
    """-> [(rule, manifest line)], writing each broken record's WAV. One record per rule (hi_chars/schema: several)."""
    out, n = [], 100

    def mk(rule, fn=lambda r: None, sents=(S1,), dur=2.0, audio=dict(seconds=None)):
        nonlocal n
        n += 1
        r = record(f"rec-{n:06d}", list(sents), dur, REAL(100.0 + 10 * n, 100.0 + 10 * n + dur))
        if audio is not None:
            wav(root / r["audio"], **{"seconds": dur, **{k: v for k, v in audio.items() if v is not None}})
        fn(r)
        out.append((rule, json.dumps(r, ensure_ascii=False)))
        return r

    def set_(path, value):
        def f(r):
            o = r
            for k in path[:-1]:
                o = o[k]
            o[path[-1]] = value
        return f

    def hi(text):  # replace sentence 0's hi_text and keep hi_words consistent with it
        def f(r):
            r["sentences"][0]["hi_text"] = text
            for wd, t in zip(r["hi_words"], [x for x in validate.tokens(text) if x]):
                wd["w"] = t
        return f

    out.append(("json", '{"id": "rec-000099", '))
    mk("id", lambda r: r.update(id="rec-12", audio="audio/rec-12.wav"))
    (root / "audio/rec-12.wav").write_bytes((root / f"audio/rec-{n:06d}.wav").read_bytes())
    dup = copy.deepcopy(VALID[0]); out.append(("id_unique", json.dumps(dup, ensure_ascii=False)))
    mk("schema", lambda r: r.pop("translation_method"))
    mk("schema", set_(["_qa"], {"asr_cer": 0.1}))  # internal keys are stripped before release
    mk("audio_path", audio=None)
    mk("audio_format", lambda r: (root / r["audio"]).write_bytes(b"RIFF....not a wav"))
    mk("audio_channels", audio=dict(ch=2))
    mk("audio_rate", audio=dict(rate=16000))
    mk("audio_depth", audio=dict(width=3))
    mk("duration_max", dur=61.0)
    mk("duration_match", audio=dict(seconds=1.0))
    mk("words_match", lambda r: r["hi_words"].pop(3))
    mk("word_times", lambda r: r["hi_words"][2].update(end=r["hi_words"][3]["start"] + 0.001))  # 1 ms overlap
    mk("sentence_times", set_(["sentences", 0, "hi_start"], 0.0))
    mk("hi_chars", hi("मैंने कल अपना पुराना bank खाता बंद करवा दिया।"))
    mk("hi_chars", hi("मैंने कल अपना पुराना बैंक खाता बंद करवा 2।"))
    mk("hi_chars", hi("मैंने कल अपना पुराना बैंक खाता बंद करवा २।"))
    mk("hi_chars", hi("मैंने कल अपना पुराना बैंक खाता बंद करवा दि\u200dया।"))
    mk("hi_chars", hi("मैंने कल [अं] पुराना बैंक खाता बंद करवा दिया।"))
    mk("hi_nfc", hi("मैंने कल अपना पुराना बैंक खाता बंद \u095bरा दिया।"))  # precomposed ज़
    mk("hi_final", hi("मैंने कल अपना पुराना बैंक खाता बंद करवा दिया|"))  # ASCII pipe, not U+0964
    mk("hi_final", hi("मैंने कल अपना पुराना बैंक खाता बंद करवा दिया"))
    mk("hi_final", hi("मैंने कल? अपना पुराना बैंक खाता बंद करवा दिया।"))  # two sentences in one entry
    mk("hi_space", hi("मैंने कल अपना पुराना बैंक खाता बंद करवा दिया ।"))
    mk("en_text", set_(["sentences", 0, "en_text"], "Yesterday I closed my खाता."))
    mk("source_empty", set_(["source", "corpus"], ""))
    mk("source_synthetic", set_(["source", "model"], "bulbul:v3"))
    mk("source_interval", lambda r: r["source"].update(end=r["source"]["end"] + 1))
    r = mk("source_reuse", dur=1.5, sents=(S2,))
    r["source"] = SYN("tts/rec-400001.wav", 1.5)
    out[-1] = ("source_reuse", json.dumps(r, ensure_ascii=False))
    mk("time_precision", lambda r: r["hi_words"][4].update(end=r["hi_words"][4]["end"] + 0.0001))
    return out


def run(lines, root, **kw):
    viol, hours = validate.validate(lines, root, **kw)
    by_line = {}
    for n, _, rule, _ in viol:
        by_line.setdefault(n, set()).add(rule)
    return viol, by_line, hours


def main():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        for r in VALID:
            wav(root / r["audio"], r["duration"])
        good = [json.dumps(r, ensure_ascii=False) for r in VALID]

        viol, _, hours = run(good, root)
        assert not viol, viol
        assert abs(hours["real"] * 3600 - 3.0) < 1e-9 and abs(hours["synthetic"] * 3600 - 3.0) < 1e-9, hours

        bad = broken(root)
        lines = good + [line for _, line in bad]
        _, by_line, _ = run(lines, root)
        assert all(n not in by_line for n in range(1, len(good) + 1)), by_line
        for i, (rule, _) in enumerate(bad, len(good) + 1):
            assert by_line.get(i) == {rule}, f"line {i}: expected {{{rule!r}}}, got {by_line.get(i)}"
        caught = {rule for rule, _ in bad}

        validate.MAX_HOURS, cap = 1 / 3600, validate.MAX_HOURS  # 1 s cap: the 6 s valid set must trip it
        viol, _, _ = run(good, root)
        validate.MAX_HOURS = cap
        assert [v[2] for v in viol] == ["hours_cap"], viol
        caught.add("hours_cap")

        # stage files: absent fields skipped, `_` keys allowed; the same record fails as a release record
        stage = copy.deepcopy(VALID[0])
        del stage["translation_method"], stage["sentences"][0]["en_text"]
        stage["_qa"] = {"asr_cer": 0.1}
        stage_line = [json.dumps(stage, ensure_ascii=False)]
        assert not run(stage_line, root, partial=True)[0]
        assert {v[2] for v in run(stage_line, root)[0]} == {"schema"}

        # CLI: exit codes and one line per violation
        m = root / "good.jsonl"; m.write_text("\n".join(good) + "\n", encoding="utf-8")
        p = subprocess.run([sys.executable, HERE / "validate.py", m], capture_output=True, text=True)
        assert p.returncode == 0 and p.stdout == "", p
        m = root / "bad.jsonl"; m.write_text("\n".join(lines) + "\n", encoding="utf-8")
        p = subprocess.run([sys.executable, HERE / "validate.py", m, "--audio-dir", root], capture_output=True, text=True)
        assert p.returncode == 1 and len(p.stdout.splitlines()) == len(bad), p.stdout

        rules = {"json", "id", "id_unique", "schema", "audio_path", "audio_format", "audio_channels", "audio_rate",
                 "audio_depth", "duration_max", "duration_match", "words_match", "word_times", "sentence_times",
                 "hi_chars", "hi_nfc", "hi_final", "hi_space", "en_text", "source_empty", "source_synthetic",
                 "source_interval", "source_reuse", "time_precision", "hours_cap"}
        assert caught == rules, rules ^ caught
    print(f"ok: valid set passes; {len(bad)} broken records + the hours cap caught, each by exactly its rule ({len(rules)} rules)")


if __name__ == "__main__":
    main()
