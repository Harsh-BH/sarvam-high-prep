"""Build the W3 TTS bake-off probe set: w3/probe.tsv -> data/w3/probe.jsonl.

    uv run --no-project python w3/probe.py

Markup in probe.tsv: {देवनागरी|latin} marks English words, [..] marks an entity span that must survive.
Commas are pause hints for the TTS input only; they are not words, so the transcript drops them.
Per item: tts_dev (TTS input, English in Devanagari), tts_lat (English in Latin), hi_text (SPEC §2 transcript,
checked with validate.py), sentences, entities, en_words.
"""
import json
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from validate import check_record  # noqa: E402

EN = re.compile(r"\{([^|{}]+)\|([^{}]+)\}")
ENT = re.compile(r"\[([^\[\]]+)\]")
SENT_END = re.compile(r"(?<=[।?!])\s+")


def nfc(s):
    return unicodedata.normalize("NFC", s)


def transcript(s):
    """TTS input -> SPEC transcript: no commas, single spaces."""
    return " ".join(s.replace(",", " ").split())


def parse(line):
    pid, domain, tags, marked = line.rstrip("\n").split("\t")
    marked = nfc(marked)
    no_ent = ENT.sub(r"\1", marked)
    dev = EN.sub(r"\1", no_ent)
    lat = EN.sub(r"\2", no_ent)
    hi = transcript(dev)
    item = {
        "pid": pid, "domain": domain, "tags": tags.split(),
        "tts_dev": dev, "tts_lat": lat, "hi_text": hi,
        "sentences": SENT_END.split(hi),
        "entities": [transcript(EN.sub(r"\1", e)) for e in ENT.findall(marked)],
        "en_words": [transcript(d) for d, _ in EN.findall(marked)],
        "chars": len(hi.replace(" ", "")),
    }
    errs = check_record({"id": "rec-000000", "sentences": [{"hi_text": t} for t in item["sentences"]]}, ".", True)
    assert not errs, (pid, errs)
    assert not re.search(r"[\[\]{}|]", dev + lat), (pid, "unbalanced markup")
    return item


def main():
    out = ROOT / "data/w3/probe.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    items = [parse(l) for l in open(ROOT / "w3/probe.tsv", encoding="utf-8") if l.strip() and not l.startswith("#")]
    assert len({i["pid"] for i in items}) == len(items)
    with open(out, "w", encoding="utf-8") as f:
        for i in items:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    c = sorted(i["chars"] for i in items)
    print(f"{len(items)} items -> {out}; chars p10/50/90 {c[len(c)//10]}/{c[len(c)//2]}/{c[9*len(c)//10]}, "
          f"total {sum(c)}; entities {sum(len(i['entities']) for i in items)}; "
          f"English spans {sum(len(i['en_words']) for i in items)}")


if __name__ == "__main__":
    t = parse("x\td\tt\tमैंने {ऑर्डर|order} [साढ़े तीन लाख], नहीं, [{पी एन आर|PNR}] दिया।")
    assert t["hi_text"] == "मैंने ऑर्डर साढ़े तीन लाख नहीं पी एन आर दिया।", t
    assert t["tts_lat"] == "मैंने order साढ़े तीन लाख, नहीं, PNR दिया।" and t["entities"] == ["साढ़े तीन लाख", "पी एन आर"]
    main()
