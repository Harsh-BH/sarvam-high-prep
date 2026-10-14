# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "boto3"]
# ///
"""S1 `cut` stage: stream one IndicVoices Hindi *train* archive (ETag-pinned), cut new single-speaker 10-60 s clips
with W1's cutter and drop rule (w1/cut_pilot.py, pinned ffmpeg), and ingest every source it cuts from, write-once with
a sha256 sidecar, to raw/ (pipeline/ingest_iv.py). Nothing lands on the laptop: run it with --sm.

    uv run pipeline/run.py cut --in stages/dedupe/20261014T083657Z-68a7f35 --sm ml.m5.large --spot -- \\
        --archive v1 --part 0/2 --id-base 1001 --quota Conversation=1.6,Extempore=1.6,Read=0.4

--in is ignored (run.py needs one): w1/ids.tsv is the register of every span ever cut, burned ones included, and a
new clip never overlaps a registered span. Files are taken in archive order. A file is processed if its speaker hashes
to --part, is not held out (an IV-valid speaker, valid_speakers.txt, or a person a W3 voice clones: w8c draws its
held-out persons from those), and its scenario still has --quota hours left; its clips are kept while the speaker stays
under --speaker-cap seconds. Streaming stops once every quota is full. Ids count up from --id-base, a block of BLOCK
per shard so shards run in parallel; OUT/ids_new.tsv holds the rows to append to w1/ids.tsv. OUT/cut_stats.json
counts what was skipped and why. Resumable: a restart re-streams the archive and skips files already done.
"""
import argparse, collections as C, http.client, io, json, re, shutil, sys, tarfile, tempfile, time, urllib.request, zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path[:0] = [str(ROOT / "w1"), str(ROOT / "pipeline"), str(ROOT / "w8c")]
import cut_pilot as W1, ingest_iv  # noqa: E402
from split import is_heldout  # noqa: E402

BLOCK = 1000  # ids per shard; a 3.6 h shard needs ~650
HERE = Path(__file__).resolve().parent


def urlopen(req):
    return urllib.request.urlopen(req, timeout=120)  # a stalled read raises TimeoutError, and Stream reconnects


def held_out():
    """Speaker ids no train cut may use: every IV-valid speaker, and every IV person a W3 voice clones. A superset of
    w8c's held-out persons (drawn from these), so it stays safe if w8c grows its set; is_heldout() is checked too."""
    valid = {l.strip() for l in open(HERE / "valid_speakers.txt") if l.strip() and not l.startswith("#")}
    return valid | {l.split("\t")[0].removeprefix("iv-") for l in open(ROOT / "w3/voices.tsv") if l.startswith("iv-")}


class Stream(io.RawIOBase):
    """A remote file read front to back. After a network error or an early close it reconnects with a Range header
    at the current offset, and every connection must carry the pinned ETag, so a 50 GB stream survives a dropped
    connection without restarting and never mixes bytes of two upstream versions."""

    def __init__(self, url, etag, opener=urlopen, retries=8, sleep=time.sleep):
        self.url, self.etag, self.opener, self.retries, self.sleep = url, etag, opener, retries, sleep
        self.pos, self.size, self.r = 0, None, None

    def readable(self):
        return True

    def _open(self):
        self.r = self.opener(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-"} if self.pos else {}))
        if (got := self.r.headers.get("ETag", "").strip('"')) != self.etag:
            raise SystemExit(f"{self.url}: ETag {got!r} != pinned {self.etag!r}; upstream changed, refusing")
        if self.size is None:
            self.size = int(self.r.headers["Content-Length"])

    def readinto(self, b):
        for attempt in range(self.retries + 1):
            try:
                if self.r is None:
                    self._open()
                n = self.r.readinto(b)
                if n == 0 and len(b) and self.pos < self.size:  # http.client returns 0 on an early close
                    raise ConnectionError(f"closed at {self.pos} of {self.size}")
                self.pos += n
                return n
            except (OSError, http.client.HTTPException) as e:
                self.r = None
                if attempt == self.retries:
                    raise
                print(f"stream: {e!r} at byte {self.pos}, reconnecting", file=sys.stderr)
                self.sleep(2 ** attempt)


def group_key(name):
    """A call ships as <uuid>_0 / <uuid>_1 (json + wav each), next to each other in the sorted archive."""
    return re.sub(r"_[01]$", "", Path(name).stem)


class Shard:
    def __init__(self, out, quota, part, nparts, id_base, speaker_cap, held=frozenset()):
        self.out, self.quota, self.part, self.nparts, self.cap, self.held = out, quota, part, nparts, speaker_cap, held
        self.used = C.defaultdict(list)  # source_file -> registered (start, end) s, never cut again
        for (f, a, b), _ in W1.load_ids().items():
            self.used[f].append((a / 1000, b / 1000))
        assert id_base > max([int(v[4:]) for v in W1.load_ids().values()] or [0]), "id block overlaps w1/ids.tsv"
        self.base, self.ids = id_base, {}  # (source_file, a_ms, b_ms) -> id, this shard's allocations
        reg = out / "ids_new.tsv"
        if reg.exists():
            for l in open(reg):
                rid, f, a, b = l.rstrip("\n").split("\t")
                self.ids[(f, int(a), int(b))] = rid
        self.hours, self.secs, self.done, self.st = C.Counter(), C.Counter(), set(), C.Counter()
        for p in ("records.jsonl", "rejects.jsonl"):
            if (q := out / p).exists() and (b := q.read_bytes()) and not b.endswith(b"\n"):
                q.write_bytes(b[: b.rfind(b"\n") + 1])  # a crash tore the last line: that file is redone
        if (out / "records.jsonl").exists():
            for l in open(out / "records.jsonl", encoding="utf-8"):
                r = json.loads(l)
                self.hours[r["_w1"]["scenario"]] += r["duration"] / 3600
                self.secs[r["speaker_id"]] += r["duration"]
                self.done.add(r["source"]["source_file"])
        if (out / "seen.txt").exists():
            self.done |= {l.strip() for l in open(out / "seen.txt")}

    def full(self):
        return all(self.hours[k] >= q for k, q in self.quota.items())

    def wanted(self, d, source_file):
        """Process this file? d = its IV JSON. Counts the reason when not."""
        kind, spk = d["scenario"], d["speaker_id"]
        why = ("other part" if zlib.crc32(spk.encode()) % self.nparts != self.part else
               "done" if source_file in self.done else
               "held-out speaker (IV valid / W3 voice)" if spk in self.held or is_heldout({"speaker_id": f"spk-indicvoices-{spk}"}) else
               "kind quota full" if self.hours[kind] >= self.quota.get(kind, 0) else
               "speaker cap" if self.secs[f"spk-indicvoices-{spk}"] >= self.cap else None)
        if why:
            self.st[f"file skipped: {why}"] += 1
        return why is None

    def take(self, cands):
        """Candidates of one file (in time order) kept under the kind quota and the speaker cap."""
        out = []
        for c in sorted(cands, key=lambda c: c["a_ms"]):
            d, spk = (c["b_ms"] - c["a_ms"]) / 1000, f"spk-indicvoices-{c['speaker']}"
            if self.hours[c["kind"]] >= self.quota.get(c["kind"], 0) or self.secs[spk] + d > self.cap:
                self.st["clip over quota / speaker cap"] += 1
                continue
            self.hours[c["kind"]] += d / 3600
            self.secs[spk] += d
            out.append(c)
        return out

    def assign(self, cands):
        """Ids from this shard's block; a span allocated before a restart keeps its id."""
        nxt = max([self.base] + [int(v[4:]) + 1 for v in self.ids.values()])
        with open(self.out / "ids_new.tsv", "a") as f:
            for c in cands:
                k = (c["source_file"], c["a_ms"], c["b_ms"])
                if k not in self.ids:
                    assert nxt < self.base + BLOCK, f"id block {self.base}+{BLOCK} exhausted"
                    self.ids[k] = f"rec-{nxt:06d}"
                    f.write(f"{self.ids[k]}\t{k[0]}\t{k[1]}\t{k[2]}\n")
                    nxt += 1
                c["id"] = self.ids[k]


def process(sh, files, tmp, url, etag, put=ingest_iv._put):
    """files: one recording group, member name -> bytes (JSONs always, WAVs only if wanted)."""
    for name, data in files.items():
        (tmp / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp / name).write_bytes(data)
    recs = {Path(r["file"]).stem: r for r in W1.load_iv(tmp)}
    for name in sorted(n for n in files if n.endswith(".wav")):
        stem, sf = Path(name).stem, f"indicvoices/{name}"
        cands, st = W1.file_candidates(stem, recs[stem], recs, tmp, 2.0, True, sh.used[sf])
        sh.st.update(st)
        sh.st[f"processed: {recs[stem]['kind']}"] += 1
        sh.st[f"candidate h: {recs[stem]['kind']}"] += sum(c["b_ms"] - c["a_ms"] for c in cands) / 3.6e6
        if kept := sh.take(cands):
            for m in (name, name[:-4] + ".json"):
                put(m, files[m], url, etag, None)  # write-once + sidecar, before any record names it
            sh.assign(kept)
            lines = [W1.make_record(c, sh.out) for c in kept]
            with open(sh.out / "records.jsonl", "a", encoding="utf-8") as f:
                f.writelines(lines)
        with open(sh.out / "seen.txt", "a") as f:
            f.write(sf + "\n")
        sh.done.add(sf)
    shutil.rmtree(tmp / "Hindi", ignore_errors=True)


def run(sh, archive, tmp, opener=urlopen, put=ingest_iv._put):
    url, etag = W1.IV_URL.format(v=archive, split="train"), W1.IV_REV[archive].strip('"')
    assert url.endswith("_train.tgz")  # never the valid split: w8c's held-out speakers come from it
    stream = Stream(url, etag, opener)
    cur, files, t0 = None, {}, time.time()
    with tarfile.open(fileobj=io.BufferedReader(stream, 1 << 20), mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            if (k := group_key(m.name)) != cur:
                if files:
                    process(sh, files, tmp, url, etag, put)
                    if sh.st["files seen"] % 500 == 0:
                        print(f"{stream.pos / 1e9:.1f}/{stream.size / 1e9:.1f} GB, {time.time() - t0:.0f} s, kept h "
                              f"{ {k: round(v, 2) for k, v in sh.hours.items()} }", file=sys.stderr, flush=True)
                    if sh.full():
                        break
                cur, files = k, {}
            if m.name.endswith(".json"):
                files[m.name] = tar.extractfile(m).read()
                sh.st["files seen"] += 1
            elif m.name.endswith(".wav") and (j := files.get(m.name[:-4] + ".json")) and sh.wanted(json.loads(j), f"indicvoices/{m.name}"):
                files[m.name] = tar.extractfile(m).read()
        else:
            if files:
                process(sh, files, tmp, url, etag, put)
    sh.st.update({"bytes streamed": stream.pos, "archive bytes": stream.size, "seconds": round(time.time() - t0)})
    sh.st.update({f"kept h: {k}": round(v, 4) for k, v in sh.hours.items()})
    (sh.out / "cut_stats.json").write_text(json.dumps(dict(sorted(sh.st.items())), indent=1))
    (sh.out / "rejects.jsonl").touch()  # nothing is rejected here: quota and rule drops are counted in cut_stats.json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", type=Path, required=True, help="ignored (see above)")
    ap.add_argument("--audio-dir", type=Path, required=True, help="ignored")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--archive", required=True, choices=sorted(W1.IV_REV))
    ap.add_argument("--part", default="0/1", help="k/N: this shard takes speakers with crc32 %% N == k")
    ap.add_argument("--id-base", type=int, required=True)
    ap.add_argument("--quota", required=True, help="hours per scenario, e.g. Conversation=1.6,Extempore=1.6,Read=0.4")
    ap.add_argument("--speaker-cap", type=float, default=240.0, help="seconds of clips per speaker in this shard")
    a = ap.parse_args()
    ingest_iv.gate("indicvoices")
    k, n = map(int, a.part.split("/"))
    quota = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.quota.split(",")}
    (a.out / "audio").mkdir(parents=True, exist_ok=True)
    sh = Shard(a.out, quota, k, n, a.id_base, a.speaker_cap, held_out())
    with tempfile.TemporaryDirectory() as tmp:  # not under --out: on SageMaker that is the synced checkpoint dir
        run(sh, a.archive, Path(tmp))
    print(f"cut {a.archive} part {a.part}: {dict(sh.hours)} h; stats in cut_stats.json", file=sys.stderr)


if __name__ == "__main__":
    main()
