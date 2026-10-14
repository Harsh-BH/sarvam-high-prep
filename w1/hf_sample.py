"""Measure gated HF parquet corpora on samples: ranged reads of a few row groups, nothing written to disk.

    uv run --no-project --with pyarrow --with huggingface_hub --with numpy python -I w1/hf_sample.py \
        datasets/ARTPARK-IISc/Vaani-transcription-part/audio/Hindi/train-00000-of-00177.parquet [more shards] \
        --row-groups 3 --out data/w1/hf/vaani_tr.jsonl

Token: HF_TOKEN from .env (never printed). Per row it keeps every non-audio column plus the audio's format,
sample rate, channels and duration (parsed from the embedded bytes, ffprobe for non-WAV), and the audio `path`.

--columns audio.path,transcript,...: read only those column chunks of EVERY row group (exact byte ranges, no
audio bytes), shards in parallel. ~0.1-6 MB per shard instead of ~500 MB, so a whole corpus can be indexed.
"""
import argparse, concurrent.futures as cf, io, json, os, subprocess, sys, wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from tar_scan import wav_info  # noqa: E402


def audio_meta(b):
    if b[:4] == b"RIFF":
        info = wav_info(b[:4096])
        if info:
            sr, ch, bits, data = info
            data = min(data, len(b) - 44)  # some writers leave the size field at 0xFFFFFFFF
            return {"fmt": "wav", "sr": sr, "ch": ch, "dur": round(data / (sr * ch * bits / 8), 3)}
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,sample_rate,channels:format=duration",
                        "-of", "json", "-"], input=b, capture_output=True)
    j = json.loads(p.stdout or b"{}")
    s = (j.get("streams") or [{}])[0]
    return {"fmt": s.get("codec_name"), "sr": int(s.get("sample_rate", 0)), "ch": s.get("channels"),
            "dur": float(j.get("format", {}).get("duration", 0) or 0)}


class Ranged(io.RawIOBase):
    """Seekable remote file that fetches exactly the byte ranges asked for; wrap in BufferedReader."""

    def __init__(self, fs, path):
        self.fs, self.path, self.size, self.pos = fs, path, fs.size(path), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        b = self.fs.cat_file(self.path, start=self.pos, end=self.pos + n) if n > 0 else b""
        buf[:len(b)] = b
        self.pos += len(b)
        return len(b)


def read_columns(fs, shard, columns):
    import pyarrow.parquet as pq
    f = io.BufferedReader(Ranged(fs, shard), buffer_size=256 * 1024)
    t = pq.ParquetFile(f, pre_buffer=False).read(columns=columns)
    rows = t.to_pylist()
    for r in rows:
        if isinstance(r.get("audio"), dict):
            r["_audio_path"] = r.pop("audio").get("path")
        r["_shard"] = shard.split("/")[-1]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shards", nargs="+", help="HfFileSystem paths, e.g. datasets/<repo>/<file>.parquet")
    ap.add_argument("--row-groups", type=int, default=2, help="row groups read per shard (spread evenly)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--columns", help="comma-separated; metadata-only mode over all row groups (shard globs allowed)")
    a = ap.parse_args()
    os.environ.setdefault("HF_TOKEN", next(l.split("=", 1)[1].strip().strip('"') for l in open(".env") if l.startswith("HF_TOKEN=")))
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.columns:
        shards = sorted(x for g in a.shards for x in (fs.glob(g) if "*" in g else [g]))
        n = 0
        with open(a.out, "w") as out, cf.ThreadPoolExecutor(int(os.environ.get("HF_THREADS", 8))) as ex:
            for rows in ex.map(lambda sh: read_columns(fs, sh, a.columns.split(",")), shards):
                out.writelines(json.dumps(r, ensure_ascii=False, default=str) + "\n" for r in rows)
                n += len(rows)
        print(f"{n} rows from {len(shards)} shards -> {a.out}", file=sys.stderr)
        return
    n = 0
    with open(a.out, "w") as out:
        for shard in a.shards:
            pf = pq.ParquetFile(fs.open(shard, block_size=8 * 2**20))
            k = pf.num_row_groups
            picks = sorted({round(i * (k - 1) / max(1, a.row_groups - 1)) for i in range(min(a.row_groups, k))})
            print(f"{shard}: {pf.metadata.num_rows} rows in {k} row groups; reading {picks}; schema {pf.schema_arrow.names}", file=sys.stderr)
            for g in picks:
                for row in pf.read_row_group(g).to_pylist():
                    au = row.pop("audio", None) or row.pop("audio_filepath", None) or {}
                    rec = {k2: v for k2, v in row.items() if not isinstance(v, (bytes, bytearray))}
                    rec["_shard"], rec["_rg"] = shard.split("/")[-1], g
                    if isinstance(au, dict) and au.get("bytes"):
                        rec["_audio_path"] = au.get("path")
                        rec.update({f"_{k2}": v for k2, v in audio_meta(au["bytes"]).items()})
                    out.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    n += 1
    print(f"{n} rows -> {a.out}", file=sys.stderr)


if __name__ == "__main__":
    hdr = b"RIFF" + (36 + 32000).to_bytes(4, "little") + b"WAVE" + b"fmt " + (16).to_bytes(4, "little") + \
        (1).to_bytes(2, "little") + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little") + \
        (2).to_bytes(2, "little") + (16).to_bytes(2, "little") + b"data" + (32000).to_bytes(4, "little")
    assert audio_meta(hdr + bytes(32000)) == {"fmt": "wav", "sr": 16000, "ch": 1, "dur": 1.0}
    main()
