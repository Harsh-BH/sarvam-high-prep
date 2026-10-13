"""Stream a remote .tar.gz once: log every member, extract all non-audio files, keep a sample of WAVs.

    uv run python -I w1/tar_scan.py URL OUT_DIR [--keep-every N] [--cap-gb G]

The server can't seek inside gzip, so the transcripts at the tail are only reachable by
streaming the whole archive. Audio is discarded unless crc32(name) % N == 0 and under the cap.
Writes OUT_DIR/members.tsv: name, bytes, sample_rate, channels, bits, seconds, kept.
"""
import argparse, shutil, struct, sys, tarfile, urllib.request, zlib
from pathlib import Path


def wav_info(head: bytes):
    """(sample_rate, channels, bits, data_bytes) from a RIFF header, or None."""
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return None
    i, fmt, data = 12, None, None
    while i + 8 <= len(head):
        cid, size = head[i:i + 4], struct.unpack("<I", head[i + 4:i + 8])[0]
        if cid == b"fmt ":
            _, ch, sr, _, _, bits = struct.unpack("<HHIIHH", head[i + 8:i + 24])
            fmt = (sr, ch, bits)
        elif cid == b"data":
            data = size
            break
        i += 8 + size + (size & 1)
    return (*fmt, data) if fmt and data is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--keep-every", type=int, default=8)
    ap.add_argument("--cap-gb", type=float, default=1.5)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    cap, kept_bytes = a.cap_gb * 1e9, 0
    log = open(a.out / "members.tsv", "w")
    log.write("name\tbytes\tsample_rate\tchannels\tbits\tseconds\tkept\n")
    req = urllib.request.Request(a.url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for m in tar:
            if not m.isfile():
                continue
            f = tar.extractfile(m)
            dest = a.out / m.name
            if not m.name.lower().endswith(".wav"):
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as o:
                    shutil.copyfileobj(f, o)
                log.write(f"{m.name}\t{m.size}\t\t\t\t\t1\n")
                continue
            head = f.read(4096)
            info = wav_info(head)
            sr, ch, bits, data = info if info else ("", "", "", 0)
            secs = round(data / (sr * ch * bits / 8), 3) if info else ""
            keep = zlib.crc32(m.name.encode()) % a.keep_every == 0 and kept_bytes + m.size <= cap
            if keep:
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as o:
                    o.write(head)
                    shutil.copyfileobj(f, o)
                kept_bytes += m.size
            log.write(f"{m.name}\t{m.size}\t{sr}\t{ch}\t{bits}\t{secs}\t{int(keep)}\n")
            log.flush()
    print(f"done; kept {kept_bytes / 1e9:.2f} GB", file=sys.stderr)


if __name__ == "__main__":
    # self-check of the header parser on a synthetic 1 s, 16 kHz mono WAV
    hdr = b"RIFF" + struct.pack("<I", 36 + 32000) + b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, 16000, 32000, 2, 16) + b"data" + struct.pack("<I", 32000)
    assert wav_info(hdr) == (16000, 1, 16, 32000)
    main()
