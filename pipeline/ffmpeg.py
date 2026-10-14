"""The one ffmpeg every cut and rebuild uses (design §4.1). Byte-identical audio depends on it."""
import functools, hashlib, subprocess, tarfile, wave
from pathlib import Path
import s3 as store

TOOLS = Path(__file__).resolve().parent.parent / "data/tools"
# BtbN FFmpeg-Builds month-end release (kept long-term), static libs + --enable-libsoxr, glibc dynamic:
# https://github.com/BtbN/FFmpeg-Builds/releases/download/autobuild-2026-09-30-13-08/ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0.tar.xz
VERSION = "n9.0.2-17-g2a571b6068-20260930"
TOOL_KEY = "raw/_tools/ffmpeg-n9.0.2-17-g2a571b6068-linux64-gpl-9.0.tar.xz"
TOOL_SHA256 = "68ee646831adaae2495618346f3bba94ff207ff83bbd34d643e7004730d66269"  # = BtbN's checksums.sha256
RATE = 24000


@functools.cache  # hash the 151 MB tarball once per process, not once per cut
def binary(s3=None):
    tgz = TOOLS / Path(TOOL_KEY).name
    if not tgz.exists():
        TOOLS.mkdir(parents=True, exist_ok=True)
        tgz.write_bytes(store.get(TOOL_KEY, s3))
    if hashlib.sha256(tgz.read_bytes()).hexdigest() != TOOL_SHA256:
        raise SystemExit(f"{tgz}: sha256 mismatch, refusing to cut with an unpinned ffmpeg")
    exe = TOOLS / tgz.name.split(".tar")[0] / "bin/ffmpeg"
    if not exe.exists():
        with tarfile.open(tgz) as t:  # only the binary: the docs, ffprobe and ffplay add ~350 MB
            t.extract(exe.relative_to(TOOLS).as_posix(), TOOLS, filter="data")
    return exe


def cut(src, start, end, out, s3=None):
    """Same arithmetic as w1/cut_pilot.py cut(): resample the whole file, keep 24 kHz samples [a, b)."""
    a, b = round(start * 1000) * RATE // 1000, round(end * 1000) * RATE // 1000
    subprocess.run([str(binary(s3)), "-nostdin", "-v", "error", "-y", "-i", str(src), "-af",
                    f"aresample={RATE}:resampler=soxr,atrim=start_sample={a}:end_sample={b}",
                    "-ac", "1", "-c:a", "pcm_s16le", "-fflags", "+bitexact", "-flags:a", "+bitexact", str(out)], check=True)
    with wave.open(str(out)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()) == (RATE, 1, 2, b - a), out
