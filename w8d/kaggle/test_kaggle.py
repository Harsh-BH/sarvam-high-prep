"""Checks for the Kaggle run (no Kaggle, no AWS): the train selection's guards, that a failed download never echoes
its presigned URL, the generated kernel, the arm schedule and the paired bootstrap.
    python3 w8d/kaggle/test_kaggle.py
"""
import base64, functools, http.server, io, re, sys, tarfile, tempfile, threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ctl, kernel  # noqa: E402


def rec(i, spk="S1", voice=None, model=None, synthetic=False, pid=None):
    r = {"id": i, "audio": f"audio/{i}.wav", "duration": 10.0, "speaker_id": f"spk-indicvoices-{spk}",
         "source": {"synthetic": synthetic, "voice": voice, "model": model}, "sentences": [{"en_text": "Hi."}, {"en_text": "Bye."}]}
    return r | ({"_w2": {"pid": pid}} if pid else {})


def main():
    ok = [rec("rec-000001"), rec("rec-400100", "x", "iv-S9", "k2-fsa/OmniVoice", True, "p999")]
    held = [rec("rec-400002"),                                                # dev id
            rec("rec-000002", "S4256027900336560"),                           # held-out real speaker
            rec("rec-400101", "x", "iv-S4256027900336560", "k2-fsa/OmniVoice", True),  # clone of that speaker
            rec("rec-400102", "x", "kavya", "bulbul:v3", True),               # held-out voice
            rec("rec-400103", "x", "iv-S9", "k2-fsa/OmniVoice", True, "p004"),  # held-out script
            rec("rec-400051", "x", "iv-S9", "k2-fsa/OmniVoice", True)]        # burned id
    rows = ctl.select([("stages/tts/r/audio/", ok + held)])
    assert [r["id"] for r in rows] == ["rec-000001", "rec-400100"], rows
    assert rows[1] == {"id": "rec-400100", "en": "Hi. Bye.", "duration": 10.0, "synthetic": True, "key": "stages/tts/r/audio/rec-400100.wav"}
    for bad in ([("a/", [ok[0]]), ("b/", [ok[0]])], [("a/", [rec("rec-000003") | {"sentences": [{"en_text": ""}]}])]):
        try:
            ctl.select(bad)
            raise AssertionError("select accepted a duplicate id / a missing en_text")
        except SystemExit:
            pass

    with tempfile.TemporaryDirectory() as d:  # one good URL, one 404 whose path stands in for a signature
        (Path(d) / "srv").mkdir()
        (Path(d) / "srv/a.wav").write_bytes(b"RIFF\0\0\0\0WAVEfmt ")
        quiet = type("Q", (http.server.SimpleHTTPRequestHandler,), {"log_message": lambda *a: None})
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(quiet, directory=str(Path(d) / "srv")))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            kernel.fetch(["a", "b"], {"a": f"{base}/a.wav", "b": f"{base}/X-Amz-Signature=SECRET"}, Path(d) / "out", tries=1)
            raise AssertionError("fetch ignored a failed download")
        except SystemExit as e:
            assert "SECRET" not in str(e) and "127.0.0.1" not in str(e) and "b: HTTPError 404" in str(e), str(e)
        assert (Path(d) / "out/a.wav").read_bytes()[:4] == b"RIFF" and not (Path(d) / "out/b.wav").exists()
        srv.shutdown()
        logs = Path(d) / "logs"  # the GPU queue: per-GPU command lists, all must exit 0
        assert kernel.queue([[["true"], ["true"]], [["true"]]], logs) and not kernel.queue([[["true"]], [["false"]]], logs)

    cfg = {"mode": "train", "run": "R", "sha": "abc", "limit": None, "arms": ctl.arms(7901, None), "base_dev": True}
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        t.addfile(tarfile.TarInfo("w8d/kaggle/kernel.py"), io.BytesIO(b""))
    src = ctl.boot(cfg, buf.getvalue())
    compile(src, "boot.py", "exec")
    code = re.search(r"CODE = '([^']*)'", src).group(1)
    assert tarfile.open(fileobj=io.BytesIO(base64.b64decode(code))).getnames() == ["w8d/kaggle/kernel.py"]
    assert eval(re.search(r"CFG = (.*)\n", src).group(1)) == cfg
    assert cfg["arms"]["A"][:2] == ["--steps", "3952"] and cfg["arms"]["B"][:2] == ["--steps", "1976"]  # 4 and 2 epochs of 988 steps

    m, lo, hi = ctl.bootstrap([0.5, 0.6, 0.7], [0.6, 0.7, 0.8])
    assert abs(m - 0.1) < 1e-9 and abs(lo - 0.1) < 1e-9 and abs(hi - 0.1) < 1e-9
    m, lo, hi = ctl.bootstrap([0.0] * 50, [0.1, -0.1] * 25)
    assert abs(m) < 1e-9 and lo < 0 < hi
    print("ok: w8d/kaggle")


if __name__ == "__main__":
    main()
