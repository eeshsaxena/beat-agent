"""One-time setup: Python packages, headless Chromium, model weights, and the stand-in test pack.

    python scripts/setup.py            (run with the Python 3.12 you want to use; creates .venv)

Idempotent: re-running skips what is already there.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

# Beat This! weights, from the authors' server (https://github.com/CPJKU/beat_this). That server
# throttles each connection to ~15 KB/s, so the 81 MB file is fetched in parallel byte ranges.
CKPT_URL = "https://cloud.cp.jku.at/public.php/dav/files/7ik4RrBKTS273gp/final0.ckpt"
CKPT_SIZE = 81058141


def run(*cmd):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.check_call([str(c) for c in cmd], cwd=ROOT)


def find_python312() -> list[str] | None:
    """The dependency set is pinned and tested on CPython 3.12. Find one if we are not it."""
    if sys.version_info[:2] == (3, 12):
        return [sys.executable]
    import shutil

    candidates = [["py", "-3.12"]] if os.name == "nt" else []
    candidates += [["python3.12"], ["uv", "run", "--python", "3.12", "--no-project", "python"]]
    for c in candidates:
        if shutil.which(c[0]):
            try:
                out = subprocess.run(c + ["-c", "import sys; print(sys.version_info[:2])"], capture_output=True,
                                     text=True, timeout=120)
                if out.returncode == 0 and "(3, 12)" in out.stdout:
                    return c
            except (OSError, subprocess.TimeoutExpired):
                continue
    return None


def fetch_ranges(url: str, size: int, dest: Path, parts: int = 24):
    if dest.exists() and dest.stat().st_size == size:
        print(f"ok: {dest.name} already present")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    chunk = (size + parts - 1) // parts
    bufs: list[bytes | None] = [None] * parts

    def get(i):
        a, b = i * chunk, min(size, (i + 1) * chunk) - 1
        data = b""
        for _ in range(10):  # resume a dropped range where it stopped
            req = urllib.request.Request(url, headers={"Range": f"bytes={a + len(data)}-{b}"})
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    data += r.read()
            except Exception as e:  # noqa: BLE001
                print(f"  range {i}: {e}; retrying", flush=True)
            if len(data) == b - a + 1:
                break
        bufs[i] = data

    print(f"downloading {dest.name} ({size / 1e6:.0f} MB) in {parts} parallel ranges ...", flush=True)
    ts = [threading.Thread(target=get, args=(i,)) for i in range(parts)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    blob = b"".join(b or b"" for b in bufs)
    if len(blob) != size:
        raise SystemExit(f"download incomplete: {len(blob)} of {size} bytes; run setup again")
    dest.write_bytes(blob)


def main():
    if not PY.exists():
        py312 = find_python312()
        if py312 is None:
            raise SystemExit(
                f"Beat Agent needs Python 3.12 (this is {sys.version.split()[0]}). Install it (python.org, pyenv, "
                "or `uv python install 3.12`) and re-run, or use Docker: `docker compose up`.")
        run(*py312, "-m", "venv", VENV)
    run(PY, "-m", "pip", "install", "-q", "--upgrade", "pip")
    # CPU-only torch first, from the PyTorch index (otherwise Linux may pull the multi-GB CUDA build)
    run(PY, "-m", "pip", "install", "-q", "torch==2.15.0", "torchaudio==2.11.1",
        "--index-url", "https://download.pytorch.org/whl/cpu")
    run(PY, "-m", "pip", "install", "-q", "-r", ROOT / "requirements.txt")
    run(PY, "-m", "pip", "install", "-q", "--no-deps", "basic-pitch==0.4.0")
    run(PY, "-m", "playwright", "install", "chromium")
    hub = Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub" / "checkpoints"
    fetch_ranges(CKPT_URL, CKPT_SIZE, hub / "beat_this-final0.ckpt")
    run(PY, "-c", "from beat_this.inference import Audio2Frames; Audio2Frames('final0', 'cpu'); print('ok: Beat This! loads')")
    run(PY, "-c", "from demucs.pretrained import get_model; get_model('htdemucs'); print('ok: Demucs htdemucs loads')")
    if not (ROOT / "testpack" / "stand-in" / "manifest.json").exists():
        run(PY, "-m", "testpack.generate")
    print("\nSetup complete.")


if __name__ == "__main__":
    main()
