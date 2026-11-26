"""Web-app plumbing that doesn't need a model: file containment, upload limits, the grid cache key."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

from beatagent import server


def _wav(seconds=0.5, sr=22050) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(seconds * sr)), sr, format="WAV")
    return buf.getvalue()


def test_files_stay_inside_runtime(monkeypatch, tmp_path):
    rt = tmp_path / "runtime"
    (rt / "sync").mkdir(parents=True)
    (rt / "sync" / "mix.wav").write_bytes(b"ok")
    (tmp_path / "runtime_old").mkdir()
    (tmp_path / "runtime_old" / "secret.txt").write_text("no")
    monkeypatch.setattr(server, "RUNTIME", rt)
    assert str(server.files("sync/mix.wav").path).endswith("mix.wav")
    for bad in ("../runtime_old/secret.txt", "../../etc/passwd", "sync"):  # sibling with a shared prefix, escape, a folder
        with pytest.raises(HTTPException):
            server.files(bad)


def test_upload_never_shadows_a_pack_track_and_has_a_size_cap(monkeypatch, tmp_path):
    pack, uploads = tmp_path / "pack", tmp_path / "uploads"
    pack.mkdir()
    (pack / "A1.wav").write_bytes(_wav())
    monkeypatch.setattr(server, "PACK", pack)
    monkeypatch.setattr(server, "UPLOADS", uploads)
    c = TestClient(server.app)
    r = c.post("/api/upload", files={"file": ("A1.wav", _wav(), "audio/wav")})
    assert r.status_code == 200 and r.json()["id"] == "A1-upload"
    assert server._library()["A1"] == pack / "A1.wav"
    r = c.post("/api/upload", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400
    r = c.post("/api/upload", files={"file": ("noise.wav", b"not audio at all", "audio/wav")})
    assert r.status_code == 400 and not (uploads / "noise.wav").exists()
    monkeypatch.setattr(server, "MAX_UPLOAD_MB", 0)
    r = c.post("/api/upload", files={"file": ("big.wav", _wav(), "audio/wav")})
    assert r.status_code == 413 and not (uploads / "big.wav").exists()


def test_grid_cache_key_follows_the_file_not_its_name(tmp_path):
    p = tmp_path / "song.wav"
    p.write_bytes(_wav(0.5))
    k1 = server._grid_key(p)
    p.write_bytes(_wav(0.7))  # a different file uploaded under the same name
    assert server._grid_key(p) != k1
