"""Render a recreation page to WAV in headless Chromium (OfflineAudioContext), so it can be measured."""
from __future__ import annotations

import base64
from pathlib import Path

import numpy as np

from ..audio_io import save


def render_page(html: Path, out_wav: Path, loops: int = 2, sample_rate: int = 44100, timeout_s: float = 120) -> dict:
    from playwright.sync_api import sync_playwright

    html = Path(html).resolve()
    requests, errors, console = [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.on("request", lambda r: requests.append(r.url))
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: console.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        page.goto(html.as_uri())
        page.wait_for_function("window.BeatAgentPage !== undefined", timeout=timeout_s * 1000)
        res = page.evaluate("async ([l, s]) => await window.BeatAgentPage.renderOffline(l, s)", [loops, sample_rate])
        browser.close()
    left = np.frombuffer(base64.b64decode(res["left"]), dtype="<i2").astype(np.float32) / 32768.0
    right = np.frombuffer(base64.b64decode(res["right"]), dtype="<i2").astype(np.float32) / 32768.0
    y = np.stack([left, right], axis=1)
    save(out_wav, y, sample_rate, "WAV", "PCM_16")
    external = [u for u in requests if not u.startswith("file:")]
    return {"wav": str(out_wav), "seconds": round(len(y) / sample_rate, 3), "loop_seconds": res["loopSeconds"],
            "loops": loops, "peak": round(float(np.abs(y).max()), 4), "network_requests": external,
            "page_errors": errors, "console": console[:10]}
