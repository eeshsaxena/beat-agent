"""Click audition: the original audio with a click on every beat, accented on downbeats."""
from __future__ import annotations

import numpy as np

from .grid import BeatGrid


def _click(sr: int, freq: float, dur: float = 0.03, decay: float = 0.008) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    return (np.sin(2 * np.pi * freq * t) * np.exp(-t / decay)).astype(np.float32)


def click_track(grid: BeatGrid, n_samples: int, sr: int) -> np.ndarray:
    beat = _click(sr, 1000.0) * 0.5
    accent = _click(sr, 2000.0, 0.04, 0.012) * 1.0
    out = np.zeros(n_samples, dtype=np.float32)
    for t, pos in zip(grid.times, grid.beat_in_bar):
        c = accent if pos == 1 else beat
        i = int(round(t * sr))
        if 0 <= i < n_samples:
            e = min(n_samples, i + len(c))
            out[i:e] += c[: e - i]
    return out


def audition(samples: np.ndarray, sr: int, grid: BeatGrid, music_gain_db: float = -4.0) -> np.ndarray:
    """Mix clicks over the music (both channels). Scaled down if the mix would clip."""
    clicks = click_track(grid, len(samples), sr)
    mix = samples * 10 ** (music_gain_db / 20) + clicks[:, None] * 0.6
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix.astype(np.float32)
