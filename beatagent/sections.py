"""Measured section analysis, so subjective requests ("the boring part") are grounded in data.

Per bar we measure the energy in five bands (sub/kick, bass/low-mid, mid, presence, air).
Instruments entering or leaving change that profile, so section boundaries are where the
4-bar-block average jumps (relative to the track's own typical bar-to-bar change). Each
section gets plain observations (no kick, thin arrangement, very repetitive, intro/outro).
The agent shows these numbers and asks the user to choose; it never decides what is
"boring" on its own.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .grid import BeatGrid

BANDS = [("sub", 20, 150), ("low-mid", 150, 600), ("mid", 600, 3000), ("presence", 3000, 8000), ("air", 8000, 16000)]


def bar_features(x: np.ndarray, sr: int, grid: BeatGrid) -> tuple[np.ndarray, list[tuple[int, float, float]]]:
    mono = x.mean(axis=1) if x.ndim > 1 else x
    hi = min(16000, sr / 2 - 500)
    ys = [signal.sosfilt(signal.butter(4, (lo, min(h, hi)), "bandpass", fs=sr, output="sos"), mono) for _, lo, h in BANDS]
    bars = grid.bars
    F, info = [], []
    for bar in range(1, grid.last_bar + 1):
        idx = np.flatnonzero(bars == bar)
        if len(idx) == 0:
            continue
        t0, t1 = float(grid.times[idx[0]]), float(grid.beat_pos_time(idx[-1] + 1))
        a, b = int(t0 * sr), int(t1 * sr)
        if b - a < 64:
            continue
        F.append([10 * np.log10(np.mean(y[a:b] ** 2) + 1e-12) for y in ys])
        info.append((bar, t0, t1))
    return np.array(F), info


def find_sections(x: np.ndarray, sr: int, grid: BeatGrid, phrase: int = 4, min_bars: int = 4) -> list[dict]:
    F, info = bar_features(x, sr, grid)
    if len(F) == 0:
        return []
    Z = (F - F.mean(0)) / (F.std(0) + 1e-9)
    nb = len(F)
    blocks = [(i, min(i + phrase, nb)) for i in range(0, nb, phrase)]
    means = np.array([Z[a:b].mean(0) for a, b in blocks])
    d = np.linalg.norm(np.diff(means, axis=0), axis=1) if len(means) > 1 else np.zeros(0)
    thr = max(1.5, 4 * float(np.median(d))) if len(d) else np.inf
    cuts = [0] + [blocks[i + 1][0] for i in np.flatnonzero(d >= thr)] + [nb]
    segs = [(a, b) for a, b in zip(cuts, cuts[1:]) if b > a]
    merged: list[tuple[int, int]] = []
    for s in segs:  # a section shorter than min_bars joins the neighbour it resembles more
        if merged and s[1] - s[0] < min_bars:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(s)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_bars:
        merged[1] = (merged[0][0], merged[1][1])
        merged.pop(0)
    med = np.median(F, axis=0)
    out = []
    for a, b in merged:
        f = F[a:b].mean(0) - med
        var = float(np.linalg.norm(np.diff(Z[a:b], axis=0), axis=1).mean()) if b - a > 1 else 0.0
        sec = {
            "bars": [info[a][0], info[b - 1][0]], "time_s": [round(info[a][1], 3), round(info[b - 1][2], 3)],
            "n_bars": b - a,
            "band_energy_db_vs_track": {name: round(float(v), 1) for (name, _, _), v in zip(BANDS, f)},
            "bar_to_bar_variation": round(var, 2),
        }
        tags = []
        if f[0] < -9:
            tags.append("no kick/bass (sub band far below the rest of the track)")
        if np.mean(f[1:3]) < -6:
            tags.append("thin arrangement (little low-mid and mid content: few instruments)")
        if np.mean(f) > 2:
            tags.append("fullest/loudest part")
        if var < 0.25:
            tags.append("very repetitive (bars nearly identical)")
        if a == 0:
            tags.append("intro")
        if b == nb:
            tags.append("outro")
        sec["observations"] = tags
        out.append(sec)
    return out
