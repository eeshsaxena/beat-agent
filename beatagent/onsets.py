"""High-resolution percussive onset detection (measurement, not a model).

Used to (1) put the beat grid on the physical attack of the beat-carrying hits,
(2) measure kick alignment in sync renders, and (3) measure onset timing of
Web Audio renders against the original.

Method: band-split the signal, take the energy envelope in ~1.5 ms frames,
and pick peaks of its positive slope. Onset time is the frame where the
slope peaks, which for fast percussive attacks is within ~1-2 ms of the true
start (measured on synthetic hits in evals/onset_calibration.json).
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from .audio_io import resample

_SR = 22050
HOP = 32  # 1.45 ms at 22.05 kHz

BANDS = {
    "low": ("lowpass", 150.0),
    "mid": ("bandpass", (150.0, 2500.0)),
    "high": ("highpass", 2500.0),
}


def _env(x: np.ndarray, sr: int, kind, freq, causal: bool = False) -> np.ndarray:
    sos = signal.butter(4, freq, btype=kind, fs=sr, output="sos")
    # zero-phase by default (no group delay); causal when pre-ringing ahead of an attack matters
    y = signal.sosfilt(sos, x) if causal else signal.sosfiltfilt(sos, x)
    # smooth the power over a window spanning the band's lowest content; unsmoothed 1.5 ms
    # frames on the low band ripple at twice a kick's pitch and fire several onsets per kick
    win_ms = {"lowpass": 12.0, "bandpass": 4.0, "highpass": 2.0}[kind]
    w = np.hanning(int(win_ms * 1e-3 * sr) + 2)[1:-1]
    p = np.convolve(y ** 2, w / w.sum(), mode="same")  # centred, so the smoothing adds no delay
    y = np.sqrt(np.maximum(p, 0.0))
    e = y ** 2
    n = len(e) // HOP
    e = e[: n * HOP].reshape(n, HOP).sum(axis=1)
    return np.sqrt(e)


def onset_function(x: np.ndarray, sr: int, bands=("low", "mid", "high"), causal: bool = False) -> tuple[np.ndarray, float]:
    """Return (novelty curve, frame rate). Frame i covers samples [i*HOP, (i+1)*HOP)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = resample(x.astype(np.float32), sr, _SR) if sr != _SR else x.astype(np.float32)
    nov = None
    for b in bands:
        kind, freq = BANDS[b]
        env = _env(x, _SR, kind, freq, causal)
        env = env / (np.percentile(env, 99) + 1e-9)
        d = np.maximum(0.0, np.diff(env, prepend=env[:1]))
        nov = d if nov is None else nov + d
    return nov, _SR / HOP


def detect(x: np.ndarray, sr: int, bands=("low", "mid", "high"), min_gap_s: float = 0.03,
           rel_threshold: float = 0.08) -> np.ndarray:
    """Onset times in seconds."""
    nov, fps = onset_function(x, sr, bands)
    if nov.max() <= 0:
        return np.zeros(0)
    # adaptive threshold: local median over ~0.5 s plus a fraction of the global peak level
    med = signal.medfilt(nov, 2 * int(0.25 * fps) + 1) if len(nov) > 64 else np.zeros_like(nov)
    thr = med + rel_threshold * np.percentile(nov, 99.5)
    peaks, _ = signal.find_peaks(nov, height=thr, distance=max(1, int(min_gap_s * fps)))
    # parabolic interpolation for sub-frame precision; frame i spans [i, i+1) hops, slope at its centre
    t = peaks.astype(float)
    ok = (peaks > 0) & (peaks < len(nov) - 1)
    p = peaks[ok]
    a, b, c = nov[p - 1], nov[p], nov[p + 1]
    den = a - 2 * b + c
    t[ok] += np.where(np.abs(den) > 1e-12, 0.5 * (a - c) / den, 0.0)
    return (t + 0.5) * HOP / _SR - 0.5 * HOP / _SR


def detect_kicks(x: np.ndarray, sr: int, rise: float = 2.0, level: float = 0.2) -> np.ndarray:
    """Kick onsets: broadband onset times (precise) filtered to those with a strong
    low-band (<150 Hz) energy rise (classification only, so filter smear does not
    bias the timing)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    on = detect(x, sr)
    if len(on) == 0:
        return on
    y = resample(x.astype(np.float32), sr, _SR) if sr != _SR else x.astype(np.float32)
    low = _env(y, _SR, "lowpass", 150.0)
    fps = _SR / HOP
    ref = np.percentile(low, 99) + 1e-9
    keep = []
    for t in on:
        i = int(t * fps)
        pre = low[max(0, i - int(0.045 * fps)):max(1, i - int(0.008 * fps))]
        post = low[i:i + int(0.045 * fps)]
        if len(post) == 0:
            continue
        p0 = float(pre.mean()) if len(pre) else 0.0
        if post.max() > rise * p0 and post.max() > level * ref:
            keep.append(t)
    return np.array(keep)


def kick_attacks_near(x: np.ndarray, sr: int, beats: np.ndarray, half: float = 0.04, cutoff: float = 150.0,
                      level: float = 0.25) -> np.ndarray:
    """Kick attack time near each beat (NaN where there is no kick).

    Attack = steepest rise of the low-band (<150 Hz) amplitude envelope within +-`half`
    of the beat. Every track measured this way goes through the identical zero-phase
    filter, so filter delay cancels when two tracks' kicks are compared."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    y = resample(x.astype(np.float32), sr, _SR) if sr != _SR else x.astype(np.float32)
    lp = signal.sosfiltfilt(signal.butter(4, cutoff, "lowpass", fs=_SR, output="sos"), y)
    env = np.abs(signal.hilbert(lp))
    k = int(0.001 * _SR) | 1
    env = np.convolve(env, np.hanning(k + 2)[1:-1] / np.hanning(k + 2)[1:-1].sum(), mode="same")
    slope = np.diff(env, prepend=env[:1])
    ref = np.percentile(env, 99.5) + 1e-9
    out = np.full(len(beats), np.nan)
    h = int(half * _SR)
    for i, b in enumerate(beats):
        c = int(round(b * _SR))
        lo, hi = max(1, c - h), min(len(env) - 1, c + h)
        if hi - lo < 8 or env[lo:hi + h // 2].max() < level * ref:
            continue
        j = lo + int(np.argmax(slope[lo:hi]))
        a, bb, cc = slope[j - 1], slope[j], slope[j + 1]
        den = a - 2 * bb + cc
        frac = 0.5 * (a - cc) / den if abs(den) > 1e-12 else 0.0
        out[i] = (j + np.clip(frac, -0.5, 0.5)) / _SR
    return out


def kick_onsets_near(x: np.ndarray, sr: int, beats: np.ndarray, half: float = 0.04) -> np.ndarray:
    """Physical start of the kick near each beat (NaN where there is no kick).

    The low-band attack (above) finds the kick but lands 6-12 ms after its start, by an
    amount that depends on the kick's pitch sweep. The start itself is the earliest
    broadband transient in the 25 ms before that attack (the beater click), which does
    not depend on the kick's timbre."""
    att = kick_attacks_near(x, sr, beats, half)
    # high band, causal filter: zero-phase filters with a low edge pre-ring ~17 ms ahead of a
    # kick, and the beater click (the physical start) lives above 2.5 kHz anyway
    nov, fps = onset_function(x, sr, bands=("high",), causal=True)
    out = np.full(len(beats), np.nan)
    for i, a in enumerate(att):
        if np.isnan(a):
            continue
        lo, hi = int((a - 0.025) * fps), int((a + 0.003) * fps) + 1
        lo = max(lo, 1)
        hi = min(hi, len(nov) - 1)
        seg = nov[lo:hi]
        if len(seg) < 3 or seg.max() <= 0:
            out[i] = a
            continue
        pk, _ = signal.find_peaks(np.concatenate([[0.0], seg, [0.0]]), height=0.3 * seg.max())
        j = lo + int(pk[0]) - 1 if len(pk) else lo + int(np.argmax(seg))
        p, q, r = nov[j - 1], nov[j], nov[j + 1]
        den = p - 2 * q + r
        frac = 0.5 * (p - r) / den if abs(den) > 1e-12 else 0.0
        out[i] = (j + np.clip(frac, -0.5, 0.5)) / fps
    return out


def match(ref: np.ndarray, est: np.ndarray, window: float = 0.05) -> np.ndarray:
    """For each ref onset, signed offset (est - ref) to the nearest est within `window`; NaN if none."""
    if len(est) == 0:
        return np.full(len(ref), np.nan)
    j = np.clip(np.searchsorted(est, ref), 1, len(est) - 1)
    left, right = est[j - 1], est[j]
    near = np.where(np.abs(ref - left) <= np.abs(right - ref), left, right)
    off = near - ref
    off[np.abs(off) > window] = np.nan
    return off
