"""Beat and downbeat tracking -> a beat grid a DJ app can trust.

Pipeline (all measurement; no LLM involved):

1. Beat This! (CPJKU, ISMIR 2024) gives frame-wise beat and downbeat logits at
   50 fps. Peaks are picked with parabolic interpolation (sub-frame precision).
2. Raw beats get integer beat numbers that tolerate missed or spurious beats.
3. Tempo model:
   * constant  - one least-squares line t = t0 + k*P through all beats. Averaging
                 over hundreds of beats removes the tracker's ~10 ms frame jitter,
                 which is what makes "every beat within 10 ms over 6 minutes"
                 reachable.
   * piecewise - dynamic-programming segmentation into a few constant-tempo
                 segments (abrupt tempo changes).
   * variable  - locally smoothed beats (live playing, drifting tempo).
4. The grid is restricted to the region with rhythmic evidence (an ambient intro
   with no pulse gets no beats).
5. Phase calibration: the grid is shifted by the median offset to the measured
   percussive onsets that coincide with beats, so grid lines sit on the attacks.
6. Meter and downbeat phase: the (beats-per-bar, phase) pair with the highest
   downbeat-activation contrast; compound metre (6/8) from onset energy at
   1/3-beat vs 1/2-beat positions.
"""
from __future__ import annotations

import os
import time

import numpy as np
from scipy import signal
from scipy.ndimage import median_filter

from . import onsets as onset_mod
from .audio_io import Audio
from .grid import BeatGrid

FPS = 50.0
COMPOUND_REGROUP_BPM = 190.0  # above this, a ternary grouping is read as compound metre (6/8 in two)
_MODELS: dict = {}


def _model(checkpoint: str, device: str):
    key = (checkpoint, device)
    if key not in _MODELS:
        from beat_this.inference import Audio2Frames

        _MODELS[key] = Audio2Frames(checkpoint_path=checkpoint, device=device)
    return _MODELS[key]


def frame_activations(mono: np.ndarray, sr: int, checkpoint: str = "final0", device: str = "cpu"):
    b, d = _model(checkpoint, device)(mono, sr)
    return b.cpu().numpy().astype(np.float64), d.cpu().numpy().astype(np.float64)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def peak_pick(logits: np.ndarray, fps: float = FPS, min_gap_frames: int = 4) -> np.ndarray:
    """Beat This!'s 'minimal' rule (local max of logit > 0) plus parabolic sub-frame refinement."""
    peaks, _ = signal.find_peaks(logits, height=0.0, distance=min_gap_frames)
    t = peaks.astype(float)
    ok = (peaks > 0) & (peaks < len(logits) - 1)
    p = peaks[ok]
    a, b, c = logits[p - 1], logits[p], logits[p + 1]
    den = a - 2 * b + c
    t[ok] += np.where(np.abs(den) > 1e-9, np.clip(0.5 * (a - c) / den, -0.5, 0.5), 0.0)
    return t / fps


def nearest_dist(ref: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Distance from each x to the nearest element of sorted `ref` (inf if ref is empty)."""
    if len(ref) == 0:
        return np.full(len(x), np.inf)
    j = np.clip(np.searchsorted(ref, x), 1, len(ref) - 1) if len(ref) > 1 else np.zeros(len(x), int)
    d = np.abs(ref[j] - x)
    if len(ref) > 1:
        d = np.minimum(d, np.abs(ref[j - 1] - x))
    return d


def _window_max(act: np.ndarray, times: np.ndarray, fps: float = FPS, half: int = 2) -> np.ndarray:
    f = np.round(times * fps).astype(int)
    out = np.zeros(len(times))
    for i, fi in enumerate(f):
        lo, hi = max(0, fi - half), min(len(act), fi + half + 1)
        out[i] = act[lo:hi].max() if hi > lo else 0.0
    return out


# ----------------------------------------------------------------------------
# Integer beat numbering
# ----------------------------------------------------------------------------
def number_beats(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Assign integer beat numbers k to raw beat times b, skipping off-grid detections
    and leaving gaps where beats were missed."""
    if len(b) < 4:
        return np.arange(len(b)), b.copy()
    ibi = np.diff(b)
    ref = median_filter(ibi, size=9, mode="nearest")
    keep, ks = [0], [0]
    for i in range(1, len(b)):
        p = ref[min(i - 1, len(ref) - 1)]
        steps = (b[i] - b[keep[-1]]) / p
        n = int(round(steps))
        if n < 1 or abs(steps - n) > 0.3:
            continue
        keep.append(i)
        ks.append(ks[-1] + n)
    return np.array(ks), b[np.array(keep)]


def _linfit(k, t):
    A = np.stack([np.ones_like(k, dtype=float), k.astype(float)], axis=1)
    coef, *_ = np.linalg.lstsq(A, t, rcond=None)
    return coef  # (t0, P)


# ----------------------------------------------------------------------------
# Constant-tempo fit
# ----------------------------------------------------------------------------
def fit_constant(b: np.ndarray, period_guess: float, inlier_s: float = 0.04):
    """Phase-coherence search for the period, then least squares on inliers."""
    ref = b[0]
    x = b - ref
    periods = np.linspace(period_guess * 0.985, period_guess * 1.015, 1201)
    z = np.exp(2j * np.pi * x[None, :] / periods[:, None]).mean(axis=1)
    i = int(np.argmax(np.abs(z)))
    P = periods[i]
    t0 = (np.angle(z[i]) / (2 * np.pi)) * P
    for _ in range(4):
        k = np.round((x - t0) / P)
        res = x - (t0 + k * P)
        inl = np.abs(res) < inlier_s
        if inl.sum() < 4:
            break
        t0, P = _linfit(k[inl], x[inl])
    k = np.round((x - t0) / P).astype(int)
    res = x - (t0 + k * P)
    return t0 + ref, P, k, res


# ----------------------------------------------------------------------------
# Piecewise-constant segmentation (dynamic programming on t = c + P*k)
# ----------------------------------------------------------------------------
def segment_piecewise(k: np.ndarray, t: np.ndarray, sigma: float, min_len: int = 12):
    n = len(t)
    kk = k.astype(float) - k[0]
    tt = t - t[0]
    S = lambda v: np.concatenate([[0.0], np.cumsum(v)])  # noqa: E731
    SK, ST, SKK, SKT, STT = S(kk), S(tt), S(kk * kk), S(kk * tt), S(tt * tt)

    def sse(i, j):  # vectorised over i (array), points i..j-1
        m = j - i
        sk, st = SK[j] - SK[i], ST[j] - ST[i]
        skk, skt, stt = SKK[j] - SKK[i], SKT[j] - SKT[i], STT[j] - STT[i]
        vk = skk - sk * sk / m
        cv = skt - sk * st / m
        vt = stt - st * st / m
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(vk > 1e-12, vt - cv * cv / vk, vt)
        return np.maximum(out, 0.0)

    lam = 8.0 * sigma ** 2 * np.log(max(n, 2))
    dp = np.full(n + 1, np.inf)
    dp[0] = 0.0
    prev = np.zeros(n + 1, dtype=int)
    for j in range(min_len, n + 1):
        i = np.arange(0, j - min_len + 1)
        i = i[np.isfinite(dp[i])]
        if len(i) == 0:
            continue
        c = dp[i] + sse(i, j) + lam
        a = int(np.argmin(c))
        dp[j], prev[j] = c[a], i[a]
    if not np.isfinite(dp[n]):
        return [(0, n)]
    segs, j = [], n
    while j > 0:
        segs.append((prev[j], j))
        j = prev[j]
    return segs[::-1]


def smooth_variable(k: np.ndarray, t: np.ndarray, half_window: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Fill missed beats and smooth with a local linear fit (Gaussian weights)."""
    full_k = np.arange(k[0], k[-1] + 1)
    out = np.empty(len(full_k))
    for j, kj in enumerate(full_k):
        m = np.abs(k - kj) <= half_window
        kk, tt = k[m], t[m]
        if len(kk) < 3:
            out[j] = np.interp(kj, k, t)
            continue
        w = np.exp(-0.5 * ((kk - kj) / (half_window / 1.6)) ** 2)
        A = np.stack([np.ones(len(kk)), kk - kj], axis=1) * np.sqrt(w)[:, None]
        coef, *_ = np.linalg.lstsq(A, tt * np.sqrt(w), rcond=None)
        out[j] = coef[0]
    return full_k, out


# ----------------------------------------------------------------------------
# Region with rhythmic evidence
# ----------------------------------------------------------------------------
def supported_region(grid_t: np.ndarray, raw: np.ndarray, tol: float = 0.05, win: int = 10, need: int = 8):
    """Index range [lo, hi) of grid beats from the first to the last window where at
    least `need` of `win` grid beats have a raw tracker beat within `tol` seconds."""
    if len(grid_t) == 0 or len(raw) == 0:
        return 0, len(grid_t)
    j = np.clip(np.searchsorted(raw, grid_t), 1, len(raw) - 1)
    d = np.minimum(np.abs(raw[j - 1] - grid_t), np.abs(raw[j] - grid_t))
    sup = (d <= tol).astype(int)
    if len(sup) < win:
        return 0, len(grid_t)
    run = np.convolve(sup, np.ones(win, dtype=int), mode="valid")
    good = np.flatnonzero(run >= need)
    if len(good) == 0:
        return 0, len(grid_t)
    lo = good[0] + int(np.argmax(sup[good[0]:good[0] + win]))
    last = good[-1]
    hi = last + win - int(np.argmax(sup[last:last + win][::-1]))
    return int(lo), int(hi)


# ----------------------------------------------------------------------------
# Onset alignment: put grid lines on the measured attacks
# ----------------------------------------------------------------------------
def smoothed_novelty(nov: np.ndarray, sigma_frames: float = 1.5) -> np.ndarray:
    nov = np.minimum(nov, np.percentile(nov, 99.5))
    r = int(4 * sigma_frames)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_frames) ** 2)
    return np.convolve(nov, k / k.sum(), mode="same")


def _nov_at(nov_s: np.ndarray, fps: float, t: np.ndarray) -> np.ndarray:
    return np.interp(t * fps, np.arange(len(nov_s)), nov_s, left=0.0, right=0.0)


def align_constant(times: np.ndarray, nov_s: np.ndarray, fps: float, search: float = 0.05, slope_ms: float = 30.0):
    """Shift (d0) and tilt (d1 at the ends) a constant grid so it best matches the onset curve."""
    K = len(times)
    c = (np.arange(K) - (K - 1) / 2) / max((K - 1) / 2, 1)

    def best(d0s, d1s):
        bs, b0, b1 = -1.0, 0.0, 0.0
        for d1 in d1s:
            T = times[None, :] + d0s[:, None] + d1 * c[None, :]
            s = _nov_at(nov_s, fps, T.ravel()).reshape(T.shape).sum(axis=1)
            i = int(np.argmax(s))
            if s[i] > bs:
                bs, b0, b1 = float(s[i]), float(d0s[i]), float(d1)
        return b0, b1, bs

    d0, d1, _ = best(np.arange(-search, search + 1e-9, 0.001), np.arange(-slope_ms, slope_ms + 1e-9, 2.0) / 1000)
    d0, d1, _ = best(np.arange(d0 - 0.0015, d0 + 0.0015 + 1e-9, 0.0001), np.arange(d1 - 0.003, d1 + 0.003 + 1e-9, 0.0002))
    return times + d0 + d1 * c, d0, d1


def local_offsets(times: np.ndarray, nov_s: np.ndarray, fps: float, win: int = 16, search: float = 0.05):
    """Best onset-alignment shift per window of `win` beats, with its relative strength."""
    d0s = np.arange(-search, search + 1e-9, 0.0005)
    out = []
    for s in range(0, len(times) - win + 1, win // 2):
        tw = times[s:s + win]
        T = tw[None, :] + d0s[:, None]
        sc = _nov_at(nov_s, fps, T.ravel()).reshape(T.shape).mean(axis=1)
        i = int(np.argmax(sc))
        out.append((s, float(d0s[i]), float(sc[i]), float(np.median(sc))))
    return out


def extend_by_onsets(g: np.ndarray, P: float, nov_s: np.ndarray, fps: float, block: int = 4,
                     min_ratio: float = 1.5, phase_tol: float = 0.010) -> tuple[np.ndarray, int, int]:
    """Extend a constant grid outward, `block` beats at a time, while the onset curve at the
    extended beats peaks at the grid's own phase (within phase_tol) and stands out against
    the other phases of the beat by `min_ratio`. Returns (grid, beats added before, after)."""
    t_max = len(nov_s) / fps
    phases = np.linspace(-0.5, 0.5, 81)[:-1] * P

    def clear(ts) -> bool:
        T = ts[None, :] + phases[:, None]
        sc = _nov_at(nov_s, fps, T.ravel()).reshape(T.shape).mean(axis=1)
        i = int(np.argmax(sc))
        return abs(phases[i]) <= phase_tol and sc[i] >= min_ratio * float(np.median(sc)) and sc[i] > 0

    added_before = added_after = 0
    # Trim edge blocks the onsets do not support (a tracker that sees the future can turn
    # confident a few beats before the drums come in). At most 32 beats per side.
    for _ in range(8):
        if len(g) > 4 * block and not clear(g[:block]):
            g, added_before = g[block:], added_before - block
        else:
            break
    for _ in range(8):
        if len(g) > 4 * block and not clear(g[-block:]):
            g, added_after = g[:-block], added_after - block
        else:
            break
    # Single trailing/leading beats with almost no onset energy (e.g. one beat into a reverb tail)
    strength = _nov_at(nov_s, fps, g)
    med = float(np.median(strength))
    for _ in range(block):
        if len(g) > 4 * block and strength[-1] < 0.15 * med:
            g, strength, added_after = g[:-1], strength[:-1], added_after - 1
        else:
            break
    for _ in range(block):
        if len(g) > 4 * block and strength[0] < 0.15 * med:
            g, strength, added_before = g[1:], strength[1:], added_before - 1
        else:
            break
    if added_before < 0 or added_after < 0:
        return g, added_before, added_after
    while g[0] - block * P > 0.0:
        cand = g[0] - P * np.arange(block, 0, -1)
        if not clear(cand):
            break
        g = np.concatenate([cand, g])
        added_before += block
    while g[-1] + block * P < t_max - 0.05:
        cand = g[-1] + P * np.arange(1, block + 1)
        if not clear(cand):
            break
        g = np.concatenate([g, cand])
        added_after += block
    return g, added_before, added_after


def raw_window_mismatch(grid_t: np.ndarray, raw: np.ndarray, win: int = 16, tol: float = 0.06) -> tuple[int, int]:
    """Windows of grid beats where the tracker has beats but they do not line up with the grid.
    (A window with no tracker beats, e.g. a breakdown, is not evidence either way.)"""
    bad = checked = 0
    for s in range(0, len(grid_t) - win + 1, win):
        w = grid_t[s:s + win]
        rb = raw[(raw >= w[0] - tol) & (raw <= w[-1] + tol)]
        if len(rb) < win // 2:
            continue
        checked += 1
        if np.mean(nearest_dist(rb, w) <= tol) < 0.5:
            bad += 1
    return bad, checked


def align_variable(times: np.ndarray, nov_s: np.ndarray, fps: float, search: float = 0.04, half: int = 4):
    """Per-beat onset alignment for a variable grid, smoothed over +-half beats."""
    d0s = np.arange(-search, search + 1e-9, 0.0005)
    T = times[:, None] + d0s[None, :]
    sc = _nov_at(nov_s, fps, T.ravel()).reshape(T.shape)
    best = d0s[np.argmax(sc, axis=1)]
    strength = sc.max(axis=1)
    ok = strength > 0.5 * np.median(strength)
    sm = np.zeros(len(times))
    for i in range(len(times)):
        lo, hi = max(0, i - half), min(len(times), i + half + 1)
        m = ok[lo:hi]
        sm[i] = np.median(best[lo:hi][m]) if m.any() else 0.0
    return times + sm, sm


# ----------------------------------------------------------------------------
# Metre and downbeats
# ----------------------------------------------------------------------------
def choose_meter(down_score: np.ndarray, candidates=(2, 3, 4)):
    n = len(down_score)
    idx = np.arange(n)
    results = []
    for m in candidates:
        if n < 2 * m:
            continue
        for ph in range(m):
            mask = (idx - ph) % m == 0
            s = down_score[mask].mean() - down_score[~mask].mean()
            results.append((s + (0.005 if m == 4 else 0.0), m, ph, s))
    if not results:
        return 4, 0, 0.0, 0.0
    results.sort(reverse=True)
    best = results[0]
    second = next((r for r in results[1:] if r[1] != best[1] or r[2] != best[2]), (0, 0, 0, 0.0))
    return best[1], best[2], best[3], best[3] - second[3]


def is_compound(grid_t: np.ndarray, nov: np.ndarray, nov_fps: float) -> tuple[bool, float]:
    """Compare onset energy at 1/3 and 2/3 of each beat with energy at 1/2 of the beat."""
    if len(grid_t) < 8:
        return False, 1.0
    a, b = grid_t[:-1], grid_t[1:]
    w = max(1, int(0.012 * nov_fps))

    def at(fr):
        f = np.round((a + fr * (b - a)) * nov_fps).astype(int)
        return np.array([nov[max(0, x - w):x + w + 1].max() if x < len(nov) else 0.0 for x in f])

    trip = 0.5 * (at(1 / 3) + at(2 / 3)).mean()
    dup = at(0.5).mean()
    ratio = float(trip / (dup + 1e-9))
    return ratio > 1.5, ratio


TIME_SIGS = {(2, False): "2/4", (3, False): "3/4", (4, False): "4/4",
             (2, True): "6/8", (3, True): "9/8", (4, True): "12/8"}


# ----------------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------------
def analyze(audio: Audio, checkpoint: str | None = None, device: str | None = None,
            calibrate_onsets: bool = True, progress=None) -> BeatGrid:
    checkpoint = checkpoint or os.environ.get("BEAT_AGENT_CHECKPOINT", "final0")
    device = device or os.environ.get("BEAT_AGENT_DEVICE", "cpu")
    say = progress or (lambda *_: None)
    timings = {}
    t_all = time.perf_counter()
    mono = audio.mono()

    say("Running the Beat This! neural tracker")
    t = time.perf_counter()
    beat_logit, down_logit = frame_activations(mono, audio.sr, checkpoint, device)
    timings["tracker_s"] = round(time.perf_counter() - t, 2)
    beat_act, down_act = _sigmoid(beat_logit), _sigmoid(down_logit)
    raw = peak_pick(beat_logit)
    raw_down = peak_pick(down_logit)

    say("Measuring percussive onsets")
    t = time.perf_counter()
    nov, nov_fps = onset_mod.onset_function(mono, audio.sr)
    nov_s = smoothed_novelty(nov)
    timings["onsets_s"] = round(time.perf_counter() - t, 2)

    say("Fitting the tempo model and aligning it to the onsets")
    t = time.perf_counter()
    raw_conf = beat_act[np.clip(np.round(raw * FPS).astype(int), 0, len(beat_act) - 1)]
    fit = _fit_tempo(raw, nov_s if calibrate_onsets else None, nov_fps, raw_conf)
    timings["fit_s"] = round(time.perf_counter() - t, 2)
    times = fit["times"]
    shifts = fit["info"].get("onset_alignment")

    say("Finding downbeats and metre")
    down_score = _window_max(down_act, times)
    m, ph, contrast, margin = choose_meter(down_score)
    level_note = None
    if m == 3 and len(times) > 12 and 60 / np.median(np.diff(times)) > COMPOUND_REGROUP_BPM:
        # The tracker followed a fast ternary level (e.g. eighth notes of a 6/8). Musicians and
        # most annotations count 6/8 "in two": regroup three tracker beats into one beat.
        fast_bpm = 60 / np.median(np.diff(times))
        fit = _fit_tempo(times[ph::3], nov_s if calibrate_onsets else None, nov_fps)
        fit["info"]["regrouped_from"] = {"tracker_level_bpm": round(float(fast_bpm), 2), "grouping": 3}
        times = fit["times"]
        shifts = fit["info"].get("onset_alignment")
        down_score = _window_max(down_act, times)
        m, ph, contrast, margin = choose_meter(down_score, candidates=(2, 4))
        level_note = (f"Tracker beats were at {fast_bpm:.1f} BPM grouped in threes; the grid counts the "
                      f"dotted-quarter pulse ({60 / np.median(np.diff(times)):.1f} BPM) as the beat.")
    compound, comp_ratio = is_compound(times, nov, nov_fps)
    idx = np.arange(len(times))
    beat_in_bar = ((idx - ph) % m) + 1
    strength = _window_max(beat_act, times)

    # confidence components
    support = float(np.mean(nearest_dist(raw, times) <= 0.05)) if len(times) else 0.0
    dts = times[beat_in_bar == 1]
    down_agree = float(np.mean(nearest_dist(raw_down, dts) <= 0.07)) if len(dts) else 0.0
    rms = fit["rms_ms"]
    conf = (0.6 * support + 0.25 * down_agree + 0.15 * float(np.exp(-rms / 20.0))) if len(times) else 0.0
    timings["total_s"] = round(time.perf_counter() - t_all, 2)

    return BeatGrid(
        times=times, beat_in_bar=beat_in_bar, beats_per_bar=m, time_signature=TIME_SIGS[(m, compound)],
        grid_type=fit["type"], duration_s=audio.duration, sample_rate=audio.sr, confidence=conf,
        confidence_detail={
            "beat_support": round(support, 3), "downbeat_agreement": round(down_agree, 3),
            "fit_rms_ms": round(rms, 2), "downbeat_contrast": round(float(contrast), 3),
            "downbeat_margin": round(float(margin), 3),
            "how": "0.6*beat_support + 0.25*downbeat_agreement + 0.15*exp(-fit_rms_ms/20). A heuristic, not a "
                   "calibrated probability.",
        },
        strength=strength,
        meta={
            "tracker": f"Beat This! ({checkpoint}) on {device}", "raw_beats": int(len(raw)),
            "raw_downbeats": int(len(raw_down)), "tempo_model": fit["type"], "fit": fit["info"],
            "region_s": [round(float(times[0]), 3), round(float(times[-1]), 3)] if len(times) else None,
            "onset_calibration_ms": shifts, "compound_ratio": round(comp_ratio, 2), "timings": timings,
            "metrical_level_note": level_note,
        },
    )


def _fit_tempo(raw: np.ndarray, nov_s: np.ndarray | None = None, fps: float = 0.0,
               conf: np.ndarray | None = None) -> dict:
    if len(raw) < 8:
        return {"times": raw.copy(), "type": "variable", "segments": None, "rms_ms": 0.0,
                "info": {"note": "too few beats for a tempo model"}}
    k, t = number_beats(raw)
    d2 = t[2:] - 2 * t[1:-1] + t[:-2]
    consec = (k[2:] - k[1:-1] == 1) & (k[1:-1] - k[:-2] == 1)
    sigma = float(max(1.4826 * np.median(np.abs(d2[consec])) / np.sqrt(6), 0.003)) if consec.sum() > 4 else 0.01
    P0 = float(np.median(np.diff(t) / np.diff(k)))

    # Beats the tracker is confident about (activation ~1.0). Free-time passages such as an
    # ambient intro give weak, scattered peaks (~0.6) that must not define the grid's extent.
    conf_raw = raw if conf is None else raw[conf >= 0.9]
    if len(conf_raw) < 8:
        conf_raw = raw
    fk0, ft0 = smooth_variable(k, t)
    tlo, thi = supported_region(ft0, conf_raw)
    track_span = (float(ft0[tlo]), float(ft0[thi - 1]))

    # constant model over the supported region
    t0, P, kc, res = fit_constant(raw, P0)
    kmin, kmax = int(kc.min()), int(kc.max())
    grid_all = t0 + np.arange(kmin, kmax + 1) * P
    lo, hi = supported_region(grid_all, conf_raw)
    g = grid_all[lo:hi]
    covers = g[0] <= track_span[0] + 8 * P and g[-1] >= track_span[1] - 8 * P
    in_region = (raw >= g[0] - 0.05) & (raw <= g[-1] + 0.05)
    r_in = res[in_region]
    inl = np.abs(r_in) < 0.04
    frac = float(inl.mean()) if len(r_in) else 0.0
    rms = float(np.sqrt(np.mean(r_in[inl] ** 2)) * 1000) if inl.any() else 99.0
    trend = median_filter(r_in[inl], size=min(33, max(3, int(inl.sum()) // 2 * 2 - 1)), mode="nearest") if inl.sum() > 8 else np.zeros(1)
    drift_ms = float(np.max(np.abs(trend)) * 1000)
    info = {"sigma_ms": round(sigma * 1000, 2), "constant_fit": {"bpm": round(60 / P, 4), "inlier_frac": round(frac, 3),
            "rms_ms": round(rms, 2), "max_local_drift_ms_raw": round(drift_ms, 2)}}

    if nov_s is not None:
        # The tracker proposes tempo; the onsets decide phase and confirm constancy.
        ga, d0, d1 = align_constant(g, nov_s, fps)
        offs = local_offsets(ga, nov_s, fps)
        strong = [o for o in offs if o[2] >= 1.5 * o[3] and o[2] >= 0.5 * np.median([w[2] for w in offs])]
        spread = max((abs(o[1]) for o in strong), default=None)
        info["onset_alignment"] = {"shift_ms": round(d0 * 1000, 2), "end_tilt_ms": round(d1 * 1000, 2),
                                   "windows": len(offs), "windows_with_clear_onsets": len(strong),
                                   "max_window_offset_ms": round(spread * 1000, 2) if spread is not None else None}
        # Accept a constant grid when the onsets agree with it everywhere they are clear. The raw
        # inlier fraction only needs to be moderate: trackers add extra beats on fast hi-hat rolls.
        bad, checked = raw_window_mismatch(ga, conf_raw)
        info["onset_alignment"]["windows_disagreeing_with_tracker"] = f"{bad}/{checked}"
        info["constant_fit"]["covers_tracked_span"] = bool(covers)
        if (frac >= 0.6 and covers and spread is not None and len(strong) >= 0.5 * len(offs) and spread <= 0.012
                and bad == 0):
            Pg = float(np.median(np.diff(ga)))
            ga, nb, na = extend_by_onsets(ga, Pg, nov_s, fps)
            info["onset_alignment"]["extended_beats"] = {"before": nb, "after": na}
            return {"times": ga, "type": "constant", "segments": None, "rms_ms": rms, "info": info}
        if spread is None and covers and frac >= 0.85 and drift_ms <= 30 and rms <= 25:
            return {"times": g.copy(), "type": "constant", "segments": None, "rms_ms": rms, "info": info}
    elif covers and frac >= 0.85 and drift_ms <= 30.0 and rms <= 25.0:
        return {"times": g.copy(), "type": "constant", "segments": None, "rms_ms": rms, "info": info}

    # piecewise / variable: the region comes from the tracker's confident beats, not the constant line
    m = (t >= ft0[tlo] - 0.05) & (t <= ft0[thi - 1] + 0.05)
    k, t = k[m], t[m]
    segs = _merge_similar(segment_piecewise(k, t, sigma), k, t)
    if len(segs) == 1 and nov_s is not None:
        c0, Ps = _linfit(k, t)
        one = c0 + Ps * np.arange(k[0], k[-1] + 1)
        one, d0, d1 = align_constant(one, nov_s, fps)
        bad, checked = raw_window_mismatch(one, conf_raw)
        info["single_segment"] = {"bpm": round(60 / Ps, 4), "windows_disagreeing_with_tracker": f"{bad}/{checked}"}
        if bad == 0:
            return {"times": one, "type": "constant", "segments": None, "rms_ms": rms, "info": info}
    seg_info = []
    ok = len(segs) <= 4
    for s, e in segs:
        c0, Ps = _linfit(k[s:e], t[s:e])
        r = t[s:e] - (c0 + Ps * k[s:e])
        seg_rms = float(np.sqrt(np.mean(r ** 2)))
        seg_info.append({"beats": int(k[e - 1] - k[s] + 1), "bpm": round(60 / Ps, 3), "rms_ms": round(seg_rms * 1000, 2)})
        if (k[e - 1] - k[s] + 1) < 24 or seg_rms > max(2.5 * sigma, 0.025):
            ok = False
    info["segments"] = seg_info
    if ok and len(segs) >= 2:
        times, bounds = [], []
        for si, (s, e) in enumerate(segs):
            c0, Ps = _linfit(k[s:e], t[s:e])
            k_start = k[s]
            k_end = k[segs[si + 1][0]] if si + 1 < len(segs) else k[-1] + 1
            ks = np.arange(k_start, k_end)
            seg_t = c0 + Ps * ks
            if nov_s is not None and len(seg_t) >= 8:
                seg_t, d0, d1 = align_constant(seg_t, nov_s, fps, slope_ms=10.0)
                seg_info[si]["onset_shift_ms"] = round(d0 * 1000, 2)
            bounds.append((len(times), len(times) + len(ks)))
            times.extend(seg_t)
        times = np.array(times)
        rms_all = float(np.mean([si["rms_ms"] for si in seg_info]))
        return {"times": times, "type": "piecewise", "segments": bounds, "rms_ms": rms_all, "info": info}

    fk, ft = smooth_variable(k, t)
    rr = t - np.interp(k, fk, ft)
    if nov_s is not None:
        ft, sm = align_variable(ft, nov_s, fps)
        info["onset_alignment_variable_ms"] = {"median": round(float(np.median(sm)) * 1000, 2),
                                               "p10": round(float(np.percentile(sm, 10)) * 1000, 2),
                                               "p90": round(float(np.percentile(sm, 90)) * 1000, 2)}
    ft = _strictly_increasing(ft)
    return {"times": ft, "type": "variable", "segments": None, "rms_ms": float(np.sqrt(np.mean(rr ** 2)) * 1000),
            "info": info}


def _strictly_increasing(t: np.ndarray, min_frac: float = 0.4) -> np.ndarray:
    """Drop beats that fall closer than min_frac * median IBI after the previous kept beat."""
    if len(t) < 3:
        return t
    t = np.sort(t)
    gap = min_frac * float(np.median(np.diff(t)))
    keep = [0]
    for i in range(1, len(t)):
        if t[i] - t[keep[-1]] >= gap:
            keep.append(i)
    return t[np.array(keep)]


def _merge_similar(segs, k, t, rel: float = 0.004):
    """Merge neighbouring segments whose tempi differ by less than `rel` (tracker phase
    bias that changes with the arrangement shows up as spurious breakpoints)."""
    segs = list(segs)
    changed = True
    while changed and len(segs) > 1:
        changed = False
        for i in range(len(segs) - 1):
            (a, b), (c, d) = segs[i], segs[i + 1]
            Pa, Pb = _linfit(k[a:b], t[a:b])[1], _linfit(k[c:d], t[c:d])[1]
            short = (k[b - 1] - k[a] + 1) < 24 or (k[d - 1] - k[c] + 1) < 24
            if abs(Pa / Pb - 1) < rel or (short and abs(Pa / Pb - 1) < 3 * rel):
                segs[i:i + 2] = [(a, d)]
                changed = True
                break
    return segs
