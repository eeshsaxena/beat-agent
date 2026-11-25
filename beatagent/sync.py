"""Two-track beat-matching: stretch track B onto track A's grid and render the overlap."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import onsets
from .audio_io import Audio, resample
from .grid import BeatGrid
from .stretch import stretch_piecewise


class SyncError(ValueError):
    pass


@dataclass
class SyncResult:
    mix: np.ndarray
    a_stem: np.ndarray
    b_stem: np.ndarray
    sr: int
    report: dict = field(default_factory=dict)
    b_grid_out: np.ndarray | None = None  # B's beat times inside the render (seconds)
    b_labels: np.ndarray | None = None
    a_grid_out: np.ndarray | None = None
    a_labels: np.ndarray | None = None
    anchors_b_s: np.ndarray | None = None  # B times (s) pinned by the stretch ...
    anchors_out_s: np.ndarray | None = None  # ... and where they land in the render (s)


def _period(grid: BeatGrid, i0: int, i1: int) -> float:
    k = np.arange(i0, i1)
    return float(np.polyfit(k, grid.times[i0:i1], 1)[0])


def _downbeat_time(grid: BeatGrid, bar: int, which: str) -> tuple[int, float]:
    if not 1 <= bar <= grid.last_bar:
        raise SyncError(f"{which} has bars 1-{grid.last_bar}; bar {bar} does not exist.")
    i = grid.bar_span(bar)[0]
    return i, float(grid.times[i])


def _near(x: np.ndarray, beats: np.ndarray | None, tol: float) -> np.ndarray:
    if beats is None or len(x) == 0 or len(beats) == 0:
        return x
    j = np.clip(np.searchsorted(beats, x), 1, max(len(beats) - 1, 1))
    d = np.minimum(np.abs(beats[j - 1] - x), np.abs(beats[np.minimum(j, len(beats) - 1)] - x))
    return x[d <= tol]


def kick_alignment(a: np.ndarray, b: np.ndarray, sr: int, window: float = 0.04,
                   beats_a: np.ndarray | None = None, beats_b: np.ndarray | None = None, near: float = 0.05) -> dict:
    """Kick onsets of A vs B, measured from the audio. Each track's kicks are searched for
    near its own beats (so offbeat bass notes are not counted as kicks); the onset is the
    physical start of the kick (see onsets.kick_onsets_near), validated against the
    stand-in's ground truth in evals (median 0.1 ms, max 0.6 ms on unstretched audio)."""
    if beats_a is None or beats_b is None:
        ka, kb = onsets.detect_kicks(a, sr), onsets.detect_kicks(b, sr)
    else:
        ka = onsets.kick_onsets_near(a, sr, beats_a)
        kb = onsets.kick_onsets_near(b, sr, beats_b)
        ka, kb = np.sort(ka[~np.isnan(ka)]), np.sort(kb[~np.isnan(kb)])
    off = onsets.match(ka, kb, window)
    ok = ~np.isnan(off)
    if ok.sum() == 0:
        return {"kicks_a": int(len(ka)), "kicks_b": int(len(kb)), "matched": 0}
    o = off[ok] * 1000
    return {"kicks_a": int(len(ka)), "kicks_b": int(len(kb)), "matched": int(ok.sum()),
            "median_offset_ms": round(float(np.median(o)), 2), "median_abs_offset_ms": round(float(np.median(np.abs(o))), 2),
            "max_abs_offset_ms": round(float(np.max(np.abs(o))), 2), "p95_abs_offset_ms": round(float(np.percentile(np.abs(o), 95)), 2)}


def sync(a: Audio, ga: BeatGrid, b: Audio, gb: BeatGrid, start_bar_a: int = 33, start_bar_b: int = 1,
         duration: float = 60.0, refine: bool = False, gain_a_db: float = -3.0, gain_b_db: float = -3.0,
         progress=None) -> SyncResult:
    say = progress or (lambda *_: None)
    sr = a.sr
    ia, tA = _downbeat_time(ga, start_bar_a, "Track A")
    ib, tB = _downbeat_time(gb, start_bar_b, "Track B")
    notes = []
    if ga.beats_per_bar != gb.beats_per_bar:
        notes.append(f"Bar lengths differ (A has {ga.beats_per_bar} beats per bar, B has {gb.beats_per_bar}); "
                     "downbeats will only coincide at the start.")

    avail_a = a.duration - tA
    if avail_a < duration:
        notes.append(f"Track A has only {avail_a:.1f} s after bar {start_bar_a}; the overlap is shortened.")
        duration = avail_a
    n_out = int(round(duration * sr))

    # Beats of A inside the render window (plus one bar past it) define where B's beats land.
    m = ga.beats_per_bar
    a_idx = np.flatnonzero((ga.times >= tA - 1e-6) & (ga.times < tA + duration))
    a_ext = np.flatnonzero(ga.times >= tA - 1e-6)[: len(a_idx) + m]
    # Map B beat ib+j -> A beat ia+j (beat-for-beat), as far as both grids go.
    nb = min(len(a_ext), gb.n - ib)
    if nb < 4:
        raise SyncError("Not enough beats in track B after its start bar to sync.")
    b_src = gb.times[ib:ib + nb]
    a_dst = ga.times[a_ext[:nb]] - tA  # seconds into the render
    constant = ga.grid_type == "constant" and gb.grid_type == "constant"
    ratio = float(np.polyfit(np.arange(nb), a_dst, 1)[0] / np.polyfit(np.arange(nb), b_src, 1)[0])

    say("Time-stretching track B (Rubber Band, pitch preserved)")
    xb = resample(b.samples, b.sr, sr) if b.sr != sr else b.samples
    # Rubber Band realises a ratio with integer hop increments and can let transients drift
    # ~8 ms over a minute before correcting (measured). pylibrb can't take a key-frame map,
    # so the time map is pinned at anchors: every 4 bars for constant grids, every bar otherwise.
    step = 4 * gb.beats_per_bar if constant else gb.beats_per_bar
    anchors = list(range(0, nb, step))
    if anchors[-1] != nb - 1:
        anchors.append(nb - 1)
    y = stretch_piecewise(xb, sr, b_src[anchors], a_dst[anchors])
    change = abs(1 / ratio - 1)
    if change > 0.08:
        notes.append(f"Large tempo change for B ({change * 100:.0f}%: {gb.tempo_bpm:.1f} -> {ga.tempo_bpm:.1f} BPM). "
                     "Stretches beyond ~8% start to sound unnatural; a DJ would usually pick a closer track or mix at "
                     "half/double time.")
    mapping = (f"anchored every {step} beats ({'constant ratio' if constant else 'beat map followed bar by bar'})")
    b_stem = np.zeros((n_out, a.channels), np.float32)
    y = y[:n_out]
    if y.shape[1] != a.channels:
        y = np.repeat(y.mean(axis=1, keepdims=True), a.channels, axis=1)
    b_stem[:len(y)] = y
    fade = int(0.002 * sr)
    b_stem[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)[:, None]
    i0 = int(round(tA * sr))
    a_stem = a.samples[i0:i0 + n_out].copy()

    say("Measuring kick alignment")
    edge = duration - 0.05  # a kick right at the end of the render has no audio after it to measure
    a_beats = ga.times[a_idx] - tA
    a_beats = a_beats[a_beats < edge]
    b_out = a_dst.copy()  # by construction B's beats land on A's beats
    before = kick_alignment(a_stem, b_stem, sr, beats_a=a_beats, beats_b=b_out[b_out < edge])
    shift_ms = 0.0
    # Small nudge (like a DJ's jog wheel) when measured kicks disagree with the grid mapping;
    # capped so a measurement problem can't move B by more than 10 ms.
    if refine and before.get("matched", 0) >= 8 and 0.25 < abs(before["median_offset_ms"]) <= 10.0:
        shift = -int(round(before["median_offset_ms"] * 1e-3 * sr))
        shift_ms = shift / sr * 1000
        b_stem = np.roll(b_stem, shift, axis=0)
        if shift > 0:
            b_stem[:shift] = 0
        else:
            b_stem[shift:] = 0
    b_out = b_out + shift_ms / 1000
    after = kick_alignment(a_stem, b_stem, sr, beats_a=a_beats, beats_b=b_out[b_out < edge]) if shift_ms else before

    mix = a_stem * 10 ** (gain_a_db / 20) + b_stem * 10 ** (gain_b_db / 20)
    peak = float(np.abs(mix).max())
    if peak > 0.99:
        mix *= 0.99 / peak
    report = {
        "track_a": {"start_bar": start_bar_a, "start_s": round(tA, 4), "bpm": round(ga.tempo_bpm, 3), "grid": ga.grid_type},
        "track_b": {"start_bar": start_bar_b, "start_s": round(tB, 4), "bpm": round(gb.tempo_bpm, 3), "grid": gb.grid_type,
                    "resampled_from_hz": b.sr if b.sr != sr else None},
        "time_stretch_ratio": round(float(ratio), 6),
        "time_stretch_ratio_meaning": "output duration / input duration of track B (pitch preserved)",
        "tempo_change_pct": round((1 / ratio - 1) * 100, 3), "mapping": mapping,
        "offset_s": round(tA - tB, 4),
        "offset_meaning": "B's start-bar downbeat (B time tB) plays at A's start-bar downbeat (A time tA). "
                          "B time t plays at A time tA + (t - tB) * ratio (+ onset refinement). offset_s = tA - tB.",
        "duration_s": round(duration, 3), "sample_rate": sr,
        "kick_alignment_grid_only": before, "onset_refinement_ms": round(shift_ms, 2),
        "kick_alignment_final": after, "notes": notes,
    }
    inside = b_out < duration
    return SyncResult(mix.astype(np.float32), a_stem, b_stem, sr, report, b_out[inside],
                      gb.beat_in_bar[ib:ib + nb][inside], ga.times[a_idx] - tA, ga.beat_in_bar[a_idx],
                      b_src[anchors], a_dst[anchors])
