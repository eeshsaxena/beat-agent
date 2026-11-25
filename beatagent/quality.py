"""Measurements of edit quality: clicks at splices."""
from __future__ import annotations

import numpy as np

from .edl import Timeline, render


def _d2(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1) if x.ndim > 1 else x
    return np.abs(np.diff(m, n=2, prepend=[m[0], m[0]]))


def splice_click_scores(tl: Timeline, src: np.ndarray, out: np.ndarray, sr: int) -> list[dict]:
    """Discontinuity at each splice: |2nd difference| of the edited audio over the 3 samples at
    the splice point, divided by the same quantity at the two joined points of the *source*.
    A continuous join scores about 1 (the music itself is no smoother or rougher there);
    a waveform jump (a click) scores far above 1."""
    d_out, d_src = _d2(out), _d2(src)
    res = []
    o, prev = 0, None
    for p in tl.pieces:
        if prev is not None and prev.src_end != p.src_start:
            w = d_out[max(0, o - 1): o + 2]
            a = d_src[max(0, prev.src_end - 1): prev.src_end + 2]
            b = d_src[max(0, p.src_start - 1): p.src_start + 2]
            ref = max(float(a.max()) if len(a) else 0.0, float(b.max()) if len(b) else 0.0, 1e-7)
            res.append({"at_s": round(o / sr, 4), "ratio": round(float(w.max()) / ref, 2)})
        prev = p
        o += p.length
    return res


def butt_splice_scores(tl: Timeline, src: np.ndarray, sr: int) -> list[dict]:
    """Same measurement on a naive render with no crossfades (the control that shows the
    metric catches clicks)."""
    return splice_click_scores(tl, src, render(tl, src, sr, xfade_ms=0.0), sr)
