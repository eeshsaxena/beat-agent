"""Pitch-preserving time-stretch with Rubber Band (R3 'finer' engine, offline mode)."""
from __future__ import annotations

import numpy as np


def DEFAULT_OPTIONS():
    from pylibrb import Option

    return Option.PROCESS_OFFLINE | Option.ENGINE_FINER | Option.CHANNELS_TOGETHER


def stretch_piecewise(x: np.ndarray, sr: int, src_t: np.ndarray, dst_t: np.ndarray, margin: float = 0.25,
                      xfade_ms: float = 10.0) -> np.ndarray:
    """Variable time map: src_t[i] (seconds in x) lands at dst_t[i] (seconds in the output).

    pylibrb 0.1.2 can't take a key-frame map from Python (its std::map caster is not
    compiled in), so each span between consecutive anchors is stretched with its own
    constant ratio, with context margins, and joined with a short equal-power crossfade
    placed just before each anchor (so the transient on the anchor stays intact)."""
    n_out = int(round(dst_t[-1] * sr))
    ch = x.shape[1]
    out = np.zeros((n_out, ch), np.float32)
    X = int(xfade_ms * 1e-3 * sr)
    ramp = np.sin((np.arange(X) + 0.5) / X * np.pi / 2)[:, None].astype(np.float32)
    for i in range(len(src_t) - 1):
        s0, s1, d0, d1 = src_t[i], src_t[i + 1], dst_t[i], dst_t[i + 1]
        r = (d1 - d0) / (s1 - s0)
        a = max(0, int(round((s0 - margin) * sr)))
        b = min(len(x), int(round((s1 + margin) * sr)))
        y = stretch(x[a:b], sr, r)
        lead = int(round((s0 - a / sr) * r * sr))
        o0, o1 = int(round(d0 * sr)), int(round(d1 * sr))
        seg = y[lead:lead + (o1 - o0)]
        if i == 0 or X == 0 or lead < X:
            out[o0:o0 + len(seg)] = seg
            continue
        pre = y[lead - X:lead]
        out[o0 - X:o0] = out[o0 - X:o0] * ramp[::-1] + pre * ramp
        out[o0:o0 + len(seg)] = seg
    return out


def stretch(x: np.ndarray, sr: int, ratio: float, options: int | None = None) -> np.ndarray:
    """Time-stretch `x` (shape (n, channels)). `ratio` = output duration / input duration."""
    from pylibrb import RubberBandStretcher

    n, ch = x.shape
    opts = options if options is not None else DEFAULT_OPTIONS()
    st = RubberBandStretcher(sr, ch, opts, initial_time_ratio=float(ratio))
    a = np.ascontiguousarray(x.T.astype(np.float32))
    block = min(int(st.get_process_size_limit()), 1 << 18)
    st.set_expected_input_duration(n)
    st.set_max_process_size(block)
    starts = list(range(0, n, block)) or [0]
    for s in starts:
        st.study(np.ascontiguousarray(a[:, s:s + block]), final=(s == starts[-1]))
    chunks = []

    def drain():
        while True:
            av = st.available()
            if av <= 0:
                return
            chunks.append(st.retrieve(av))

    for s in starts:
        st.process(np.ascontiguousarray(a[:, s:s + block]), final=(s == starts[-1]))
        drain()
    drain()
    out = np.concatenate(chunks, axis=1).T if chunks else np.zeros((0, ch), np.float32)
    return np.ascontiguousarray(out, dtype=np.float32)
