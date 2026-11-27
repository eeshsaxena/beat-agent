"""Edit decision list (EDL): typed edit operations addressed in beats or bars.

The LLM produces these operations; everything below is deterministic code.

Editing is non-destructive. The edited track is a `Timeline`: an ordered list of
`Piece`s, each a slice of the *source* audio with gain ramps attached. Every op
is resolved against the current beat grid into output-sample ranges and turned
into a new Timeline. The beat grid of the result is derived from the pieces, so
it always matches the edited audio.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import BaseModel, Field

from .grid import BeatGrid


# =============================================================================
# Typed operations (the contract between the LLM and the code)
# =============================================================================
class BeatRange(BaseModel):
    """Half-open beat range [start_beat, end_beat). 'beat 16-32' -> 16, 32 (16 beats)."""

    unit: Literal["beats"] = "beats"
    start_beat: int = Field(description="0-based index of the first beat in the range (inclusive).")
    end_beat: int = Field(description="Beat position where the range stops (exclusive). Use the track's beat "
                                      "count N to mean 'to the end of the track'.")


class BarRange(BaseModel):
    """Inclusive bar range. 'bar 5' -> 5, 5. 'bars 1-4' -> 1, 4."""

    unit: Literal["bars"] = "bars"
    first_bar: int = Field(description="1-based first bar (inclusive). Bar 0 is the pickup, if any.")
    last_bar: int = Field(description="1-based last bar (inclusive).")


Range = Annotated[Union[BeatRange, BarRange], Field(discriminator="unit")]


class BeatPosition(BaseModel):
    unit: Literal["beats"] = "beats"
    beat: int = Field(description="Insert before this beat position (0..N, N = end of track).")


class BarPosition(BaseModel):
    unit: Literal["bars"] = "bars"
    bar: int = Field(description="Insert before the first beat of this bar. last_bar + 1 = end of track.")


Position = Annotated[Union[BeatPosition, BarPosition], Field(discriminator="unit")]


class Remove(BaseModel):
    op: Literal["remove"] = "remove"
    range: Range


class KeepOnly(BaseModel):
    op: Literal["keep_only"] = "keep_only"
    range: Range


class Repeat(BaseModel):
    """The range plays `total_plays` times in a row, in place; the track then continues."""

    op: Literal["repeat"] = "repeat"
    range: Range
    total_plays: int = Field(ge=2, le=64, description="How many times the range plays in total, including the "
                                                      "original. 'repeat it twice' -> 2, 'loop four times' -> 4.")


class Move(BaseModel):
    op: Literal["move"] = "move"
    range: Range
    to: Position


class InsertCopy(BaseModel):
    op: Literal["insert_copy"] = "insert_copy"
    range: Range
    at: Position
    copies: int = Field(default=1, ge=1, le=64)


class Fade(BaseModel):
    op: Literal["fade"] = "fade"
    range: Range
    direction: Literal["in", "out"]
    shape: Literal["equal_power", "linear", "exponential"] = "equal_power"


class Gain(BaseModel):
    op: Literal["gain"] = "gain"
    range: Range
    db: float = Field(ge=-60, le=24)


class Mute(BaseModel):
    """Silence a range but keep its length (the grid is unchanged)."""

    op: Literal["mute"] = "mute"
    range: Range


EditOp = Annotated[Union[Remove, KeepOnly, Repeat, Move, InsertCopy, Fade, Gain, Mute], Field(discriminator="op")]

SHIFTING_OPS = {"remove", "keep_only", "repeat", "move", "insert_copy"}


class EditError(ValueError):
    """A request the code refuses: out of range, impossible or ambiguous. Message is user-facing."""


# =============================================================================
# Non-destructive timeline
# =============================================================================
@dataclass(frozen=True)
class Ramp:
    """A gain curve over a whole piece, sampled at curve positions u0..u1."""

    kind: str  # "fade_in" | "fade_out" | "gain"
    shape: str = "equal_power"
    u0: float = 0.0
    u1: float = 1.0
    db: float = 0.0

    def curve(self, u: np.ndarray) -> np.ndarray:
        if self.kind == "gain":
            return np.full_like(u, 10 ** (self.db / 20))
        v = u if self.kind == "fade_in" else 1 - u
        if self.shape == "linear":
            return v
        if self.shape == "exponential":  # linear in dB down to -60 dB, then to silence
            return np.where(v > 0, 10 ** (-60 * (1 - v) / 20) * np.minimum(1, v * 20), 0.0)
        return np.sin(v * np.pi / 2)  # equal power

    def split(self, frac: float) -> tuple["Ramp", "Ramp"]:
        um = self.u0 + (self.u1 - self.u0) * frac
        return replace(self, u1=um), replace(self, u0=um)


@dataclass(frozen=True)
class Piece:
    src_start: int
    src_end: int
    ramps: tuple[Ramp, ...] = ()

    @property
    def length(self) -> int:
        return self.src_end - self.src_start

    def split(self, at: int) -> tuple["Piece", "Piece"]:
        frac = at / self.length
        left, right = [], []
        for r in self.ramps:
            a, b = r.split(frac)
            left.append(a)
            right.append(b)
        return (Piece(self.src_start, self.src_start + at, tuple(left)),
                Piece(self.src_start + at, self.src_end, tuple(right)))

    def gain(self) -> np.ndarray:
        g = np.ones(self.length)
        if not self.ramps:
            return g
        pos = np.arange(self.length) / max(self.length, 1)
        for r in self.ramps:
            g *= r.curve(r.u0 + (r.u1 - r.u0) * pos)
        return g


@dataclass(frozen=True)
class Timeline:
    pieces: tuple[Piece, ...]

    @property
    def length(self) -> int:
        return sum(p.length for p in self.pieces)

    def offsets(self) -> list[int]:
        out, o = [], 0
        for p in self.pieces:
            out.append(o)
            o += p.length
        return out

    def split_at(self, pos: int) -> "Timeline":
        out, o = [], 0
        for p in self.pieces:
            if o < pos < o + p.length:
                a, b = p.split(pos - o)
                out += [a, b]
            else:
                out.append(p)
            o += p.length
        return Timeline(tuple(out))

    def slice(self, a: int, b: int) -> tuple[Piece, ...]:
        tl = self.split_at(a).split_at(b)
        out, o = [], 0
        for p in tl.pieces:
            if o >= a and o + p.length <= b:
                out.append(p)
            o += p.length
        return tuple(out)

    def with_ramp(self, a: int, b: int, ramp: Ramp) -> "Timeline":
        tl = self.split_at(a).split_at(b)
        out, o = [], 0
        for p in tl.pieces:
            if o >= a and o + p.length <= b:
                u0, u1 = (o - a) / (b - a), (o + p.length - a) / (b - a)
                p = Piece(p.src_start, p.src_end, p.ramps + (replace(ramp, u0=u0, u1=u1),))
            out.append(p)
            o += p.length
        return Timeline(tuple(out))


# =============================================================================
# Rendering and grid derivation
# =============================================================================
XFADE_MS = 10.0  # splice crossfade, placed just before the boundary
GAIN_SMOOTH_MS = 6.0  # smoothing of gain steps (e.g. a gain op starting mid-signal)
EDGE_FADE_MS = 2.0  # fade at the very start/end when they are cut mid-signal


def splice_points(tl: Timeline) -> list[int]:
    """Output sample positions where the source is not contiguous (real splices)."""
    pts, o = [], 0
    prev = None
    for p in tl.pieces:
        if prev is not None and prev.src_end != p.src_start:
            pts.append(o)
        prev = p
        o += p.length
    return pts


def render(tl: Timeline, src: np.ndarray, sr: int, xfade_ms: float = XFADE_MS) -> np.ndarray:
    """Render a timeline from the source audio (float32, shape (n, ch))."""
    n_out = tl.length
    ch = src.shape[1]
    out = np.zeros((n_out, ch), dtype=np.float32)
    gain = np.ones(n_out, dtype=np.float64)
    X = int(round(xfade_ms * 1e-3 * sr))
    t = (np.arange(X) + 0.5) / max(X, 1)
    fade_in, fade_out = np.sin(t * np.pi / 2)[:, None], np.cos(t * np.pi / 2)[:, None]
    o = 0
    prev = None
    for p in tl.pieces:
        out[o:o + p.length] = src[p.src_start:p.src_end]
        gain[o:o + p.length] = p.gain()
        if prev is not None and prev.src_end != p.src_start and X > 0:
            # Equal-power crossfade in the X samples *before* the boundary: the outgoing piece
            # fades out over its last X samples while the audio just before the incoming
            # piece's start fades in, so the incoming transient (usually a beat) is intact
            # and every piece keeps its exact length (the grid does not move).
            x = min(X, o)
            pre = np.zeros((x, ch), dtype=np.float32)
            avail = min(x, p.src_start)
            if avail:
                pre[x - avail:] = src[p.src_start - avail:p.src_start]
            out[o - x:o] = out[o - x:o] * fade_out[X - x:] + pre * fade_in[X - x:]
        prev = p
        o += p.length
    # smooth gain discontinuities (gain/mute ops that start mid-signal)
    k = int(GAIN_SMOOTH_MS * 1e-3 * sr) | 1
    if k > 1 and np.any(np.abs(np.diff(gain)) > 1e-3):
        win = np.hanning(k + 2)[1:-1]
        win /= win.sum()
        gain = np.convolve(np.pad(gain, (k // 2, k // 2), mode="edge"), win, mode="valid")
    out *= gain[:, None].astype(np.float32)
    E = int(EDGE_FADE_MS * 1e-3 * sr)
    if tl.pieces and tl.pieces[0].src_start > 0 and E:
        out[:E] *= np.linspace(0, 1, E, dtype=np.float32)[:, None]
    if tl.pieces and tl.pieces[-1].src_end < len(src) and E:
        out[-E:] *= np.linspace(1, 0, E, dtype=np.float32)[:, None]
    return out


def derive_grid(tl: Timeline, src_grid: BeatGrid, sr: int) -> BeatGrid:
    """Beat grid of the edited audio: source beats carried through every piece."""
    src_samp = src_grid.times * sr
    src_round = np.round(src_samp).astype(np.int64)
    times, labels, strength = [], [], []
    o = 0
    for p in tl.pieces:
        m = (src_round >= p.src_start) & (src_round < p.src_end)
        idx = np.flatnonzero(m)
        times.append((o + (src_samp[idx] - p.src_start)) / sr)
        labels.append(src_grid.beat_in_bar[idx])
        if src_grid.strength is not None:
            strength.append(src_grid.strength[idx])
        o += p.length
    times = np.concatenate(times) if times else np.zeros(0)
    # A cut is at a whole sample, the beat between two: when the edit starts the file on a beat (keep_only),
    # that beat can come out a fraction of a sample below 0. It starts at the file's first sample.
    times = np.maximum(times, 0.0)
    labels = np.concatenate(labels) if labels else np.zeros(0, int)
    st = np.concatenate(strength) if strength else None
    ibi = np.diff(times)
    if src_grid.grid_type == "constant" and len(ibi) and np.ptp(ibi) < 1e-3:
        gtype = "constant"
    elif len(ibi) and np.ptp(ibi) < 1e-3:
        gtype = "constant"
    else:
        gtype = "variable" if src_grid.grid_type == "variable" else "piecewise"
    return BeatGrid(times=times, beat_in_bar=labels, beats_per_bar=src_grid.beats_per_bar,
                    time_signature=src_grid.time_signature, grid_type=gtype, duration_s=tl.length / sr,
                    sample_rate=sr, confidence=src_grid.confidence, confidence_detail=src_grid.confidence_detail,
                    strength=st, meta={**src_grid.meta, "derived_from_edit": True})


# =============================================================================
# Resolving ops against the current grid
# =============================================================================
@dataclass
class Resolved:
    op: BaseModel
    k0: int  # beat positions [k0, k1)
    k1: int
    a: int  # output samples [a, b)
    b: int
    pos_k: int | None = None
    pos: int | None = None


def _range_text(grid: BeatGrid, k0: int, k1: int) -> str:
    n = k1 - k0
    bars = grid.bars
    first_bar, last_bar = int(bars[k0]), int(bars[k1 - 1])
    bar_part = f"bar {first_bar}" if first_bar == last_bar else f"bars {first_bar}-{last_bar}"
    t0, t1 = grid.beat_pos_time(k0), grid.beat_pos_time(k1)
    return f"beats {k0}-{k1 - 1} ({n} beat{'s' if n != 1 else ''}, {bar_part}, {t0:.3f}-{t1:.3f} s)"


def resolve_range(grid: BeatGrid, r) -> tuple[int, int]:
    N = grid.n
    if N == 0:
        raise EditError("This track has no beat grid, so beat or bar edits are not possible.")
    if isinstance(r, BeatRange):
        a, b = r.start_beat, r.end_beat
        if a < 0 or b > N:
            raise EditError(f"Beats {a}-{b} are out of range: this track has beats 0-{N - 1} "
                            f"(beat position {N} is the end of the track).")
        if b <= a:
            raise EditError(f"Beat range {a}-{b} is empty or reversed. Ranges run from a start beat to a later end beat.")
        return a, b
    first, last = r.first_bar, r.last_bar
    lo = 0 if grid.pickup_beats else 1
    if first > last:
        raise EditError(f"Bar range {first}-{last} is reversed.")
    if first < lo or last > grid.last_bar:
        extra = " plus a pickup (bar 0)" if grid.pickup_beats else ""
        raise EditError(f"Bars {first}-{last} are out of range: this track has bars 1-{grid.last_bar}{extra} "
                        f"({grid.duration_s:.1f} s at about {grid.tempo_bpm:.1f} BPM).")
    return grid.bar_span(first)[0], grid.bar_span(last)[1]


def resolve_position(grid: BeatGrid, p) -> int:
    N = grid.n
    if isinstance(p, BeatPosition):
        if not 0 <= p.beat <= N:
            raise EditError(f"Beat position {p.beat} is out of range (0-{N}).")
        return p.beat
    if p.bar == grid.last_bar + 1:
        return N
    if not (0 if grid.pickup_beats else 1) <= p.bar <= grid.last_bar:
        raise EditError(f"Bar {p.bar} is out of range: this track has bars 1-{grid.last_bar}.")
    return grid.bar_span(p.bar)[0]


def beat_sample(grid: BeatGrid, k: int, total: int, sr: int) -> int:
    return total if k == grid.n else int(round(grid.times[k] * sr))


# =============================================================================
# Applying a turn
# =============================================================================
@dataclass
class TurnResult:
    timeline: Timeline
    grid: BeatGrid
    echo: list[str]
    warnings: list[str] = field(default_factory=list)


def apply_ops(tl: Timeline, grid: BeatGrid, src_grid: BeatGrid, ops: list, sr: int,
              peak_fn=None) -> TurnResult:
    """Apply one turn's ops. All ops are addressed against the grid at the start of the turn."""
    if not ops:
        raise EditError("No edit operations were given.")
    total = tl.length
    res: list[Resolved] = []
    for op in ops:
        k0, k1 = resolve_range(grid, op.range)
        r = Resolved(op, k0, k1, beat_sample(grid, k0, total, sr), beat_sample(grid, k1, total, sr))
        if isinstance(op, (Move, InsertCopy)):
            target = op.to if isinstance(op, Move) else op.at
            r.pos_k = resolve_position(grid, target)
            r.pos = beat_sample(grid, r.pos_k, total, sr)
            if isinstance(op, Move) and k0 < r.pos_k < k1:
                raise EditError(f"Can't move {_range_text(grid, k0, k1)} to beat {r.pos_k}: the destination is inside "
                                f"the range being moved.")
            if isinstance(op, Move) and r.pos_k in (k0, k1):
                raise EditError(f"Moving {_range_text(grid, k0, k1)} to beat {r.pos_k} would leave the track unchanged.")
        res.append(r)

    shifting = [r for r in res if r.op.op in SHIFTING_OPS]
    if any(r.op.op == "keep_only" for r in shifting) and len(shifting) > 1:
        raise EditError("'Keep only' can't be combined with other cuts in the same message. Send them one at a time.")

    def span(r: Resolved) -> tuple[int, int]:
        lo, hi = r.k0, r.k1
        if r.pos_k is not None:
            lo, hi = min(lo, r.pos_k), max(hi, r.pos_k)
        return lo, hi

    spans = sorted(span(r) for r in shifting)
    for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
        if b0 < a1:
            raise EditError("Two of the edits in this message overlap, so the order would change the result. "
                            "Please send them as separate messages.")

    echo, warnings = [], []
    # Non-shifting ops first (positions do not move), then shifting ops right to left so
    # that the turn-start positions of the remaining ops stay valid.
    for r in [r for r in res if r.op.op not in SHIFTING_OPS]:
        op = r.op
        if op.op == "fade":
            tl = tl.with_ramp(r.a, r.b, Ramp("fade_" + op.direction, op.shape))
            where = _range_text(grid, r.k0, r.k1)
            echo.append(f"FADE {op.direction.upper()} ({op.shape.replace('_', '-')}) over {where}.")
            if op.direction == "out" and r.k1 != grid.n:
                warnings.append(f"The fade-out ends at beat {r.k1}; audio after it plays at full level again.")
            if op.direction == "in" and r.k0 != 0:
                warnings.append(f"The fade-in starts at beat {r.k0}; audio before it is unchanged.")
        elif op.op == "gain":
            tl = tl.with_ramp(r.a, r.b, Ramp("gain", db=op.db))
            echo.append(f"GAIN {op.db:+.1f} dB on {_range_text(grid, r.k0, r.k1)}.")
        elif op.op == "mute":
            tl = tl.with_ramp(r.a, r.b, Ramp("gain", db=-200.0))
            echo.append(f"MUTE {_range_text(grid, r.k0, r.k1)} (length kept, so the grid does not move).")

    for r in sorted(shifting, key=lambda r: span(r)[0], reverse=True):
        op = r.op
        pieces = tl.pieces
        where = _range_text(grid, r.k0, r.k1)
        dur = (r.b - r.a) / sr
        if op.op == "remove":
            if r.a == 0 and r.b == total or (r.k0 == 0 and r.k1 == grid.n and grid.times[0] * sr < 1):
                raise EditError("That would remove the whole track.")
            tl = Timeline(tl.slice(0, r.a) + tl.slice(r.b, tl.length))
            echo.append(f"REMOVE {where}: {dur:.3f} s cut; what followed now starts at {r.a / sr:.3f} s.")
        elif op.op == "keep_only":
            tl = Timeline(tl.slice(r.a, r.b))
            echo.append(f"KEEP ONLY {where}; everything else (including any pre-roll and tail) is removed.")
        elif op.op == "repeat":
            seg = tl.slice(r.a, r.b)
            tl = Timeline(tl.slice(0, r.a) + seg * op.total_plays + tl.slice(r.b, tl.length))
            echo.append(f"REPEAT {where} so it plays {op.total_plays}x in total ({op.total_plays - 1} extra "
                        f"cop{'y' if op.total_plays == 2 else 'ies'} inserted at {r.b / sr:.3f} s, "
                        f"+{dur * (op.total_plays - 1):.3f} s); the track then continues.")
        elif op.op == "insert_copy":
            seg = tl.slice(r.a, r.b)
            tl = Timeline(tl.slice(0, r.pos) + seg * op.copies + tl.slice(r.pos, tl.length))
            echo.append(f"INSERT {op.copies} cop{'y' if op.copies == 1 else 'ies'} of {where} before beat {r.pos_k} "
                        f"({r.pos / sr:.3f} s), +{dur * op.copies:.3f} s.")
        elif op.op == "move":
            seg = tl.slice(r.a, r.b)
            if r.pos < r.a:
                tl = Timeline(tl.slice(0, r.pos) + seg + tl.slice(r.pos, r.a) + tl.slice(r.b, tl.length))
            else:
                tl = Timeline(tl.slice(0, r.a) + tl.slice(r.b, r.pos) + seg + tl.slice(r.pos, tl.length))
            echo.append(f"MOVE {where} to just before beat {r.pos_k} ({r.pos / sr:.3f} s in the pre-edit track).")
        if op.op in ("remove", "repeat", "insert_copy", "move"):
            m = grid.beats_per_bar
            aligned = grid.beat_in_bar[r.k0] == 1 and ((r.k1 - r.k0) % m == 0) and \
                (r.k1 == grid.n or grid.beat_in_bar[r.k1] == 1)
            if not aligned:
                warnings.append(f"{op.op} range {where} is not whole bars, so the bar lines after it shift; "
                                f"the new grid keeps each beat's original position-in-bar label.")
        if tl.length == 0:
            raise EditError("That would leave an empty track.")
        del pieces

    new_grid = derive_grid(tl, src_grid, sr)
    if peak_fn is not None:
        peak = peak_fn(tl)
        if peak > 1.0:
            raise EditError(f"That would clip: the peak would be {20 * np.log10(peak):+.1f} dBFS. "
                            f"Try a smaller gain (at most {20 * np.log10(1 / peak) + _max_gain(ops):+.1f} dB).")
    echo.append(f"Track length {total / sr:.3f} s -> {tl.length / sr:.3f} s; grid now {new_grid.n} beats, "
                f"bars 1-{new_grid.last_bar}.")
    return TurnResult(tl, new_grid, echo, warnings)


def _max_gain(ops) -> float:
    g = [op.db for op in ops if getattr(op, "op", None) == "gain"]
    return max(g) if g else 0.0
