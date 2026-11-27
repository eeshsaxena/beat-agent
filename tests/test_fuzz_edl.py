"""Randomised multi-turn edit sessions on synthetic grids, checking the engine's invariants.

After every applied turn: render length == timeline length == grid duration; beat times strictly
increasing and inside the file; with structural ops only, every grid beat lands on a click in the
rendered audio; undo and redo restore the exact render and grid. Anything but an EditError fails.
The fuller version (more seeds) is how a sub-sample negative beat time after keep_only was found.
"""
from __future__ import annotations

import numpy as np
import pytest

from beatagent.edl import (BarPosition, BarRange, BeatPosition, BeatRange, EditError, Fade, Gain, InsertCopy,
                           KeepOnly, Move, Mute, Remove, Repeat)
from beatagent.grid import BeatGrid
from beatagent.session import EditSession
from tests.test_core import SR, make_audio


def grid(rng):
    n = int(rng.integers(12, 70))
    bpm = float(rng.choice([90, 120, 128, 140]))
    P = 60 / bpm
    pickup = int(rng.integers(0, 4))
    jitter = rng.choice([0.0, 0.004])
    times = 0.3 + np.arange(n) * P + rng.normal(0, jitter, n) * (np.arange(n) > 0)
    times = np.maximum.accumulate(times)
    pos = (np.arange(n) - pickup) % 4 + 1
    tail = float(rng.choice([0.05, P, 1.0]))
    return BeatGrid(times=times, beat_in_bar=pos, beats_per_bar=4, time_signature="4/4",
                    grid_type="constant" if jitter == 0 else "variable",
                    duration_s=float(times[-1] + tail), sample_rate=SR)


def rrange(rng, g):
    if rng.random() < 0.5:
        a = int(rng.integers(-2, g.n + 2))
        b = a + int(rng.integers(-1, 20))
        return BeatRange(start_beat=a, end_beat=b)
    lb = g.last_bar
    a = int(rng.integers(-1, lb + 3))
    return BarRange(first_bar=a, last_bar=a + int(rng.integers(-1, 5)))


def rpos(rng, g):
    if rng.random() < 0.5:
        return BeatPosition(beat=int(rng.integers(-1, g.n + 2)))
    return BarPosition(bar=int(rng.integers(-1, g.last_bar + 3)))


def rop(rng, g, structural_only):
    kinds = ["remove", "keep_only", "repeat", "move", "insert_copy"] + ([] if structural_only else ["fade", "gain", "mute"])
    k = rng.choice(kinds)
    r = rrange(rng, g)
    if k == "remove":
        return Remove(range=r)
    if k == "keep_only":
        return KeepOnly(range=r)
    if k == "repeat":
        return Repeat(range=r, total_plays=int(rng.integers(2, 5)))
    if k == "move":
        return Move(range=r, to=rpos(rng, g))
    if k == "insert_copy":
        return InsertCopy(range=r, at=rpos(rng, g), copies=int(rng.integers(1, 3)))
    if k == "fade":
        return Fade(range=r, direction=str(rng.choice(["in", "out"])),
                    shape=str(rng.choice(["equal_power", "linear", "exponential"])))
    if k == "gain":
        return Gain(range=r, db=float(rng.choice([-12, -3, 0, 2])))
    return Mute(range=r)


def check(s, structural, where):
    y = s.render()
    g = s.grid
    L = s.state.timeline.length
    assert len(y) == L, f"{where}: render {len(y)} != timeline {L}"
    assert abs(L - round(g.duration_s * SR)) <= 1, f"{where}: timeline {L} vs grid duration {g.duration_s * SR:.1f}"
    t = np.asarray(g.times)
    assert len(t) == g.n, f"{where}: n mismatch"
    if len(t):
        assert np.all(np.diff(t) > 0), f"{where}: beat times not increasing"
        assert t[0] >= -1e-9 and t[-1] < g.duration_s + 1e-9, f"{where}: beat outside file ({t[0]}, {t[-1]}, {g.duration_s})"
    assert len(g.beat_in_bar) == g.n and len(g.bars) == g.n, f"{where}: label arrays"
    if structural:
        m = y[:, 0]
        for i, tt in enumerate(t):
            k = int(round(tt * SR))
            w = m[max(0, k - 8):k + 40]
            assert np.abs(w).max() > 0.3, f"{where}: no click at beat {i} ({tt:.4f} s)"


def run(seed, structural):
    rng = np.random.default_rng(seed)
    g = grid(rng)
    s = EditSession(make_audio(g), g, "fuzz")
    applied = refused = 0
    for turn in range(1, 9):
        if s.grid.n == 0:
            break
        ops = [rop(rng, s.grid, structural) for _ in range(int(rng.integers(1, 3)))]
        before_y, before_t = s.render().copy(), np.asarray(s.grid.times).copy()
        try:
            s.apply(ops, turn_id=turn)
        except EditError:
            refused += 1
            continue
        applied += 1
        where = f"seed {seed} turn {turn} ops {[o.model_dump() for o in ops]}"
        check(s, structural, where)
        after_y, after_t = s.render().copy(), np.asarray(s.grid.times).copy()
        if rng.random() < 0.4:
            s.undo()
            assert np.array_equal(s.render(), before_y), f"{where}: undo render differs"
            assert np.array_equal(np.asarray(s.grid.times), before_t), f"{where}: undo grid differs"
            s.redo()
            assert np.array_equal(s.render(), after_y), f"{where}: redo render differs"
            assert np.array_equal(np.asarray(s.grid.times), after_t), f"{where}: redo grid differs"
    return applied, refused


@pytest.mark.parametrize("seed", range(12))
def test_random_edit_sessions_keep_the_invariants(seed):
    applied, _ = run(2 * seed, structural=True)
    run(2 * seed + 1, structural=False)
    assert applied >= 0


def test_keep_only_never_puts_a_beat_before_the_file_start():
    """Beats between two samples, a cut rounded up past one: the first beat used to come out at -0.00004 s."""
    P = 0.5
    times = 0.5 + 0.7 / SR + np.arange(32) * P
    g = BeatGrid(times=times, beat_in_bar=np.arange(32) % 4 + 1, beats_per_bar=4, time_signature="4/4",
                 grid_type="constant", duration_s=float(times[-1] + P), sample_rate=SR)
    s = EditSession(make_audio(g), g, "t")
    s.apply([KeepOnly(range=BarRange(first_bar=2, last_bar=3))], turn_id=1)
    assert s.grid.times[0] == 0.0 and np.all(s.grid.times >= 0)
