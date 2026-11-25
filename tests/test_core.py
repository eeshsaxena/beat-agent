"""Fast, model-free tests of the conventions and invariants the agent relies on.

    python -m pytest -q
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from beatagent import audio_io, score
from beatagent.edl import (BarPosition, BarRange, BeatRange, EditError, Fade, Gain, InsertCopy, Move, Piece, Remove,
                           Repeat, Timeline, render, resolve_range)
from beatagent.grid import BeatGrid
from beatagent.quality import butt_splice_scores, splice_click_scores
from beatagent.session import EditSession
from beatagent.tracking import fit_constant, number_beats, segment_piecewise

SR = 8000


def make_grid(n=64, bpm=120.0, t0=0.5, pickup=0, sr=SR, dur=None) -> BeatGrid:
    P = 60 / bpm
    times = t0 + np.arange(n) * P
    pos = (np.arange(n) - pickup) % 4 + 1
    return BeatGrid(times=times, beat_in_bar=pos, beats_per_bar=4, time_signature="4/4", grid_type="constant",
                    duration_s=dur or float(times[-1] + P + 0.5), sample_rate=sr)


def make_audio(g: BeatGrid, sr=SR) -> audio_io.Audio:
    n = int(g.duration_s * sr)
    rng = np.random.default_rng(0)
    x = 0.05 * rng.standard_normal(n)
    for t in g.times:  # a decaying click on every beat
        i = int(round(t * sr))
        k = np.arange(min(400, n - i))
        x[i:i + len(k)] += 0.5 * np.sin(2 * np.pi * 200 * k / sr) * np.exp(-k / 80)
    x = np.stack([x, x], axis=1).astype(np.float32)
    return audio_io.Audio(x, sr, "PCM_24", "WAV")


# ------------------------------------------------------------------ conventions
def test_beat_ranges_are_zero_based_half_open():
    g = make_grid()
    assert resolve_range(g, BeatRange(start_beat=16, end_beat=32)) == (16, 32)  # "beat 16-32" = 16 beats


def test_bar_ranges_are_one_based_inclusive():
    g = make_grid()
    assert resolve_range(g, BarRange(first_bar=5, last_bar=5)) == (16, 20)
    assert resolve_range(g, BarRange(first_bar=1, last_bar=4)) == (0, 16)


def test_pickup_beats_are_bar_zero():
    g = make_grid(pickup=2)
    assert g.pickup_beats == 2 and g.bars[0] == 0 and g.bars[2] == 1
    assert resolve_range(g, BarRange(first_bar=1, last_bar=1)) == (2, 6)


@pytest.mark.parametrize("r, msg", [(BarRange(first_bar=200, last_bar=210), "out of range"),
                                    (BeatRange(start_beat=32, end_beat=16), "empty or reversed"),
                                    (BeatRange(start_beat=0, end_beat=999), "out of range")])
def test_bad_ranges_are_refused_with_a_reason(r, msg):
    with pytest.raises(EditError, match=msg):
        resolve_range(make_grid(), r)


# ------------------------------------------------------------------ edits
def session(**kw):
    g = make_grid(**kw)
    return EditSession(make_audio(g), g, "t")


def test_repeat_twice_means_two_plays_in_total():
    s = session()
    n0, len0 = s.grid.n, s.state.timeline.length
    s.apply([Repeat(range=BeatRange(start_beat=16, end_beat=32), total_plays=2)], turn_id=1)
    assert s.grid.n == n0 + 16
    assert s.state.timeline.length - len0 == pytest.approx(16 * 0.5 * SR, abs=2)


def test_grid_stays_true_after_edits():
    s = session()
    s.apply([Repeat(range=BeatRange(start_beat=4, end_beat=12), total_plays=3)], turn_id=1)
    s.apply([Remove(range=BarRange(first_bar=2, last_bar=2))], turn_id=2)
    s.apply([Move(range=BarRange(first_bar=1, last_bar=1), to=BarPosition(bar=6))], turn_id=3)
    # constant grid stays constant (equal inter-beat intervals across every splice)
    assert np.ptp(np.diff(s.grid.times)) < 1e-3
    # every grid beat lands on a click in the rendered audio
    y = s.render()[:, 0]
    for t in s.grid.times[:40]:
        i = int(round(t * SR))
        assert np.abs(y[i:i + 40]).max() > 0.3


def test_ops_in_one_turn_use_turn_start_numbering():
    s = session()
    t2 = float(s.grid.times[resolve_range(s.grid, BarRange(first_bar=4, last_bar=4))[1]])
    s.apply([Remove(range=BarRange(first_bar=2, last_bar=2)), Remove(range=BarRange(first_bar=4, last_bar=4))], turn_id=1)
    assert s.grid.n == 64 - 8
    # original bar 5 now starts where original bar 3 started (two bars earlier)
    assert s.grid.times[8] == pytest.approx(t2 - 2 * 4 * 0.5, abs=1e-6)


def test_overlapping_ops_in_one_turn_are_refused():
    s = session()
    with pytest.raises(EditError, match="overlap"):
        s.apply([Remove(range=BarRange(first_bar=2, last_bar=3)), Repeat(range=BarRange(first_bar=3, last_bar=4),
                                                                         total_plays=2)], turn_id=1)


def test_undo_redo_is_per_turn():
    s = session()
    s.apply([Remove(range=BarRange(first_bar=2, last_bar=2))], turn_id=1)
    s.apply([Fade(range=BeatRange(start_beat=s.grid.n - 8, end_beat=s.grid.n), direction="out")], turn_id=2)
    s.apply([Gain(range=BarRange(first_bar=1, last_bar=1), db=-3)], turn_id=2)  # same turn: same undo step
    assert len(s.history) == 3
    s.undo()
    assert s.grid.n == 60 and s.can_redo()
    s.redo()
    s.undo()
    s.undo()
    assert s.grid.n == 64 and not s.can_undo()
    with pytest.raises(EditError):
        s.undo()


def test_move_into_its_own_range_is_refused():
    s = session()
    with pytest.raises(EditError, match="inside"):
        s.apply([Move(range=BarRange(first_bar=2, last_bar=4), to=BarPosition(bar=3))], turn_id=1)


def test_clipping_gain_is_refused():
    s = session()
    with pytest.raises(EditError, match="clip"):
        s.apply([Gain(range=BarRange(first_bar=1, last_bar=2), db=24)], turn_id=1)


def test_insert_copy_and_non_bar_aligned_warning():
    s = session()
    res = s.apply([InsertCopy(range=BeatRange(start_beat=0, end_beat=3), at=BarPosition(bar=3), copies=1)], turn_id=1)
    assert s.grid.n == 67
    assert any("not whole bars" in w for w in res.warnings)


# ------------------------------------------------------------------ audio quality and format
def test_untouched_samples_round_trip_bit_exact(tmp_path):
    rng = np.random.default_rng(1)
    x = (rng.integers(-2 ** 23, 2 ** 23, size=(4000, 2)) / 2 ** 23).astype(np.float32)
    p = tmp_path / "in.wav"
    sf.write(p, x, 48000, subtype="PCM_24")
    a = audio_io.load(p)
    out = audio_io.save_like(tmp_path / "out", a.samples, a)
    b = sf.read(out, dtype="int32")[0]
    assert sf.info(out).subtype == "PCM_24" and sf.info(out).samplerate == 48000
    assert np.array_equal(b, sf.read(p, dtype="int32")[0])


def test_crossfaded_splice_has_no_click_where_a_butt_splice_does():
    t = np.arange(SR * 2) / SR
    x = np.stack([0.5 * np.sin(2 * np.pi * 220 * t)] * 2, axis=1).astype(np.float32)
    tl = Timeline((Piece(0, 6000), Piece(6017, len(x))))  # join two different phases mid-cycle
    good = splice_click_scores(tl, x, render(tl, x, SR), SR)[0]["ratio"]
    bad = butt_splice_scores(tl, x, SR)[0]["ratio"]
    assert good < 5 < bad


# ------------------------------------------------------------------ score / page
def _score():
    return {"schema": score.SCHEMA, "title": "t", "tempo_bpm": 100.0, "time_signature": [4, 4], "beats_per_bar": 4,
            "bars": 1, "loop_beats": 4, "tracks": [
                {"id": "bass", "name": "Bass", "kind": "pitched", "patch": dict(score.DEFAULT_PATCHES["bass"]),
                 "notes": [{"start": 0.0, "duration": 1.0, "pitch": 40, "velocity": 0.8}]}]}


def test_page_cannot_be_broken_out_of(tmp_path):
    s = _score()
    s["title"] = "</script><script>alert(1)</script>"
    html = score.build_page(s, tmp_path / "p.html").read_text(encoding="utf-8")
    assert "<script>alert(1)" not in html
    assert "default-src 'none'" in html


def test_score_validation_rejects_bad_data():
    s = _score()
    s["tracks"][0]["notes"][0]["pitch"] = "60; fetch('x')"
    with pytest.raises(score.ScoreError):
        score.validate(s)


def test_score_ops():
    s, echo = score.apply_score_ops(_score(), [score.SetTempo(bpm=140), score.Brightness(track="bass", factor=1.5),
                                               score.SetMute(track="bass", muted=True)])
    assert s["tempo_bpm"] == 140 and s["tracks"][0]["mute"] is True
    assert s["tracks"][0]["patch"]["cutoff_hz"] == pytest.approx(score.DEFAULT_PATCHES["bass"]["cutoff_hz"] * 1.5)
    with pytest.raises(score.ScoreError, match="no track"):
        score.apply_score_ops(_score(), [score.SetMute(track="drums", muted=True)])


# ------------------------------------------------------------------ tracking maths
def test_constant_fit_recovers_tempo_through_jitter_and_missing_beats():
    rng = np.random.default_rng(3)
    true = 0.3 + np.arange(400) * (60 / 128)
    raw = np.delete(true + rng.normal(0, 0.008, len(true)), [10, 11, 200, 333])
    raw = np.sort(np.concatenate([raw, [5.123, 77.7]]))  # two spurious detections
    t0, P, k, res = fit_constant(raw, float(np.median(np.diff(raw))))
    assert 60 / P == pytest.approx(128, abs=0.01)
    grid = t0 + np.arange(k.min(), k.max() + 1) * P
    assert np.abs(grid[:len(true)] - true).max() < 0.004


def test_number_beats_leaves_gaps_for_missed_beats():
    t = np.arange(20) * 0.5
    k, tt = number_beats(np.delete(t, [7]))
    assert list(k[:9]) == [0, 1, 2, 3, 4, 5, 6, 8, 9]


def test_piecewise_segmentation_finds_a_tempo_change():
    a = np.arange(64) * 0.5
    b = a[-1] + 0.5 + np.arange(64) * (60 / 128)
    t = np.concatenate([a, b])
    segs = segment_piecewise(np.arange(len(t)), t, sigma=0.005)
    assert len(segs) == 2 and abs(segs[1][0] - 64) <= 1


# ------------------------------------------------------------------ answer key grading
def test_grading_compares_resolved_ranges_not_syntax():
    from evals.answer_key import KEY, grade

    g = make_grid()
    entry = KEY["s1_A1"][3]  # "Remove bar 5."
    as_beats = {"events": [{"type": "applied", "ops": [Remove(range=BeatRange(start_beat=16, end_beat=20)).model_dump()]}]}
    wrong = {"events": [{"type": "applied", "ops": [Remove(range=BeatRange(start_beat=20, end_beat=24)).model_dump()]}]}
    assert grade(entry, as_beats, g)["ok"]
    assert not grade(entry, wrong, g)["ok"]


def test_grid_json_round_trip():
    g = make_grid(pickup=1)
    d = json.loads(json.dumps(g.to_dict()))
    g2 = BeatGrid.from_dict(d)
    assert np.allclose(g2.times, g.times, atol=1e-6) and list(g2.bars) == list(g.bars)
    assert d["conventions"]["beat_ranges"].startswith("'beat a to beat b' is half-open")


def test_edl_export_describes_the_edit(tmp_path):
    s = session()
    s.apply([Repeat(range=BeatRange(start_beat=16, end_beat=32), total_plays=2)], turn_id=1)
    s.apply([Remove(range=BarRange(first_bar=1, last_bar=1))], turn_id=2)
    s.undo()
    paths = s.export(tmp_path, "t", listen_copy=False)
    edl = json.loads(open(paths["edl"]).read())
    assert [h["active"] for h in edl["history"]] == [True, False]
    tl = edl["timeline"]
    assert tl[0]["out_start_s"] == 0 and tl[-1]["out_end_s"] == pytest.approx(edl["duration_s"])
    assert len(edl["splices_s"]) == 1 and edl["history"][0]["ops"][0]["total_plays"] == 2
