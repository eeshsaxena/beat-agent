"""The offline grammar: paraphrases other than the brief's commands, and its refusals."""
from __future__ import annotations

import numpy as np
import pytest

from beatagent.grid import BeatGrid
from beatagent.offline import interpret


def grid(n=128, pickup=0):
    t = 0.5 + np.arange(n) * 0.5
    return BeatGrid(times=t, beat_in_bar=(np.arange(n) - pickup) % 4 + 1, beats_per_bar=4, time_signature="4/4",
                    grid_type="constant", duration_s=float(t[-1] + 1), sample_rate=44100)


def ops(text, g=None):
    p = interpret(text, g or grid(), "T1", ("T2",))
    assert p.calls, p.reply
    tool, inp = p.calls[0]
    return tool, inp.get("ops", inp)


@pytest.mark.parametrize("text, expect", [
    ("loop bars 9 through 12 three times", {"op": "repeat", "range": {"unit": "bars", "first_bar": 9, "last_bar": 12},
                                            "total_plays": 3}),
    ("please repeat beats 4-12 twice", {"op": "repeat", "range": {"unit": "beats", "start_beat": 4, "end_beat": 12},
                                        "total_plays": 2}),
    ("delete beat 3", {"op": "remove", "range": {"unit": "beats", "start_beat": 3, "end_beat": 4}}),
    ("get rid of the last 2 bars", {"op": "remove", "range": {"unit": "bars", "first_bar": 31, "last_bar": 32}}),
    ("fade in the first four bars", {"op": "fade", "range": {"unit": "bars", "first_bar": 1, "last_bar": 4},
                                     "direction": "in"}),
    ("make bars 5-8 quieter by 4.5 dB", {"op": "gain", "range": {"unit": "bars", "first_bar": 5, "last_bar": 8},
                                         "db": -4.5}),
    ("keep only bars 17 to 24", {"op": "keep_only", "range": {"unit": "bars", "first_bar": 17, "last_bar": 24}}),
    ("silence bar 7", {"op": "mute", "range": {"unit": "bars", "first_bar": 7, "last_bar": 7}}),
    ("move bars 1-4 after bar 16", {"op": "move", "range": {"unit": "bars", "first_bar": 1, "last_bar": 4},
                                    "to": {"unit": "bars", "bar": 17}}),
])
def test_paraphrases(text, expect):
    tool, o = ops(text)
    assert tool == "apply_edits" and o == [expect]


def test_undo_redo_and_subjective():
    assert interpret("undo that", grid(), "T1").calls == [("undo", {})]
    assert interpret("Redo.", grid(), "T1").calls == [("redo", {})]
    assert interpret("cut the dull section", grid(), "T1").calls == [("find_sections", {})]


def test_sync():
    p = interpret("sync T2 to T1 at bar 9", grid(), "T1", ("T2",))
    assert p.calls == [("sync_tracks", {"other_track": "T2", "start_bar_this": 9, "start_bar_other": 1})]
    assert not interpret("sync T3 to T1", grid(), "T1", ("T2",)).calls  # unknown track: refused


@pytest.mark.parametrize("text", ["remove bar 5 and bar 9",  # two ranges: refuse rather than drop one
                                  "make it groovier",  # no range, not a known subjective target
                                  "bars 3-4"])  # range but no action
def test_refuses_what_it_cannot_parse(text):
    p = interpret(text, grid(), "T1")
    assert not p.calls and p.reply.startswith("Offline grammar")


def test_repeat_without_count_is_two_plays_and_says_so():
    p = interpret("repeat bar 2", grid(), "T1")
    assert p.calls[0][1]["ops"][0]["total_plays"] == 2 and "twice in total" in p.reply


def _score():
    from beatagent import score as S

    return {"schema": S.SCHEMA, "title": "t", "tempo_bpm": 110.0, "time_signature": [4, 4], "beats_per_bar": 4,
            "bars": 1, "loop_beats": 4, "tracks": [
                {"id": "drums", "name": "Drums", "kind": "drums", "patch": dict(S.DEFAULT_PATCHES["drums"]), "notes": []},
                {"id": "bass", "name": "Bass", "kind": "pitched", "patch": dict(S.DEFAULT_PATCHES["bass"]), "notes": []}]}


@pytest.mark.parametrize("text, op", [
    ("mute the drums", {"op": "set_mute", "track": "drums", "muted": True}),
    ("make the bass brighter", {"op": "brightness", "track": "bass", "factor": 1.5}),
    ("play it at 140", {"op": "set_tempo", "bpm": 140.0}),
    ("bass up an octave", {"op": "transpose", "track": "bass", "semitones": 12}),
    ("turn the bassline down 6 dB", {"op": "set_gain", "track": "bass", "db": -6.0}),
])
def test_recreation_grammar(text, op):
    from beatagent.offline import interpret_score

    p = interpret_score(text, _score())
    assert p.calls == [("apply_score_ops", {"ops": [op]})]


def test_recreation_grammar_refuses_inventing_notes():
    from beatagent.offline import interpret_score

    p = interpret_score("add a guitar", _score())
    assert not p.calls and "inventing notes" in p.reply


def test_confirm_mode_never_applies_before_the_user_does():
    from beatagent.agent import EditAgent
    from beatagent.session import EditSession
    from tests.test_core import make_audio, make_grid

    g = make_grid()
    s = EditSession(make_audio(g), g, "T1")
    ag = EditAgent(s)
    ag.interpreter, ag.confirm = "offline", True
    r = ag.ask("remove bar 2")
    assert s.grid.n == 64 and not s.can_undo()  # nothing changed
    assert [e["type"] for e in r.events] == ["proposed"] and r.events[0]["ops"][0]["op"] == "remove"
    ag.confirm = False
    ag.ask("remove bar 2")
    assert s.grid.n == 60
