"""Malformed input of any shape is refused cleanly, never crashes: the score validator (the gate before
a score is embedded in a page), the offline grammars, and the section analysis on low sample rates.
_private-style fuzzers found the cases pinned here."""
from __future__ import annotations

import copy
import json
import math
import random
from pathlib import Path

import pytest

from beatagent import score as S
from beatagent.agent import EditAgent
from beatagent.offline import interpret_score
from beatagent.sections import find_sections
from beatagent.session import EditSession
from tests.test_core import make_audio, make_grid

SCORE = json.loads((Path(__file__).resolve().parents[1] / "evals" / "out" / "part3" / "P2" / "score.json").read_text())


@pytest.mark.parametrize("path,value", [
    ((), []), (("tracks",), 5), (("tracks",), None), (("tracks", 0), "drums"), (("tracks", 0, "id"), ["x"]),
    (("loop_beats",), math.nan), (("loop_beats",), math.inf), (("tempo_bpm",), math.nan), (("tempo_bpm",), True),
    (("tracks", 0, "gain_db"), 1e20), (("tracks", 0, "gain_db"), "loud"), (("tracks", 0, "mute"), "yes"),
    (("tracks", 0, "notes", 0, "duration"), math.inf), (("tracks", 0, "notes", 0, "pitch"), math.nan),
    (("tracks", 0, "notes"), {"a": 1}), (("tracks", 0, "notes", 0), 7), (("tracks", 0, "patch"), None),
    (("tracks", 1, "patch", "cutoff_hz"), math.inf), (("tracks", 0, "name"), "x" * 500),
])
def test_validator_refuses_malformed_scores_with_a_score_error(path, value):
    bad = copy.deepcopy(SCORE)
    if not path:
        bad = value
    else:
        node = bad
        for k in path[:-1]:
            node = node[k]
        node[path[-1]] = value
    with pytest.raises(S.ScoreError):
        S.validate(bad)


def test_validator_never_raises_anything_but_score_error():
    rng = random.Random(0)
    junk = [None, "x" * 300, -1, math.inf, math.nan, [], {}, True, 10**20, "</script>"]

    def corrupt(x, depth=0):
        if isinstance(x, dict) and x:
            x = dict(x)
            k = rng.choice(list(x))
            x[k] = corrupt(x[k], depth + 1) if depth < 4 and rng.random() < 0.5 else rng.choice(junk)
            return x
        if isinstance(x, list) and x:
            x = list(x)
            i = rng.randrange(len(x))
            x[i] = corrupt(x[i], depth + 1) if depth < 4 else rng.choice(junk)
            return x
        return rng.choice(junk)

    S.validate(copy.deepcopy(SCORE))
    for _ in range(400):
        try:
            S.validate(corrupt(copy.deepcopy(SCORE)))
        except S.ScoreError:
            pass


@pytest.mark.parametrize("sr", [8000, 16000, 22050])
def test_sections_work_on_low_sample_rates(sr):
    """The 'air' band (8-16 kHz) is above Nyquist at 8 and 16 kHz; it used to crash the filter design."""
    g = make_grid(48, sr=sr)
    secs = find_sections(make_audio(g, sr=sr).samples, sr, g)
    assert secs and all(set(s["band_energy_db_vs_track"]) == {"sub", "low-mid", "mid", "presence", "air"} for s in secs)


@pytest.mark.parametrize("text", ["chop out the boring part", "remove bar 5 and bar 9", "remove bars 1-999999999999",
                                  "repeat bar 3 64 times", "\x00 remove \\b bar", "fade out the last ∞ beats", ""])
def test_offline_edit_grammar_refuses_or_applies_but_never_crashes(text):
    g = make_grid(64)
    ag = EditAgent(EditSession(make_audio(g), g, "A1"))
    ag.interpreter = "offline"
    r = ag.ask(text)
    assert r.text or r.events


@pytest.mark.parametrize("text", ["play it at 0", "play it at 99999", "transpose the drums up 3", "make the keys brighter",
                                  "much darker drums", "solo the bass", "", "<script>"])
def test_recreation_grammar_never_crashes(text):
    plan = interpret_score(text, copy.deepcopy(SCORE))
    assert plan.reply
