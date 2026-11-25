"""Expected interpretation of every chat command in the brief's sessions.

The key encodes the conventions in DECISIONS.md (0-based half-open beats, 1-based inclusive
bars, "repeat twice" = plays twice in total, "cut X and repeat" = take X, subjective requests
are answered with measured candidates and a question, out-of-range requests are refused).

Two uses:
* reference mode: the engine executes the key's ops directly, so Part 2's audio, grid and
  click checks run without the LLM (clearly labelled as such in the results);
* grading: each live (or replayed) agent turn is compared with the key. Comparison is on the
  ops *resolved against the grid* (operation, beat range, parameters), so "bar 5" and
  "beats 16-20" count as the same edit.
"""
from __future__ import annotations

from beatagent.edl import BarRange, BeatRange, Fade, Remove, Repeat, resolve_range
from beatagent.grid import BeatGrid


def _last_full_bar_before_first_change(g: BeatGrid) -> int:
    segs = g.tempo_segments()
    if len(segs) < 2:
        raise ValueError("no tempo change in this grid")
    return g._last_full_bar_before(segs[1]["start_index"])


KEY: dict[str, list[dict]] = {
    "s1_A1": [
        {"expect": "apply", "ops": lambda g: [Repeat(range=BeatRange(start_beat=16, end_beat=32), total_plays=2)],
         "why": "take beats 16-31 and play them twice in total, in place"},
        {"expect": "apply", "ops": lambda g: [Fade(range=BeatRange(start_beat=g.n - 8, end_beat=g.n), direction="out")],
         "why": "the last 8 beats of the current (edited) track, to the end of the file"},
        {"expect": "undo", "why": "undo the fade"},
        {"expect": "apply", "ops": lambda g: [Remove(range=BarRange(first_bar=5, last_bar=5))],
         "why": "bar 5 of the current grid (the first bar of the repeated section)"},
    ],
    "s2_A2": [
        {"expect": "apply", "ops": lambda g: [Repeat(range=BarRange(first_bar=1, last_bar=4), total_plays=4)],
         "why": "bars 1-4 play four times in total, then the track continues"},
    ],
    "s3_A5": [
        {"expect": "apply", "ops": lambda g: [Repeat(range=BarRange(first_bar=_last_full_bar_before_first_change(g),
                                                                   last_bar=_last_full_bar_before_first_change(g)),
                                                     total_plays=2)],
         "why": "the last full bar of the first tempo segment, played twice in total"},
    ],
    "s4_A1": [
        {"expect": "ask", "why": "'boring' is subjective: show measured candidate sections and ask; no edit"},
    ],
    "s5_A3": [
        {"expect": "apply", "ops": lambda g: [Remove(range=BeatRange(start_beat=0, end_beat=8))],
         "why": "remove beats 0-7; the pre-roll before beat 0 stays"},
        {"expect": "refuse", "literal_ops": lambda g: [Remove(range=BarRange(first_bar=200, last_bar=210))],
         "why": "bars 200-210 do not exist; say so with the real range, no edit"},
    ],
    "s6_A1": [
        {"expect": "sync", "args": {"other_track": "A7", "start_bar_this": 33, "start_bar_other": 1},
         "why": "A1 is the master; A7 enters at its bar 1 on A1's bar 33"},
    ],
}


def _resolved(ops, g: BeatGrid) -> list[tuple]:
    out = []
    for op in ops:
        k0, k1 = resolve_range(g, op.range)
        extra = tuple(sorted((k, v) for k, v in op.model_dump().items() if k not in ("op", "range")))
        out.append((op.op, k0, k1, extra))
    return sorted(out)


def grade(entry: dict, reply: dict, grid_before: BeatGrid) -> dict:
    """Did the agent's turn match the key? Returns {ok, expected, got, note}."""
    from beatagent.edl import EditOp
    from pydantic import TypeAdapter

    ev = reply.get("events", [])
    applied = [e for e in ev if e["type"] == "applied"]
    kinds = {e["type"] for e in ev}
    exp = entry["expect"]
    if exp == "apply":
        want = _resolved(entry["ops"](grid_before), grid_before)
        got_ops = [o for e in applied for o in e["ops"]]
        try:
            got = _resolved(TypeAdapter(list[EditOp]).validate_python(got_ops), grid_before) if got_ops else []
        except Exception as e:  # noqa: BLE001
            got = [f"unresolvable: {e}"]
        return {"ok": got == want, "expected": str(want), "got": str(got)}
    if exp == "undo":
        return {"ok": "undo" in kinds and not applied, "expected": "undo", "got": sorted(kinds)}
    if exp == "ask":
        return {"ok": not applied and "sections" in kinds, "expected": "measured sections + a question, no edit",
                "got": sorted(kinds)}
    if exp == "refuse":
        return {"ok": not applied, "expected": "no edit, explanation", "got": sorted(kinds)}
    if exp == "sync":
        calls = [c for c in reply.get("tool_calls", []) if c["tool"] == "sync_tracks"]
        args = calls[0]["input"] if calls else {}
        ok = "sync" in kinds and all(args.get(k, 1 if k == "start_bar_other" else None) == v
                                     for k, v in entry["args"].items())
        return {"ok": ok, "expected": entry["args"], "got": args}
    return {"ok": False, "expected": exp, "got": "unknown expectation"}
