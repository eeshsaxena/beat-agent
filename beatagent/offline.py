"""Offline interpreter: a small deterministic grammar used when no LLM is configured.

It turns a documented subset of phrasings into exactly the tool calls the LLM would make
(apply_edits, undo, redo, find_sections, sync_tracks), so validation, echoes and undo are
the same code paths. Anything outside the grammar is refused with a list of what it
understands; it never guesses. It is a fallback so the app and the evals still run without
an API key, not a replacement for the LLM: the LLM handles paraphrase, context and
ambiguity, which a grammar cannot.

Conventions are the same as everywhere else: beats 0-based and half-open, bars 1-based and
inclusive, "repeat N times" = N plays in total.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .grid import BeatGrid

WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
         "eleven": 11, "twelve": 12, "sixteen": 16, "thirty-two": 32, "a": 1, "an": 1}
TIMES = {"once": 1, "twice": 2, "thrice": 3}
SUBJECTIVE = r"\b(boring|dull|best|good bit|good part|highlight|drop|chorus|verse|hook|breakdown|build|interesting)\b"

HELP = ("Without a language model I understand a fixed set of phrasings: \"repeat/loop <range> N times\", "
        "\"remove/delete/cut <range>\", \"keep only <range>\", \"fade in/out <range>\", \"make <range> louder/quieter "
        "[by N dB]\", \"mute <range>\", \"move <range> to bar N\", \"undo\", \"redo\", \"sync <track> to <this track> at "
        "bar N\", and subjective targets like \"the boring part\" (I measure sections and ask). Ranges: \"beat 16-32\", "
        "\"beats 0 to 8\", \"bar 5\", \"bars 1-4\", \"the first/last N bars|beats\", \"the last bar before the tempo change\".")


@dataclass
class Plan:
    calls: list[tuple[str, dict]] = field(default_factory=list)
    reply: str = ""


def interpret_score(text: str, score: dict) -> Plan:
    """Offline grammar for chat changes to a recreation (the ScoreOps in score.py)."""
    t = _normalise(text)
    tracks = {tr["id"]: tr for tr in score["tracks"]}
    names = {tr["name"].lower(): tr["id"] for tr in score["tracks"]}
    synonyms = {"drum": "drums", "beat": "drums", "percussion": "drums", "bassline": "bass", "piano": "keys",
                "chords": "keys", "melody": "lead", "synth": "lead"}

    def which() -> str | None:
        for w in re.findall(r"[a-z]+", t):
            w2 = synonyms.get(w, synonyms.get(w.rstrip("s"), w))
            for cand in (w2, w2.rstrip("s"), w2 + "s"):
                if cand in tracks:
                    return cand
                if cand in names:
                    return names[cand]
        return None

    if re.match(r"^(undo|undo that)\b", t):
        return Plan([("undo", {})], "Offline grammar: undo.")
    if m := re.search(r"\b(?:play it |tempo |at |to )(\d+(?:\.\d+)?)(?: ?bpm)?\b", t):
        return Plan([("apply_score_ops", {"ops": [{"op": "set_tempo", "bpm": float(m.group(1))}]})],
                    f"Offline grammar: tempo {float(m.group(1)):g} BPM.")
    if re.search(r"\b(faster|slower)\b", t) and not which():
        f = 1.1 if "faster" in t else 1 / 1.1
        bpm = round(score["tempo_bpm"] * f, 2)
        return Plan([("apply_score_ops", {"ops": [{"op": "set_tempo", "bpm": bpm}]})],
                    f"Offline grammar: {'faster' if f > 1 else 'slower'} by 10% = {bpm} BPM.")
    if re.search(r"\b(add|insert|new)\b", t):
        return Plan(reply="Offline grammar: the recreation can only change the tracks that were transcribed; adding "
                          "an instrument would mean inventing notes. Tracks: "
                          + ", ".join(f"{tr['id']} ({tr['name']})" for tr in score["tracks"]) + ".")
    tid = which()
    if tid is None:
        return Plan(reply="Offline grammar: I couldn't tell which track you mean. Tracks: "
                          + ", ".join(f"{tr['id']} ({tr['name']})" for tr in score["tracks"]) + ".")
    tr = tracks[tid]
    if re.search(r"\bunmute\b", t):
        op = {"op": "set_mute", "track": tid, "muted": False}
    elif re.search(r"\b(mute|drop|remove|kill|silence)\b", t):
        op = {"op": "set_mute", "track": tid, "muted": True}
    elif re.search(r"\bunsolo\b", t):
        op = {"op": "set_solo", "track": tid, "solo": False}
    elif re.search(r"\b(solo|only)\b", t):
        op = {"op": "set_solo", "track": tid, "solo": True}
    elif m := re.search(r"\b(brighter|darker|duller)\b", t):
        f = (2.0 if "much" in t else 1.5) if m.group(1) == "brighter" else (0.5 if "much" in t else 0.67)
        op = {"op": "brightness", "track": tid, "factor": f}
    elif m := re.search(r"\b(?:transpose|shift)\b.*?(-?\d+)|\b(up|down) an octave\b", t):  # before "up"/"down" = level
        semis = int(m.group(1)) if m.group(1) else (12 if m.group(2) == "up" else -12)
        op = {"op": "transpose", "track": tid, "semitones": semis}
    elif m := re.search(r"\b(louder|quieter|softer|up|down)\b", t):
        db = re.search(r"(\d+(?:\.\d+)?) ?db", t)
        step = float(db.group(1)) if db else 3.0
        sign = 1 if m.group(1) in ("louder", "up") else -1
        op = {"op": "set_gain", "track": tid, "db": max(-36.0, min(6.0, tr.get("gain_db", 0.0) + sign * step))}
    else:
        return Plan(reply=f"Offline grammar: I found the {tr['name']} track but not what to change. I understand "
                          "mute/unmute, solo, louder/quieter [by N dB], brighter/darker, up/down an octave, "
                          "\"play it at N\", faster/slower and undo.")
    return Plan([("apply_score_ops", {"ops": [op]})], f"Offline grammar: {op['op'].replace('_', ' ')} on {tr['name']}.")


def _num(s: str) -> int:
    s = s.lower()
    return int(s) if s.isdigit() else WORDS[s]


_N = r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|sixteen|thirty-two)"
_TO = r"\s*(?:-|to|through|thru|until)\s*"


def _normalise(text: str) -> str:
    t = text.lower().replace("–", "-").replace("—", "-").replace("’", "'")
    return re.sub(r"\s+", " ", t).strip()


def find_range(t: str, g: BeatGrid) -> tuple[dict | None, str, tuple[int, int]]:
    """(range dict for the ops, how it was read, matched span). None if no range is mentioned."""
    m = re.search(r"last (?:full )?bar before the (?:first )?tempo change", t)
    if m:
        segs = g.tempo_segments()
        if len(segs) < 2:
            return {"error": "This track has no tempo change in its grid."}, "", m.span()
        bar = g._last_full_bar_before(segs[1]["start_index"])
        return ({"unit": "bars", "first_bar": bar, "last_bar": bar},
                f"the last full bar before the tempo change = bar {bar}", m.span())
    m = re.search(r"(first|last) " + _N + r" (bars?|beats?)", t)
    if m:
        k, unit = _num(m.group(2)), m.group(3)
        if unit.startswith("bar"):
            if m.group(1) == "first":
                return {"unit": "bars", "first_bar": 1, "last_bar": k}, f"the first {k} bars = bars 1-{k}", m.span()
            return ({"unit": "bars", "first_bar": g.last_bar - k + 1, "last_bar": g.last_bar},
                    f"the last {k} bars = bars {g.last_bar - k + 1}-{g.last_bar}", m.span())
        if m.group(1) == "first":
            return {"unit": "beats", "start_beat": 0, "end_beat": k}, f"the first {k} beats = beats 0-{k - 1}", m.span()
        return ({"unit": "beats", "start_beat": g.n - k, "end_beat": g.n},
                f"the last {k} beats = beats {g.n - k}-{g.n - 1}", m.span())
    m = re.search(r"beats? " + _N + _TO + r"(?:beat )?" + _N, t)
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        return {"unit": "beats", "start_beat": a, "end_beat": b}, f"beats {a} to {b} = beats {a}-{b - 1} (end exclusive)", m.span()
    m = re.search(r"bars? " + _N + _TO + r"(?:bar )?" + _N, t)
    if m:
        a, b = _num(m.group(1)), _num(m.group(2))
        return {"unit": "bars", "first_bar": a, "last_bar": b}, f"bars {a}-{b} (inclusive)", m.span()
    m = re.search(r"\bbeat " + _N, t)
    if m:
        a = _num(m.group(1))
        return {"unit": "beats", "start_beat": a, "end_beat": a + 1}, f"beat {a}", m.span()
    m = re.search(r"\bbar " + _N, t)
    if m:
        a = _num(m.group(1))
        return {"unit": "bars", "first_bar": a, "last_bar": a}, f"bar {a}", m.span()
    return None, "", (0, 0)


def _count(t: str) -> int | None:
    for w, n in TIMES.items():
        if re.search(rf"\b{w}\b", t):
            return n
    m = re.search(_N + r" times", t)
    return _num(m.group(1)) if m else None


def interpret(text: str, g: BeatGrid, this_track: str = "", library: tuple[str, ...] = ()) -> Plan:
    t = _normalise(text)
    if re.match(r"^(undo|undo that|go back)\b", t):
        return Plan([("undo", {})], "Offline grammar: undo.")
    if re.match(r"^redo\b", t):
        return Plan([("redo", {})], "Offline grammar: redo.")
    m = re.search(r"\bsync (\w+) (?:to|with|onto) (\w+)", t)
    if m:
        b, a = m.group(1).upper(), m.group(2).upper()
        if a != this_track.upper():
            return Plan(reply=f"Offline grammar: this session's track is {this_track}; open {a} to make it the master.")
        if b not in {x.upper() for x in library}:
            return Plan(reply=f"Offline grammar: {b} is not an analysed track here. Available: {', '.join(library) or 'none'}.")
        bar = re.search(r"bar " + _N, t)
        bar_b = re.search(rf"{b.lower()}'s bar " + _N, t)
        args = {"other_track": b, "start_bar_this": _num(bar.group(1)) if bar else 1,
                "start_bar_other": _num(bar_b.group(1)) if bar_b else 1}
        return Plan([("sync_tracks", args)], f"Offline grammar: {b} enters at its bar {args['start_bar_other']} on "
                                              f"{a}'s bar {args['start_bar_this']}.")

    rng, how, span = find_range(t, g)
    if rng and "error" in rng:
        return Plan(reply="Offline grammar: " + rng["error"])
    # one range per message: another beat/bar mention outside the matched range (other than a move
    # target "to/after/before bar N") would be silently dropped, so refuse instead
    rest = (t[:span[0]] + " " + t[span[1]:]) if rng else t
    rest = re.sub(r"\b(?:to|before|after) bar " + _N, "", rest)
    if rng and re.search(r"\b(?:bars?|beats?) " + _N, rest):
        return Plan(reply="Offline grammar: I can only handle one range per message. Please send them one at a time.")
    if rng is None:
        if re.search(SUBJECTIVE, t) or re.search(r"\bpart\b", t):
            return Plan([("find_sections", {})], "Offline grammar: that target is a matter of taste, so I measured the "
                                                 "sections below. Which bars do you mean?")
        return Plan(reply="Offline grammar: I couldn't find a beat or bar range in that. " + HELP)

    ops, reading = [], [how]
    if re.search(r"\b(repeat|loop|double)\b", t):
        n = _count(t)
        if n is None:
            n = 2
            reading.append("no count given, so it plays twice in total")
        elif re.search(r"\brepeat\b", t) and n == 1:
            n = 2
            reading.append("'repeat once' = plays twice in total")
        else:
            reading.append(f"plays {n} times in total")
        ops.append({"op": "repeat", "range": rng, "total_plays": n})
    elif re.search(r"\bkeep (?:only|just)\b|\bonly keep\b", t):
        ops.append({"op": "keep_only", "range": rng})
    elif m := re.search(r"\bfade (in|out)\b", t):
        ops.append({"op": "fade", "range": rng, "direction": m.group(1)})
    elif re.search(r"\b(mute|silence)\b", t):
        ops.append({"op": "mute", "range": rng})
    elif m := re.search(r"\b(louder|quieter|softer|boost|lower|turn (?:it )?(?:up|down))\b", t):
        db = re.search(r"(\d+(?:\.\d+)?) ?db", t)
        amount = float(db.group(1)) if db else 3.0
        sign = -1 if m.group(1) in ("quieter", "softer", "lower") or "down" in m.group(1) else 1
        if not db:
            reading.append("no amount given, so 3 dB")
        ops.append({"op": "gain", "range": rng, "db": sign * amount})
    elif m := re.search(r"\b(?:move|put)\b.* (?:to|before|after) bar " + _N, t):
        target = _num(m.group(1)) + (1 if re.search(r"\bafter bar\b", t) else 0)
        ops.append({"op": "move", "range": rng, "to": {"unit": "bars", "bar": target}})
        reading.append(f"move to just before bar {target}")
    elif re.search(r"\b(remove|delete|cut|chop|drop|take out|get rid of)\b", t):
        ops.append({"op": "remove", "range": rng})
    else:
        return Plan(reply=f"Offline grammar: I found {how}, but not what to do with it. " + HELP)
    return Plan([("apply_edits", {"reading": "offline grammar: " + "; ".join(reading), "ops": ops})],
                "Offline grammar read it as: " + "; ".join(reading) + ".")
