"""The edit agent: an LLM (Claude or Gemini) interprets chat messages into typed edit operations.

Division of labour:
* The LLM reads the user's words plus a factual summary of the beat grid (computed by
  code) and calls tools with typed operations. It never sees or produces audio, times
  it measured, or tempo guesses.
* Code validates every operation against the grid, applies it, and writes the exact
  echo (beats, bars, seconds) the user sees.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from pydantic import TypeAdapter, ValidationError

from .edl import EditError, EditOp
from .llm import LLMUnavailable, Tool, any_llm_key, make_loop, schema_with_defs, is_permanent
from .sections import find_sections
from .session import EditSession

SYSTEM = """You are the interpreter inside Beat Agent, a chat tool for editing a music track on its beat grid.

Your only job is to understand what the user means and express it as typed operations by calling tools. Deterministic code measures the audio, owns the beat grid, validates every operation, applies it and writes the exact echo the user sees. You never estimate tempo, times or positions yourself: use only numbers from the <track_state> block, the user's words and tool results.

Addressing conventions. Apply them exactly, and when the words allow another reading, say which reading you used:
- Beats are numbered from 0. Beat ranges are half-open: "beat 16-32", "beats 16 to 32" and "from beat 16 to beat 32" all mean start_beat=16, end_beat=32, which is 16 beats ending where beat 32 starts. "Beat 0 to beat 8" is 8 beats. A single "beat 7" is start_beat=7, end_beat=8.
- Bars are numbered from 1; bar 0 is a pickup when the state lists one. Bar ranges include both ends: "bar 5" is first_bar=5, last_bar=5; "bars 200-210" is 200 to 210 (11 bars); "the first 4 bars" is 1 to 4; "the last 2 bars" uses the last bar number from the state.
- "The last 8 beats" is start_beat=N-8, end_beat=N, where N is the beat count in the state (beat position N is the end of the track).
- Counting repeats: "repeat X twice", "loop X four times" and "play X 3 times" set total_plays to 2, 4 and 3: the number of times X is heard in total, including the original. Say "plays twice in total" (or similar) in your reply.
- "Cut X and repeat/loop it" means take X and repeat it in place; it does not delete X. "Cut X", "cut out X", "remove X", "delete X" or "chop X" on their own mean remove.
- "Then let the track play on" or "then continue" needs no extra operation: a repeat already continues with the rest of the track.
- All operations in one apply_edits call are addressed against the grid at the start of the user's message. If a later part of the message must refer to the result of an earlier part, apply the first part, then call apply_edits again with the second part using the new state the tool returns. Both calls form one undo step.

Worked examples (on a 4/4 track with 512 beats, bars 1-128, no pickup):
- "loop beats 4 to 12 three times" -> repeat {beats, start_beat 4, end_beat 12}, total_plays 3.
- "drop bars 9-12" -> remove {bars, first_bar 9, last_bar 12} (4 bars = beats 32-47).
- "keep just the second half" -> ask: "half" is ambiguous on a track with a pickup or an odd bar count; or, if the state makes it exact (bars 65-128), keep_only with those bars and say so.
- "make bar 3 quieter" -> gain {bars 3-3}, db -6, and say you chose -6 dB.
- "silence beat 100" -> mute {beats 100-101}.
- "put bars 1-4 after bar 16" -> move {bars 1-4} to {bar 17} (before the first beat of bar 17).
- "fade in the first 4 bars" -> fade in {bars 1-4}.
- "cut bars 500-510" on this track -> do not edit; bars 500-510 do not exist (the last bar is 128).

Behaviour:
- A clear request: call apply_edits straight away. Do not ask for confirmation of clear edits; the user can say "undo".
- "Undo" or "redo": call undo or redo. Never rebuild an edit yourself.
- A subjective or under-specified target ("the boring part", "the drop", "the good bit", "make it shorter"): call find_sections, then reply with the one to three most plausible candidates, quoting their measured evidence (bars, seconds, observations), and ask which one to use. Do not edit until the user picks. Taste is the user's call, not yours.
- Out-of-range or impossible requests: answer directly when the state already shows the problem, or call apply_edits and let the code check. Explain plainly and offer the closest valid alternative, such as the real last bar. Never quietly change the user's numbers to make an edit fit.
- If a tool returns an error, do not retry with different numbers unless the error shows you misread the request; explain the error instead.
- If a message is ambiguous in a way these conventions do not settle, ask one short question instead of guessing.
- Beat-matching another track to this one ("sync X to this track", "mix X in at bar 33"): call sync_tracks. This track is the master (A). Unless the user names B's start, B enters at its own bar 1; say so in your reply.
- If a message is not about editing or syncing this track, say briefly what you can do.

Reply style: one to three short sentences of plain text, no headings or lists. The code's exact echo (beats, bars, seconds) is shown to the user right under your reply, so do not restate numbers you did not get from a tool result. Mention any reading you chose and any warning the tool returned."""

_OPS = TypeAdapter(list[EditOp])
_ops_schema = _OPS.json_schema(ref_template="#/$defs/{model}")
_DEFS = _ops_schema.pop("$defs", {})
_READING = {"type": "string", "description": "One sentence: how you read the user's words, naming any convention you "
                                              "applied (e.g. 'repeat twice = plays twice in total')."}


@dataclass
class AgentReply:
    user: str
    text: str
    events: list[dict] = field(default_factory=list)  # applied / proposed / undo / redo / sections / error
    error: str | None = None
    usage: dict = field(default_factory=dict)
    tool_calls: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"user": self.user, "reply": self.text, "events": self.events, "error": self.error, "usage": self.usage,
                "tool_calls": self.tool_calls}


class EditAgent:
    def __init__(self, session: EditSession, model: str | None = None, effort: str | None = None,
                 library: dict | None = None, out_dir=None, replay: list[dict] | None = None):
        """library: {name: (Audio, BeatGrid)} of other analysed tracks available for sync.
        replay: recorded turns ({"user", "reply", "tool_calls"}) to re-run without calling the LLM."""
        self.session = session
        self.turn = 0
        self._events: list[dict] = []
        self.library = library or {}
        self.out_dir = out_dir
        self.replay = list(replay) if replay is not None else None
        self.recorded: list[dict] = []
        # "auto": LLM when a key is configured, otherwise (or if the LLM is unreachable) the offline grammar;
        # "llm": LLM only; "offline": grammar only.
        self.interpreter = os.environ.get("BEAT_AGENT_INTERPRETER", "auto")
        self._llm_down: str | None = None
        self.confirm = False  # when True, apply_edits only previews: the user applies from the echo
        tools = [
            Tool("apply_edits", "Validate and apply typed edit operations to the track now. Returns the exact echo "
                                "(beats, bars, seconds) or an error explaining why nothing was changed.",
                 schema_with_defs({"reading": _READING, "ops": _ops_schema}, ["reading", "ops"], _DEFS), self._apply),
            Tool("propose_edits", "Validate edit operations and describe exactly what they would do, without applying "
                                  "them. Use when the user should confirm first.",
                 schema_with_defs({"reading": _READING, "ops": _ops_schema}, ["reading", "ops"], _DEFS), self._propose),
            Tool("undo", "Undo the most recent edit turn.", schema_with_defs({}, []), self._undo),
            Tool("redo", "Redo the most recently undone edit turn.", schema_with_defs({}, []), self._redo),
            Tool("find_sections", "Measure the track's sections: bars, seconds, energy per band (sub, low-mid, mid, "
                                  "presence, air) relative to the whole track, and how repetitive the bars are, with "
                                  "plain observations (no kick, thin arrangement, intro, outro). Use for subjective "
                                  "targets such as 'the boring part' before asking the user to choose.",
                 schema_with_defs({}, []), self._sections),
        ]
        if self.library:
            tools.append(Tool(
                "sync_tracks", "Beat-match another analysed track (B) to this track (A, the master): code computes "
                               "the time-stretch ratio and offset from both beat grids, stretches B with pitch "
                               "preserved so its downbeats land on A's downbeats, renders the overlap and measures "
                               "the kick alignment.",
                schema_with_defs({
                    "other_track": {"type": "string", "enum": sorted(self.library), "description": "Track B."},
                    "start_bar_this": {"type": "integer", "description": "Bar of this track (A) where the overlap starts."},
                    "start_bar_other": {"type": "integer", "description": "Bar of track B that lands there (default 1)."},
                    "seconds": {"type": "number", "description": "Overlap length in seconds (default 60)."},
                }, ["other_track", "start_bar_this"]), self._sync))
        self.loop = make_loop(SYSTEM, tools, model=model, effort=effort, tag=f"edit:{session.name}")

    # ----------------------------------------------------------------- tools
    def _parse(self, inp: dict):
        try:
            return _OPS.validate_python(inp.get("ops", []))
        except ValidationError as e:
            raise EditError("The operations were malformed: " + "; ".join(
                f"{'.'.join(map(str, err['loc']))}: {err['msg']}" for err in e.errors()[:4]))

    def _apply(self, inp: dict) -> dict:
        if self.confirm:  # enforced in code, not only asked of the model
            out = self._propose(inp)
            if "error" not in out:
                out["note"] = "Confirmation mode: proposed only; the user applies it from the echo."
            return out
        try:
            ops = self._parse(inp)
            res = self.session.apply(ops, turn_id=self.turn)
        except EditError as e:
            self._events.append({"type": "error", "message": str(e), "ops": inp.get("ops")})
            return {"error": str(e), "changed": False}
        ev = {"type": "applied", "reading": inp.get("reading", ""), "ops": [o.model_dump() for o in ops],
              "echo": res.echo, "warnings": res.warnings}
        self._events.append(ev)
        return {"applied": True, "echo": res.echo, "warnings": res.warnings, "new_state": self._state_text()}

    def _propose(self, inp: dict) -> dict:
        try:
            ops = self._parse(inp)
            res = self.session.preview(ops)
        except EditError as e:
            self._events.append({"type": "error", "message": str(e), "ops": inp.get("ops")})
            return {"error": str(e), "changed": False}
        self._events.append({"type": "proposed", "reading": inp.get("reading", ""), "ops": [o.model_dump() for o in ops],
                             "echo": res.echo, "warnings": res.warnings})
        return {"would_do": res.echo, "warnings": res.warnings, "applied": False}

    def _undo(self, _inp: dict) -> dict:
        try:
            msg = self.session.undo()
        except EditError as e:
            self._events.append({"type": "error", "message": str(e)})
            return {"error": str(e)}
        self._events.append({"type": "undo", "echo": [msg]})
        return {"done": msg, "new_state": self._state_text()}

    def _redo(self, _inp: dict) -> dict:
        try:
            msg = self.session.redo()
        except EditError as e:
            self._events.append({"type": "error", "message": str(e)})
            return {"error": str(e)}
        self._events.append({"type": "redo", "echo": [msg]})
        return {"done": msg, "new_state": self._state_text()}

    def _sections(self, _inp: dict) -> dict:
        secs = find_sections(self.session.render(), self.session.sr, self.session.grid)
        self._events.append({"type": "sections", "sections": secs})
        return {"sections": secs, "note": "Measured, not judged: 'boring' is the user's call."}

    def _sync(self, inp: dict) -> dict:
        from pathlib import Path

        from . import audio_io
        from . import sync as syncmod

        name = inp.get("other_track")
        if name not in self.library:
            return {"error": f"Unknown track {name!r}. Available: {', '.join(sorted(self.library))}."}
        entry = self.library[name]
        if callable(entry):  # loaded lazily
            entry = self.library[name] = entry()
        b_audio, b_grid = entry
        s = self.session
        a_audio = audio_io.Audio(s.render(), s.sr, s.audio.subtype, s.audio.format, s.audio.path)
        try:
            r = syncmod.sync(a_audio, s.grid, b_audio, b_grid, start_bar_a=int(inp["start_bar_this"]),
                             start_bar_b=int(inp.get("start_bar_other") or 1), duration=float(inp.get("seconds") or 60))
        except syncmod.SyncError as e:
            self._events.append({"type": "error", "message": str(e)})
            return {"error": str(e)}
        paths = {}
        if self.out_dir:
            d = Path(self.out_dir) / f"sync_{name}_to_{s.name}_bar{inp['start_bar_this']}"
            paths = {"mix": str(audio_io.save(d / "overlap_mix.wav", r.mix, r.sr)),
                     "stem_b": str(audio_io.save(d / "stem_B_stretched.wav", r.b_stem, r.sr)),
                     "mix_mp3": str(audio_io.save(d / "overlap_mix.mp3", r.mix, r.sr, "MP3", "MPEG_LAYER_III"))}
            (d / "report.json").write_text(dumps(r.report))
        rep = r.report
        k = rep["kick_alignment_final"]
        echo = [f"SYNC {name} -> {s.name}: {name} bar {rep['track_b']['start_bar']} ({rep['track_b']['start_s']:.3f} s) "
                f"lands on {s.name} bar {rep['track_a']['start_bar']} ({rep['track_a']['start_s']:.3f} s); time-stretch "
                f"ratio {rep['time_stretch_ratio']:.6f} ({rep['track_b']['bpm']:.2f} -> {rep['track_a']['bpm']:.2f} BPM, "
                f"pitch preserved); {rep['duration_s']:.1f} s overlap rendered.",
                f"Measured kick alignment: median {k.get('median_abs_offset_ms')} ms, max {k.get('max_abs_offset_ms')} ms "
                f"over {k.get('matched')} kicks."]
        self._events.append({"type": "sync", "echo": echo, "report": rep, "paths": paths, "warnings": rep["notes"]})
        return {"done": echo, "notes": rep["notes"]}

    # ------------------------------------------------------------------ turn
    def _state_text(self) -> str:
        s = self.session
        return (f"{s.grid.summary()}\nEdit history:\n{s.history_text()}\n"
                f"Undo available: {'yes' if s.can_undo() else 'no'}. Redo available: {'yes' if s.can_redo() else 'no'}.")

    def ask(self, text: str) -> AgentReply:
        self.turn += 1
        self._events = []
        if self.replay is not None:
            return self._ask_replay(text)
        if self.interpreter == "offline" or (self.interpreter == "auto" and (not any_llm_key() or self._llm_down)):
            return self._ask_offline(text, note=self._llm_down or "")
        lib = f"\nOther analysed tracks available for sync: {', '.join(sorted(self.library))}." if self.library else ""
        content = f"<track_state>\nTrack: {self.session.name}{lib}\n{self._state_text()}\n</track_state>\n\n{text}"
        try:
            res = self.loop.run(content)
        except LLMUnavailable as e:
            return AgentReply(text, "", [], str(e))
        if res.unavailable and self.interpreter == "auto":
            self._events = []
            note = f"[LLM unavailable: {res.error} Using the offline grammar.] "
            if is_permanent(res.error):
                self._llm_down = note  # a missing or rejected key: stop trying for this session
            return self._ask_offline(text, note=note)  # a transient error: fall back for this message only
        calls = [{"tool": c["tool"], "input": c["input"]} for c in res.tool_calls]
        self.recorded.append({"user": text, "reply": res.text, "error": res.error, "tool_calls": calls})
        return AgentReply(text, res.text, self._events, res.error, res.usage, calls)

    def _ask_offline(self, text: str, note: str = "") -> AgentReply:
        """Deterministic grammar (beatagent/offline.py); same tool functions, same validation and echo."""
        from .offline import interpret

        plan = interpret(text, self.session.grid, self.session.name, tuple(self.library))
        calls = []
        for tool, inp in plan.calls:
            if tool in self.loop.tools:
                self.loop.tools[tool].fn(inp)
                calls.append({"tool": tool, "input": inp})
        return AgentReply(text, (note + plan.reply).strip(), self._events, None, {"interpreter": "offline"}, calls)

    def _ask_replay(self, text: str) -> AgentReply:
        """Re-run a recorded turn: the recorded tool inputs go through the same validating code."""
        if not self.replay:
            return AgentReply(text, "", [], "No recorded turn left to replay.")
        rec = self.replay.pop(0)
        if rec.get("user") != text:
            return AgentReply(text, "", [], f"Replay mismatch: the recorded message was {rec.get('user')!r}.")
        for c in rec.get("tool_calls", []):
            self.loop.tools[c["tool"]].fn(c["input"])
        return AgentReply(text, rec.get("reply", ""), self._events, rec.get("error"), {"replayed": True},
                          rec.get("tool_calls", []))


def format_reply(r: AgentReply, echo_user: bool = True) -> str:
    """Plain-text rendering used by the CLI (which already shows what was typed: echo_user=False) and the
    eval transcripts."""
    lines = [f"> {r.user}"] if echo_user else []
    if r.error:
        lines.append(f"[error] {r.error}")
    if r.text:
        lines.append(r.text)
    for ev in r.events:
        if ev["type"] in ("applied", "proposed", "undo", "redo", "sync"):
            tag = {"applied": "APPLIED", "proposed": "PROPOSED (not applied)", "undo": "UNDO", "redo": "REDO",
                   "sync": "SYNC"}[ev["type"]]
            for e in ev.get("echo", []):
                lines.append(f"  [{tag}] {e}")
            for w in ev.get("warnings", []):
                lines.append(f"  [warning] {w}")
        elif ev["type"] == "error":
            lines.append(f"  [refused by code] {ev['message']}")
        elif ev["type"] == "sections":
            lines.append(f"  [measured {len(ev['sections'])} sections]")
            if not echo_user:  # interactive: show the measured candidates, not just their count
                for sec in ev["sections"]:
                    lines.append(f"    bars {sec['bars'][0]}-{sec['bars'][1]} ({sec['time_s'][0]}-{sec['time_s'][1]} s): "
                                 f"{', '.join(sec['observations']) or 'no flags'}")
    return "\n".join(lines)


def dumps(obj) -> str:
    return json.dumps(obj, indent=1, default=str)
