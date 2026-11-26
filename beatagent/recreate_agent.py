"""Chat changes to a recreation ("mute the drums", "make the bass brighter", "play it at 140").

The LLM emits typed ScoreOps; code validates them, applies them to the score and
rebuilds the page from the fixed template. The LLM never writes JavaScript.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from .llm import LLMUnavailable, Tool, any_llm_key, make_loop, schema_with_defs, is_permanent
from .score import ScoreError, ScoreOp, apply_score_ops, build_page

SYSTEM = """You are the interpreter for Beat Agent's recreation player. A short audio clip has been transcribed into a score (tempo, tracks, notes) that plays in a Web Audio page. You change the playback only by calling apply_score_ops with typed operations; code applies them and rebuilds the page. You never write code, and you never invent notes or tempo.

Map requests to operations:
- "mute/unmute the drums", "drop the bass": set_mute. "only the keys", "solo the bass": set_solo.
- "louder/quieter": set_gain, with absolute dB (current levels are in the state): a small change is 3 dB.
- "play it at 140", "slower": set_tempo with the BPM (for "slower"/"faster" without a number, use 10% of the current tempo and say so).
- "brighter/darker": brightness with factor 1.5 or 0.67 ("much" = 2 or 0.5).
- "up an octave": transpose by 12.
If the request names a track that does not exist, or asks for something the score can't do (e.g. "add a guitar"), say so plainly and list what is possible. Reply in one or two sentences; the code's exact echo is shown under your reply."""

_OPS = TypeAdapter(list[ScoreOp])
_schema = _OPS.json_schema(ref_template="#/$defs/{model}")
_DEFS = _schema.pop("$defs", {})


class RecreateAgent:
    def __init__(self, score: dict, page_path: Path, model: str | None = None):
        self.score = score
        self.page_path = Path(page_path)
        self.history = [score]
        self._events: list[dict] = []
        self._llm_down: str | None = None
        tools = [Tool("apply_score_ops", "Apply typed operations to the score and rebuild the page.",
                      schema_with_defs({"ops": _schema}, ["ops"], _DEFS), self._apply),
                 Tool("undo", "Undo the last change to the recreation.", schema_with_defs({}, []), self._undo)]
        self.loop = make_loop(SYSTEM, tools, model=model, tag=f"recreate:{score.get('title')}")

    def _state(self) -> str:
        s = self.score
        rows = [f"Tempo {s['tempo_bpm']:.2f} BPM, {s['bars']} bars of {s['time_signature'][0]}/{s['time_signature'][1]}."]
        for t in s["tracks"]:
            rows.append(f"- {t['id']} ({t['name']}): {len(t['notes'])} notes, level {t.get('gain_db', 0):+.1f} dB, "
                        f"{'muted' if t.get('mute') else 'audible'}{', solo' if t.get('solo') else ''}")
        return "\n".join(rows)

    def _apply(self, inp: dict) -> dict:
        try:
            ops = _OPS.validate_python(inp.get("ops", []))
            new, echo = apply_score_ops(self.score, ops)
        except (ValidationError, ScoreError) as e:
            msg = str(e) if isinstance(e, ScoreError) else "Malformed operations."
            self._events.append({"type": "error", "message": msg})
            return {"error": msg}
        self.history.append(new)
        self.score = new
        build_page(new, self.page_path)
        self._events.append({"type": "applied", "echo": echo})
        return {"applied": True, "echo": echo, "state": self._state()}

    def _undo(self, _inp: dict) -> dict:
        if len(self.history) < 2:
            return {"error": "Nothing to undo."}
        self.history.pop()
        self.score = self.history[-1]
        build_page(self.score, self.page_path)
        self._events.append({"type": "undo", "echo": ["Undid the last change to the recreation."]})
        return {"done": True, "state": self._state()}

    def ask(self, text: str) -> dict:
        import os

        self._events = []
        mode = os.environ.get("BEAT_AGENT_INTERPRETER", "auto")
        if mode == "offline" or (mode == "auto" and (not any_llm_key() or self._llm_down)):
            return self._ask_offline(text, self._llm_down or "")
        try:
            res = self.loop.run(f"<score_state>\n{self._state()}\n</score_state>\n\n{text}")
        except LLMUnavailable as e:
            return {"user": text, "reply": "", "events": [], "error": str(e)}
        if res.unavailable and mode == "auto":
            note = f"[LLM unavailable: {res.error} Using the offline grammar.] "
            if is_permanent(res.error):
                self._llm_down = note  # a missing or rejected key: stop trying for this session
            self._events = []
            return self._ask_offline(text, note)
        return {"user": text, "reply": res.text, "events": self._events, "error": res.error, "usage": res.usage}

    def _ask_offline(self, text: str, note: str = "") -> dict:
        """Deterministic grammar (beatagent/offline.py); same tool functions, same validation and echo."""
        from .offline import interpret_score

        plan = interpret_score(text, self.score)
        for tool, inp in plan.calls:
            self.loop.tools[tool].fn(inp)
        return {"user": text, "reply": (note + plan.reply).strip(), "events": self._events, "error": None,
                "usage": {"interpreter": "offline"}}
