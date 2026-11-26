"""Part 3 chat: words -> typed score ops on a recreation, graded on the resulting score.

    python -m evals.run_recreate_chat            # live: an LLM interprets every message (no grammar fallback)
    python -m evals.run_recreate_chat --offline  # the offline grammar (beatagent/offline.py)

One session on P2's score (drums + bass). Each message has a check on the score it leaves
behind: mute, brightness, tempo, level and solo must change exactly what was asked; a request
the score can't do ("add a guitar") must leave it untouched; undo must restore the last state.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals" / "results"
SCORE = ROOT / "evals" / "out" / "part3" / "P2" / "score.json"


def _t(s, tid):
    return next(t for t in s["tracks"] if t["id"] == tid)


def _strip(s, keys):
    s = copy.deepcopy(s)
    for tid, key in keys:
        obj = s if tid is None else _t(s, tid)
        *parents, last = key.split(".")
        for k in parents:
            obj = obj[k]
        obj.pop(last, None)
    return s


def _same_except(a, b, keys):
    """a and b are equal apart from `keys`, a list of (track id or None for the score, dotted key)."""
    return _strip(a, keys) == _strip(b, keys)


CASES = [
    ("Mute the drums.", "drums muted, nothing else changed",
     lambda b, a: _t(a, "drums")["mute"] is True and _same_except(b, a, [("drums", "mute")])),
    ("Make the bass a bit brighter.", "bass filter cutoff raised, nothing else changed",
     lambda b, a: _t(a, "bass")["patch"]["cutoff_hz"] > _t(b, "bass")["patch"]["cutoff_hz"]
     and _same_except(b, a, [("bass", "patch.cutoff_hz")])),
    ("Play it at 140.", "tempo 140 BPM, notes untouched",
     lambda b, a: abs(a["tempo_bpm"] - 140) < 1e-6 and _same_except(b, a, [(None, "tempo_bpm")])),
    ("Turn the bass down a little.", "bass level lowered by a small step (1-6 dB)",
     lambda b, a: 1 <= _t(b, "bass").get("gain_db", 0) - _t(a, "bass").get("gain_db", 0) <= 6
     and _same_except(b, a, [("bass", "gain_db")])),
    ("Add a guitar.", "refused: the score has no guitar and notes are never invented; score unchanged",
     lambda b, a: a == b),
    ("Undo.", "the bass level change is undone",
     lambda b, a: _t(a, "bass").get("gain_db", 0) > _t(b, "bass").get("gain_db", 0)
     and _same_except(b, a, [("bass", "gain_db")])),
    ("Solo the bass.", "bass soloed, nothing else changed",
     lambda b, a: _t(a, "bass").get("solo") is True and _same_except(b, a, [("bass", "solo")])),
]


def run(mode: str) -> dict:
    os.environ["BEAT_AGENT_INTERPRETER"] = "llm" if mode == "live" else "offline"
    from beatagent.recreate_agent import RecreateAgent

    score = json.loads(SCORE.read_text())
    rows = []
    with tempfile.TemporaryDirectory() as d:
        ag = RecreateAgent(score, Path(d) / "page.html")
        for msg, expect, check in CASES:
            before = copy.deepcopy(ag.score)
            r = ag.ask(msg)
            ok = bool(check(before, ag.score)) and not r.get("error")
            row = {"user": msg, "expected": expect, "ok": ok, "reply": r.get("reply", ""), "error": r.get("error"),
                   "echo": [e for ev in r.get("events", []) for e in (ev.get("echo") or [ev.get("message", "")])]}
            if mode == "live":
                row["model"] = ag.loop.model
            rows.append(row)
            print(f"> {msg}\n  {'OK' if ok else 'DIFFERENT'}: {row['reply']}\n  " + "\n  ".join(row["echo"]), flush=True)
    return {"mode": mode, "clip": "P2", "turns": rows, "spend_usd": round(getattr(ag.loop, "spend_usd", 0.0), 4)}


def write_markdown(res: dict) -> str:
    rows = res["turns"]
    models = sorted({r["model"] for r in rows if r.get("model")})
    what = ("live LLM, no fallback to the grammar" if res["mode"] == "live"
            else "offline grammar (written with these commands in view: shows the fallback works, not that it generalises)")
    L = ["# Part 3 chat: changes to a recreation", "", f"Mode: {what}. Clip: {res['clip']} (drums + bass).",
         *([f"Model: {', '.join(models)}. LLM spend: ${res['spend_usd']:.3f} (estimated)."] if models else []), "",
         f"**{sum(r['ok'] for r in rows)}/{len(rows)} turns leave the score exactly as expected.**", "",
         "| Message | Expected | Result | Echo (written by code) |", "|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['user']} | {r['expected']} | {'OK' if r['ok'] else 'different'} | "
                 f"{'; '.join(r['echo']) or '(no change)'} |")
    L += ["", "Agent replies:", ""]
    L += [f"* **{r['user']}** {r['reply'] or ''}{' (error: ' + r['error'] + ')' if r.get('error') else ''}" for r in rows]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    mode = "offline" if args.offline else "live"
    res = run(mode)
    name = "part3_chat" if mode == "live" else "part3_chat_offline"
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"{name}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    (RES / f"{name}.md").write_text(write_markdown(res), encoding="utf-8")


if __name__ == "__main__":
    main()
