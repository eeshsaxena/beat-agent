"""Part 2 evals: the brief's chat sessions, run headlessly.

    python -m evals.run_part2               # live: Claude interprets every message (needs ANTHROPIC_API_KEY)
    python -m evals.run_part2 --replay      # re-runs the recorded tool calls of the last live run, no API calls
    python -m evals.run_part2 --reference   # no LLM: the engine executes the answer key (evals/answer_key.py)

Every turn writes the edited audio (input sample rate and bit depth), the beat grid and a click
audition, and checks:
  * interpretation - live/replay only: the agent's ops vs the answer key, compared after
                     resolving both against the grid (so "bar 5" == "beats 16-20")
  * format         - the edited file keeps the input's sample rate and subtype
  * splices        - discontinuity score at every splice (<= 5 is clean; butt splices as a control)
  * grid truth     - the edited grid vs (a) the reference beats carried through the same edit and
                     (b) a fresh analysis of the edited audio
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

from beatagent import audio_io, tracking
from beatagent.agent import AgentReply, EditAgent, format_reply
from beatagent.edl import Timeline
from beatagent.grid import BeatGrid
from beatagent.quality import butt_splice_scores, splice_click_scores
from beatagent.session import EditSession
from evals import pack
from evals.answer_key import KEY, grade

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "out" / "part2"
RES = ROOT / "evals" / "results"
REC = ROOT / "evals" / "recorded"

SESSIONS = [
    ("s1_A1", "A1", ["Cut beat 16–32 and repeat it twice.", "Now fade out the last 8 beats.", "Undo.", "Remove bar 5."]),
    ("s2_A2", "A2", ["Loop the first 4 bars four times, then let the track play on."]),
    ("s3_A5", "A5", ["Repeat the last bar before the tempo change twice."]),
    ("s4_A1", "A1", ["Chop out the boring part."]),
    ("s5_A3", "A3", ["Cut beat 0 to beat 8.", "Delete bars 200–210."]),
    ("s6_A1", "A1", ["Sync A7 to A1, starting the overlap at A1's bar 33."]),
]
MODE_TEXT = {"live": "live LLM (Claude interprets every message)",
             "replay": "replay of the recorded tool calls from the last live run (no LLM calls)",
             "reference": "reference: the engine executes the answer key; no LLM involved, so this tests the edit "
                          "engine, not the agent's interpretation",
             "offline": "offline grammar (beatagent/offline.py), no LLM. The grammar was written with these commands in "
                        "view, so its score here shows the fallback works, not that it generalises"}


def load_grid(cid: str) -> BeatGrid:
    p = ROOT / "evals" / "out" / "part1" / cid / "grid.json"
    if p.exists():
        return BeatGrid.from_dict(json.loads(p.read_text()))
    return tracking.analyze(audio_io.load(pack.audio_path(cid)))


def map_times(tl: Timeline, times: np.ndarray, sr: int) -> np.ndarray:
    """Carry source times through an edit timeline (same rule as the grid derivation)."""
    samp = times * sr
    rnd = np.round(samp).astype(np.int64)
    out, o = [], 0
    for p in tl.pieces:
        m = (rnd >= p.src_start) & (rnd < p.src_end)
        out.append((o + samp[m] - p.src_start) / sr)
        o += p.length
    return np.concatenate(out) if out else np.zeros(0)


def check_turn(sess: EditSession, cid: str, d: Path, stem: str) -> dict:
    paths = sess.export(d, stem)
    info = sf.info(paths["audio"])
    y = sess.render()
    tl = sess.state.timeline
    splices = splice_click_scores(tl, sess.audio.samples, y, sess.sr)
    control = butt_splice_scores(tl, sess.audio.samples, sess.sr)
    out = {
        "paths": {k: str(Path(v).relative_to(ROOT)) for k, v in paths.items()},
        "format": {"input": [sess.audio.sr, sess.audio.subtype], "output": [info.samplerate, info.subtype],
                   "same": info.samplerate == sess.audio.sr and info.subtype == sess.audio.output_spec()[1]},
        "splices": splices, "splices_butt_control": control,
        "max_splice_score": max([s["ratio"] for s in splices], default=None),
        "grid": {"n_beats": sess.grid.n, "n_bars": sess.grid.last_bar, "duration_s": round(sess.grid.duration_s, 3)},
    }
    ref_b, ref_d = pack.ref_beats(cid)
    if ref_b is not None:
        mapped = map_times(tl, ref_b, sess.sr)
        e = tracking.nearest_dist(mapped, sess.grid.times) * 1000
        out["grid_vs_mapped_reference"] = {
            "beat_f": round(float(mir_eval.beat.f_measure(mapped, sess.grid.times)), 4),
            "max_abs_ms_matched": round(float(e[e < 70].max()), 2) if (e < 70).any() else None}
        if ref_d is not None:
            md = map_times(tl, ref_d, sess.sr)
            out["grid_vs_mapped_reference"]["downbeat_f"] = round(float(mir_eval.beat.f_measure(md, sess.grid.downbeat_times)), 4)
    if sess.can_undo():  # independent check: analyse the edited audio from scratch
        fresh = tracking.analyze(audio_io.Audio(y, sess.sr, sess.audio.subtype, sess.audio.format))
        out["grid_vs_fresh_analysis"] = {
            "beat_f": round(float(mir_eval.beat.f_measure(fresh.times, sess.grid.times)), 4),
            "downbeat_f": round(float(mir_eval.beat.f_measure(fresh.downbeat_times, sess.grid.downbeat_times)), 4)}
    return out


def scripted_turn(agent: EditAgent, entry: dict, msg: str) -> AgentReply:
    """Reference mode: run the answer key's action through the agent's own tool functions."""
    agent.turn += 1
    agent._events = []
    g = agent.session.grid
    exp = entry["expect"]
    calls = []
    if exp in ("apply", "refuse"):
        ops = entry["ops"](g) if exp == "apply" else entry["literal_ops"](g)
        inp = {"reading": f"answer key: {entry['why']}", "ops": [o.model_dump() for o in ops]}
        agent._apply(inp)
        calls.append({"tool": "apply_edits", "input": inp})
    elif exp == "undo":
        agent._undo({})
        calls.append({"tool": "undo", "input": {}})
    elif exp == "ask":
        agent._sections({})
        calls.append({"tool": "find_sections", "input": {}})
    elif exp == "sync":
        agent._sync(dict(entry["args"]))
        calls.append({"tool": "sync_tracks", "input": entry["args"]})
    return AgentReply(msg, f"[reference mode, no LLM] {entry['why']}", agent._events, None, {}, calls)


def run(mode: str) -> dict:
    grids, library, results = {}, {}, {}
    rec_all = json.loads((REC / "part2_sessions.json").read_text()) if mode == "replay" else {}
    out_root = OUT / mode if mode in ("reference", "offline") else OUT
    for sid, cid, msgs in SESSIONS:
        print(f"== {sid} on {cid}", flush=True)
        a = audio_io.load(pack.audio_path(cid))
        grids.setdefault(cid, load_grid(cid))
        if sid == "s6_A1" and "A7" not in library:
            library["A7"] = (audio_io.load(pack.audio_path("A7")), load_grid("A7"))
        sess = EditSession(a, grids[cid], cid)
        d = out_root / sid
        d.mkdir(parents=True, exist_ok=True)
        agent = EditAgent(sess, library=library if sid == "s6_A1" else None, out_dir=d,
                          replay=rec_all.get(sid) if mode == "replay" else None)
        if mode == "offline":
            agent.interpreter = "offline"
        turns, transcript = [], []
        for i, m in enumerate(msgs, start=1):
            entry = KEY[sid][i - 1]
            grid_before = sess.grid
            r = scripted_turn(agent, entry, m) if mode == "reference" else agent.ask(m)
            text = format_reply(r)
            print(text, flush=True)
            transcript.append(text)
            t = {"turn": i, **r.to_dict(), "state_after": {"can_undo": sess.can_undo(), "can_redo": sess.can_redo(),
                                                            "history": sess.history_text()}}
            t["expected"] = {"expect": entry["expect"], "why": entry["why"]}
            if mode != "reference":
                t["interpretation"] = grade(entry, t, grid_before)
                print(f"  [interpretation vs answer key] {'OK' if t['interpretation']['ok'] else 'DIFFERENT'}", flush=True)
            if sess.can_undo() or any(ev["type"] == "undo" for ev in r.events):
                t["checks"] = check_turn(sess, cid, d, f"turn{i}")
            turns.append(t)
        (d / "transcript.txt").write_text("\n\n".join(transcript) + "\n", encoding="utf-8")
        (d / "turns.json").write_text(json.dumps(turns, indent=1, default=str), encoding="utf-8")
        results[sid] = {"track": cid, "turns": turns, "spend_usd": round(agent.loop.spend_usd, 4)}
        if mode == "live":
            rec_all[sid] = agent.recorded
    if mode == "live":
        REC.mkdir(parents=True, exist_ok=True)
        (REC / "part2_sessions.json").write_text(json.dumps(rec_all, indent=1, default=str), encoding="utf-8")
    return results


def write_markdown(results: dict, mode: str) -> str:
    L = ["# Part 2 results: chat edits on the beat grid", "", f"Mode: {MODE_TEXT[mode]}.",
         f"LLM spend this run: ${sum(r['spend_usd'] for r in results.values()):.3f}.", ""]
    rows = [(sid, t) for sid, r in results.items() for t in r["turns"]]
    if mode != "reference":
        ok = sum(1 for _, t in rows if t.get("interpretation", {}).get("ok"))
        L += [f"**Interpretation vs the answer key: {ok}/{len(rows)} turns.**", ""]
    L += ["| Session | Message | Expected | Format kept | Max splice score (butt control) | Grid vs carried reference (beat F / downbeat F) | Grid vs fresh analysis (beat F) |"
          + (" Interpretation |" if mode != "reference" else ""),
          "|---|---|---|---|---|---|---|" + ("---|" if mode != "reference" else "")]
    for sid, t in rows:
        c = t.get("checks") or {}
        g, f = c.get("grid_vs_mapped_reference", {}), c.get("grid_vs_fresh_analysis", {})
        ctl = max([s["ratio"] for s in c.get("splices_butt_control", [])], default=None)
        cells = [sid, t["user"], t["expected"]["expect"], str(c["format"]["same"]) if c else "-",
                 f"{c.get('max_splice_score')} ({ctl})" if c else "-",
                 f"{g.get('beat_f', '-')} / {g.get('downbeat_f', '-')}" if c else "-", str(f.get("beat_f", "-")) if c else "-"]
        if mode != "reference":
            cells.append("OK" if t.get("interpretation", {}).get("ok") else "different")
        L.append("| " + " | ".join(cells) + " |")
    L.append("")
    for sid, r in results.items():
        L += [f"## {sid} ({r['track']})", ""]
        for t in r["turns"]:
            L += [f"**> {t['user']}**", "", f"- expected: {t['expected']['why']}"]
            if t.get("error"):
                L.append(f"- error: {t['error']}")
            if t.get("reply"):
                L.append(f"- agent: {t['reply']}")
            for ev in t["events"]:
                for e in ev.get("echo", []):
                    L.append(f"- `{ev['type']}` {e}")
                for w in ev.get("warnings", []):
                    L.append(f"- warning: {w}")
                if ev["type"] == "error":
                    L.append(f"- refused by code: {ev['message']}")
                if ev["type"] == "sections":
                    for s in ev["sections"]:
                        L.append(f"  - bars {s['bars'][0]}-{s['bars'][1]} ({s['time_s'][0]}-{s['time_s'][1]} s): "
                                 f"{', '.join(s['observations']) or 'no flags'}")
            if "interpretation" in t:
                it = t["interpretation"]
                L.append(f"- interpretation: {'OK' if it['ok'] else 'DIFFERENT'} (expected {it['expected']}, got {it['got']})")
            L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--replay", action="store_true")
    g.add_argument("--reference", action="store_true")
    g.add_argument("--offline", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    mode = "replay" if args.replay else "reference" if args.reference else "offline" if args.offline else "live"
    if mode == "replay" and not (REC / "part2_sessions.json").exists():
        print("No recorded live run to replay (evals/recorded/part2_sessions.json); running reference mode instead.")
        mode = "reference"
    res = run(mode)
    RES.mkdir(parents=True, exist_ok=True)
    name = {"reference": "part2_reference", "offline": "part2_offline"}.get(mode, "part2")
    (RES / f"{name}.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    (RES / f"{name}.md").write_text(write_markdown(res, mode), encoding="utf-8")


if __name__ == "__main__":
    main()
