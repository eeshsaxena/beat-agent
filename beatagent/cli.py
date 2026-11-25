"""Command line.

    python -m beatagent.cli analyze  FILE [--out DIR]       beat grid JSON + click audition
    python -m beatagent.cli chat     FILE                   interactive chat edits (needs ANTHROPIC_API_KEY)
    python -m beatagent.cli sync     A B [--bar-a 33] [--bar-b 1] [--seconds 60] [--out DIR]
    python -m beatagent.cli recreate CLIP [--out DIR]       score.json + page.html + offline render
    python -m beatagent.cli evals    [--replay]             all evals (Part 2 replays recorded LLM calls with --replay)
    python -m beatagent.cli demo     [--replay]             evals, then the web app on http://127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _analyze(a):
    from . import audio_io, click, tracking

    au = audio_io.load(a.file)
    g = tracking.analyze(au, progress=lambda m: print("·", m))
    out = Path(a.out or Path(a.file).stem + "_beatagent")
    out.mkdir(parents=True, exist_ok=True)
    (out / "grid.json").write_text(json.dumps(g.to_dict({"file": Path(a.file).name}), indent=1))
    audio_io.save(out / "click_audition.wav", click.audition(au.samples, au.sr, g), au.sr)
    print(g.summary())
    print(f"wrote {out / 'grid.json'} and {out / 'click_audition.wav'}")


def _chat(a):
    from . import audio_io, tracking
    from .agent import EditAgent, format_reply
    from .session import EditSession

    au = audio_io.load(a.file)
    g = tracking.analyze(au, progress=lambda m: print("·", m))
    sess = EditSession(au, g, Path(a.file).stem)
    agent = EditAgent(sess)
    print(g.summary())
    print("Type an edit (\"export\" to write the result, \"quit\" to stop).")
    while True:
        try:
            msg = input("> ").strip()
        except EOFError:
            break
        if msg in ("quit", "exit"):
            break
        if msg == "export":
            print(sess.export(Path(f"{sess.name}_edited")))
            continue
        if msg:
            print(format_reply(agent.ask(msg)))


def _sync(a):
    from . import audio_io, sync, tracking

    A, B = audio_io.load(a.a), audio_io.load(a.b)
    r = sync.sync(A, tracking.analyze(A), B, tracking.analyze(B), a.bar_a, a.bar_b, a.seconds)
    out = Path(a.out or "sync_out")
    for name, x in (("overlap_mix", r.mix), ("stem_A", r.a_stem), ("stem_B_stretched", r.b_stem)):
        audio_io.save(out / f"{name}.wav", x, r.sr)
    (out / "report.json").write_text(json.dumps(r.report, indent=1))
    print(json.dumps(r.report, indent=1))


def _recreate(a):
    from . import audio_io, recreate, score
    from .webaudio.render import render_page

    au = audio_io.load(a.file)
    sc = recreate.build_score(au, Path(a.file).stem, progress=lambda m: print("·", m))
    out = Path(a.out or Path(a.file).stem + "_recreation")
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(sc, indent=1))
    page = score.build_page(sc, out / "page.html")
    print(render_page(page, out / "render.wav", sample_rate=au.sr))
    print(f"open {page}")


def _evals(a):
    from evals import run_part1, run_part2, run_part3

    run_part1.main([])
    replay = a.replay or not os.environ.get("ANTHROPIC_API_KEY")
    if replay:
        print("Part 2: replaying the recorded LLM tool calls (no API key, or --replay).")
    run_part2.main(["--replay"] if replay else [])
    run_part3.main([])
    print("Results: evals/results/part1.md, part2.md, part3.md")


def _demo(a):
    _evals(a)
    from .server import main as serve

    serve()


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(prog="beatagent", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("analyze"); p.add_argument("file"); p.add_argument("--out"); p.set_defaults(fn=_analyze)  # noqa: E702
    p = sub.add_parser("chat"); p.add_argument("file"); p.set_defaults(fn=_chat)  # noqa: E702
    p = sub.add_parser("sync"); p.add_argument("a"); p.add_argument("b")  # noqa: E702
    p.add_argument("--bar-a", type=int, default=33); p.add_argument("--bar-b", type=int, default=1)  # noqa: E702
    p.add_argument("--seconds", type=float, default=60.0); p.add_argument("--out"); p.set_defaults(fn=_sync)  # noqa: E702
    p = sub.add_parser("recreate"); p.add_argument("file"); p.add_argument("--out"); p.set_defaults(fn=_recreate)  # noqa: E702
    for name, fn in (("evals", _evals), ("demo", _demo)):
        p = sub.add_parser(name); p.add_argument("--replay", action="store_true"); p.set_defaults(fn=fn)  # noqa: E702
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
