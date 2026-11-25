"""Check that the numbers the README states match the result files.

    python scripts/check_claims.py

Each headline number is recomputed from evals/results/*.json, formatted the way the README
states it, and looked up in README.md. Any mismatch is printed and the exit code is 1.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals" / "results"


def load(name):
    p = RES / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles default to cp1252, which can't print "≤"
        sys.stdout.reconfigure(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    claims: list[tuple[str, str]] = []  # (what, text that must appear in the README)

    p1 = load("part1.json")
    if p1:
        clips = {c["id"]: c for c in p1["clips"]}
        for cid, c in clips.items():
            claims.append((f"{cid} beat F", f"{c['beat_f_70ms']:.3f}"))
            if c.get("downbeat_f_70ms") is not None:
                claims.append((f"{cid} downbeat F", f"{c['downbeat_f_70ms']:.3f}"))
            d = (c.get("drift") or {}).get("max_abs_ms")
            if d is not None and c["grid_type"] != "variable":
                claims.append((f"{cid} max drift", f"| {d:.2f} ms |"))
        k = p1["sync"]["kick_alignment_final"]
        claims.append(("sync kicks median", f"median {k['median_abs_offset_ms']:.1f} ms"))
        claims.append(("sync kicks max", f"max {k['max_abs_offset_ms']:.1f} ms"))
        claims.append(("sync ratio", f"{p1['sync']['time_stretch_ratio']}"))
        claims.append(("A1 analysis time", f"{clips['A1']['analysis_s']:.0f} s"))
    pb = load("part1_baselines.json")
    if pb:
        for r in pb["baselines"]:
            claims.append((f"{r['id']} librosa beat F", f"{r['librosa']['beat_f']:.2f}"))
        worst = max(s.get("max_drift_ms") or 0 for s in pb["stress_A1"])
        claims.append(("stress worst drift", f"≤ {worst:.2f} ms"))
        claims.append(("speed on 4 threads", f"{pb['speed_A1_seconds_by_threads']['4']:.0f} s on 4 threads"))
    p3 = load("part3.json")
    if p3:
        by = {r["id"]: r for r in p3}
        f = lambda cid, k: by[cid]["reference"][k]["f"]  # noqa: E731
        claims += [("P1 drums", f"P1 drums {f('P1', 'drums_note_f'):.2f}"),
                   ("P2 drums", f"drums {f('P2', 'drums_note_f'):.2f}"),
                   ("P2 bass", f"bass {f('P2', 'bass_note_f'):.2f}"),
                   ("P3 piano", f"piano {f('P3', 'pitched_note_f'):.2f}"),
                   ("P3 precision", f"precision {by['P3']['reference']['pitched_note_f']['p']:.2f}")]
        claims.append(("no network requests", "no network requests"))
        assert all(len(r["render"]["network_requests"]) == 0 for r in p3), "a page made a network request"
    p2o = load("part2_offline.json")
    if p2o:
        turns = [t for s in p2o.values() for t in s["turns"]]
        ok = sum(1 for t in turns if t.get("interpretation", {}).get("ok"))
        claims.append(("offline grammar score", f"{ok}/{len(turns)}"))
    n_tests = re.search(r"(\d+) tests? collected", subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], cwd=ROOT, capture_output=True,
        text=True).stdout)
    if n_tests:
        claims.append(("unit test count", f"{n_tests.group(1)} fast unit tests"))

    bad = [(w, s) for w, s in claims if s not in readme]
    for w, s in claims:
        print(f"{'ok ' if (w, s) not in bad else 'MISSING'}  {w:28s} {s}")
    print(f"\n{len(claims) - len(bad)}/{len(claims)} claims found in README.md as stated by the result files.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
