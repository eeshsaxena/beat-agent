# Evals

All numbers come from the synthetic stand-in pack (`testpack/`), unless you point `BEAT_AGENT_PACK` at the real one.

| Command | What it measures | Results |
|---|---|---|
| `python -m evals.run_part1` | Beat grid per clip: mir_eval beat/downbeat F (±70 ms), the full mir_eval suite, grid drift vs reference, analysis time. Sync A7 → A1 at bar 33, plus two variable-tempo partners (A5, A3). | [results/part1.md](results/part1.md) |
| `python -m evals.run_baselines` | librosa and raw Beat This! vs the full pipeline on every clip; A1 degraded four ways (MP3 96 kbps, 15 dB noise, 3 kHz low-pass, kick removed); speed with 4 threads. | [results/part1_baselines.md](results/part1_baselines.md) |
| `python -m evals.run_part2` | The brief's chat sessions through the live agent, each turn graded against [answer_key.py](answer_key.py), with the format, splice and grid checks below. Writes `recorded/part2_sessions.json`. | [results/part2.md](results/part2.md) |
| `python -m evals.run_part2 --replay` | The same, replaying the recorded tool calls (no API calls). | same |
| `python -m evals.run_part2 --offline` | The same sessions through the offline grammar (no LLM), graded against the answer key. The grammar was written with these commands in view; `tests/test_offline.py` covers other phrasings. | [results/part2_offline.md](results/part2_offline.md) |
| `python -m evals.run_part2 --reference` | The same sessions with the answer key executed by the engine (**no LLM**). Tests the edit engine, not the agent. | [results/part2_reference.md](results/part2_reference.md) |
| `python -m evals.run_part3` | Audio → score → page → offline render for P1–P4: note F vs reference MIDI (onset-only, ±50 ms), score timing vs MIDI, render onset timing vs the original, network requests and page errors. | [results/part3.md](results/part3.md) |
| `python -m pytest -q tests` | Model-free unit tests of the conventions and invariants. | |

## Part 2 checks, per turn

* **Format:** the edited file has the input's sample rate and subtype.
* **Splices:** the second difference at each splice divided by the same quantity at the joined points of the source. About 1 means continuous; a click scores far above 1. Every result row also shows a naive butt splice as a control, and `tests/test_core.py` has a case where the control scores > 5 and the crossfade < 5.
* **Grid truth:**
  * (a) the reference beats carried through the same edit, compared with the edited grid;
  * (b) a fresh tracker analysis of the edited audio, compared with the edited grid. This is weaker where the music has no kick (e.g. A2's looped intro), because the tracker itself struggles there.

## Artefacts (`out/`)

| Part | Where | Files |
|---|---|---|
| 1 | `out/part1/<ID>/` | `grid.json`, `click_audition.mp3` (+ `.wav` when regenerated), `metrics.json` |
| 1 | `out/part1/sync_<B>_to_A1/` | `overlap_mix`, `stem_A`, `stem_B_stretched` (`.mp3`, + `.wav`), `overlap_mix_click_audition`, `report.json` |
| 2 | `out/part2/<session>/` (live), `out/part2/offline/<session>/`, `out/part2/reference/<session>/` | `turn<N>.wav|flac` (edited audio at the input's format), `turn<N>.edl.json` (the edit decision list: every turn's ops and echo, plus the timeline of source pieces), `turn<N>.grid.json`, `turn<N>.click_audition` (`.mp3`, + `.wav`), `turn<N>.mp3`, `transcript.txt`, `turns.json`, and for session 6 the sync render |
| 3 | `out/part3/<ID>/` | `score.json`, `page.html` (open it in a browser), `render.wav` (2 loops from the headless offline render), `metrics.json` |

Full-resolution WAV/FLAC files are not committed (a 24-bit 6-minute edit is ~97 MB). `make evals` regenerates them deterministically, and `python scripts/package_artifacts.py` zips them with the stand-in pack for the [v1.0 release](https://github.com/eeshsaxena/beat-agent/releases/tag/v1.0), where they are attached. `python scripts/check_claims.py` checks that every number the README states matches these result files.

## What has been listened to

See [LISTENING.md](LISTENING.md). Measurements don't replace listening.
