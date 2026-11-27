# Beat Agent

One chat-driven agent for a piece of audio:

1. **Beat alignment.** A beat grid a DJ app could trust (tempo, every beat with bar and position, downbeats, metre, confidence, constant/piecewise/variable), a click audition, and two-track beat-matching with a pitch-preserving stretch.
2. **Chat editing on the grid.** "Cut beat 16–32 and repeat it twice" becomes typed edit operations that code validates, echoes in exact beats/bars/seconds, and applies without clicks. The grid stays true after every edit, and undo/redo works per turn.
3. **Audio to playable code.** A 2–16 bar clip becomes a score (JSON) and a self-contained HTML page that recreates it with Web Audio synthesis, plus an offline render so it can be measured.

**Core rule: the LLM interprets, code measures and executes.** Tempo, beat positions, onsets and notes come from signal processing and models (Beat This!, Rubber Band, Demucs, Basic Pitch). The LLM (Claude or Gemini) only turns words into typed operations; it never produces a time and never writes the JavaScript.

Without a working API key, the app falls back to an **offline grammar** ([`beatagent/offline.py`](beatagent/offline.py)) and says so in every reply.
* It turns a documented subset of phrasings into the same typed operations, through the same validation and echo.
* It refuses everything else.
* It is a fallback so the app, the demo and the evals still run; it does not replace the LLM's handling of paraphrase and ambiguity.

> **Read this first: all numbers below come from a synthetic stand-in test pack.** The brief's test pack (A1–A7, P1–P4) was not attached, and I asked for it. `testpack/generate.py` synthesises one clip per item from its description, with exact ground truth.
> * Synthetic audio is cleaner than real recordings, so these numbers are an upper bound. Stress tests with degraded copies are below.
> * The harness reads the real pack as a drop-in: set `BEAT_AGENT_PACK=path/to/pack`; the layout is in [`testpack/README.md`](testpack/README.md).

## Run it

You need Python 3.12 and about 3 GB of disk for packages and models. An `ANTHROPIC_API_KEY` or a `GEMINI_API_KEY` enables the LLM interpreter (with both, Claude is tried first); without one, chat uses the offline grammar and says so.

```bash
cp .env.example .env        # add ANTHROPIC_API_KEY or GEMINI_API_KEY
make demo                   # setup (first run only) -> all evals, live with your key (~20 min on a laptop CPU) -> web app on http://127.0.0.1:8000
```

* **Without make** (e.g. Windows): `python scripts/setup.py`, then `.venv\Scripts\python -m beatagent.cli demo`.
* **Docker:** `docker compose up`. The image builds in CI; a full `up` hasn't been run in a container (see the status table).

| Command | What it does |
|---|---|
| `make serve` | The web app alone; every result is already in `evals/results/` |
| `make evals` | Parts 1–3; Part 2 and the recreation chat call the LLM live and are graded against an answer key (replayed or offline without a key) |
| `make evals-replay` | The same, with Part 2 replaying the recorded tool calls (no API calls); `cli demo --replay` does this before starting the app |
| `python -m evals.run_part2 --reference` | Part 2's sessions run by the engine from the answer key, no LLM |
| `python -m evals.run_part2 --offline` | Part 2's sessions through the offline grammar, graded against the answer key |
| `python -m evals.run_recreate_chat [--offline]` | Chat changes to a recreation (mute, brighter, tempo, level, solo, undo, a refusal), graded on the score each turn leaves |
| `python -m evals.run_baselines` | librosa and raw Beat This! baselines, plus degraded-audio stress tests |
| `make check` | Checks that every number this README states matches `evals/results/` |
| `make test` | 123 fast unit tests: conventions, edit invariants (including randomised multi-turn sessions), formats, page safety, malformed scores and commands, tempo maths, both grammars, Gemini schema and fallback, server file and upload limits |
| `python -m beatagent.cli analyze\|chat\|sync\|recreate FILE` | Individual tools on your own audio |

Setup also downloads:
* the Beat This! weights (81 MB, fetched in parallel byte ranges because the authors' server throttles each connection to ~15 KB/s);
* the Demucs `htdemucs` weights;
* headless Chromium for the offline renders.

## Architecture

```
            words                     typed ops (JSON)                   exact echo + new state
 user ──────────────► LLM (tool use)    ───────────────► validate & apply ──────────────────────► user
                         ▲   reads only:                     (beatagent/edl.py, score.py)
                         │   grid summary, history,
                         │   measured sections
 audio ──► tracking.py ──┴──► BeatGrid (grid.py) ──► EditSession: non-destructive timeline of
           Beat This! logits     beats, bars,         source pieces + gain ramps; render with
           + onset-aligned       downbeats, metre     pre-beat crossfades; grid derived from pieces
           tempo model                                         │
                                                               ├──► click.py  (audition)
                                                               └──► sync.py   (Rubber Band R3, pinned at bar anchors)

 clip ──► recreate.py: loop grid (tracker) · Demucs stems · drum onsets + classifier · Basic Pitch
          └──► score.json ──► webaudio/template.html (fixed synth engine; score embedded as data; CSP: no network)
                                   └──► webaudio/render.py (headless Chromium, OfflineAudioContext) ──► render.wav
```

| Module | What it does |
|---|---|
| `tracking.py` | Beat This! frame logits → raw beats → constant / piecewise / variable tempo model. The tracker proposes the tempo; the measured onset curve decides phase, slope and whether the tempo really is constant. Also metre, downbeat phase, compound-metre regrouping, grid extent and confidence. |
| `grid.py` | `BeatGrid`: the JSON format, the indexing conventions, and the factual summary the LLM reads. |
| `edl.py` | Typed ops (`remove`, `keep_only`, `repeat`, `move`, `insert_copy`, `fade`, `gain`, `mute`) in beats or bars, a non-destructive timeline, the click-free renderer and grid derivation. |
| `session.py`, `agent.py`, `llm.py` | Undo/redo per chat turn; the Claude and Gemini tool loops (same tools, same schemas; busy Gemini models are skipped for the next); the system prompt with the conventions. |
| `sections.py` | Measured sections (band energy per bar) for subjective requests like "the boring part". |
| `sync.py`, `stretch.py` | Ratio and offset from the two grids; a Rubber Band stretch pinned at bar anchors; measured kick alignment. |
| `recreate.py`, `score.py`, `webaudio/` | Audio → score → page → offline render, plus typed score ops for chat ("mute the drums"). |
| `server.py`, `static/index.html` | The web app: waveform with the grid, chat with echo cards, direct (no-LLM) edits, undo/redo, sync, and the recreation next to the original. |
| `evals/`, `tests/` | The harnesses and the answer key ([evals/README.md](evals/README.md)); unit tests. |

## Status

| Area | Status | Evidence |
|---|---|---|
| Beat grid JSON (tempo, beats with index/bar/position, downbeats, metre, confidence, grid type, conventions) | Works | `evals/out/part1/*/grid.json` |
| Click audition | Works. I have not listened to every file yet (see `evals/LISTENING.md`). | `evals/out/part1/*/click_audition.mp3` |
| Beat/downbeat accuracy (mir_eval) | On the stand-in: beat F ≥ 0.995 on all 7 clips, downbeat F ≥ 0.99. librosa reaches 0.23–1.0. | `evals/results/part1.md`, `part1_baselines.md` |
| Grid drift, 6-minute constant clip | 0.88 ms max over 736 beats (target 10 ms). Stays ≤ 1.34 ms under MP3, noise, low-pass and kick removal. | same |
| Speed | 6-minute track in 12 s on a laptop CPU, also 12 s on 4 threads (target 60 s) | same |
| Two-track sync | A7 → A1: measured kick onsets median 0.8 ms, max 1.9 ms (targets 5 and 15 ms). Variable-tempo partners also work; stretches over 8% are flagged. | `evals/results/part1.md` |
| Typed edit ops, echo, clean splices, sample rate and bit depth kept, grid stays true, undo/redo | Works: all brief sessions run through the engine (reference mode) plus unit tests | `evals/results/part2_reference.md`, `tests/` |
| Chat sessions from the brief through the live LLM | **10/10 turns match the answer key**, live through Gemini (`gemini-3.5-flash-lite`) with no grammar fallback, for about $0.06. The edited audio is byte-identical (sha256) to the reference run's. The recorded tool calls replay with no API calls, also 10/10. Ten turns is a small sample, and the larger Gemini models were overloaded, so this is the smallest one. **Not run through Claude**: the Anthropic key on the build machine was rejected (401). | `evals/results/part2.md`, `part2_replay.md`, `evals/recorded/` |
| Same sessions through the offline grammar | 10/10 turns match the answer key. I wrote the grammar with these commands in view, so this shows the fallback works, not that it generalises. Unit tests cover other phrasings and refusals. | `evals/results/part2_offline.md`, `tests/test_offline.py` |
| Edits on a variable-tempo grid | Works (A3 live band, A5 tempo change in reference mode) | same |
| Preview player with waveform and grid; direct edits; "confirm before applying" mode | Works. In confirm mode, edits are proposed with their exact echo and applied only on the user's click; this is enforced in code. | web app, Edit tab |
| Audio → score → Web Audio page → offline render | Works for P1–P4; no network requests, no page errors. Reproducible: a full re-run gives byte-identical scores and identical metrics; the Chromium renders differ by at most one 16-bit step (−90 dB) on under 0.05% of samples | `evals/results/part3.md` |
| Note F vs reference MIDI | P1 drums 0.85; P2 drums 0.89, bass 0.98; P3 piano 0.78. Score timing vs MIDI: 0.3–1.1 ms median. | same |
| Chat changes to the recreation, per-track mixer | The mixer works. Recreation chat (mute, brighter, tempo, level, solo, undo, and a refusal for "add a guitar"): live: 7/7 turns leave the score exactly as expected (Gemini); offline grammar: 7/7. | `evals/results/part3_chat.md`, `part3_chat_offline.md` |
| `docker compose up` | The image **builds in CI** (`docker compose build`, including setup inside it: packages, Chromium, both model downloads and the stand-in pack; [run](https://github.com/eeshsaxena/beat-agent/actions/runs/45824763842)). **A full `up` (evals + app) has not been run in a container**; the same commands run natively. | `Dockerfile`, `.github/workflows/ci.yml` |
| Full-resolution artefacts | In the repo: every grid, EDL, report, score, page, Part 3 render and an MP3 of every audio file. Full-resolution edits, stems and the stand-in pack are attached to the [v1.0 release](https://github.com/eeshsaxena/beat-agent/releases/tag/v1.0) (built by `make package`); `make evals` regenerates them deterministically. | `evals/README.md` |
| Real test pack | **Missing**: it was not provided; the loader is ready | `evals/pack.py` |

## Results (stand-in pack)

**Part 1** ([table](evals/results/part1.md), [baselines and stress tests](evals/results/part1_baselines.md)):

| Clip | Grid | Tempo (ref) | Metre | Beat F | Downbeat F | Max drift | librosa beat F | Beat This! raw: max drift / downbeat F |
|---|---|---|---|---|---|---|---|---|
| A1 house, 6 min | constant | 124.00 (124.0) | 4/4 | 1.000 | 1.000 | 0.88 ms | 0.23 | 54 ms / 0.83 |
| A2 hip-hop, half-time | constant | 140.00 (140.0) | 4/4 | 0.995 | 1.000 | 0.46 ms | 0.67 | 63 ms / 0.98 |
| A3 live band | variable | 93.02 (93.0) | 4/4 | 1.000 | n/a | n/a | 0.99 | n/a |
| A4 ambient intro | constant | 120.00 (120.0) | 4/4 | 0.995 | 1.000 | 0.56 ms | 1.00 | 50 ms / 0.81 |
| A5 tempo change | piecewise | 120 → 128 | 4/4 | 0.998 | 0.993 | 1.08 ms | 0.84 | n/a / 0.77 |
| A6 6/8 dholak | constant | 68.00 (68.0) | 6/8 | 0.996 | 0.992 | 1.61 ms | 0.40 | 21 ms / 0.98 |
| A7 house | constant | 128.00 (128.0) | 4/4 | 1.000 | 1.000 | 1.59 ms | 0.99 | 39 ms / 0.91 |

**Sync** A7 → A1 from A1's bar 33: stretch ratio 1.032253 (128 → 124 BPM); measured kick onsets median 0.8 ms, max 1.9 ms over 124 kicks. The variable-tempo partners also land on the grid:
* A5 → A1: reference beats within 0.3 ms.
* A3 → A1: reference beats within 1.5 ms median, 5.6 ms max, with a warning that its 35% stretch is musically unreasonable.

**Part 2, live LLM** ([table](evals/results/part2.md)): Gemini read every command the way the answer key does, including the two pushbacks (code refused bars 200–210; "the boring part" got measured candidates and a question).

**Part 2**, engine on the answer key, no LLM ([table](evals/results/part2_reference.md)):
* The offline grammar's edits are byte-identical (sha256) to the answer key's.
* Every edit keeps the input's format.
* Every splice scores ≤ 1.02, where about 1 means as continuous as the music itself.
* The edited grid matches the reference beats carried through the edit at beat F ≥ 0.995.
* "Delete bars 200–210" is refused with the real range, and "the boring part" returns measured sections.

**Part 3** ([table](evals/results/part3.md)): P1 drums 0.85 · P2 drums 0.89, bass 0.98 · P3 piano 0.78 · P4 renders (no reference).
* Score timing vs MIDI is a median 0.3–1.1 ms.
* Render onsets vs the original scatter about 10 ms, with a mean near 0, from the synth voices' attack shapes.

## Top three problems I know about

1. **Every number comes from synthetic audio.** The stand-in has exact timing and clean attacks, so real recordings will score lower. The degraded-copy stress tests hold up (drift ≤ 1.34 ms), but they don't replace real music. Onset alignment needs clear percussive onsets; without them the grid keeps the tracker's own placement. On this pack that placement lagged the onsets by 15–36 ms on five of the seven clips and by ~0 on the live band and the 6/8 clip (see `analysis.fit.onset_alignment` in each grid JSON).
2. **Drum transcription is a hand-tuned rule classifier, tuned on the same P1/P2 it is scored on.** Expect overfitting. Open hats are missed, and hats under snares can't be separated by band energy (measured: the snare's noise dominates). A trained drum transcription model is the fix.
3. **Polyphonic transcription (P3) is the weakest part.** Basic Pitch reports harmonics and decays as notes (precision 0.64, recall 0.99). Timbre is a few measured numbers (filter cutoff, kick pitch) feeding generic synth voices.

See [DECISIONS.md](DECISIONS.md) for the main choices, every assumption, and what I did not build. [AI_USAGE.md](AI_USAGE.md) covers how AI was used and seventeen cases where it was wrong and how that was caught.
