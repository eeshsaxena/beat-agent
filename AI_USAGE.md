# AI usage

> **Author's note (fill in before submitting).** The sections marked **TODO (you)** must be written from your own review. The "caught during the build" cases below are real and reproducible: each lists the command that shows it. Re-run them, confirm them, and add the cases you find yourself while reviewing the code line by line.

## Tools

* **Claude Code** (Claude Opus 5.5, Claude desktop app) as the coding assistant for the whole build.
* **Claude Opus 5.5 inside the product**: the chat agent's interpreter (`beatagent/llm.py`), behind typed tool calls.

## What I delegated

* Nearly all the code: the beat-grid post-processing, the edit engine, sync, transcription, the Web Audio engine, the web app and the eval harnesses.
* The synthetic stand-in test pack (`testpack/generate.py`).
* First drafts of README.md and DECISIONS.md.

## What I reviewed line by line

**TODO (you):** list the files you read in full and what you checked in each. Suggested priority:
1. `beatagent/edl.py`: range conventions, the crossfade placement, grid derivation.
2. `beatagent/tracking.py`: `_fit_tempo` and `align_constant`, the constancy tests.
3. `beatagent/agent.py`: the system prompt (it defines how commands are read).
4. `beatagent/webaudio/template.html`: the code that runs in a user's browser.
5. `evals/run_part1.py` and `run_part3.py`: what exactly each metric measures.

## Where the AI was wrong or subtly off

### Caught during the build, by measurement (verify each)

1. **It assumed the neural tracker's beats are unbiased in time.** A least-squares grid on raw Beat This! beats gave 43 ms max drift on A1. Per-section measurement showed the tracker 8 ms late in kick-and-hat sections but 35–42 ms late where the offbeat bass played, so the lag depends on the arrangement. Fixed by fitting the grid's phase and tilt to the onset curve: max drift 0.88 ms.
   *Check:* `onset_alignment.shift_ms` in `evals/out/part1/A1/metrics.json` (−30.5 ms).
2. **A "fix" made a different clip wrong.** After relaxing the constant-tempo test so A1 passed, A5, the clip with a tempo change, was accepted as constant 128 BPM. The grid region had silently dropped the 120 BPM half, so every check passed on the half that was left. Beat F 0.72 exposed it. Fixed by requiring the constant grid to cover everything the tracker confidently tracked, and to agree with it window by window.
   *Check:* A5 is `piecewise` in `evals/results/part1.md`.
3. **The sync "refinement" made alignment worse.** The first kick detector used the low band, whose attack timing depends on the kick's pitch sweep (6 ms for A1's kick, 11.5 ms for A7's). An automatic nudge built on it shifted B by −4.2 ms to "fix" a detector artefact, and the true downbeat alignment from the reference annotations got 4 ms worse. Caught by comparing the detector against ground-truth kick positions. Fixed with a validated detector (median 0.1 ms error); the nudge is off by default.
4. **A filter choice created a phantom onset 17 ms before every kick.** The zero-phase (filtfilt) band filter with a 150 Hz edge pre-rings ahead of an attack. Caught by printing the onset curve around known kicks. Fixed with a causal high-band filter for attack timing.
5. **The Rubber Band stretch looked right but wasn't.** A test on a click train with a "nice" ratio passed, but the real ratio (1.032261) let kicks drift 7.6 ms over 60 s while total length stayed exact. Caught by measuring kick positions along the render. Fixed by pinning the time map at anchors every 4 bars, which also works around pylibrb not accepting key-frame maps.
6. **A plausible transcription filter deleted real notes.** "Merge re-triggered notes" removed genuine repeated piano chords: P3 recall fell from 0.94 to 0.69. Caught by measuring each filter separately. Removed.
7. **A3's grid stopped at 120 s** because the region step used the constant-tempo line on a drifting live take. Caught when beat F came out at 0.84. Fixed by taking the region from the tracker's own confident beats.
8. **The score validator crashed on exactly the input it exists to reject.** A string where a number belongs raised a `TypeError` inside a comparison instead of a clean `ScoreError`. Caught by the unit test `test_score_validation_rejects_bad_data`. Fixed by checking types before values.
   *Check:* `python -m pytest -q tests`.
9. **The "boring part" analysis was too coarse to be useful.** Its onset-density feature read ~10 onsets per beat everywhere (detector noise), so A1's thin intro and outro were merged into the full sections. Caught by reading the sections the reference run produced. Rebuilt on per-bar band energies: A1 now splits exactly at its arrangement changes (bars 17, 65, 77, 81, 169).
10. **A metric that mixed two things.** Part 3's "render onset timing" (~10 ms median) looked like a transcription timing problem. Measuring the score against the reference MIDI showed the transcription is within 0.3–1.1 ms; the scatter comes from the synth voices' attack shapes. Both numbers are now reported, with the explanation.
11. **A failed eval looked like a successful one.** In a chained run, `run_part3 | grep … && run_part2 …` crashed (a variable that held the output folder was reused for a list), but the chain continued because a pipeline's exit status is grep's. Caught because `part3.md` still had an old timestamp. Fixed the variable, and chained runs now use `set -o pipefail`.
12. **Results that changed between identical runs.** P2's bass F went from 1.0 to 0.98 with no code change: Demucs applies one random time shift by default (`shifts=1`). Fixed with `shifts=0`; two runs now produce byte-identical scores.
13. **A regex that could never match.** A patch script turned `` (word boundary) into a literal backspace character in the offline grammar. As a result, "remove bar 5 and bar 9" silently removed only bar 5 instead of being refused. Caught by the unit test `test_refuses_what_it_cannot_parse`, then a scan of every source file for control characters (none elsewhere). Fixed, and later patches avoid shell-quoted escapes.

### Caught in my own review

**TODO (you):** add at least one case you found yourself while reading the code, with the file, the line, what was off and how you noticed. Look closely at the edit conventions in `edl.py` against the system prompt in `agent.py`, and at the open-hat rule in `recreate.py`.

## Verification I did myself

**TODO (you):**
* Which click auditions and renders you listened to, and what you heard (fill in `evals/LISTENING.md`).
* Whether you re-ran `make evals` and the numbers matched `evals/results/`.
