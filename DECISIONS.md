# Decisions

Each decision lists the options I considered, what I chose, the trade-off, and what would change my mind. Assumptions about unclear wording, and what I chose not to build, are at the end.

## 1. Build a synthetic stand-in test pack rather than wait

The brief's test pack (A1–A7 with beat and downbeat annotations, P1–P4 with MIDI) was not attached to the brief or the email. I asked for it.

* **Options:** wait for it; use public datasets (GTZAN/Ballroom beat annotations, Groove MIDI, MAESTRO); synthesise a stand-in.
* **Chose:** `testpack/generate.py`, which writes one clip per item that mirrors its description, with exact ground truth.
  * Each clip carries the trap its item implies: a breakdown with no kick, a half-time feel with 32nd and triplet hi-hat rolls, a drifting live tempo with per-hit jitter and a pickup, a free-time ambient intro, an abrupt 120 → 128 BPM change, 6/8 with dholak strokes, and a 48 kHz sync partner.
  * The clips also use mixed formats: 16- and 24-bit WAV, FLAC and MP3.
  * The loader (`evals/pack.py`) takes the real pack as a drop-in.
  * Public datasets would have meant large downloads with their own licences, and I would still have had to build the brief's specific cases.
* **Trade-off:** the numbers are optimistic. Synthetic audio has exact timing and clean attacks. I state this at the top of the README and in every results file, and run stress tests with degraded copies (`evals/results/part1_baselines.md`).
* **Would change my mind:** the real pack arriving. That is one environment variable and one re-run.

## 2. The tracker proposes the tempo; the onsets decide phase and constancy

* **Options:** librosa (the brief's baseline), madmom (unmaintained on Python 3.12), Beat This! with its own peak picking or a DBN, BeatNet.
* **Chose:** Beat This! (`final0`) for frame-wise beat and downbeat activations, followed by my own post-processing:
  * a least-squares tempo model: one constant line, a few piecewise segments, or local smoothing for live playing;
  * the grid's phase and tilt fitted to the measured onset-strength curve;
  * a constant grid accepted only when every local window of onsets agrees, and the tracker agrees wherever it is confident.
* **Why:** I measured the raw tracker on the stand-in.
  * Its beats were 8 ms late in kick-and-hat sections but 35–42 ms late wherever offbeat bass and sidechained pads played. The lag depends on the arrangement, which no constant correction fixes.
  * Raw Beat This! drifts 21–63 ms on the constant clips; with onset alignment, A1's max drift over 6 minutes is 0.88 ms.
  * librosa scores beat F 0.23–1.0.
* **Trade-off:** phase precision depends on clear percussive onsets. Without them (pads, rubato) the grid falls back to the tracker's own placement, with its lag. The post-processing is also more code to trust than calling the model's own peak picking.
* **Would change my mind:** real annotations showing that annotators tap later than the physical attack. Then the onset step would need a calibrated offset.

## 3. Grid conventions

* **Options:** beats and bars both 0-based; both 1-based; beats 0-based with bars 1-based. Inclusive or half-open ranges.
* **Chose:**
  * Beats are **0-based** and beat ranges are **half-open**: "beat 16–32" is 16 beats, ending where beat 32 starts. The brief itself says "beat 0", so beats must start at 0.
  * Bars are **1-based** and bar ranges are **inclusive**: "bars 1–4" is 4 bars. This matches DAWs, and with it "A1's bar 33" is the start of the second 32-bar phrase.
  * Beats before the first downbeat are a **pickup in bar 0**.
  * Beat position N (the number of beats) means the end of the file, so the last beat runs to the end of the file, tail included.
  * Audio before beat 0 is a pre-roll that beat ranges never touch.
* **Trade-off:** mixed bases are easy to confuse. That's mitigated by writing the conventions into every grid JSON (`conventions`), into the LLM's system prompt and into the typed field names (`end_beat`, `last_bar`), and by having every echo restate the resolved beats, bars and seconds.
* **Would change my mind:** Soundverse's existing tools using another convention. Matching the house convention beats any internal preference; it would be one place in `edl.resolve_range` plus the prompt.

## 4. Non-destructive edit decision list

* **Options:** destructive array edits with snapshot undo; a list of source pieces.
* **Chose:** a timeline of source pieces with gain ramps attached to each piece.
  * Fades and gains travel with the audio they were applied to.
  * Undo is popping a list, cheap even for 6-minute 24-bit files.
  * The output grid is derived from the pieces, so it cannot drift from the audio.
  * The EDL is exported per turn (`turn<N>.edl.json`).
* **Splices:** a 10 ms equal-power crossfade placed *before* the beat, so the incoming transient (usually the kick) is untouched and every piece keeps its exact length.
* **Several ops in one message:** they are addressed against the grid at the start of that message and applied right to left. Overlapping ops are refused with a request to send them separately.
* **Trade-off:**
  * Rendering is from the source each time (fast enough here: well under a second for 6 minutes).
  * A pre-beat crossfade borrows 10 ms of the audio before the incoming beat, which is inaudible in tests but is not a pure butt splice.
  * Splice quality is measured, not assumed: a score per splice, with a butt-splice control that shows the metric catches clicks.
* **Would change my mind:** edits that need sample-level processing (time-stretching a section, reversing). The timeline would need per-piece processors, or a render-and-freeze step.

## 5. LLM boundary: typed operations and a code-written echo

* **Options:** the LLM returns times or audio actions directly; the LLM returns free-text intent parsed by code; the LLM calls typed tools that code validates.
* **Chose:** typed tools.
  * Claude Opus 5.5 through the Messages API with tool use, or Gemini through Google's API with function calling. `BEAT_AGENT_LLM` picks one; with both keys, Claude is tried first. Model and effort are environment variables.
    * Both loops expose the same tools, built from the same pydantic schemas. For Gemini the schema is reduced to the subset its function declarations accept: `$ref` inlined, `oneOf` → `anyOf`, `const` → one-value `enum`.
    * Gemini models are often overloaded (503s, 60-second deadlines). The loop ranks this key's stable models (Flash newest first, then Flash-Lite, then Pro), skips a busy one for the next, and records which model answered each turn. The last model that answered is remembered for six hours (`runtime/gemini_last_model.json`), so a restart doesn't wait through the busy ones again.
    * A request that fails partway through a turn rolls back the whole turn from the model's history: a tool call left without its result would make the API reject every later message. Edits the turn already applied stay applied, and the next state summary reports them.
  * The LLM sees only a factual grid summary computed by code (beats, bars, tempo segments, the last full bar before each tempo change, irregular bars) and the edit history.
  * It returns pydantic-typed ops. Code validates every op and writes the echo the user sees; the LLM's words are shown separately.
  * **Subjective requests** ("the boring part"): the LLM must call `find_sections`, which measures band energy per bar, section boundaries and repetitiveness, then ask the user to choose. Taste is not delegated to the model.
  * **Measuring interpretation separately from the engine:** `evals/answer_key.py` holds the expected ops for every command in the brief's sessions.
    * `--reference` runs the engine on the key with no LLM.
    * A live run grades each agent turn against the key, after resolving both against the grid.
    * The system prompt's worked examples deliberately use other commands.
    * Every live run records its tool calls for `--replay`, and spend is logged per call.
  * **Offline fallback:** with no key, or a key the API rejects, a deterministic grammar (`beatagent/offline.py`) produces the same typed ops and refuses anything else. Every reply says which interpreter answered.
* **Trade-off:**
  * An extra round trip per message and a dependency on an API key.
  * The fallback grammar handles only a documented subset of phrasings.
  * I chose it over a local open model: a small model on a laptop CPU is slow, still fallible, and a multi-GB download.
* **Would change my mind:** measured interpretation errors on the live sessions. I would add tool-level few-shot examples or a second check that compares the LLM's resolved ops with the grammar's where the grammar understands the message. So far: 10/10 live turns through Gemini Flash-Lite and 7/7 recreation-chat turns, with no errors to act on yet; the sample is small.

## 6. Sync: grid-based mapping, Rubber Band pinned at anchors, measured kicks

* **Options:**
  * a constant ratio over the whole overlap;
  * a Rubber Band key-frame map;
  * stretching between anchors.
  * For checking: trust the grids, or measure the kicks.
* **Chose:**
  * The ratio and offset come from the two grids: B's start bar lands on A's start bar, beat for beat.
  * Rubber Band (R3 "finer" engine, pitch preserved) through `pylibrb`, with the time map pinned at anchors: every 4 bars for constant grids, every bar for variable ones. Each span is stretched separately and joined with pre-anchor crossfades.
  * Kick alignment is measured with a detector validated against ground truth (median 0.1 ms, max 0.6 ms on unstretched audio).
* **Why:**
  * With a ratio that isn't a simple fraction (1.032261), a single stretch let transients drift 7.6 ms over 60 s, although total length was exact.
  * pylibrb 0.1.2 can't pass a key-frame map from Python (its map type caster isn't compiled in).
  * A first kick detector read the low band and had a 6–12 ms bias that depended on the kick's timbre. An automatic "nudge" built on it made the true alignment worse, so the nudge is off by default.
* **Trade-off:** anchors add a crossfade every 4 bars inside the stretched track. It's placed before a downbeat and inaudible in my tests, but it is a seam that a single stretch wouldn't have.
* **Would change my mind:** a Rubber Band binding with working key-frame maps, which would remove the seams.

## 7. Recreation: the LLM never writes JavaScript

* **Options:** the LLM writes the page; a fixed engine plus data.
* **Chose:** one hand-written synth engine (`webaudio/template.html`) that reads the score as inert JSON.
  * The score is validated first (numbers and short names only), and its `<` characters are escaped so it can't close its script tag.
  * The page's Content-Security-Policy is `default-src 'none'`, so it can't make network requests. The offline render checks this.
  * The same voice code drives live playback and the OfflineAudioContext render.
  * The score stores time in beats from the first downbeat, so tempo is a single number and the page loops at exactly the clip's length.
* **Why:** generated code in a user's browser can do anything: exfiltrate data, misbehave, or differ on every run. The model also adds nothing to timing accuracy, which comes from measurement. Chat changes ("mute the drums", "play it at 140") are typed score ops, not code.
* **Trade-off:** timbre is limited to the voices I wrote (subtractive bass, FM keys, a lead, a pad, synthesised drums). A model-written patch might sound closer on some clips.
* **Would change my mind:** a need for richer timbre would lead to more voices and parameters in the engine. The model would still never write the engine.

## 8. Transcription stack

* **Options:** Basic Pitch or MT3 for notes; a drum transcription model or onsets plus a classifier for drums; with or without source separation.
* **Chose:**
  * **Stems:** Demucs `htdemucs` on the clip tiled three times, so loop edges have context. It uses `shifts=0`, so results are reproducible.
  * **Pitched parts:** Basic Pitch through its bundled ONNX model. Its pinned dependencies don't build on Python 3.12, so it is installed with `--no-deps` and run with onnxruntime.
  * **Drums:** a measured rule classifier on band envelopes (low / 300 Hz–5 kHz / above 7 kHz), with each hit's time refined to the causal attack.
    * Every rule came from a measured failure: low-band ripple turned one kick into five onsets; kick clicks were read as snares; zero-phase smoothing made hits 10–15 ms early.
  * **Tried and removed:** a re-trigger merge for pitched notes. It deleted real repeated chord attacks (P3 recall fell from 0.94 to 0.69).
* **Trade-off:**
  * The drum rules are tuned on the same P1/P2 they are scored on, so expect overfitting.
  * Hats under a snare can't be separated by band energy (measured).
  * Polyphonic precision is 0.64.
* **Would change my mind:** a trained drum transcription model (ADTOF-style), or MT3 if CPU time allowed. Either would be a drop-in replacement behind the same score format.

## 9. Compound metre

* **Options:** always report the tracker's own level; regroup by a tempo threshold; ask the user.
* **Chose:** regroup by threshold. When the tracker locks to a fast ternary level (3 beats per bar above 190 BPM), three tracker beats become one dotted-quarter beat, and the metre is reported as 6/8 or 12/8.
  * A6's tracker ran at 204 BPM in "3/4". Regrouped, it is 68 BPM 6/8 with beat F 0.996 against a reference annotated at the dotted quarter.
* **Trade-off:** it's a heuristic. The 190 BPM threshold keeps Viennese waltz (about 177 BPM quarter notes in common datasets) as 3/4. A real 3/4 piece faster than that would be misread.
* **Would change my mind:** a reference annotated at the eighth-note level, or a fast 3/4 clip in the real pack. The threshold, or the whole rule, would go.

## 10. Output formats

* **Options:** write everything as 16-bit WAV; keep the input's container and subtype.
* **Chose:** keep the input's sample rate and subtype (16-bit, 24-bit or float), and WAV or FLAC container. MP3 inputs are written as 16-bit WAV, since MP3 has no bit depth and re-encoding would add loss.
* **Trade-off:** 24-bit 6-minute edits are ~97 MB, too big for git. The repo carries MP3 listening copies and all JSON. The full-resolution files are regenerated deterministically by `make evals` and packaged for a release by `scripts/package_artifacts.py`.
* **Would change my mind:** a requirement to keep MP3 in and MP3 out (e.g. a web product). Then re-encode at the input's bitrate and say so.

## Assumptions where the brief or a command was unclear

| Where | Reading I chose |
|---|---|
| "Cut beat 16–32 and repeat it twice" | Take beats 16–31 (half-open) and repeat them in place so they play **twice in total**; the track continues. "Cut" here means take, not delete. The agent says which reading it used. |
| "Now fade out the last 8 beats" | Beats N−8 … N, ending at the end of the file, tail included. Equal-power curve. |
| "Undo." | Undo the last chat turn. |
| "Remove bar 5" (after the repeat) | Bar 5 of the *current* grid, which is the first bar of the repeated section. |
| "Loop the first 4 bars four times, then let the track play on" | Bars 1–4 play 4 times in total, then the track continues. |
| "Repeat the last bar before the tempo change twice" | The last *full* bar of the first tempo segment, which code lists in the grid summary, played twice in total. |
| "Chop out the boring part" | Ask. The agent shows measured candidate sections and edits only after the user picks one. |
| "Echo before you cut" | Every applied edit is echoed by code in exact beats, bars and seconds, with undo available. An optional "confirm before applying" mode shows the echo first and applies only on the user's click. |
| "Cut beat 0 to beat 8" | Remove beats 0–7. The pre-roll before beat 0 stays. |
| "Delete bars 200–210" on A3 | Refused with the real bar range; never clamped. |
| "Sync A7 to A1, starting the overlap at A1's bar 33" | A1 is the master; A7 enters at its own bar 1 on A1's bar 33; 60 s overlap; both at −3 dB. |
| Ambient intro (A4) | Beats are annotated only where the drums play. The grid does not extend into a section with no pulse; it is trimmed and extended only where the onsets support it. |
| Half-time hip-hop (A2) | The reference is at the notated level (140 BPM). AMLt is reported too, to show whether a miss is a metrical-level error. |
| 6/8 (A6) | The beat is the dotted quarter (2 per bar). |
| "Edits keep the input's bit depth" | Untouched samples round-trip exactly (24-bit fits in float32). Edited samples are written at the same subtype. |
| Deadline | The email's (Friday 27 November, 11:59 PM IST) replaces the document's. |

## What I chose not to build

* Key detection, harmonic mixing, EQ and filter transitions in the sync render. The brief asks for an aligned overlap, not a finished DJ transition.
* A local open-model interpreter. The offline grammar covers running without a key; a model would add gigabytes and seconds per message for uncertain gains.
* Recovering hats masked by snares with a "pattern prior". That would be inferring notes I can't measure.
* LLM-written synth patches or code. Timbre comes from a few measured parameters.
* A public deployment.
