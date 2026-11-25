# Part 3 results: audio to playable Web Audio

Note F is onset-only, ±50 ms, via mir_eval.transcription against the reference MIDI (P4 has none). Render timing compares onsets detected in the page's offline render with onsets in the original clip.

| Clip | Tempo | Bars | Tracks (notes) | Drums note F | Drums any-class F | Pitched note F | All notes F | Score timing vs MIDI (median ms) | Render onset F | Render median offset ms | Net requests |
|---|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 96.00 | 2 | drums (24) | 0.8519 | 0.8889 | - | 0.8519 | 0.87 | 0.75 | 10.43 | 0 |
| P2 | 110.00 | 4 | drums (71), bass (23) | 0.8917 | 0.9045 | 0.9778 | 0.9109 | 1.13 | 0.9141 | 9.4 | 0 |
| P3 | 84.00 | 8 | keys (123) | - | - | 0.7783 | 0.7783 | 0.29 | 0.8318 | 9.36 | 0 |
| P4 | 124.00 | 8 | drums (133), bass (37), keys (200) | - | - | - | - | - | 0.9216 | 8.63 | 0 |

Score timing compares the transcribed note times with the reference MIDI (matched notes only), so it measures the analysis. Render timing compares onsets detected in the page's audio with onsets in the original, so it also includes the synth voices' attack shapes, which differ from the original sounds (the ~10 ms scatter there has a mean near 0).

Per-class drum results:

* P1: kick F 1.0 (ref 7, est 7); snare F 0.9231 (ref 7, est 6); closed hat F 0.7692 (ref 15, est 11); open hat F 0.0 (ref 1, est 0)
* P2: kick F 1.0 (ref 14, est 14); snare F 1.0 (ref 8, est 8); closed hat F 0.8571 (ref 64, est 48); open hat F 0.0 (ref 0, est 1)
