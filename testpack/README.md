# Test pack

The brief's pack (A1–A7, P1–P4) was not provided, so `generate.py` builds a **synthetic stand-in** with exact ground truth:

```bash
python -m testpack.generate          # writes testpack/stand-in/ (about 230 MB; deterministic, seed 7)
```

| ID | Stand-in | Format | Reference |
|---|---|---|---|
| A1 | House, 124 BPM, 5:59, breakdown with no kick (bars 65–80), first downbeat at 0.18 s | WAV 24-bit 44.1 kHz | beats + downbeats |
| A2 | Trap, notated 140 BPM, snare on beat 3 (half-time feel), 16th/32nd/triplet hi-hats, bars 1–4 with no kick | WAV 16-bit 48 kHz | beats + downbeats (notated level) |
| A3 | Live band, tempo drifting 92 → 97 BPM with wobble, 7–12 ms per-hit jitter, 2-beat pickup, fills | FLAC 16-bit | beats only |
| A4 | 45 s free-time ambient intro (no pulse), then 120 BPM | MP3 | beats + downbeats (from where the drums enter) |
| A5 | Techno, 120 BPM for 32 bars, then 128 BPM from bar 33 | WAV 16-bit | beats + downbeats |
| A6 | Folk-pop in 6/8 with dholak, guitar, harmonium, flute, 68 dotted-quarter BPM | FLAC 24-bit | beats (dotted quarters) + downbeats |
| A7 | House, 128 BPM, 3:33, sync partner for A1 | WAV 16-bit 48 kHz | beats + downbeats |
| P1 | 2-bar drum loop, 96 BPM, ghost notes, open hat | WAV | MIDI |
| P2 | 4-bar drums + bass, 110 BPM | WAV | MIDI |
| P3 | 8-bar piano melody with chords, 84 BPM | WAV | MIDI |
| P4 | 8-bar dense mix, 124 BPM | WAV | none |

## Using the real pack

Put the files in a folder and point the harness at it: `BEAT_AGENT_PACK=/path/to/pack make evals`.

* Audio: `<ID>.wav|.flac|.mp3`.
* Beats: `<ID>.beats` or `<ID>.beats.txt`, one beat per line, either `time` or `time<TAB>position` (position 1 = downbeat). Optionally `<ID>.downbeats(.txt)`.
* MIDI: `<ID>.mid`.

`evals/pack.py` is the only place that reads these; adapt it there if the real pack uses another layout.
