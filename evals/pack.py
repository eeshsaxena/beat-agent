"""Test-pack loader. Works for the synthetic stand-in and for the real pack.

Layout expected in BEAT_AGENT_PACK (default testpack/stand-in):
    <ID>.wav|.flac|.mp3        audio, e.g. A1.wav
    <ID>.beats[.txt]           one beat per line: "time" or "time<TAB>position" (1 = downbeat)
    <ID>.downbeats[.txt]       optional, one downbeat time per line
    <ID>.mid|.midi             reference MIDI for P clips
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PACK = Path(os.environ.get("BEAT_AGENT_PACK", ROOT / "testpack" / "stand-in"))
AUDIO_EXT = (".wav", ".flac", ".mp3", ".aiff", ".ogg")


def pack_dir() -> Path:
    return PACK


def ids(prefix: str = "") -> list[str]:
    found = sorted({p.stem for p in PACK.iterdir() if p.suffix.lower() in AUDIO_EXT and p.stem.startswith(prefix)})
    return found


def audio_path(cid: str) -> Path:
    for ext in AUDIO_EXT:
        p = PACK / f"{cid}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No audio for {cid} in {PACK}")


def _first(*names) -> Path | None:
    for n in names:
        p = PACK / n
        if p.exists():
            return p
    return None


def ref_beats(cid: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """(beat times, downbeat times or None)."""
    p = _first(f"{cid}.beats.txt", f"{cid}.beats", f"{cid}.txt")
    if p is None:
        return None, None
    rows = [line.split() for line in p.read_text().splitlines() if line.strip() and not line.startswith("#")]
    beats = np.array([float(r[0]) for r in rows])
    down = None
    if rows and len(rows[0]) >= 2:
        pos = np.array([float(r[1]) for r in rows])
        down = beats[pos == 1]
    dp = _first(f"{cid}.downbeats.txt", f"{cid}.downbeats")
    if dp is not None:
        down = np.array([float(x.split()[0]) for x in dp.read_text().splitlines() if x.strip()])
    return beats, down


def ref_midi(cid: str):
    p = _first(f"{cid}.mid", f"{cid}.midi")
    if p is None:
        return None
    import pretty_midi

    return pretty_midi.PrettyMIDI(str(p))
