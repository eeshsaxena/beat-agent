"""Score: the intermediate representation between analysis and playback.

Times are in beats from the loop start (first downbeat), so changing tempo is a
single number and the page loops at exactly `loop_beats`. Notes and onsets come
from measurement and transcription models; the LLM can only change the score
through the typed ScoreOps below, which code validates and applies.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

SCHEMA = "beat-agent/score@1"
TEMPLATE = Path(__file__).parent / "webaudio" / "template.html"

DEFAULT_PATCHES = {
    "drums": {"f_start": 140.0, "f_end": 50.0, "sweep": 0.06, "decay": 0.22, "click": 0.3, "tone": 185.0,
              "noise_hz": 3500.0, "hp_hz": 7000.0, "open_decay": 0.18},
    "bass": {"voice": "bass", "wave": "sawtooth", "cutoff_hz": 700.0, "resonance": 2.0, "env_amount": 1.5, "sub": 0.5,
             "attack": 0.004, "decay": 0.18, "sustain": 0.6, "release": 0.05},
    "keys": {"voice": "keys", "fm_ratio": 1.0, "fm_index": 1.2, "decay": 1.4, "release": 0.25},
    "lead": {"voice": "lead", "wave": "square", "cutoff_hz": 2800.0, "resonance": 1.0, "attack": 0.008, "decay": 0.12,
             "sustain": 0.7, "release": 0.08},
    "pad": {"voice": "pad", "cutoff_hz": 1500.0, "attack": 0.4, "decay": 0.8, "sustain": 0.85, "release": 1.0},
}


class ScoreError(ValueError):
    pass


def _num(v, lo: float, hi: float) -> bool:
    """A real, finite number in [lo, hi] (bools and NaN are not numbers here)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and lo <= v <= hi


def validate(score: dict) -> dict:
    """Check the score before it is embedded in a page (it is data, never code). Anything malformed, of
    any shape, raises ScoreError; every number the page reads must be finite and in range, since one
    NaN or infinite value would silence or blow up the audio graph."""
    if not isinstance(score, dict) or score.get("schema") != SCHEMA:
        raise ScoreError("Not a beat-agent score.")
    tb = score.get("tempo_bpm")
    if not _num(tb, 20, 400):
        raise ScoreError(f"Tempo {tb!r} is outside 20-400 BPM.")
    lb = score.get("loop_beats")
    if not _num(lb, 1e-3, 1024):
        raise ScoreError("loop_beats must be a positive number of beats (at most 1024).")
    if not _num(score.get("beats_per_bar", 4), 1, 32):
        raise ScoreError("beats_per_bar must be 1-32.")
    if not _num(score.get("master_gain_db", 0.0), -60, 12):
        raise ScoreError("master_gain_db must be between -60 and +12 dB.")
    if not isinstance(score.get("title", ""), str) or len(score.get("title", "")) > 80:
        raise ScoreError("The title must be a short string.")
    tracks = score.get("tracks", [])
    if not isinstance(tracks, list) or len(tracks) > 16:
        raise ScoreError("tracks must be a list of at most 16 tracks.")
    ids = set()
    for t in tracks:
        if not isinstance(t, dict) or not isinstance(t.get("id"), str) or not 0 < len(t["id"]) <= 24:
            raise ScoreError("Every track needs a short string id.")
        if t["id"] in ids:
            raise ScoreError(f"Duplicate track id {t['id']}.")
        ids.add(t["id"])
        if not isinstance(t.get("name", t["id"]), str) or len(t.get("name", t["id"])) > 40:
            raise ScoreError(f"Track {t['id']}: the name must be a short string.")
        if t.get("kind") not in ("drums", "pitched"):
            raise ScoreError(f"Unknown track kind {t.get('kind')!r}.")
        if not _num(t.get("gain_db", 0.0), -60, 12):
            raise ScoreError(f"Track {t['id']}: gain_db must be between -60 and +12 dB.")
        if not all(isinstance(t.get(k, False), bool) for k in ("mute", "solo")):
            raise ScoreError(f"Track {t['id']}: mute and solo are true or false.")
        notes = t.get("notes")
        if not isinstance(notes, list) or len(notes) > 5000:
            raise ScoreError(f"Track {t['id']}: notes must be a list of at most 5000 notes.")
        for n in notes:
            # types first: a string where a number belongs must be rejected, not crash the comparison
            if not isinstance(n, dict) or set(n) != {"start", "duration", "pitch", "velocity"} or \
                    not all(_num(v, -1e9, 1e9) for v in n.values()):
                raise ScoreError(f"Invalid note in track {t['id']}: notes have exactly start, duration, pitch, velocity "
                                 "as finite numbers.")
            if not (0 <= n["start"] < lb + 1e-6 and 0 < n["duration"] <= lb and 0 <= n["pitch"] <= 127
                    and 0 <= n["velocity"] <= 1):
                raise ScoreError(f"Invalid note in track {t['id']}: {n}")
        patch = t.get("patch")
        if not isinstance(patch, dict) or len(patch) > 32:
            raise ScoreError(f"Track {t['id']}: patch must be a small object.")
        for k, v in patch.items():
            if not isinstance(k, str) or len(k) > 24 or not (
                    _num(v, -1e6, 1e6) or (isinstance(v, str) and len(v) <= 24)):
                raise ScoreError("Patch values must be short names or finite numbers.")
    return score


def to_seconds(score: dict, track_ids=None) -> list[tuple[str, int, float, float, float]]:
    """(track, pitch, onset_s, offset_s, velocity) for one pass through the loop, at the score tempo,
    in the original clip's time (adds back the first-downbeat offset)."""
    spb = 60.0 / score["tempo_bpm"]
    t0 = score.get("source", {}).get("first_downbeat_s", 0.0)
    out = []
    for t in score["tracks"]:
        if track_ids and t["id"] not in track_ids:
            continue
        for n in t["notes"]:
            s = t0 + n["start"] * spb
            out.append((t["id"], int(n["pitch"]), s, s + n["duration"] * spb, float(n["velocity"])))
    return out


def build_page(score: dict, path: Path) -> Path:
    validate(score)
    data = json.dumps(score, separators=(",", ":"))
    # The score sits inside <script type="application/json">: escape "<" so no string can close the tag.
    data = data.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    title = (score.get("title") or "Recreation").replace("<", "").replace(">", "").replace("&", "and")
    html = TEMPLATE.read_text(encoding="utf-8").replace("__SCORE_JSON__", data).replace("__TITLE__", title)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Chat changes to the recreation: typed operations the LLM may emit
# ---------------------------------------------------------------------------
class SetMute(BaseModel):
    op: Literal["set_mute"] = "set_mute"
    track: str
    muted: bool


class SetSolo(BaseModel):
    op: Literal["set_solo"] = "set_solo"
    track: str
    solo: bool


class SetGain(BaseModel):
    op: Literal["set_gain"] = "set_gain"
    track: str
    db: float = Field(ge=-36, le=6, description="Track level in dB (absolute, 0 = as transcribed).")


class SetTempo(BaseModel):
    op: Literal["set_tempo"] = "set_tempo"
    bpm: float = Field(ge=30, le=300)


class Brightness(BaseModel):
    op: Literal["brightness"] = "brightness"
    track: str
    factor: float = Field(default=1.5, ge=0.25, le=4.0, description="Multiply the track's filter cutoff / brightness. "
                                                                    "'brighter' = 1.5, 'much brighter' = 2, 'darker' = 0.67.")


class Transpose(BaseModel):
    op: Literal["transpose"] = "transpose"
    track: str
    semitones: int = Field(ge=-24, le=24)


ScoreOp = Annotated[Union[SetMute, SetSolo, SetGain, SetTempo, Brightness, Transpose], Field(discriminator="op")]


def apply_score_ops(score: dict, ops: list) -> tuple[dict, list[str]]:
    s = copy.deepcopy(score)
    by_id = {t["id"]: t for t in s["tracks"]}
    names = ", ".join(f"{t['id']} ({t['name']})" for t in s["tracks"])
    echo = []
    for op in ops:
        if op.op != "set_tempo" and op.track not in by_id:
            raise ScoreError(f"There is no track '{op.track}'. Tracks: {names}.")
        if op.op == "set_mute":
            by_id[op.track]["mute"] = op.muted
            echo.append(f"{'MUTE' if op.muted else 'UNMUTE'} {by_id[op.track]['name']}.")
        elif op.op == "set_solo":
            by_id[op.track]["solo"] = op.solo
            echo.append(f"{'SOLO' if op.solo else 'UNSOLO'} {by_id[op.track]['name']}.")
        elif op.op == "set_gain":
            old = by_id[op.track].get("gain_db", 0.0)
            by_id[op.track]["gain_db"] = op.db
            echo.append(f"LEVEL {by_id[op.track]['name']}: {old:+.1f} -> {op.db:+.1f} dB.")
        elif op.op == "set_tempo":
            old = s["tempo_bpm"]
            s["tempo_bpm"] = op.bpm
            echo.append(f"TEMPO {old:.2f} -> {op.bpm:.2f} BPM (loop {s['loop_beats'] * 60 / old:.3f} -> "
                        f"{s['loop_beats'] * 60 / op.bpm:.3f} s; pitch unchanged).")
        elif op.op == "brightness":
            tr = by_id[op.track]
            p = tr["patch"]
            if tr["kind"] == "drums":
                old = (p["hp_hz"], p["noise_hz"])
                p["hp_hz"] = min(16000.0, p["hp_hz"] * op.factor)
                p["noise_hz"] = min(12000.0, p["noise_hz"] * op.factor)
                echo.append(f"BRIGHTNESS {tr['name']}: hat high-pass {old[0]:.0f} -> {p['hp_hz']:.0f} Hz, snare noise "
                            f"{old[1]:.0f} -> {p['noise_hz']:.0f} Hz.")
            elif "cutoff_hz" in p:
                old = p["cutoff_hz"]
                p["cutoff_hz"] = float(min(16000.0, max(60.0, old * op.factor)))
                echo.append(f"BRIGHTNESS {tr['name']}: filter cutoff {old:.0f} -> {p['cutoff_hz']:.0f} Hz.")
            else:
                old = p.get("fm_index", 1.0)
                p["fm_index"] = float(min(8.0, old * op.factor))
                echo.append(f"BRIGHTNESS {tr['name']}: FM index {old:.2f} -> {p['fm_index']:.2f}.")
        elif op.op == "transpose":
            tr = by_id[op.track]
            if tr["kind"] == "drums":
                raise ScoreError("Drums can't be transposed.")
            for n in tr["notes"]:
                n["pitch"] = int(min(127, max(0, n["pitch"] + op.semitones)))
            echo.append(f"TRANSPOSE {tr['name']} by {op.semitones:+d} semitones.")
    return validate(s), echo
