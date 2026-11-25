"""Synthetic stand-in for the Soundverse Beat Agent test pack.

The real test pack (A1-A7, P1-P4) was not shipped with the brief. This script
synthesises one clip per item that mirrors its description, with exact ground
truth: every drum hit and note is placed from a known tempo map, so reference
beats, downbeats and MIDI are correct by construction.

Honest limit: synthetic audio is cleaner than real recordings, so scores on this
pack are an optimistic upper bound for the real pack. The harness reads any pack
laid out the same way (see testpack/README.md), so the real one can be dropped in.

Usage:
    python -m testpack.generate                 # all clips -> testpack/stand-in/
    python -m testpack.generate --only A1,P2
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from scipy import signal

OUT_DIR = Path(__file__).parent / "stand-in"


# --------------------------------------------------------------------------- #
# Tempo map: musical position (beats, float) -> seconds
# --------------------------------------------------------------------------- #
class TempoMap:
    """Times of integer beats; fractional positions interpolate linearly."""

    def __init__(self, beat_times: np.ndarray):
        self.bt = np.asarray(beat_times, dtype=float)

    @classmethod
    def constant(cls, bpm: float, t0: float, n_beats: int) -> "TempoMap":
        return cls(t0 + np.arange(n_beats + 2) * 60.0 / bpm)

    @classmethod
    def piecewise(cls, segments: list[tuple[float, int]], t0: float) -> "TempoMap":
        times, t = [t0], t0
        for bpm, n in segments:
            for _ in range(n):
                t += 60.0 / bpm
                times.append(t)
        last_bpm = segments[-1][0]
        times.append(t + 60.0 / last_bpm)
        return cls(np.array(times))

    @classmethod
    def curve(cls, bpm_at, t0: float, n_beats: int) -> "TempoMap":
        times, t = [t0], t0
        for _ in range(n_beats + 1):
            t += 60.0 / bpm_at(t)
            times.append(t)
        return cls(np.array(times))

    def t(self, pos):
        pos = np.asarray(pos, dtype=float)
        i = np.clip(np.floor(pos).astype(int), 0, len(self.bt) - 2)
        return self.bt[i] + (pos - i) * (self.bt[i + 1] - self.bt[i])


# --------------------------------------------------------------------------- #
# DSP helpers
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=None)
def _sos(kind: str, freq, sr: int, order: int):
    return signal.butter(order, freq, btype=kind, fs=sr, output="sos")


def filt(x, kind, freq, sr, order=2):
    return signal.sosfilt(_sos(kind, freq, sr, order), x).astype(np.float32)


def mtof(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def ramp_in(w, sr, ms=0.5):
    a = max(1, int(ms * 1e-3 * sr))
    w[:a] *= np.linspace(0, 1, a, dtype=np.float32)
    return w


def adsr(n, sr, a, d, s, r, note_dur):
    t = np.arange(n) / sr
    env = np.where(t < a, t / max(a, 1e-5), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-4)))
    if note_dur < n / sr:
        k = int(note_dur * sr)
        lvl = env[k] if k < n else env[-1]
        env[k:] = lvl * np.exp(-(t[k:] - note_dur) / max(r, 1e-4))
    return env.astype(np.float32)


# --------------------------------------------------------------------------- #
# Drum voices (onset of every voice is exactly sample 0)
# --------------------------------------------------------------------------- #
def kick(sr, rng, f_hi=150.0, f_lo=48.0, tau=0.035, decay=0.30, click=0.25, drive=1.6, dur=0.6):
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f_lo + (f_hi - f_lo) * np.exp(-t / tau)
    body = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / decay)
    body = np.tanh(drive * body) / np.tanh(drive)
    c = np.zeros(n)
    nc = int(0.004 * sr)
    c[:nc] = rng.standard_normal(nc) * np.linspace(1, 0, nc)
    w = (body + click * filt(c, "highpass", 1500, sr)).astype(np.float32)
    return ramp_in(w, sr)


def snare(sr, rng, tone=185.0, decay=0.16, dur=0.45):
    n = int(dur * sr)
    t = np.arange(n) / sr
    tonal = np.sin(2 * np.pi * tone * t) * np.exp(-t / 0.05) + 0.5 * np.sin(2 * np.pi * 1.6 * tone * t) * np.exp(-t / 0.03)
    nz = filt(rng.standard_normal(n), "bandpass", (1200, 9000), sr) * np.exp(-t / decay)
    return ramp_in((0.55 * tonal + 0.9 * nz).astype(np.float32), sr)


def clap(sr, rng, dur=0.4):
    n = int(dur * sr)
    t = np.arange(n) / sr
    env = np.zeros(n)
    for off in (0.0, 0.010, 0.021, 0.031):
        m = t >= off
        env[m] += np.exp(-(t[m] - off) / 0.005)
    m = t >= 0.031
    env[m] += 0.6 * np.exp(-(t[m] - 0.031) / 0.12)
    nz = filt(rng.standard_normal(n), "bandpass", (900, 5000), sr)
    return ramp_in((nz * env * 0.9).astype(np.float32), sr)


def hat(sr, rng, decay=0.035, hp=7000.0, dur=None):
    dur = dur or max(0.08, decay * 6)
    n = int(dur * sr)
    t = np.arange(n) / sr
    metal = sum(np.sign(np.sin(2 * np.pi * 410 * r * t + rng.uniform(0, 6.28))) for r in (2, 3, 4.16, 5.43, 6.79, 8.21))
    x = 0.6 * rng.standard_normal(n) + 0.15 * metal
    x = filt(x, "highpass", hp, sr) * np.exp(-t / decay)
    return ramp_in(x.astype(np.float32), sr)


def tom(sr, rng, f=110.0, decay=0.25, dur=0.6):
    n = int(dur * sr)
    t = np.arange(n) / sr
    fr = f * (1 + 0.5 * np.exp(-t / 0.03))
    x = np.sin(2 * np.pi * np.cumsum(fr) / sr) * np.exp(-t / decay)
    x += 0.2 * filt(rng.standard_normal(n), "bandpass", (300, 3000), sr) * np.exp(-t / 0.02)
    return ramp_in(x.astype(np.float32), sr)


def crash(sr, rng, decay=1.3):
    return hat(sr, rng, decay=decay, hp=3500.0, dur=decay * 4) * 0.8


def shaker(sr, rng, decay=0.03):
    n = int(0.12 * sr)
    t = np.arange(n) / sr
    env = np.minimum(t / 0.004, 1.0) * np.exp(-t / decay)
    return (filt(rng.standard_normal(n), "bandpass", (5000, 12000), sr) * env).astype(np.float32)


def dholak_bass(sr, rng, f0=92.0, slide=0.22, decay=0.33, dur=0.8):
    """Bass head ('ge'/'dha' low part): tone with an upward pitch glide."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    f = f0 * (1 - slide * np.exp(-t / 0.07))
    x = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / decay)
    x += 0.35 * filt(rng.standard_normal(n), "lowpass", 900, sr) * np.exp(-t / 0.008)
    return ramp_in(x.astype(np.float32), sr)


def dholak_treble(sr, rng, f0=480.0, decay=0.11, dur=0.45):
    """Treble head ('na'/'ta'): ringing inharmonic partials plus a slap."""
    n = int(dur * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for r, a, dk in ((1.0, 1.0, 1.0), (1.58, 0.6, 0.8), (2.13, 0.45, 0.6), (2.65, 0.3, 0.5), (3.15, 0.2, 0.4)):
        x += a * np.sin(2 * np.pi * f0 * r * t + rng.uniform(0, 6.28)) * np.exp(-t / (decay * dk))
    x += 0.5 * filt(rng.standard_normal(n), "bandpass", (2000, 6000), sr) * np.exp(-t / 0.008)
    return ramp_in((0.5 * x).astype(np.float32), sr)


# --------------------------------------------------------------------------- #
# Pitched voices
# --------------------------------------------------------------------------- #
def _saw(f, n, sr, phase=0.0):
    ph = (phase + f * np.arange(n) / sr) % 1.0
    return 2 * ph - 1


def bass_synth(sr, rng, midi, dur, cutoff=700.0, kind="saw", vel=1.0):
    n = int((dur + 0.12) * sr)
    f = mtof(midi)
    t = np.arange(n) / sr
    if kind == "saw":
        osc = 0.7 * _saw(f, n, sr) + 0.5 * np.sin(2 * np.pi * f * t)
    elif kind == "square":
        osc = 0.6 * np.sign(np.sin(2 * np.pi * f * t)) + 0.5 * np.sin(2 * np.pi * f * t)
    else:  # round (live electric bass)
        osc = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.12 * _saw(f, n, sr)
    x = filt(osc, "lowpass", cutoff, sr) * adsr(n, sr, 0.003, 0.18, 0.55, 0.04, dur) * vel
    return ramp_in(x.astype(np.float32), sr)


def sub808(sr, rng, midi, dur, glide_from=None, vel=1.0):
    n = int((dur + 0.06) * sr)
    t = np.arange(n) / sr
    f_target = mtof(midi)
    f = np.full(n, f_target)
    if glide_from is not None:
        f = f_target + (mtof(glide_from) - f_target) * np.exp(-t / 0.05)
    x = np.sin(2 * np.pi * np.cumsum(f) / sr) * np.exp(-t / 0.9)
    x = np.tanh(2.2 * x) / np.tanh(2.2)
    x *= adsr(n, sr, 0.002, 10, 1.0, 0.03, dur)
    return ramp_in((x * vel).astype(np.float32), sr)


def pad(sr, rng, notes, dur, cutoff=1400.0, attack=0.6, release=1.2):
    n = int((dur + release * 3) * sr)
    x = np.zeros(n)
    for m in notes:
        for det in (-0.07, 0.0, 0.07):
            x += _saw(mtof(m + det), n, sr, rng.uniform())
    x = filt(x / (3 * len(notes)), "lowpass", cutoff, sr) * adsr(n, sr, attack, 1.0, 0.9, release, dur)
    return x.astype(np.float32)


def stab(sr, rng, notes, dur=0.18, cutoff=2500.0):
    n = int((dur + 0.1) * sr)
    x = sum(_saw(mtof(m), n, sr, rng.uniform()) for m in notes) / len(notes)
    x = filt(x, "lowpass", cutoff, sr) * adsr(n, sr, 0.002, 0.09, 0.0, 0.05, dur)
    return ramp_in(x.astype(np.float32), sr)


@lru_cache(maxsize=None)
def _ks_cached(sr, midi, dur, bright, variant):
    return _karplus(sr, np.random.default_rng(1000 * midi + variant), midi, dur, bright)


def _karplus(sr, rng, midi, dur, bright=0.5, decay=0.996):
    f = mtof(midi)
    N = max(2, int(round(sr / f - 0.5)))
    total = int(dur * sr)
    out = np.zeros(total + N)
    init = rng.uniform(-1, 1, N)
    out[:N] = filt(init, "lowpass", 1500 + 7000 * bright, sr)
    s = N
    while s < total:
        e = min(s + N, total)
        cur = out[s - N:e - N]
        prev = np.concatenate([[out[s - N - 1] if s - N - 1 >= 0 else 0.0], out[s - N:e - N - 1]])
        out[s:e] = decay * 0.5 * (cur + prev)
        s = e
    out = out[:total]
    rel = int(0.08 * sr)
    out[-rel:] *= np.linspace(1, 0, rel)
    return ramp_in(out.astype(np.float32), sr, 1.0)


def pluck(sr, rng, midi, dur, bright=0.5):
    return _ks_cached(sr, int(midi), round(dur, 2), round(bright, 2), int(rng.integers(3)))


def piano(sr, rng, midi, dur, vel=0.7):
    f0 = mtof(midi)
    n = int((dur + 1.0) * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    B = 0.0004
    for k in range(1, 17):
        fk = k * f0 * np.sqrt(1 + B * k * k)
        if fk > 9000:
            break
        amp = (vel ** (0.5 + 0.12 * k)) / k ** 1.1
        tau = 1.6 * np.sqrt(261.6 / f0) / (1 + 0.45 * (k - 1))
        x += amp * np.sin(2 * np.pi * fk * t + rng.uniform(0, 6.28)) * np.exp(-t / tau)
    k = int(dur * sr)
    x[k:] *= np.exp(-(t[k:] - dur) / 0.09)
    h = int(0.003 * sr)
    x[:h] += 0.3 * vel * filt(rng.standard_normal(h), "lowpass", 2500, sr)
    return ramp_in((0.35 * x).astype(np.float32), sr)


def lead(sr, rng, midi, dur, vel=0.8, cutoff=3200.0):
    n = int((dur + 0.15) * sr)
    t = np.arange(n) / sr
    vib = 1 + 0.004 * np.sin(2 * np.pi * 5.5 * t) * np.clip((t - 0.15) / 0.2, 0, 1)
    ph = np.cumsum(mtof(midi) * vib) / sr
    x = 0.5 * np.sign(np.sin(2 * np.pi * ph)) + 0.4 * (2 * (ph % 1) - 1)
    x = filt(x, "lowpass", cutoff, sr) * adsr(n, sr, 0.008, 0.12, 0.7, 0.08, dur) * vel
    return ramp_in(x.astype(np.float32), sr)


def bell(sr, rng, midi, dur, vel=0.7):
    n = int((dur + 1.5) * sr)
    t = np.arange(n) / sr
    f = mtof(midi)
    index = 3.0 * np.exp(-t / 0.4)
    x = np.sin(2 * np.pi * f * t + index * np.sin(2 * np.pi * 3.5 * f * t)) * np.exp(-t / 1.1)
    return ramp_in((0.5 * vel * x).astype(np.float32), sr)


def flute(sr, rng, midi, dur, vel=0.6):
    n = int((dur + 0.2) * sr)
    t = np.arange(n) / sr
    vib = 1 + 0.006 * np.sin(2 * np.pi * 5.0 * t)
    ph = 2 * np.pi * np.cumsum(mtof(midi) * vib) / sr
    x = np.sin(ph) + 0.2 * np.sin(2 * ph) + 0.05 * filt(rng.standard_normal(n), "bandpass", (1500, 5000), sr)
    return (x * adsr(n, sr, 0.06, 0.2, 0.8, 0.12, dur) * vel * 0.5).astype(np.float32)


def harmonium(sr, rng, notes, dur):
    n = int((dur + 0.4) * sr)
    t = np.arange(n) / sr
    x = np.zeros(n)
    for m in notes:
        ph = (mtof(m) * t + rng.uniform()) % 1.0
        x += np.where(ph < 0.3, 1.0, -0.43)
    x = filt(x / len(notes), "lowpass", 2000, sr) * adsr(n, sr, 0.08, 0.3, 0.85, 0.15, dur)
    return x.astype(np.float32)


def noise_riser(sr, rng, dur):
    n = int(dur * sr)
    x = rng.standard_normal(n)
    out = np.zeros(n)
    blk = int(0.05 * sr)
    for i in range(0, n, blk):
        fc = 400 + 9000 * (i / n) ** 2
        out[i:i + blk] = filt(x[i:i + blk], "bandpass", (fc, min(fc * 2.5, sr / 2 - 100)), sr)
    return (out * np.linspace(0, 1, n) ** 2 * 0.5).astype(np.float32)


# --------------------------------------------------------------------------- #
# Mixing
# --------------------------------------------------------------------------- #
class Bus:
    def __init__(self, n: int, sr: int, wrap: bool):
        self.x = np.zeros((n, 2), dtype=np.float32)
        self.sr, self.wrap = sr, wrap

    def add(self, t: float, wave: np.ndarray, gain: float = 1.0, pan: float = 0.0):
        n = len(self.x)
        i = int(round(t * self.sr))
        w = wave * gain
        lr = np.array([np.cos((pan + 1) * np.pi / 4), np.sin((pan + 1) * np.pi / 4)], dtype=np.float32) * np.sqrt(2)
        if self.wrap:
            idx = (i + np.arange(len(w))) % n
            np.add.at(self.x, idx, w[:, None] * lr[None, :])
            return
        if i < 0:
            w, i = w[-i:], 0
        if i >= n:
            return
        e = min(n, i + len(w))
        self.x[i:e] += w[: e - i, None] * lr[None, :]


def reverb(x, sr, rng, t60=1.2, wet=0.2, predelay=0.012):
    nir = int(t60 * sr)
    t = np.arange(nir) / sr
    ir = rng.standard_normal((nir, 2)) * np.exp(-6.9 * t / t60)[:, None]
    ir = signal.sosfilt(_sos("lowpass", 6000, sr, 2), ir, axis=0)
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True))
    ir = np.concatenate([np.zeros((int(predelay * sr), 2)), ir])
    w = np.stack([signal.oaconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1)
    return ((1 - wet) * x + wet * w).astype(np.float32)


def sidechain(x, sr, kick_times, depth=0.6, release=0.14):
    g = np.ones(len(x), dtype=np.float32)
    L = int(release * 5 * sr)
    env = 1 - depth * np.exp(-np.arange(L) / (release * sr))
    for kt in kick_times:
        i = int(round(kt * sr))
        if i >= len(x):
            continue
        e = min(len(x), i + L)
        g[i:e] = np.minimum(g[i:e], env[: e - i])
    return x * g[:, None]


def master(x, sr, peak_db=-1.0):
    x = signal.sosfilt(_sos("highpass", 25, sr, 2), x, axis=0)
    x = np.tanh(1.1 * x / max(np.abs(x).max(), 1e-9) * 1.4)
    return (x / np.abs(x).max() * 10 ** (peak_db / 20)).astype(np.float32)


@dataclass
class Clip:
    id: str
    description: str
    sr: int
    duration: float
    wrap: bool = False
    buses: dict = field(default_factory=dict)
    beats: list = field(default_factory=list)  # (time, position-in-bar or None)
    midi_notes: list = field(default_factory=list)  # (track, is_drum, program, pitch, start, end, vel)
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.n = int(round(self.duration * self.sr))

    def bus(self, name):
        if name not in self.buses:
            self.buses[name] = Bus(self.n, self.sr, self.wrap)
        return self.buses[name]

    def note(self, track, pitch, start, dur, vel, is_drum=False, program=0):
        self.midi_notes.append((track, is_drum, program, int(pitch), float(start), float(start + dur), float(vel)))

    def mix(self, rng, fx=None, gains=None):
        fx, gains = fx or {}, gains or {}
        out = np.zeros((self.n, 2), dtype=np.float32)
        for name, b in self.buses.items():
            y = b.x
            for f in fx.get(name, []):
                y = f(y)
            out += y * gains.get(name, 1.0)
        return master(out, self.sr)


GM = {"kick": 36, "snare": 38, "clap": 39, "hat": 42, "ohat": 46, "crash": 49, "tom_lo": 45, "tom_hi": 48, "shaker": 70}


def variants(make, k=4):
    return [make() for _ in range(k)]


# --------------------------------------------------------------------------- #
# Clips
# --------------------------------------------------------------------------- #
def make_house(cid, desc, rng, sr, bpm, t0, bars, key_roots, chords, kick_kw, breakdown, intro_bars,
               outro_from, bass_style="offbeat"):
    tm = TempoMap.constant(bpm, t0, bars * 4 + 4)
    c = Clip(cid, desc, sr, float(tm.t(bars * 4)) + 2.5)
    K = kick(sr, rng, **kick_kw)
    H = variants(lambda: hat(sr, rng, 0.03))
    OH = hat(sr, rng, 0.16)
    CL = variants(lambda: clap(sr, rng), 3)
    kicks = []
    for bar in range(1, bars + 1):
        in_break = breakdown[0] <= bar <= breakdown[1]
        full = intro_bars < bar < outro_from and not in_break
        chord_i = ((bar - 1) // 2) % len(chords)
        for b in range(4):
            pos = (bar - 1) * 4 + b
            t = float(tm.t(pos))
            c.beats.append((t, b + 1))
            if not in_break:
                c.bus("kick").add(t, K, 0.95)
                kicks.append(t)
            if not in_break:
                for k, v in enumerate((0.45, 0.22, 0.75, 0.25)):
                    c.bus("hats").add(float(tm.t(pos + k / 4)), H[rng.integers(4)], v * rng.uniform(0.85, 1.0), 0.25)
            if full and b in (1, 3):
                c.bus("clap").add(t, CL[rng.integers(3)], 0.7, -0.1)
            if full and bar > intro_bars + 16:
                c.bus("hats").add(float(tm.t(pos + 0.5)), OH, 0.35, -0.2)
            if full:
                root = key_roots[chord_i]
                if bass_style == "offbeat":
                    c.bus("bass").add(float(tm.t(pos + 0.5)), bass_synth(sr, rng, root, 0.22, 650), 0.8)
                else:  # rolling 16ths, skip the downbeat 16th (kick)
                    for k in (1, 2, 3):
                        m = root + (12 if k == 2 else 0)
                        c.bus("bass").add(float(tm.t(pos + k / 4)), bass_synth(sr, rng, m, 0.1, 800), 0.65)
            if full and bar > intro_bars + 16 and b in (1, 3):
                c.bus("stab").add(float(tm.t(pos + 0.5)), stab(sr, rng, chords[chord_i]), 0.45, 0.15)
        if in_break and (bar - breakdown[0]) % 2 == 0:
            c.bus("pad").add(float(tm.t((bar - 1) * 4)), pad(sr, rng, chords[chord_i], 2 * 4 * 60 / bpm), 0.8)
        if in_break and bar >= breakdown[1] - 3 and bar == breakdown[1] - 3:
            c.bus("fx").add(float(tm.t((bar - 1) * 4)), noise_riser(sr, rng, 4 * 4 * 60 / bpm), 0.6)
        if in_break or (breakdown[1] + 32 < bar <= breakdown[1] + 64):
            scale = [0, 3, 5, 7, 10, 12]
            for k in range(0, 8, 2):
                if rng.uniform() < 0.7:
                    m = key_roots[chord_i] + 36 + scale[rng.integers(len(scale))]
                    c.bus("lead").add(float(tm.t((bar - 1) * 4 + k / 2)), lead(sr, rng, m, 0.4 * 60 / bpm * 2, 0.5), 0.35, 0.3)
    fx = {
        "clap": [lambda y: reverb(y, sr, rng, 1.0, 0.2)],
        "pad": [lambda y: sidechain(y, sr, kicks), lambda y: reverb(y, sr, rng, 2.5, 0.35)],
        "lead": [lambda y: reverb(y, sr, rng, 2.0, 0.3)],
        "bass": [lambda y: sidechain(y, sr, kicks, 0.5)],
        "stab": [lambda y: sidechain(y, sr, kicks, 0.5), lambda y: reverb(y, sr, rng, 1.2, 0.25)],
    }
    c.meta.update(tempo_bpm=bpm, first_downbeat_s=t0, bars=bars, grid="constant", time_signature="4/4",
                  sections={"intro_kick_hats_only": [1, intro_bars], "breakdown_no_kick": list(breakdown),
                            "outro_kick_hats_only": [outro_from, bars]})
    return c, rng, fx


def make_A1(rng):
    chords = [[57, 60, 64, 67], [53, 57, 60, 64], [55, 60, 64, 67], [55, 59, 62, 67]]
    c, rng, fx = make_house("A1", "Four-on-the-floor house, constant 124 BPM, about 6 minutes", rng, 44100, 124.0,
                            0.180, 184, [33, 29, 36, 31], chords, dict(f_hi=150, f_lo=48, decay=0.30),
                            breakdown=(65, 80), intro_bars=16, outro_from=169)
    c.meta["format"] = ("WAV", "PCM_24")
    return c, c.mix(rng, fx)


def make_A7(rng):
    chords = [[53, 56, 60, 63], [56, 60, 63, 67], [58, 61, 65, 68], [51, 55, 58, 63]]
    c, rng, fx = make_house("A7", "House at 128 BPM (sync partner for A1), 48 kHz", rng, 48000, 128.0, 0.520, 112,
                            [29, 32, 34, 27], chords, dict(f_hi=180, f_lo=55, decay=0.22, click=0.35),
                            breakdown=(49, 64), intro_bars=16, outro_from=105, bass_style="rolling")
    c.meta["format"] = ("WAV", "PCM_16")
    return c, c.mix(rng, fx)


def make_A2(rng):
    """Trap: notated 140 BPM, snare on beat 3 (half-time feel), 16th/32nd/triplet hats."""
    sr, bpm, t0, bars = 48000, 140.0, 0.040, 72
    tm = TempoMap.constant(bpm, t0, bars * 4 + 4)
    c = Clip("A2", "Hip-hop, half-time feel (snare on beat 3) with fast hi-hats; reference at notated 140 BPM",
             sr, float(tm.t(bars * 4)) + 3.0)
    K = kick(sr, rng, f_hi=120, f_lo=45, decay=0.18, click=0.4)
    S = snare(sr, rng, 200, 0.2)
    CL = clap(sr, rng)
    H = variants(lambda: hat(sr, rng, 0.02, 8000))
    OH = hat(sr, rng, 0.12)
    kick_pat = [[0, 0.75, 2.5, 3.25], [0, 1.5, 2.5, 3.75]]
    roots = [36, 36, 32, 34]  # C2, C2, Ab1, Bb1
    melody = [(0, 72), (0.5, 75), (1, 79), (1.75, 77), (2.5, 75), (3, 74), (4, 72), (4.5, 75), (5, 80), (6, 79), (7, 75)]
    kicks = []
    for bar in range(1, bars + 1):
        intro = bar <= 4
        brk = 41 <= bar <= 44
        base = (bar - 1) * 4
        for b in range(4):
            c.beats.append((float(tm.t(base + b)), b + 1))
        if (bar - 1) % 2 == 0:
            for p, m in melody:
                c.bus("bell").add(float(tm.t(base + p)), bell(sr, rng, m, 0.4), 0.5, 0.2)
        # hats: 16ths, with a 32nd or triplet roll on the last beat of every 2nd bar
        for b in range(4):
            if b == 3 and bar % 2 == 0:
                div = 8 if bar % 4 == 0 else 6
            else:
                div = 4
            for k in range(div):
                v = (0.7 if k == 0 else 0.45) * rng.uniform(0.8, 1.0)
                c.bus("hats").add(float(tm.t(base + b + k / div)), H[rng.integers(4)], v, 0.3)
        if intro or brk:
            continue
        pat = kick_pat[(bar - 1) % 2]
        for i, p in enumerate(pat):
            t = float(tm.t(base + p))
            c.bus("kick").add(t, K, 0.9)
            kicks.append(t)
            nxt = pat[i + 1] if i + 1 < len(pat) else 4
            glide = roots[(bar - 1) % 4] + 12 if (bar % 8 == 0 and i == len(pat) - 1) else None
            c.bus("808").add(t, sub808(sr, rng, roots[(bar - 1) % 4], (nxt - p) * 60 / bpm * 0.95, glide), 0.8)
        t3 = float(tm.t(base + 2))
        c.bus("snare").add(t3, S, 0.8)
        c.bus("snare").add(t3, CL, 0.5)
        if bar % 4 == 2:
            c.bus("hats").add(float(tm.t(base + 3.5)), OH, 0.35)
    fx = {"bell": [lambda y: reverb(y, sr, rng, 2.2, 0.35)], "snare": [lambda y: reverb(y, sr, rng, 1.0, 0.18)]}
    c.meta.update(tempo_bpm=bpm, felt_tempo_bpm=bpm / 2, first_downbeat_s=t0, bars=bars, grid="constant",
                  time_signature="4/4", format=("WAV", "PCM_16"),
                  note="Bars 1-4 have bell, hats only (no kick or snare). Reference is at the notated 140 BPM level.")
    return c, c.mix(rng, fx, {"808": 0.9})


def make_A3(rng):
    """Live band, no click: tempo drifts 92 -> 97 BPM with wobble, per-hit timing jitter, 2-beat pickup."""
    sr, bars = 44100, 64
    walk = np.cumsum(rng.normal(0, 0.15, 4000))

    def bpm_at(t):
        return 92 + 5 * min(t / 165, 1.0) + 1.4 * np.sin(2 * np.pi * t / 23) + walk[int(t * 10) % 4000] * 0.15

    pickup = 2
    tm = TempoMap.curve(bpm_at, 0.35, pickup + bars * 4 + 4)
    c = Clip("A3", "Live band (drums, bass, guitar), no click: drifting tempo, human timing, 2-beat pickup",
             sr, float(tm.t(pickup + bars * 4)) + 4.0)
    Ks = variants(lambda: kick(sr, rng, 110, 55, 0.05, 0.25, 0.15, 1.2))
    Ss = variants(lambda: snare(sr, rng, 175, 0.18))
    Hs = variants(lambda: hat(sr, rng, 0.045, 6500))
    CR = crash(sr, rng)
    T = [tom(sr, rng, f) for f in (190, 150, 115, 90)]
    chords = {"G": [43, [55, 59, 62, 67, 71]], "D": [38, [50, 57, 62, 66]], "Em": [40, [52, 55, 59, 64, 67]],
              "C": [36, [48, 52, 55, 60, 64]]}
    prog = ["G", "D", "Em", "C"]

    def hum(t, sd=0.007):
        return t + rng.normal(0, sd)

    # pickup fill: snare 8ths on beats 3-4 of bar 0
    for k in range(4):
        c.bus("drums").add(hum(float(tm.t(k / 2))), Ss[rng.integers(4)], 0.5 + 0.12 * k)
    for p in range(pickup):
        c.beats.append((float(tm.t(p)), None))
    for bar in range(1, bars + 1):
        base = pickup + (bar - 1) * 4
        root, voicing = chords[prog[(bar - 1) % 4]]
        fill = bar % 8 == 0
        if (bar - 1) % 8 == 0:
            c.bus("drums").add(hum(float(tm.t(base))), CR, 0.6, -0.3)
        for b in range(4):
            c.beats.append((float(tm.t(base + b)), None))
            if b in (0, 2):
                c.bus("drums").add(hum(float(tm.t(base + b))), Ks[rng.integers(4)], rng.uniform(0.8, 0.95))
            if b == 2 and bar % 2 == 0:
                c.bus("drums").add(hum(float(tm.t(base + 2.5))), Ks[rng.integers(4)], 0.7)
            if b in (1, 3) and not (fill and b == 3):
                c.bus("drums").add(hum(float(tm.t(base + b))), Ss[rng.integers(4)], rng.uniform(0.8, 0.95))
            if not (fill and b >= 2):
                for k in (0, 0.5):
                    c.bus("drums").add(hum(float(tm.t(base + b + k)), 0.009), Hs[rng.integers(4)],
                                       (0.6 if k == 0 else 0.35) * rng.uniform(0.85, 1.0), 0.3)
            for k in (0, 0.5):
                m = root + (7 if (b == 3 and k == 0.5) else 0)
                c.bus("bass").add(hum(float(tm.t(base + b + k)), 0.010), bass_synth(sr, rng, m, 0.27 * 60 / 94, 500, "round"), 0.75)
        if fill:
            for i, k in enumerate(np.arange(2, 4, 0.25)):
                c.bus("drums").add(hum(float(tm.t(base + k))), T[min(i // 2, 3)], 0.7)
        for p in (0, 1.5, 2.5, 3.5):
            t = hum(float(tm.t(base + p)), 0.012)
            for j, m in enumerate(voicing):
                c.bus("guitar").add(t + j * 0.006, pluck(sr, rng, m, 1.2, 0.45), 0.22, 0.4)
    # final hit
    tf = float(tm.t(pickup + bars * 4))
    c.beats.append((tf, None))
    c.bus("drums").add(tf, Ks[0], 0.9)
    c.bus("drums").add(tf, CR, 0.7)
    for j, m in enumerate(chords["G"][1]):
        c.bus("guitar").add(tf + j * 0.008, pluck(sr, rng, m, 3.5, 0.5), 0.25, 0.4)
    fx = {name: [lambda y: reverb(y, sr, rng, 0.8, 0.18)] for name in ("drums", "guitar", "bass")}
    c.meta.update(tempo_bpm_range=[90, 99], grid="variable", time_signature="4/4", pickup_beats=pickup,
                  bars=bars, format=("FLAC", "PCM_16"),
                  note="Beats-only reference, like the brief. Timing jitter 7-12 ms per hit.")
    return c, c.mix(rng, fx)


def make_A4(rng, fmt):
    """Ambient free-time intro (~45 s, no drums, no pulse), then a 120 BPM beat."""
    sr, bpm, t_entry, bars = 44100, 120.0, 45.37, 48
    tm = TempoMap.constant(bpm, t_entry, bars * 4 + 4)
    c = Clip("A4", "Ambient intro of ~45 s with no drums or pulse, then a full 120 BPM beat", sr,
             float(tm.t(bars * 4)) + 3.0)
    chords = [[50, 57, 61, 66], [47, 54, 57, 62], [43, 50, 54, 59], [52, 59, 62, 66, 69]]
    starts = [0.0, 6.8, 14.9, 21.3, 29.6, 37.2]
    for i, s in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else t_entry
        c.bus("pad").add(s, pad(sr, rng, chords[i % 4], end - s + 0.8, 1100, 2.0, 2.5), 0.9)
    t = 1.0
    while t < t_entry - 1:
        m = 62 + [0, 2, 4, 7, 9, 12, 14][rng.integers(7)] + 12
        c.bus("shimmer").add(t, bell(sr, rng, m, 0.3, 0.4), 0.3, rng.uniform(-0.8, 0.8))
        t += rng.exponential(1.6)
    K = kick(sr, rng, 150, 50, decay=0.28)
    H = variants(lambda: hat(sr, rng, 0.03))
    CL = clap(sr, rng)
    kicks = []
    for bar in range(1, bars + 1):
        base = (bar - 1) * 4
        ci = ((bar - 1) // 2) % 4
        if (bar - 1) % 2 == 0:
            c.bus("pad").add(float(tm.t(base)), pad(sr, rng, chords[ci], 8 * 60 / bpm, 1300), 0.6)
        for b in range(4):
            tb = float(tm.t(base + b))
            c.beats.append((tb, b + 1))
            c.bus("kick").add(tb, K, 0.95)
            kicks.append(tb)
            c.bus("hats").add(float(tm.t(base + b + 0.5)), H[rng.integers(4)], 0.5, 0.2)
            if bar > 8 and b in (1, 3):
                c.bus("clap").add(tb, CL, 0.6)
            if bar > 8:
                c.bus("bass").add(float(tm.t(base + b + 0.5)), bass_synth(sr, rng, chords[ci][0] - 12, 0.2, 600), 0.75)
    fx = {"pad": [lambda y: sidechain(y, sr, kicks, 0.4), lambda y: reverb(y, sr, rng, 3.0, 0.4)],
          "shimmer": [lambda y: reverb(y, sr, rng, 3.5, 0.5)], "clap": [lambda y: reverb(y, sr, rng, 1.0, 0.2)]}
    c.meta.update(tempo_bpm=bpm, beat_entry_s=t_entry, bars=bars, grid="constant", time_signature="4/4",
                  format=fmt, note="Reference beats start where the drums enter; the intro has no annotated beats.")
    return c, c.mix(rng, fx)


def make_A5(rng):
    """Techno: 120 BPM for 32 bars, then an abrupt change to 128 BPM at bar 33."""
    sr, t0 = 44100, 0.100
    segs = [(120.0, 32 * 4), (128.0, 40 * 4)]
    bars = 72
    tm = TempoMap.piecewise(segs, t0)
    c = Clip("A5", "Electronic, 120 BPM for 32 bars then 128 BPM from bar 33 (abrupt change on the downbeat)",
             sr, float(tm.t(bars * 4)) + 2.5)
    K = kick(sr, rng, 170, 52, decay=0.26, click=0.3)
    H = variants(lambda: hat(sr, rng, 0.025))
    OH = hat(sr, rng, 0.12)
    CL = clap(sr, rng)
    CR = crash(sr, rng)
    seq = [45, 45, 57, 45, 48, 45, 55, 43]
    kicks = []
    for bar in range(1, bars + 1):
        base = (bar - 1) * 4
        if bar == 33:
            c.bus("perc").add(float(tm.t(base)), CR, 0.6)
        for b in range(4):
            tb = float(tm.t(base + b))
            c.beats.append((tb, b + 1))
            c.bus("kick").add(tb, K, 0.95)
            kicks.append(tb)
            c.bus("perc").add(float(tm.t(base + b + 0.5)), OH, 0.4, -0.2)
            for k in range(4):
                c.bus("perc").add(float(tm.t(base + b + k / 4)), H[rng.integers(4)], 0.25, 0.3)
            if bar > 8 and b in (1, 3):
                c.bus("perc").add(tb, CL, 0.55)
            if bar > 4:
                for k in range(4):
                    m = seq[(b * 4 + k) % 8] + (12 if bar > 32 else 0)
                    c.bus("acid").add(float(tm.t(base + b + k / 4)), bass_synth(sr, rng, m, 0.08, 1400, "square"), 0.35, 0.1)
    fx = {"acid": [lambda y: sidechain(y, sr, kicks, 0.5), lambda y: reverb(y, sr, rng, 0.9, 0.2)]}
    c.meta.update(tempo_bpm=[120.0, 128.0], tempo_change_at_bar=33, tempo_change_at_s=float(tm.t(32 * 4)),
                  first_downbeat_s=t0, bars=bars, grid="piecewise", time_signature="4/4", format=("WAV", "PCM_16"))
    return c, c.mix(rng, fx)


def make_A6(rng):
    """Folk-pop in 6/8 with dholak; beats annotated at the dotted-quarter level (2 per bar)."""
    sr, bpm, t0, bars = 44100, 68.0, 0.25, 64  # bpm = dotted quarters per minute
    tm = TempoMap.constant(bpm, t0, bars * 2 + 2)
    c = Clip("A6", "Folk-pop in 6/8 with dholak, guitar, harmonium and flute; 68 dotted-quarter BPM", sr,
             float(tm.t(bars * 2)) + 3.0)
    DB = variants(lambda: dholak_bass(sr, rng, 92))
    DT = variants(lambda: dholak_treble(sr, rng, 480))
    DL = variants(lambda: dholak_treble(sr, rng, 520, 0.06))
    SH = shaker(sr, rng)
    chords = [[50, 57, 62, 66], [55, 59, 62, 67], [57, 61, 64, 69], [50, 57, 62, 66]]
    mel = [74, 76, 78, 76, 74, 71, 73, 74, 76, 74, 73, 69]

    def hum(t, sd=0.006):
        return t + rng.normal(0, sd)

    for bar in range(1, bars + 1):
        base = (bar - 1) * 2
        ch = chords[(bar - 1) % 4]
        for b in range(2):
            c.beats.append((float(tm.t(base + b)), b + 1))
        for e in range(6):
            te = hum(float(tm.t(base + e / 3)))
            if e in (0, 3):
                c.bus("dholak").add(te, DB[rng.integers(4)], 0.9 if e == 0 else 0.7)
                c.bus("dholak").add(te, DT[rng.integers(4)], 0.6)
            elif e in (2, 5):
                c.bus("dholak").add(te, DT[rng.integers(4)], 0.55)
            else:
                c.bus("dholak").add(te, DL[rng.integers(4)], 0.3)
            c.bus("perc").add(hum(float(tm.t(base + e / 3)), 0.008), SH, 0.25 if e % 3 else 0.4, 0.4)
            if e in (0, 2, 3, 5):
                order = ch if e in (0, 3) else ch[::-1]
                for j, m in enumerate(order):
                    c.bus("guitar").add(te + j * 0.007, pluck(sr, rng, m, 0.9, 0.5), 0.2 if e in (0, 3) else 0.12, -0.35)
        if bar > 8:
            c.bus("harmonium").add(float(tm.t(base)), harmonium(sr, rng, ch, 2 * 60 / bpm), 0.35, 0.2)
        if 17 <= bar <= 32 or 49 <= bar <= 64:
            for e in range(0, 6, 2):
                m = mel[((bar - 1) * 3 + e // 2) % len(mel)]
                c.bus("flute").add(float(tm.t(base + e / 3)), flute(sr, rng, m, 2 * 60 / bpm / 3 * 1.8), 0.4, 0.25)
    fx = {"flute": [lambda y: reverb(y, sr, rng, 1.8, 0.3)], "guitar": [lambda y: reverb(y, sr, rng, 1.0, 0.15)],
          "dholak": [lambda y: reverb(y, sr, rng, 0.7, 0.12)]}
    c.meta.update(tempo_bpm=bpm, tempo_note="dotted-quarter BPM (eighth notes at 204/min)", bars=bars,
                  first_downbeat_s=t0, grid="constant", time_signature="6/8", beats_per_bar=2,
                  format=("FLAC", "PCM_24"))
    return c, c.mix(rng, fx)


# ----------------------------- Part 3 clips ---------------------------------- #
def _drum_hit(c, tm, name, voice, pos, vel, rng, sd=0.0, pan=0.0, bus="drums"):
    t = float(tm.t(pos)) + (rng.normal(0, sd) if sd else 0.0)
    t = max(t, 0.0)
    c.bus(bus).add(t, voice, vel, pan)
    c.note("drums", GM[name], t, 0.1, vel, is_drum=True)


def make_P1(rng):
    sr, bpm, bars = 44100, 96.0, 2
    tm = TempoMap.constant(bpm, 0.0, bars * 4 + 2)
    c = Clip("P1", "2-bar drum loop, 96 BPM (kick, snare with ghosts, hats, open hat)", sr, bars * 4 * 60 / bpm, wrap=True)
    K, S, H, OH = kick(sr, rng, 140, 50, decay=0.25), snare(sr, rng), hat(sr, rng, 0.03), hat(sr, rng, 0.2)
    pats = [dict(kick=[0, 0.75, 2.5], snare=[1, 3], ghost=[1.75, 3.25]),
            dict(kick=[0, 1.5, 2.0, 2.75], snare=[1, 3], ghost=[3.75])]
    for bar, p in enumerate(pats):
        base = bar * 4
        for x in p["kick"]:
            _drum_hit(c, tm, "kick", K, base + x, 0.9, rng, 0.003)
        for x in p["snare"]:
            _drum_hit(c, tm, "snare", S, base + x, 0.85, rng, 0.003)
        for x in p["ghost"]:
            _drum_hit(c, tm, "snare", S, base + x, 0.28, rng, 0.003)
        for k in range(8):
            if bar == 1 and k == 7:
                _drum_hit(c, tm, "ohat", OH, base + k / 2, 0.6, rng, 0.003, 0.3)
            else:
                _drum_hit(c, tm, "hat", H, base + k / 2, 0.7 if k % 2 == 0 else 0.45, rng, 0.003, 0.3)
    for i in range(bars * 4):
        c.beats.append((float(tm.t(i)), i % 4 + 1))
    c.meta.update(tempo_bpm=bpm, bars=bars, time_signature="4/4", format=("WAV", "PCM_16"))
    return c, c.mix(rng, {"drums": [lambda y: reverb(y, sr, rng, 0.6, 0.1)]})


def make_P2(rng):
    sr, bpm, bars = 44100, 110.0, 4
    tm = TempoMap.constant(bpm, 0.0, bars * 4 + 2)
    c = Clip("P2", "4-bar drums and bass, 110 BPM, syncopated electric-style bass", sr, bars * 4 * 60 / bpm, wrap=True)
    K, S, H = kick(sr, rng, 130, 52, decay=0.22), snare(sr, rng, 200, 0.14), hat(sr, rng, 0.025)
    bass = [(0, 0.5, 40), (0.75, 0.25, 40), (1.5, 0.5, 43), (2.5, 0.25, 45), (3, 0.5, 47), (3.5, 0.25, 45)]
    bass2 = [(0, 0.5, 36), (0.75, 0.25, 36), (1.5, 0.5, 38), (2.5, 0.75, 38), (3.5, 0.5, 35)]
    for bar in range(bars):
        base = bar * 4
        for x in ([0, 1.75, 2.5] if bar % 2 == 0 else [0, 0.75, 2.5, 3.5]):
            _drum_hit(c, tm, "kick", K, base + x, 0.9, rng, 0.002)
        for x in (1, 3):
            _drum_hit(c, tm, "snare", S, base + x, 0.85, rng, 0.002)
        for k in range(16):
            _drum_hit(c, tm, "hat", H, base + k / 4, (0.6 if k % 2 == 0 else 0.3), rng, 0.002, 0.3)
        for p, d, m in (bass if bar < 2 else bass2):
            t = float(tm.t(base + p))
            dur = d * 60 / bpm * 0.9
            c.bus("bass").add(t, bass_synth(sr, rng, m, dur, 900, "saw"), 0.8)
            c.note("bass", m, t, dur, 0.8, program=33)
    for i in range(bars * 4):
        c.beats.append((float(tm.t(i)), i % 4 + 1))
    c.meta.update(tempo_bpm=bpm, bars=bars, time_signature="4/4", format=("WAV", "PCM_16"))
    return c, c.mix(rng, {"drums": [lambda y: reverb(y, sr, rng, 0.5, 0.08)]})


def make_P3(rng):
    sr, bpm, bars = 44100, 84.0, 8
    tm = TempoMap.constant(bpm, 0.0, bars * 4 + 2)
    c = Clip("P3", "8-bar piano: melody with left-hand chords, 84 BPM", sr, bars * 4 * 60 / bpm, wrap=True)
    chords = [[48, 52, 55], [45, 48, 52], [41, 45, 48], [43, 47, 50]]
    melody = [  # (beat position within 8 bars, duration in beats, midi)
        (0, 1, 72), (1, 0.5, 74), (1.5, 0.5, 76), (2, 1.5, 79), (3.5, 0.5, 77),
        (4, 1, 76), (5, 1, 72), (6, 1, 74), (7, 1, 72),
        (8, 0.5, 69), (8.5, 0.5, 72), (9, 1, 77), (10, 1, 76), (11, 1, 74),
        (12, 1.5, 74), (13.5, 0.5, 72), (14, 1, 71), (15, 1, 74),
        (16, 1, 76), (17, 0.5, 77), (17.5, 0.5, 79), (18, 1, 81), (19, 1, 79),
        (20, 1, 77), (21, 1, 76), (22, 1, 74), (23, 1, 72),
        (24, 1, 77), (25, 1, 76), (26, 1, 74), (27, 1, 71),
        (28, 3, 72),
    ]
    for p, d, m in melody:
        t = float(tm.t(p))
        dur = d * 60 / bpm * 0.92
        v = 0.75 if p % 1 == 0 else 0.6
        c.bus("piano").add(t, piano(sr, rng, m, dur, v), 0.9, 0.15)
        c.note("piano_rh", m, t, dur, v)
    for bar in range(bars):
        ch = chords[bar % 4]
        for half in (0, 2):
            t = float(tm.t(bar * 4 + half))
            dur = 2 * 60 / bpm * 0.95
            for m in ch:
                c.bus("piano").add(t, piano(sr, rng, m, dur, 0.5), 0.7, -0.15)
                c.note("piano_lh", m, t, dur, 0.5)
    for i in range(bars * 4):
        c.beats.append((float(tm.t(i)), i % 4 + 1))
    c.meta.update(tempo_bpm=bpm, bars=bars, time_signature="4/4", format=("WAV", "PCM_16"))
    return c, c.mix(rng, {"piano": [lambda y: reverb(y, sr, rng, 1.6, 0.22)]})


def make_P4(rng):
    sr, bpm, bars = 44100, 124.0, 8
    tm = TempoMap.constant(bpm, 0.0, bars * 4 + 2)
    c = Clip("P4", "8-bar dense full mix, 124 BPM: drums, bass, pad, stabs, lead arp, shaker, riser", sr,
             bars * 4 * 60 / bpm, wrap=True)
    K, CL, H, OH, SH = kick(sr, rng), clap(sr, rng), hat(sr, rng, 0.03), hat(sr, rng, 0.15), shaker(sr, rng)
    chords = [[57, 60, 64, 67], [53, 57, 60, 64], [55, 60, 64, 67], [55, 59, 62, 67]]
    roots = [45, 41, 48, 43]
    kicks = []
    for bar in range(bars):
        base = bar * 4
        ci = (bar // 2) % 4
        if bar % 2 == 0:
            c.bus("pad").add(float(tm.t(base)), pad(sr, rng, chords[ci], 8 * 60 / bpm), 0.5)
        for b in range(4):
            tb = float(tm.t(base + b))
            c.bus("drums").add(tb, K, 0.95)
            kicks.append(tb)
            if b in (1, 3):
                c.bus("drums").add(tb, CL, 0.6)
            c.bus("drums").add(float(tm.t(base + b + 0.5)), OH, 0.3, -0.2)
            for k in range(4):
                c.bus("drums").add(float(tm.t(base + b + k / 4)), H, 0.25, 0.3)
                c.bus("drums").add(float(tm.t(base + b + k / 4 + 0.125)), SH, 0.15, -0.4)
                m = chords[ci][(b * 4 + k) % 4] + 12
                c.bus("arp").add(float(tm.t(base + b + k / 4)), lead(sr, rng, m, 0.09, 0.4, 2500), 0.3, 0.35)
            c.bus("bass").add(float(tm.t(base + b + 0.5)), bass_synth(sr, rng, roots[ci] - 12, 0.2, 700), 0.8)
            if b == 3:
                c.bus("stab").add(float(tm.t(base + b + 0.75)), stab(sr, rng, chords[ci]), 0.4)
    c.bus("fx").add(float(tm.t(24)), noise_riser(sr, rng, 8 * 60 / bpm), 0.5)
    for i in range(bars * 4):
        c.beats.append((float(tm.t(i)), i % 4 + 1))
    fx = {"pad": [lambda y: sidechain(y, sr, kicks), lambda y: reverb(y, sr, rng, 2.5, 0.35)],
          "arp": [lambda y: reverb(y, sr, rng, 1.5, 0.3)], "bass": [lambda y: sidechain(y, sr, kicks, 0.5)]}
    c.midi_notes = []  # like the brief: no reference for P4
    c.meta.update(tempo_bpm=bpm, bars=bars, time_signature="4/4", format=("WAV", "PCM_16"))
    return c, c.mix(rng, fx)


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #
REFERENCE_KIND = {"A1": "beats+downbeats", "A2": "beats+downbeats", "A3": "beats", "A4": "beats+downbeats",
                  "A5": "beats+downbeats", "A6": "beats+downbeats", "A7": "beats+downbeats",
                  "P1": "midi", "P2": "midi", "P3": "midi", "P4": "none"}


def write_clip(c: Clip, audio: np.ndarray, out: Path):
    fmt, subtype = c.meta.pop("format")
    ext = {"WAV": "wav", "FLAC": "flac", "MP3": "mp3"}[fmt]
    audio_path = out / f"{c.id}.{ext}"
    if fmt == "MP3":
        sf.write(audio_path, audio, c.sr, format="MP3", subtype="MPEG_LAYER_III")
    else:
        sf.write(audio_path, audio, c.sr, format=fmt, subtype=subtype)
    kind = REFERENCE_KIND[c.id]
    if c.id.startswith("A"):
        with open(out / f"{c.id}.beats.txt", "w") as f:
            for t, pos in c.beats:
                f.write(f"{t:.6f}\n" if kind == "beats" else f"{t:.6f}\t{pos}\n")
    if kind == "midi":
        tempo = c.meta["tempo_bpm"]
        pm = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        pm.time_signature_changes.append(pretty_midi.TimeSignature(4, 4, 0))
        tracks = {}
        for name, is_drum, prog, pitch, s, e, v in c.midi_notes:
            if name not in tracks:
                tracks[name] = pretty_midi.Instrument(program=prog, is_drum=is_drum, name=name)
                pm.instruments.append(tracks[name])
            tracks[name].notes.append(pretty_midi.Note(int(np.clip(v * 127, 1, 127)), pitch, s, e))
        pm.write(str(out / f"{c.id}.mid"))
    return {
        "id": c.id, "description": c.description, "audio": audio_path.name, "sample_rate": c.sr,
        "format": fmt, "subtype": subtype if fmt != "MP3" else "MPEG_LAYER_III",
        "duration_s": round(len(audio) / c.sr, 3), "reference": kind, **c.meta,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--only", default="")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    mp3_ok = "MP3" in sf.available_formats()
    a4_fmt = ("MP3", None) if mp3_ok else ("FLAC", "PCM_16")
    makers = {"A1": make_A1, "A2": make_A2, "A3": make_A3, "A4": lambda r: make_A4(r, a4_fmt), "A5": make_A5,
              "A6": make_A6, "A7": make_A7, "P1": make_P1, "P2": make_P2, "P3": make_P3, "P4": make_P4}
    only = [s.strip() for s in args.only.split(",") if s.strip()] or list(makers)
    manifest_path = args.out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for cid in only:
        rng = np.random.default_rng(args.seed + int(cid[1:]) + (100 if cid[0] == "P" else 0))
        clip, audio = makers[cid](rng)
        manifest[cid] = write_clip(clip, audio, args.out)
        print(f"{cid}: {manifest[cid]['duration_s']:.1f} s, {manifest[cid]['format']} {manifest[cid]['subtype']}, "
              f"{manifest[cid]['sample_rate']} Hz")
    manifest = dict(sorted(manifest.items()))
    manifest_path.write_text(json.dumps({"_note": "Synthetic stand-in pack generated by testpack/generate.py. "
                                         "Fields other than id/audio/reference are ground truth for evals only; "
                                         "the analysis code never reads them.",
                                         **{k: v for k, v in manifest.items() if not k.startswith("_")}}, indent=2))


if __name__ == "__main__":
    main()
