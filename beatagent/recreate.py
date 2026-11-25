"""Audio -> score: measured loop grid, drum onsets and classes, pitched notes.

Nothing here comes from the LLM:
* tempo, loop length, first downbeat: the Part 1 tracker on the clip tiled 3x (a loop
  analysed with context on both sides), snapped so the clip is a whole number of bars
  when it is within 3% of one;
* stems: Demucs (htdemucs) on CPU;
* drums: band-wise onset detection (low / mid / high) and a measured classifier
  (kick = low-band rise, snare = mid-band noise rise, hat = sustained high-band rise,
  open hat = long high-band decay);
* pitched parts: Basic Pitch (ONNX) on the bass and 'other' stems, onsets snapped to
  measured onsets;
* patches (timbre, last in the fidelity order): a few numbers measured from the stems.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

from . import onsets as onset_mod
from . import tracking
from .audio_io import Audio, resample
from .score import DEFAULT_PATCHES, SCHEMA

_SR = 22050
HOP = 32


def _say(progress, msg):
    if progress:
        progress(msg)


# ----------------------------------------------------------------------------
# Loop grid
# ----------------------------------------------------------------------------
def loop_grid(audio: Audio) -> dict:
    dur = audio.duration
    reps = int(min(6, max(3, np.ceil(30.0 / dur))))
    tiled = Audio(np.tile(audio.samples, (reps, 1)), audio.sr, audio.subtype, audio.format)
    g = tracking.analyze(tiled)
    P = float(np.median(np.diff(g.times)))
    bpb = g.beats_per_bar
    bar = bpb * P
    est = dur / bar
    n_bars = max(1, int(round(est)))
    exact = abs(est - n_bars) / n_bars < 0.03
    if exact:
        P = dur / (n_bars * bpb)
        bar = bpb * P
    # first downbeat phase (circular mean of downbeats modulo a bar)
    d = g.downbeat_times
    ang = np.angle(np.mean(np.exp(2j * np.pi * (d % bar) / bar))) if len(d) else 0.0
    t0 = float((ang / (2 * np.pi)) % 1.0 * bar)
    if t0 > bar - 0.015:
        t0 -= bar
    num, den = {"2/4": (2, 4), "3/4": (3, 4), "4/4": (4, 4), "6/8": (6, 8), "9/8": (9, 8), "12/8": (12, 8)}[g.time_signature]
    return {"tempo_bpm": 60.0 / P, "beats_per_bar": bpb, "time_signature": [num, den], "bars": n_bars if exact else est,
            "loop_beats": n_bars * bpb if exact else dur / P, "first_downbeat_s": t0, "exact_loop": exact,
            "tracker_tempo_bpm": g.tempo_bpm, "grid_type": g.grid_type, "clip_duration_s": dur}


# ----------------------------------------------------------------------------
# Separation
# ----------------------------------------------------------------------------
_DEMUCS = None


def separate(audio: Audio, tile: int = 3) -> dict[str, np.ndarray]:
    """Demucs stems for the clip (tiled so the model sees the loop with context)."""
    global _DEMUCS
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    if _DEMUCS is None:
        _DEMUCS = get_model("htdemucs")
        _DEMUCS.eval()
    m = _DEMUCS
    x = audio.samples if audio.channels == 2 else np.repeat(audio.samples[:, :1], 2, axis=1)
    x = resample(x, audio.sr, m.samplerate)
    n = len(x)
    xt = np.tile(x, (tile, 1))
    wav = torch.tensor(xt.T, dtype=torch.float32)
    ref = wav.mean(0)
    wav = (wav - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        # shifts=0: no random time shift, so stems (and every Part 3 number) are reproducible run to run
        out = apply_model(m, wav[None], shifts=0, device="cpu", split=True, overlap=0.25, progress=False)[0]
    out = out * (ref.std() + 1e-8) + ref.mean()
    mid = (tile // 2) * n
    stems = {}
    for name, s in zip(m.sources, out):
        y = s.numpy().T[mid:mid + n]
        stems[name] = resample(np.ascontiguousarray(y), m.samplerate, audio.sr)
    return stems


def rms_db(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12))


# ----------------------------------------------------------------------------
# Drums
# ----------------------------------------------------------------------------
DRUM_BANDS = {"low": ("lowpass", 150.0), "mid": ("bandpass", (300.0, 5000.0)), "high": ("highpass", 7000.0)}


ENV_WINDOW_MS = {"low": 12.0, "mid": 4.0, "high": 2.0}  # >= one period of the band's lowest content


def _band_envs(x: np.ndarray, sr: int) -> tuple[dict, float]:
    """RMS envelopes per band. Each band is smoothed over a window that spans its own low
    frequencies; a 1.5 ms window on the low band ripples at twice the kick's pitch and
    turns one kick into five 'onsets' (measured on P1)."""
    y = resample(x.astype(np.float32), sr, _SR) if sr != _SR else x.astype(np.float32)
    envs = {}
    for b, (kind, f) in DRUM_BANDS.items():
        z = signal.sosfiltfilt(signal.butter(4, f, btype=kind, fs=_SR, output="sos"), y)
        w = np.hanning(int(ENV_WINDOW_MS[b] * 1e-3 * _SR) + 2)[1:-1]
        p = np.convolve(z ** 2, w / w.sum(), mode="same")
        n = len(p) // HOP
        envs[b] = np.sqrt(np.maximum(p[: n * HOP].reshape(n, HOP).mean(axis=1), 0.0))
    return envs, _SR / HOP


def transcribe_drums(x: np.ndarray, sr: int, wrap: bool = True) -> tuple[list[dict], dict]:
    """Drum hits as {time, pitch (GM), velocity, cls}. With wrap=True the clip is treated as a loop."""
    mono = x.mean(axis=1) if x.ndim > 1 else x
    n = len(mono)
    if wrap:  # analyse with one copy of context on each side so hits at the loop edge are seen
        mono = np.concatenate([mono, mono, mono])
    envs, fps = _band_envs(mono, sr)
    flux = {b: np.maximum(0, np.diff(e, prepend=e[:1])) for b, e in envs.items()}
    cand = []
    for b, f in flux.items():
        if f.max() <= 0:
            continue
        pk, _ = signal.find_peaks(f, height=0.05 * np.percentile(f, 99.9), distance=int(0.04 * fps))
        cand += [int(p) for p in pk]
    cand = sorted(cand)
    events = []
    for p in cand:
        if events and p - events[-1] < int(0.02 * fps):
            continue
        events.append(p)
    pre0, pre1, post = int(0.04 * fps), int(0.006 * fps), int(0.025 * fps)
    feats = []
    for k, p in enumerate(events):
        f = {"frame": p}
        for b, e in envs.items():
            before = e[max(0, p - pre0):max(1, p - pre1)]
            after = e[p:p + post]
            b0 = float(before.mean()) if len(before) else 0.0
            a1 = float(after.max()) if len(after) else 0.0
            f[b] = a1 - b0
            f[b + "_contrast"] = a1 / (b0 + 1e-6 * (a1 + 1e-9))
        hp = envs["high"]
        peak_i = p + int(np.argmax(hp[p:p + int(0.01 * fps)])) if p + 1 < len(hp) else p
        nxt = events[k + 1] if k + 1 < len(events) else len(hp)
        f["high_sustain"] = float(hp[peak_i + int(0.008 * fps):peak_i + int(0.035 * fps)].mean() / (hp[peak_i] + 1e-9)) \
            if peak_i + int(0.035 * fps) < len(hp) else 0.0
        # open-hat test: high-band level 60-90 ms after the peak, but only up to the next hit
        a, b2 = peak_i + int(0.06 * fps), min(peak_i + int(0.09 * fps), nxt - int(0.004 * fps))
        f["high_tail"] = float(hp[a:b2].mean() / (hp[peak_i] + 1e-9)) if b2 > a else 0.0
        feats.append(f)
    if not feats:
        return [], {}
    mx = {b: max(1e-9, max(f[b] for f in feats)) for b in DRUM_BANDS}
    hits = []
    for f in feats:
        rl, rm, rh = f["low"] / mx["low"], f["mid"] / mx["mid"], f["high"] / mx["high"]
        t = f["frame"] / fps
        is_kick = rl >= 0.4 and f["low_contrast"] >= 2.0
        if is_kick:
            hits.append({"time": t, "cls": "kick", "pitch": 36, "velocity": rl})
        # snare: noise mostly below 5 kHz; a kick's click and pitch sweep also reach this band,
        # so on a kick a snare needs a much stronger mid-band rise (measured on P2)
        if rm >= (0.45 if is_kick else 0.18) and f["mid_contrast"] >= 2.5 and f["mid"] >= 0.6 * f["high"]:
            hits.append({"time": t, "cls": "snare", "pitch": 38, "velocity": rm})
        elif rh >= 0.1 and f["high_contrast"] >= 2.5 and f["high_sustain"] >= 0.25:
            open_hat = f["high_tail"] >= 0.35
            hits.append({"time": t, "cls": "ohat" if open_hat else "hat", "pitch": 46 if open_hat else 42, "velocity": rh})
    # Refine each hit's time to the physical attack: the smoothed zero-phase envelopes put
    # hits 10-15 ms early (measured on P2); the causal mid/high novelty peak does not.
    nov, nfps = onset_mod.onset_function(mono, sr, bands=("mid", "high"), causal=True)
    for h in hits:
        lo, hi = int((h["time"] - 0.008) * nfps), int((h["time"] + 0.025) * nfps)
        lo, hi = max(lo, 0), min(hi, len(nov))
        if hi - lo > 2 and nov[lo:hi].max() > 0:
            seg = nov[lo:hi]
            pk, _ = signal.find_peaks(np.concatenate([[0.0], seg, [0.0]]), height=0.3 * seg.max())
            j = lo + (int(pk[0]) - 1 if len(pk) else int(np.argmax(seg)))
            h["time"] = j / nfps
    # keep the middle copy only
    if wrap:
        dur = n / sr
        hits = [dict(h, time=h["time"] - dur) for h in hits if dur - 0.02 <= h["time"] < 2 * dur - 0.02]
    # de-duplicate same class within 30 ms
    hits.sort(key=lambda h: (h["cls"], h["time"]))
    out = []
    for h in hits:
        if out and out[-1]["cls"] == h["cls"] and h["time"] - out[-1]["time"] < 0.03:
            if h["velocity"] > out[-1]["velocity"]:
                out[-1] = h
            continue
        out.append(h)
    out.sort(key=lambda h: h["time"])
    for h in out:
        h["velocity"] = float(np.clip(0.3 + 0.7 * h["velocity"], 0.05, 1.0))
    return out, _drum_patch(x, sr, out)


def _drum_patch(x: np.ndarray, sr: int, hits: list[dict]) -> dict:
    p = dict(DEFAULT_PATCHES["drums"])
    mono = x.mean(axis=1) if x.ndim > 1 else x
    kicks = [h["time"] for h in hits if h["cls"] == "kick"]
    if kicks:
        specs = []
        for t in kicks[:16]:
            a = int((t + 0.02) * sr)
            seg = mono[a:a + int(0.15 * sr)]
            if len(seg) < 256:
                continue
            S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 14))
            specs.append(S)
        if specs:
            S = np.mean(specs, axis=0)
            fr = np.fft.rfftfreq(1 << 14, 1 / sr)
            band = (fr >= 35) & (fr <= 140)
            p["f_end"] = float(fr[band][np.argmax(S[band])])
            p["f_start"] = float(min(300.0, p["f_end"] * 2.8))
    return p


# ----------------------------------------------------------------------------
# Pitched parts
# ----------------------------------------------------------------------------
_BP = None


def _basic_pitch_model():
    global _BP
    if _BP is None:
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import Model

        _BP = Model(build_icassp_2022_model_path(FilenameSuffix.onnx))
    return _BP


def transcribe_pitched(x: np.ndarray, sr: int, role: str, wrap: bool = True) -> list[dict]:
    """Notes {start, end, pitch, velocity} (seconds in the clip) via Basic Pitch."""
    from basic_pitch.inference import predict

    mono = x.mean(axis=1) if x.ndim > 1 else x
    dur = len(mono) / sr
    y = np.concatenate([mono, mono, mono]) if wrap else mono
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        sf.write(tmp, y, sr, subtype="FLOAT")
        kw = dict(onset_threshold=0.5, frame_threshold=0.3, minimum_note_length=58, melodia_trick=True)
        if role == "bass":
            kw.update(minimum_frequency=28.0, maximum_frequency=400.0)
        else:
            kw.update(minimum_frequency=60.0)
        _, _, events = predict(tmp, _basic_pitch_model(), **kw)
    finally:
        os.remove(tmp)
    notes = [{"start": float(s), "end": float(e), "pitch": int(p), "amp": float(a)} for s, e, p, a, *_ in events]
    if wrap:
        notes = [dict(nn, start=nn["start"] - dur, end=nn["end"] - dur) for nn in notes
                 if dur - 0.03 <= nn["start"] < 2 * dur - 0.03]
    if not notes:
        return []
    amax = max(nn["amp"] for nn in notes)
    notes = [nn for nn in notes if nn["amp"] >= 0.2 * amax]
    notes.sort(key=lambda nn: (nn["start"], -nn["amp"]))
    if role == "bass":  # monophonic: one note at a time, the stronger one wins a near-tie
        mono_notes = []
        for nn in notes:
            if mono_notes and nn["start"] - mono_notes[-1]["start"] < 0.03:
                if nn["amp"] > mono_notes[-1]["amp"]:
                    mono_notes[-1] = nn
                continue
            if mono_notes and mono_notes[-1]["end"] > nn["start"]:
                mono_notes[-1]["end"] = nn["start"]
            mono_notes.append(nn)
        notes = mono_notes
    # snap onsets to measured onsets in the stem (Basic Pitch frames are ~11.6 ms)
    on = onset_mod.detect(np.concatenate([mono, mono, mono]) if wrap else mono, sr) - (dur if wrap else 0.0)
    kept = []
    for nn in notes:
        has_onset = False
        if len(on):
            j = int(np.argmin(np.abs(on - nn["start"])))
            if abs(on[j] - nn["start"]) <= 0.03:
                shift = on[j] - nn["start"]
                nn["start"] += shift
                nn["end"] = max(nn["start"] + 0.03, nn["end"] + shift)
                has_onset = True
        nn["velocity"] = float(np.clip(0.35 + 0.65 * nn["amp"] / amax, 0.05, 1.0))
        # Polyphonic parts: a note with no measured attack is a decay artefact (P3: 35 of 61 extras).
        if role == "bass" or has_onset:
            kept.append(nn)
    notes = kept
    if role != "bass":
        # (_merge_retriggers was tried and removed: it merged real repeated chord attacks; P3 recall 0.94 -> 0.69)
        notes = _drop_harmonics(notes)
    return notes


def _merge_retriggers(notes: list[dict], tol: float = 0.03) -> list[dict]:
    """A same-pitch note that starts while the previous one still sounds, and is quieter,
    is the same note re-detected (Basic Pitch splits sustained notes), not a new attack."""
    out: list[dict] = []
    for nn in sorted(notes, key=lambda n: n["start"]):
        prev = next((p for p in reversed(out) if p["pitch"] == nn["pitch"]), None)
        if prev is not None and prev["end"] >= nn["start"] - tol and nn["amp"] <= prev["amp"]:
            prev["end"] = max(prev["end"], nn["end"])
            continue
        out.append(nn)
    return out


def _drop_harmonics(notes: list[dict], tol: float = 0.03) -> list[dict]:
    """Drop a note an octave, twelfth or two octaves above a louder note with the same onset
    (transcription models often report strong harmonics of chord tones as notes)."""
    drop = set()
    for i, a in enumerate(notes):
        for j, b in enumerate(notes):
            if i != j and abs(a["start"] - b["start"]) <= tol and (a["pitch"] - b["pitch"]) in (12, 19, 24) \
                    and a["amp"] < 0.7 * b["amp"]:
                drop.add(i)
    return [n for k, n in enumerate(notes) if k not in drop]


def _cutoff_from(x: np.ndarray, sr: int, mult: float, lo: float, hi: float) -> float:
    mono = x.mean(axis=1) if x.ndim > 1 else x
    f, P = signal.welch(mono, sr, nperseg=4096)
    c = float((f * P).sum() / (P.sum() + 1e-12))
    return float(np.clip(c * mult, lo, hi))


def _polyphony(notes: list[dict]) -> float:
    if not notes:
        return 0.0
    ev = sorted([(n["start"], 1) for n in notes] + [(n["end"], -1) for n in notes])
    cur = best = 0
    for t, d in ev:
        cur += d
        best = max(best, cur)
    return float(best)


# ----------------------------------------------------------------------------
# Whole clip -> score
# ----------------------------------------------------------------------------
def build_score(audio: Audio, title: str = "", use_separation: bool = True, progress=None) -> dict:
    _say(progress, "Measuring tempo, bars and the first downbeat")
    lg = loop_grid(audio)
    spb = 60.0 / lg["tempo_bpm"]
    L = lg["loop_beats"]
    t0 = lg["first_downbeat_s"]

    def beats(t):
        return float(((t - t0) / spb) % L)

    mix_db = rms_db(audio.samples)
    stems = {}
    if use_separation:
        _say(progress, "Separating stems with Demucs (CPU)")
        stems = separate(audio)
    stem_db = {k: round(rms_db(v) - mix_db, 1) for k, v in stems.items()}
    tracks, prov = [], {"grid": lg, "stem_level_db_vs_mix": stem_db}

    _say(progress, "Transcribing drums")
    drum_src = stems.get("drums", audio.samples) if stems else audio.samples
    if not stems or stem_db.get("drums", -99) > -18:
        hits, patch = transcribe_drums(drum_src, audio.sr)
        if len(hits) >= 3:
            tracks.append({"id": "drums", "name": "Drums", "kind": "drums", "gain_db": 0.0, "mute": False,
                           "patch": patch,
                           "notes": [{"start": round(beats(h["time"]), 4), "duration": 0.25, "pitch": h["pitch"],
                                      "velocity": round(h["velocity"], 3)} for h in hits]})
            prov["drums"] = {"source": "demucs drums stem" if stems else "full mix", "hits": len(hits)}

    pitched_parts = []
    if stems:
        if stem_db.get("bass", -99) > -20:
            pitched_parts.append(("bass", stems["bass"], "bass"))
        other = stems.get("other")
        if stem_db.get("other", -99) > -20:
            pitched_parts.append(("other", other, "other"))
    else:
        pitched_parts.append(("other", audio.samples, "other"))
    for sid, x, role in pitched_parts:
        _say(progress, f"Transcribing the {sid} part (Basic Pitch)")
        notes = transcribe_pitched(x, audio.sr, role)
        if len(notes) < 2:
            continue
        if role == "bass":
            patch = dict(DEFAULT_PATCHES["bass"], cutoff_hz=_cutoff_from(x, audio.sr, 2.5, 150, 3000))
            tid, name = "bass", "Bass"
        else:
            poly = _polyphony(notes)
            if poly >= 2:
                patch, tid, name = dict(DEFAULT_PATCHES["keys"]), "keys", "Keys"
            else:
                patch = dict(DEFAULT_PATCHES["lead"], cutoff_hz=_cutoff_from(x, audio.sr, 3.0, 800, 8000))
                tid, name = "lead", "Lead"
        tracks.append({"id": tid, "name": name, "kind": "pitched", "gain_db": 0.0, "mute": False, "patch": patch,
                       "notes": [{"start": round(beats(nn["start"]), 4),
                                  "duration": round(max(0.05, (nn["end"] - nn["start"]) / spb), 4),
                                  "pitch": int(nn["pitch"]), "velocity": round(nn["velocity"], 3)} for nn in notes]})
        prov[tid] = {"source": f"demucs {sid} stem" if stems else "full mix", "model": "basic-pitch 0.4.0 (ONNX)",
                     "notes": len(notes)}
    for t in tracks:
        t["notes"].sort(key=lambda n: n["start"])
    return {
        "schema": SCHEMA, "title": title or (audio.path.stem if audio.path else "Recreation"),
        "tempo_bpm": round(lg["tempo_bpm"], 4), "time_signature": lg["time_signature"],
        "beats_per_bar": lg["beats_per_bar"], "bars": lg["bars"], "loop_beats": L, "master_gain_db": -9.0,
        "source": {"file": audio.path.name if audio.path else None, "duration_s": round(audio.duration, 4),
                   "first_downbeat_s": round(t0, 4), "exact_loop": lg["exact_loop"]},
        "tracks": tracks, "provenance": prov,
    }
