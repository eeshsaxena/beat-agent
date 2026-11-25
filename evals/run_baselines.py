"""Part 1 baselines, ablation and stress tests.

    python -m evals.run_baselines

* Baselines on every A clip: librosa's tracker (the brief's "baseline to beat"), Beat This! raw
  output (its own peak picking and downbeats), and the full pipeline.
* Stress tests on A1: the stand-in is clean synthetic audio, so it is degraded on purpose
  (MP3 at 96 kbps, pink noise at 15 dB SNR, 3 kHz low-pass, kick removed by a 200 Hz high-pass)
  to see how much the numbers depend on that cleanliness.
* Speed with 4 CPU threads (a typical laptop) as well as all threads.
"""
from __future__ import annotations

import io
import json
import time
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import soundfile as sf
import torch
from scipy import signal

from beatagent import audio_io, tracking
from evals import pack
from evals.run_part1 import drift_stats

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evals" / "results"


def score(ref_b, ref_d, beats, downs, constant: bool) -> dict:
    out = {"beat_f": round(float(mir_eval.beat.f_measure(ref_b, beats)), 4) if len(beats) else 0.0}
    full = mir_eval.beat.evaluate(ref_b, beats) if len(beats) > 1 else {}
    out["amlt"] = round(float(full.get("Any Metric Level Total", 0.0)), 4)
    if ref_d is not None and downs is not None:
        out["downbeat_f"] = round(float(mir_eval.beat.f_measure(ref_d, downs)), 4) if len(downs) else 0.0
    if constant and len(beats) > 8:
        out["max_drift_ms"] = drift_stats(np.asarray(beats), ref_b)["max_abs_ms"]
    return out


def librosa_beats(a: audio_io.Audio) -> np.ndarray:
    y = librosa.resample(a.mono(), orig_sr=a.sr, target_sr=22050)
    _, frames = librosa.beat.beat_track(y=y, sr=22050, hop_length=512)
    return librosa.frames_to_time(frames, sr=22050, hop_length=512)


def beat_this_raw(a: audio_io.Audio) -> tuple[np.ndarray, np.ndarray]:
    from beat_this.inference import Postprocessor

    b, d = tracking.frame_activations(a.mono(), a.sr)
    pp = Postprocessor(type="minimal", fps=50)
    beats, downs = pp(torch.tensor(b), torch.tensor(d))
    return np.asarray(beats), np.asarray(downs)


def degrade(a: audio_io.Audio, kind: str) -> audio_io.Audio:
    x, sr = a.samples.copy(), a.sr
    if kind == "mp3_96k":
        buf = io.BytesIO()
        sf.write(buf, x, sr, format="MP3", subtype="MPEG_LAYER_III", compression_level=0.75)
        buf.seek(0)
        x, _ = sf.read(buf, dtype="float32", always_2d=True)
    elif kind == "noise_15db":
        rng = np.random.default_rng(0)
        w = rng.standard_normal((len(x), 2))
        f = np.fft.rfftfreq(len(x), 1 / sr)
        pink = np.fft.irfft(np.fft.rfft(w, axis=0) / np.sqrt(np.maximum(f, 20.0))[:, None], n=len(x), axis=0)
        pink *= np.sqrt(np.mean(x ** 2) / np.mean(pink ** 2)) * 10 ** (-15 / 20)
        x = (x + pink).astype(np.float32)
    elif kind == "lowpass_3k":
        x = signal.sosfiltfilt(signal.butter(6, 3000, "lowpass", fs=sr, output="sos"), x, axis=0).astype(np.float32)
    elif kind == "no_kick_hp200":
        x = signal.sosfiltfilt(signal.butter(6, 200, "highpass", fs=sr, output="sos"), x, axis=0).astype(np.float32)
    return audio_io.Audio(x, sr, a.subtype, a.format, a.path)


def main():
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    meta = json.loads((pack.pack_dir() / "manifest.json").read_text()) if (pack.pack_dir() / "manifest.json").exists() else {}
    rows = []
    for cid in pack.ids("A"):
        a = audio_io.load(pack.audio_path(cid))
        ref_b, ref_d = pack.ref_beats(cid)
        constant = meta.get(cid, {}).get("grid") == "constant"
        lb = librosa_beats(a)
        rb, rd = beat_this_raw(a)
        g = tracking.analyze(a)
        row = {"id": cid, "librosa": score(ref_b, None, lb, None, constant),
               "beat_this_raw": score(ref_b, ref_d, rb, rd, constant),
               "full_pipeline": score(ref_b, ref_d, g.times, g.downbeat_times, constant)}
        rows.append(row)
        print(cid, json.dumps(row), flush=True)

    stress = []
    a1 = audio_io.load(pack.audio_path("A1"))
    ref_b, ref_d = pack.ref_beats("A1")
    for kind in ("clean", "mp3_96k", "noise_15db", "lowpass_3k", "no_kick_hp200"):
        a = a1 if kind == "clean" else degrade(a1, kind)
        g = tracking.analyze(a)
        s = score(ref_b, ref_d, g.times, g.downbeat_times, True)
        s.update(kind=kind, grid=g.grid_type, tempo=round(g.tempo_bpm, 3),
                 onset_shift_ms=(g.meta.get("onset_calibration_ms") or {}).get("shift_ms"))
        stress.append(s)
        print("A1", json.dumps(s), flush=True)

    speed = {}
    for threads in (4, torch.get_num_threads()):
        torch.set_num_threads(threads)
        t = time.perf_counter()
        tracking.analyze(a1)
        speed[threads] = round(time.perf_counter() - t, 2)
    print("speed", speed, flush=True)

    out = {"baselines": rows, "stress_A1": stress, "speed_A1_seconds_by_threads": speed}
    RES.mkdir(parents=True, exist_ok=True)
    (RES / "part1_baselines.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    L = ["# Part 1: baselines, ablation and stress tests", "",
         "Beat F at ±70 ms (mir_eval). \"Beat This! raw\" is the model's own minimal post-processing; "
         "\"full pipeline\" adds the tempo model and onset alignment (beatagent/tracking.py). "
         "Max drift is reported for constant-tempo clips.", "",
         "| Clip | librosa beat F | Beat This! raw beat F / downbeat F / max drift | Full pipeline beat F / downbeat F / max drift |",
         "|---|---|---|---|"]
    for r in rows:
        bt, fp = r["beat_this_raw"], r["full_pipeline"]
        L.append(f"| {r['id']} | {r['librosa']['beat_f']} | {bt['beat_f']} / {bt.get('downbeat_f', '-')} / "
                 f"{bt.get('max_drift_ms', '-')} ms | {fp['beat_f']} / {fp.get('downbeat_f', '-')} / {fp.get('max_drift_ms', '-')} ms |")
    L += ["", "## Stress test on A1 (degraded copies of the 6-minute house clip)", "",
          "| Degradation | Grid | Tempo | Beat F | Downbeat F | Max drift |", "|---|---|---|---|---|---|"]
    for s in stress:
        L.append(f"| {s['kind']} | {s['grid']} | {s['tempo']} | {s['beat_f']} | {s.get('downbeat_f')} | {s.get('max_drift_ms')} ms |")
    L += ["", f"Speed, A1 (6 min) on CPU: {speed} seconds by number of torch threads.", ""]
    (RES / "part1_baselines.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
