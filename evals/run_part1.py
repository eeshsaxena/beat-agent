"""Part 1 evals: beat grid, click audition, mir_eval accuracy, drift, speed, and A7->A1 sync.

    python -m evals.run_part1 [--only A1,A3] [--no-sync]

Writes per-clip artefacts to evals/out/part1/<ID>/ and the results table to
evals/results/part1.{json,md}.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import mir_eval
import numpy as np

from beatagent import audio_io, click, tracking
from beatagent import sync as syncmod
from beatagent.tracking import nearest_dist
from evals import pack

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "out" / "part1"
RES = ROOT / "evals" / "results"


def save_listen(path: Path, x: np.ndarray, sr: int, subtype: str = "PCM_16"):
    """Full-resolution WAV plus an MP3 listening copy (the MP3 is what the repo carries)."""
    audio_io.save(path.with_suffix(".wav"), x, sr, "WAV", subtype)
    audio_io.save(path.with_suffix(".mp3"), x, sr, "MP3", "MPEG_LAYER_III")


def signed_nearest(ref: np.ndarray, est: np.ndarray) -> np.ndarray:
    """For each est value, signed error (est - nearest ref)."""
    j = np.clip(np.searchsorted(ref, est), 1, len(ref) - 1)
    left, right = ref[j - 1], ref[j]
    near = np.where(np.abs(est - left) <= np.abs(right - est), left, right)
    return est - near


def drift_stats(grid_t: np.ndarray, ref_t: np.ndarray) -> dict:
    m = (grid_t >= ref_t[0] - 0.07) & (grid_t <= ref_t[-1] + 0.07)
    e = signed_nearest(ref_t, grid_t[m]) * 1000
    matched = np.abs(e) <= 70
    em = e[matched]
    t = grid_t[m][matched]
    per_min = {}
    for k in range(int(t.max() // 60) + 1 if len(t) else 0):
        s = np.abs(em[(t >= 60 * k) & (t < 60 * (k + 1))])
        if len(s):
            per_min[f"{k}-{k + 1} min"] = round(float(s.max()), 2)
    miss_ref = int(np.sum(nearest_dist(grid_t, ref_t) > 0.07))
    return {
        "grid_beats_in_ref_span": int(m.sum()), "grid_beats_matched": int(matched.sum()),
        "ref_beats_without_grid_beat": miss_ref,
        "max_abs_ms": round(float(np.abs(em).max()), 2) if len(em) else None,
        "p95_abs_ms": round(float(np.percentile(np.abs(em), 95)), 2) if len(em) else None,
        "median_abs_ms": round(float(np.median(np.abs(em))), 2) if len(em) else None,
        "mean_signed_ms": round(float(em.mean()), 2) if len(em) else None,
        "beats_over_10ms": int(np.sum(np.abs(em) > 10)),
        "max_abs_ms_per_minute": per_min,
    }


def eval_clip(cid: str, timings_out: dict) -> dict:
    path = pack.audio_path(cid)
    a = audio_io.load(path)
    t = time.perf_counter()
    g = tracking.analyze(a)
    analysis_s = time.perf_counter() - t
    d = OUT / cid
    d.mkdir(parents=True, exist_ok=True)
    src = {"file": path.name, "sample_rate": a.sr, "channels": a.channels, "subtype": a.subtype, "format": a.format,
           "duration_s": round(a.duration, 4)}
    (d / "grid.json").write_text(json.dumps(g.to_dict(src), indent=1))
    save_listen(d / "click_audition", click.audition(a.samples, a.sr, g), a.sr)

    ref_b, ref_d = pack.ref_beats(cid)
    row = {"id": cid, "duration_s": round(a.duration, 1), "analysis_s": round(analysis_s, 2),
           "realtime_factor": round(a.duration / analysis_s, 1), "grid_type": g.grid_type, "tempo_bpm": round(g.tempo_bpm, 3),
           "time_signature": g.time_signature, "n_beats": g.n, "n_bars": g.last_bar, "pickup_beats": g.pickup_beats,
           "confidence": round(g.confidence, 3), "onset_calibration_ms": g.meta.get("onset_calibration_ms"),
           "tempo_fit": g.meta.get("fit")}
    if ref_b is not None:
        row["beat_f_70ms"] = round(float(mir_eval.beat.f_measure(ref_b, g.times, 0.07)), 4)
        full = mir_eval.beat.evaluate(ref_b, g.times)  # mir_eval's standard: trims beats before 5 s
        row["mir_eval_beat_trimmed5s"] = {k: round(float(v), 4) for k, v in full.items()}
        row["ref_tempo_bpm"] = round(float(60 / np.median(np.diff(ref_b))), 3)
        row["drift"] = drift_stats(g.times, ref_b)
    if ref_d is not None and len(ref_d):
        row["downbeat_f_70ms"] = round(float(mir_eval.beat.f_measure(ref_d, g.downbeat_times, 0.07)), 4)
    (d / "metrics.json").write_text(json.dumps(row, indent=1))
    timings_out[cid] = g.meta.get("timings")
    return row


def eval_sync(a_id="A1", b_id="A7", start_bar_a=33, start_bar_b=1, duration=60.0) -> dict:
    a, b = audio_io.load(pack.audio_path(a_id)), audio_io.load(pack.audio_path(b_id))
    ga = tracking.analyze(a)
    gb = tracking.analyze(b)
    r = syncmod.sync(a, ga, b, gb, start_bar_a=start_bar_a, start_bar_b=start_bar_b, duration=duration)
    d = OUT / f"sync_{b_id}_to_{a_id}"
    d.mkdir(parents=True, exist_ok=True)
    save_listen(d / "overlap_mix", r.mix, r.sr)
    save_listen(d / "stem_A", r.a_stem, r.sr)
    save_listen(d / "stem_B_stretched", r.b_stem, r.sr)
    # downbeats of B (mapped through the stretch) vs downbeats of A, inside the render
    bd = r.b_grid_out[r.b_labels == 1]
    ad = r.a_grid_out[r.a_labels == 1]
    off = signed_nearest(ad, bd) * 1000 if len(ad) > 1 and len(bd) else np.array([np.nan])
    rep = dict(r.report)
    rep["b_downbeats_vs_a_downbeats_ms"] = {"n": int(len(bd)), "max_abs": round(float(np.nanmax(np.abs(off))), 2),
                                            "median_abs": round(float(np.nanmedian(np.abs(off))), 2)}
    # independent check against the reference annotations: where do B's reference downbeats land?
    rb, rdb = pack.ref_beats(b_id)
    ra, rda = pack.ref_beats(a_id)
    tA = r.report["track_a"]["start_s"]
    shift = r.report["onset_refinement_ms"] / 1000
    use_down = rdb is not None and rda is not None
    rb_ref, ra_ref = (rdb, rda) if use_down else (rb, ra)
    if rb_ref is not None and ra_ref is not None:
        inside = (rb_ref >= r.anchors_b_s[0] - 1e-6) & (rb_ref <= r.anchors_b_s[-1])
        mapped = tA + np.interp(rb_ref[inside], r.anchors_b_s, r.anchors_out_s) + shift
        inwin = mapped[mapped <= tA + r.report["duration_s"] - 0.05]
        e = signed_nearest(ra_ref, inwin) * 1000
        rep["reference_check"] = {"what": f"B's reference {'downbeats' if use_down else 'beats'} mapped through the "
                                          f"stretch vs A's reference {'downbeats' if use_down else 'beats'}",
                                  "n": int(len(inwin)), "max_abs_ms": round(float(np.abs(e).max()), 2),
                                  "median_abs_ms": round(float(np.median(np.abs(e))), 2)}
    with_clicks = click.audition(r.mix, r.sr, _render_grid(r))
    save_listen(d / "overlap_mix_click_audition", with_clicks, r.sr)
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    return rep


def _render_grid(r):
    from beatagent.grid import BeatGrid

    return BeatGrid(times=r.a_grid_out, beat_in_bar=r.a_labels, beats_per_bar=4, time_signature="4/4",
                    grid_type="constant", duration_s=len(r.mix) / r.sr, sample_rate=r.sr)


def write_markdown(rows: list[dict], sync_rep: dict | None, env: dict, extra: dict | None = None) -> str:
    L = ["# Part 1 results: beat grids and sync", "",
         f"Pack: `{env['pack']}`. Machine: {env['cpu']} ({env['threads']} threads), device `cpu`.", "",
         "| Clip | Grid | Tempo (ref) | Sig | Beat F (±70 ms) | Downbeat F | AMLt | Max drift ms | Beats >10 ms | Analysis s | Conf |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        dr = r.get("drift", {})
        ml = r.get("mir_eval_beat_trimmed5s", {})
        L.append(f"| {r['id']} | {r['grid_type']} | {r['tempo_bpm']:.2f} ({r.get('ref_tempo_bpm', '-')}) | "
                 f"{r['time_signature']} | {r.get('beat_f_70ms', '-')} | {r.get('downbeat_f_70ms', '-')} | "
                 f"{ml.get('Any Metric Level Total', '-')} | {dr.get('max_abs_ms', '-')} | {dr.get('beats_over_10ms', '-')} | "
                 f"{r['analysis_s']} | {r['confidence']} |")
    if sync_rep:
        k = sync_rep["kick_alignment_final"]
        k0 = sync_rep["kick_alignment_grid_only"]
        L += ["", "## Sync A7 -> A1 (overlap from A1 bar 33, A7 bar 1, 60 s)", "",
              f"* Time-stretch ratio (B output/input duration): **{sync_rep['time_stretch_ratio']}** "
              f"({sync_rep['track_b']['bpm']} -> {sync_rep['track_a']['bpm']} BPM), mapping: {sync_rep['mapping']}.",
              f"* Offset: A1 bar 33 at {sync_rep['track_a']['start_s']} s, A7 bar 1 at {sync_rep['track_b']['start_s']} s.",
              f"* Kick onsets, grid only: median |offset| {k0.get('median_abs_offset_ms')} ms, max {k0.get('max_abs_offset_ms')} ms "
              f"({k0.get('matched')} matched kicks).",
              f"* Automatic onset nudge: {'off' if not sync_rep['onset_refinement_ms'] else str(sync_rep['onset_refinement_ms']) + ' ms'} "
              f"(p95 |offset| {k.get('p95_abs_offset_ms')} ms).",
              f"* B downbeats vs A downbeats in the render: max {sync_rep['b_downbeats_vs_a_downbeats_ms']['max_abs']} ms."]
        if "reference_check" in sync_rep:
            rc = sync_rep["reference_check"]
            L.append(f"* Reference check ({rc['what']}): max {rc['max_abs_ms']} ms, median {rc['median_abs_ms']} ms.")
    if extra:
        L += ["", "## Variable-tempo partners synced onto A1 bar 33 (bar-by-bar stretch)", "",
              "| B | B grid | Ratio | Kick onsets median / max | Reference check median / max | Notes |",
              "|---|---|---|---|---|---|"]
        for b, rep in extra.items():
            k2, rc = rep["kick_alignment_final"], rep.get("reference_check", {})
            L.append(f"| {b} | {rep['track_b']['grid']} | {rep['time_stretch_ratio']} | {k2.get('median_abs_offset_ms')} / "
                     f"{k2.get('max_abs_offset_ms')} ms | {rc.get('median_abs_ms')} / {rc.get('max_abs_ms')} ms | "
                     f"{' '.join(rep['notes'])} |")
        L += ["", "A3 is a live take whose kicks scatter 7-12 ms around the beat by design, so its kick-onset numbers "
              "measure the drummer as much as the sync; the reference check (intended beat times) measures the mapping."]
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-sync", action="store_true")
    args = ap.parse_args(argv)
    import torch

    env = {"pack": str(pack.pack_dir().relative_to(ROOT)) if pack.pack_dir().is_relative_to(ROOT) else str(pack.pack_dir()),
           "cpu": platform.processor() or platform.machine(), "threads": torch.get_num_threads(),
           "python": platform.python_version(), "torch": torch.__version__}
    t = time.perf_counter()
    tracking._model(os.environ.get("BEAT_AGENT_CHECKPOINT", "final0"), "cpu")
    env["model_load_s"] = round(time.perf_counter() - t, 2)
    ids = [s for s in args.only.split(",") if s] or pack.ids("A")
    rows, timings = [], {}
    for cid in ids:
        row = eval_clip(cid, timings)
        rows.append(row)
        print(f"{cid}: {row['grid_type']}, {row['tempo_bpm']} BPM, beatF={row.get('beat_f_70ms')}, "
              f"downF={row.get('downbeat_f_70ms')}, drift max={row.get('drift', {}).get('max_abs_ms')} ms, "
              f"{row['analysis_s']} s", flush=True)
    sync_rep, extra_syncs = None, {}
    if not args.no_sync and {"A1", "A7"} <= set(pack.ids("A")):
        sync_rep = eval_sync()
        print("sync:", json.dumps(sync_rep["kick_alignment_final"]), flush=True)
        for b in ("A5", "A3"):  # variable-tempo partners exercise the bar-by-bar stretch
            if b in pack.ids("A"):
                extra_syncs[b] = eval_sync(b_id=b)
                print(f"sync {b}->A1:", json.dumps(extra_syncs[b]["kick_alignment_final"]), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    prev = {}
    pj = RES / "part1.json"
    if pj.exists() and args.only:
        prev = {r["id"]: r for r in json.loads(pj.read_text())["clips"]}
    for r in rows:
        prev[r["id"]] = r
    allrows = [prev[k] for k in sorted(prev)]
    if sync_rep is None and pj.exists():  # keep the previous sync results when only some clips were re-run
        old = json.loads(pj.read_text())
        sync_rep, extra_syncs = old.get("sync"), old.get("extra_syncs", {})
    out = {"env": env, "clips": allrows, "sync": sync_rep, "extra_syncs": extra_syncs, "timings": timings}
    pj.write_text(json.dumps(out, indent=1))
    (RES / "part1.md").write_text(write_markdown(allrows, sync_rep, env, extra_syncs), encoding="utf-8")


if __name__ == "__main__":
    main()
