"""Part 3 evals: audio -> score -> self-contained Web Audio page -> offline render -> measurements.

    python -m evals.run_part3 [--only P1,P2] [--no-separation]

Per clip: score.json, page.html, render.wav (2 loops, rendered in headless Chromium with an
OfflineAudioContext) and metrics:
  * note F-measure vs the reference MIDI, onset only, +-50 ms (mir_eval.transcription);
    drums per class and class-agnostic, pitched parts merged and per role;
  * onset timing of the rendered audio vs the original (same detector on both);
  * the page made no network requests and raised no errors.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mir_eval
import numpy as np

from beatagent import audio_io, onsets, recreate
from beatagent import score as S
from beatagent.webaudio.render import render_page
from evals import pack

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evals" / "out" / "part3"
RES = ROOT / "evals" / "results"
DRUM_CLASS = {35: 36, 36: 36, 37: 38, 38: 38, 39: 38, 40: 38, 42: 42, 44: 42, 46: 46, 49: 49, 51: 42, 57: 49}
NAMES = {36: "kick", 38: "snare", 42: "closed hat", 46: "open hat", 49: "crash"}


def _f(ref_t, est_t, win=0.05):
    if len(ref_t) == 0 and len(est_t) == 0:
        return None
    if len(ref_t) == 0 or len(est_t) == 0:
        return {"f": 0.0, "p": 0.0 if len(est_t) else None, "r": 0.0 if len(ref_t) else None, "ref": len(ref_t), "est": len(est_t)}
    f, p, r = mir_eval.onset.f_measure(np.sort(ref_t) + 0.1, np.sort(est_t) + 0.1, window=win)  # +0.1 s: mir_eval rejects negative times
    return {"f": round(f, 4), "p": round(p, 4), "r": round(r, 4), "ref": len(ref_t), "est": len(est_t)}


def note_f(ref: list[tuple], est: list[tuple], win=0.05):
    """ref/est: (onset_s, offset_s, midi). Onset-only note F (pitch within 50 cents)."""
    if not ref or not est:
        return {"f": 0.0, "p": 0.0, "r": 0.0, "ref": len(ref), "est": len(est)}
    ri = np.array([[a, max(b, a + 0.01)] for a, b, _ in ref]) + 0.1  # same constant shift for ref and est
    rp = mir_eval.util.midi_to_hz(np.array([m for *_, m in ref], float))
    ei = np.array([[a, max(b, a + 0.01)] for a, b, _ in est]) + 0.1
    ep = mir_eval.util.midi_to_hz(np.array([m for *_, m in est], float))
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, onset_tolerance=win, offset_ratio=None)
    return {"f": round(f, 4), "p": round(p, 4), "r": round(r, 4), "ref": len(ref), "est": len(est)}


def _wrap_window(t: np.ndarray, loop: float, lo: float = -0.05) -> np.ndarray:
    return ((t - lo) % loop) + lo


def eval_clip(cid: str, use_sep: bool) -> dict:
    a = audio_io.load(pack.audio_path(cid))
    d = OUT / cid
    d.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    sc = recreate.build_score(a, cid, use_separation=use_sep)
    build_s = time.perf_counter() - t
    (d / "score.json").write_text(json.dumps(sc, indent=1), encoding="utf-8")
    page = S.build_page(sc, d / "page.html")
    t = time.perf_counter()
    rr = render_page(page, d / "render.wav", loops=2, sample_rate=a.sr)
    render_s = time.perf_counter() - t
    loop_s = sc["loop_beats"] * 60 / sc["tempo_bpm"]
    row = {"id": cid, "tempo_bpm": sc["tempo_bpm"], "bars": sc["bars"], "exact_loop": sc["source"]["exact_loop"],
           "first_downbeat_s": sc["source"]["first_downbeat_s"], "loop_s": round(loop_s, 4),
           "tracks": {tr["id"]: len(tr["notes"]) for tr in sc["tracks"]}, "build_s": round(build_s, 1),
           "render_s": round(render_s, 1), "render": rr, "stems_db": sc["provenance"].get("stem_level_db_vs_mix")}

    ref = pack.ref_midi(cid)
    est_all = S.to_seconds(sc)
    if ref is not None:
        r_drums = [(n.start, DRUM_CLASS.get(n.pitch, n.pitch)) for i in ref.instruments if i.is_drum for n in i.notes]
        r_pitch = [(n.start, n.end, n.pitch) for i in ref.instruments if not i.is_drum for n in i.notes]
        e_drums = [(s, DRUM_CLASS.get(p, p)) for tid, p, s, e, v in est_all if tid == "drums"]
        e_pitch = [(s, e, p) for tid, p, s, e, v in est_all if tid != "drums"]
        e_drums = [(float(_wrap_window(np.array([s]), loop_s)[0]), c) for s, c in e_drums]
        e_pitch = [(float(_wrap_window(np.array([s]), loop_s)[0]), e - s + float(_wrap_window(np.array([s]), loop_s)[0]), p)
                   for s, e, p in e_pitch]
        m = {}
        if r_drums:
            m["drums_any_class"] = _f(np.array([s for s, _ in r_drums]), np.array([s for s, _ in e_drums]))
            m["drums_note_f"] = note_f([(s, s + 0.1, c) for s, c in r_drums], [(s, s + 0.1, c) for s, c in e_drums])
            m["drums_per_class"] = {NAMES.get(c, str(c)): _f(np.array([s for s, k in r_drums if k == c]),
                                                              np.array([s for s, k in e_drums if k == c]))
                                    for c in sorted({c for _, c in r_drums} | {c for _, c in e_drums})}
        if r_pitch:
            m["pitched_note_f"] = note_f(r_pitch, e_pitch)
            by_inst = {i.name: [(n.start, n.end, n.pitch) for n in i.notes] for i in ref.instruments if not i.is_drum}
            if "bass" in by_inst:
                m["bass_note_f"] = note_f(by_inst["bass"], [(s, e, p) for (s, e, p), (tid, *_ ) in
                                                             zip(e_pitch, [x for x in est_all if x[0] != "drums"]) if tid == "bass"])
        m["all_notes_note_f"] = note_f([(s, s + 0.1, c) for s, c in r_drums] + r_pitch,
                                       [(s, s + 0.1, c) for s, c in e_drums] + e_pitch)
        # timing of the score itself: matched notes (same class/pitch, within 50 ms) vs the reference MIDI
        refs = [(s, c, True) for s, c in r_drums] + [(s, p, False) for s, _, p in r_pitch]
        ests = [(s, c, True) for s, c in e_drums] + [(s, p, False) for s, _, p in e_pitch]
        dts = []
        for t, p, dr in refs:
            cand = [te - t for te, pe, de in ests if pe == p and de == dr and abs(te - t) <= 0.05]
            if cand:
                dts.append(min(cand, key=abs) * 1000)
        if dts:
            dts = np.array(dts)
            m["score_timing_vs_reference_ms"] = {"matched": len(dts), "of": len(refs),
                                                 "median_abs": round(float(np.median(np.abs(dts))), 2),
                                                 "mean_signed": round(float(dts.mean()), 2),
                                                 "p90_abs": round(float(np.percentile(np.abs(dts), 90)), 2)}
        row["reference"] = m
    # onset timing: rendered audio (first loop) vs the original clip, same detector on both
    rend = audio_io.load(d / "render.wav")
    t0 = sc["source"]["first_downbeat_s"]
    orig_on = onsets.detect(a.samples, a.sr, rel_threshold=0.15)
    ren_on = onsets.detect(rend.samples[: int((loop_s + 0.05) * rend.sr)], rend.sr, rel_threshold=0.15) + t0
    ren_on = ren_on[ren_on < a.duration]
    off = onsets.match(orig_on, ren_on, 0.05) * 1000
    ok = ~np.isnan(off)
    row["render_onset_timing"] = {**(_f(orig_on, ren_on) or {}),
                                  "median_abs_offset_ms": round(float(np.median(np.abs(off[ok]))), 2) if ok.any() else None,
                                  "mean_signed_offset_ms": round(float(np.mean(off[ok])), 2) if ok.any() else None}
    (d / "metrics.json").write_text(json.dumps(row, indent=1, default=str), encoding="utf-8")
    return row


def write_markdown(rows: list[dict]) -> str:
    L = ["# Part 3 results: audio to playable Web Audio", "",
         "Note F is onset-only, ±50 ms, via mir_eval.transcription against the reference MIDI (P4 has none). "
         "Render timing compares onsets detected in the page's offline render with onsets in the original clip.", "",
         "| Clip | Tempo | Bars | Tracks (notes) | Drums note F | Drums any-class F | Pitched note F | All notes F | Score timing vs MIDI (median ms) | Render onset F | Render median offset ms | Net requests |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        m = r.get("reference", {})
        g = lambda k: (m.get(k) or {}).get("f", "-") if m else "-"  # noqa: E731
        ro = r["render_onset_timing"]
        L.append(f"| {r['id']} | {r['tempo_bpm']:.2f} | {r['bars']} | "
                 f"{', '.join(f'{k} ({v})' for k, v in r['tracks'].items())} | {g('drums_note_f')} | {g('drums_any_class')} | "
                 f"{g('pitched_note_f')} | {g('all_notes_note_f')} | "
                 f"{(m.get('score_timing_vs_reference_ms') or {}).get('median_abs', '-') if m else '-'} | "
                 f"{ro.get('f', '-')} | {ro.get('median_abs_offset_ms', '-')} | "
                 f"{len(r['render']['network_requests'])} |")
    L += ["", "Score timing compares the transcribed note times with the reference MIDI (matched notes only), so it "
          "measures the analysis. Render timing compares onsets detected in the page's audio with onsets in the "
          "original, so it also includes the synth voices' attack shapes, which differ from the original sounds "
          "(the ~10 ms scatter there has a mean near 0).", "", "Per-class drum results:", ""]
    for r in rows:
        pc = (r.get("reference") or {}).get("drums_per_class")
        if pc:
            L.append(f"* {r['id']}: " + "; ".join(f"{k} F {v['f']} (ref {v['ref']}, est {v['est']})"
                                                for k, v in pc.items() if v))
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-separation", action="store_true")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ids = [s for s in args.only.split(",") if s] or pack.ids("P")
    rows = []
    for cid in ids:
        r = eval_clip(cid, not args.no_separation)
        rows.append(r)
        m = r.get("reference", {})
        print(f"{cid}: tempo {r['tempo_bpm']:.2f}, tracks {r['tracks']}, drums F {(m.get('drums_note_f') or {}).get('f')}, "
              f"pitched F {(m.get('pitched_note_f') or {}).get('f')}, render onset F {r['render_onset_timing'].get('f')}, "
              f"net {len(r['render']['network_requests'])}, errors {len(r['render']['page_errors'])}", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    prev = {}
    pj = RES / "part3.json"
    if pj.exists() and args.only:
        prev = {r["id"]: r for r in json.loads(pj.read_text())}
    for r in rows:
        prev[r["id"]] = r
    allrows = [prev[k] for k in sorted(prev)]
    pj.write_text(json.dumps(allrows, indent=1, default=str), encoding="utf-8")
    (RES / "part3.md").write_text(write_markdown(allrows), encoding="utf-8")


if __name__ == "__main__":
    main()
