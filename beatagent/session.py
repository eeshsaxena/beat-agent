"""Edit session: the analysed track, its edit history, per-turn undo/redo, and exports."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio_io, click
from .audio_io import Audio
from .edl import EditError, Piece, Timeline, TurnResult, apply_ops, render, splice_points
from .grid import BeatGrid


@dataclass
class HistoryEntry:
    timeline: Timeline
    grid: BeatGrid
    label: str
    turn_id: int | None = None
    ops: list[dict] = field(default_factory=list)
    echo: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class EditSession:
    def __init__(self, audio: Audio, grid: BeatGrid, name: str = "track"):
        self.audio, self.src_grid, self.name = audio, grid, name
        tl0 = Timeline((Piece(0, len(audio.samples)),))
        self.history: list[HistoryEntry] = [HistoryEntry(tl0, grid, "original")]
        self.cursor = 0
        self._cache: dict[int, np.ndarray] = {}

    # ----------------------------------------------------------------- state
    @property
    def state(self) -> HistoryEntry:
        return self.history[self.cursor]

    @property
    def grid(self) -> BeatGrid:
        return self.state.grid

    @property
    def sr(self) -> int:
        return self.audio.sr

    def can_undo(self) -> bool:
        return self.cursor > 0

    def can_redo(self) -> bool:
        return self.cursor < len(self.history) - 1

    # ----------------------------------------------------------------- edits
    def preview(self, ops: list) -> TurnResult:
        """Resolve and check ops without committing (for 'propose, then confirm')."""
        return apply_ops(self.state.timeline, self.state.grid, self.src_grid, ops, self.sr, peak_fn=self._peak)

    def apply(self, ops: list, turn_id: int | None = None) -> TurnResult:
        res = self.preview(ops)
        entry = HistoryEntry(res.timeline, res.grid,
                             "; ".join(e.split(":")[0].rstrip(".") for e in res.echo[:-1]) or "edit",
                             turn_id, [o.model_dump() for o in ops], res.echo, res.warnings)
        top = self.history[self.cursor]
        if turn_id is not None and top.turn_id == turn_id:
            # a second apply in the same chat turn joins that turn's undo step
            entry.echo = top.echo + entry.echo
            entry.ops = top.ops + entry.ops
            entry.label = top.label + "; " + entry.label
            self.history[self.cursor] = entry
            self.history = self.history[: self.cursor + 1]
        else:
            self.history = self.history[: self.cursor + 1] + [entry]
            self.cursor += 1
        return res

    def undo(self) -> str:
        if not self.can_undo():
            raise EditError("There is nothing to undo; the track is in its original state.")
        undone = self.state
        self.cursor -= 1
        return (f"Undid: {undone.label}. The track is back to {self.grid.duration_s:.3f} s, {self.grid.n} beats, "
                f"bars 1-{self.grid.last_bar}.")

    def redo(self) -> str:
        if not self.can_redo():
            raise EditError("There is nothing to redo.")
        self.cursor += 1
        return (f"Redid: {self.state.label}. The track is now {self.grid.duration_s:.3f} s, {self.grid.n} beats, "
                f"bars 1-{self.grid.last_bar}.")

    def history_text(self) -> str:
        lines = []
        for i, h in enumerate(self.history[1:], start=1):
            status = "" if i <= self.cursor else " (undone; redo available)"
            lines.append(f"  {i}. {h.label}{status}")
        return "\n".join(lines) if lines else "  (no edits yet)"

    # ------------------------------------------------------------- rendering
    def render(self, entry: HistoryEntry | None = None) -> np.ndarray:
        entry = entry or self.state
        key = id(entry.timeline)
        if key not in self._cache:
            if len(self._cache) > 6:
                self._cache.clear()
            self._cache[key] = render(entry.timeline, self.audio.samples, self.sr)
        return self._cache[key]

    def _peak(self, tl: Timeline) -> float:
        return float(np.abs(render(tl, self.audio.samples, self.sr)).max())

    def splices(self) -> list[float]:
        return [p / self.sr for p in splice_points(self.state.timeline)]

    def edl_dict(self) -> dict:
        """The edit decision list: every turn's typed ops and echo, plus the resulting timeline of
        source pieces (what plays, in order, with gain ramps), in samples and seconds."""
        sr = self.sr
        pieces, o = [], 0
        for p in self.state.timeline.pieces:
            pieces.append({
                "out_start_s": round(o / sr, 6), "out_end_s": round((o + p.length) / sr, 6),
                "src_start_s": round(p.src_start / sr, 6), "src_end_s": round(p.src_end / sr, 6),
                "src_start_sample": p.src_start, "src_end_sample": p.src_end,
                "ramps": [{"kind": r.kind, "shape": r.shape, "u0": round(r.u0, 6), "u1": round(r.u1, 6),
                           **({"db": r.db} if r.kind == "gain" else {})} for r in p.ramps],
            })
            o += p.length
        return {
            "schema": "beat-agent/edl@1",
            "source": {"file": self.audio.path.name if self.audio.path else None, "sample_rate": sr,
                       "subtype": self.audio.subtype, "duration_s": round(self.audio.duration, 6)},
            "conventions": "Ops address the grid at the start of their turn: beats 0-based half-open, bars 1-based "
                           "inclusive (see the grid JSON). Splices use a 10 ms equal-power crossfade just before the "
                           "boundary; piece lengths are exact, so the grid does not move.",
            "history": [{"step": i, "label": h.label, "ops": h.ops, "echo": h.echo, "warnings": h.warnings,
                         "active": i <= self.cursor} for i, h in enumerate(self.history[1:], start=1)],
            "timeline": pieces,
            "splices_s": [round(s, 6) for s in self.splices()],
            "duration_s": round(self.state.timeline.length / sr, 6),
        }

    def export(self, out_dir: Path, stem: str = "edited", listen_copy: bool = True) -> dict:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        y = self.render()
        audio_path = audio_io.save_like(out_dir / stem, y, self.audio)
        fmt, subtype, _ = self.audio.output_spec()
        src = {"file": audio_path.name, "edited_from": self.audio.path.name if self.audio.path else None,
               "sample_rate": self.sr, "format": fmt, "subtype": subtype}
        grid_path = out_dir / f"{stem}.grid.json"
        grid_path.write_text(json.dumps(self.grid.to_dict(src), indent=1))
        aud = click.audition(y, self.sr, self.grid)
        click_path = audio_io.save(out_dir / f"{stem}.click_audition.wav", aud, self.sr, "WAV", "PCM_16")
        edl_path = out_dir / f"{stem}.edl.json"
        edl_path.write_text(json.dumps(self.edl_dict(), indent=1))
        paths = {"audio": str(audio_path), "grid": str(grid_path), "click_audition": str(click_path),
                 "edl": str(edl_path)}
        if listen_copy:
            paths["audio_mp3"] = str(audio_io.save(out_dir / f"{stem}.mp3", y, self.sr, "MP3", "MPEG_LAYER_III"))
            paths["click_audition_mp3"] = str(audio_io.save(out_dir / f"{stem}.click_audition.mp3", aud, self.sr, "MP3",
                                                            "MPEG_LAYER_III"))
        return paths
