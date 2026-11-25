"""Beat grid data model, JSON format and indexing conventions.

Conventions (also written into every grid JSON under "conventions"):

* time         seconds from the start of the file.
* index        beat index, 0-based. "Beat 16" is index 16.
* bar          1-based. Bar 1 starts at the first downbeat. Beats before the
               first downbeat are a pickup and belong to bar 0.
* beat_in_bar  1-based; 1 is the downbeat.
* beat ranges  "beat a to beat b" is half-open [a, b): it starts at beat a and
               ends where beat b starts, so "beat 16-32" is 16 beats.
* bar ranges   "bars a-b" includes both bars, so "bars 1-4" is 4 bars.
* track end    beat position N (the number of beats) is the end of the file;
               the last beat's span runs to the end of the file.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SCHEMA = "beat-agent/grid@1"

CONVENTIONS = {
    "time": "Seconds from the start of the audio file.",
    "index": "Beat index, 0-based. 'Beat 16' means index 16.",
    "bar": "1-based. Bar 1 starts at the first downbeat; beats before it form a pickup, bar 0.",
    "beat_in_bar": "1-based position in the bar; 1 is the downbeat.",
    "beat_ranges": "'beat a to beat b' is half-open [a, b): from the start of beat a to the start of beat b.",
    "bar_ranges": "'bars a-b' is inclusive of both bars.",
    "track_end": "Beat position N (= number of beats) is the end of the file; the last beat runs to the end.",
}


@dataclass
class BeatGrid:
    times: np.ndarray  # float64 seconds, strictly increasing
    beat_in_bar: np.ndarray  # int, 1-based (1 = downbeat)
    beats_per_bar: int
    time_signature: str
    grid_type: str  # "constant" | "piecewise" | "variable"
    duration_s: float
    sample_rate: int
    confidence: float = 0.0
    confidence_detail: dict = field(default_factory=dict)
    strength: np.ndarray | None = None  # per-beat tracker support in [0, 1]
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.beat_in_bar = np.asarray(self.beat_in_bar, dtype=int)
        if self.strength is not None:
            self.strength = np.asarray(self.strength, dtype=float)

    # ------------------------------------------------------------------ basics
    @property
    def n(self) -> int:
        return len(self.times)

    @property
    def bars(self) -> np.ndarray:
        """Bar number of every beat (0 for pickup beats)."""
        return np.cumsum(self.beat_in_bar == 1)

    @property
    def downbeat_times(self) -> np.ndarray:
        return self.times[self.beat_in_bar == 1]

    @property
    def last_bar(self) -> int:
        return int(self.bars[-1]) if self.n else 0

    @property
    def pickup_beats(self) -> int:
        return int(np.sum(self.bars == 0))

    @property
    def tempo_bpm(self) -> float:
        if self.n < 2:
            return 0.0
        return float(60.0 / np.median(np.diff(self.times)))

    def beat_pos_time(self, k: int) -> float:
        """Time of beat position k in 0..N; position N is the end of the file."""
        if k == self.n:
            return self.duration_s
        return float(self.times[k])

    def bar_span(self, bar: int) -> tuple[int, int]:
        """[first beat index, one past last beat index) of a bar number."""
        idx = np.flatnonzero(self.bars == bar)
        if len(idx) == 0:
            raise IndexError(bar)
        return int(idx[0]), int(idx[-1]) + 1

    def bar_lengths(self) -> dict[int, int]:
        b = self.bars
        nums, counts = np.unique(b, return_counts=True)
        return {int(k): int(v) for k, v in zip(nums, counts)}

    # --------------------------------------------------------------- segments
    def tempo_segments(self, tol: float = 0.012, min_beats: int = 8) -> list[dict]:
        """Runs of near-constant tempo, measured from the beat times.

        A new segment starts where the inter-beat interval moves more than `tol`
        (relative) away from the running segment's median for 3+ beats in a row.
        """
        if self.n < 3:
            return [self._seg(0, self.n)]
        ibi = np.diff(self.times)
        segs, start = [], 0
        i = 1
        while i < len(ibi):
            ref = np.median(ibi[start:i])
            run = ibi[i:i + 3]
            if len(run) == 3 and np.all(np.abs(run / ref - 1) > tol) and np.all(np.sign(run - ref) == np.sign(run[0] - ref)) \
                    and i - start >= min_beats:
                segs.append((start, i))
                start = i
                i += 3
                continue
            i += 1
        segs.append((start, len(ibi)))
        out = []
        for s, e in segs:
            out.append(self._seg(s, e + 1 if e == len(ibi) else e))
        # the last segment owns the final beat
        out[-1]["end_index"] = self.n
        return out

    def _seg(self, s: int, e: int) -> dict:
        ibi = np.diff(self.times[s:e]) if e - s > 1 else np.array([np.nan])
        bars = self.bars
        return {
            "start_index": int(s), "end_index": int(e),
            "bpm": round(float(60 / np.median(ibi)), 3) if np.isfinite(np.median(ibi)) else None,
            "bpm_min": round(float(60 / np.max(ibi)), 2) if len(ibi) and np.isfinite(ibi).all() else None,
            "bpm_max": round(float(60 / np.min(ibi)), 2) if len(ibi) and np.isfinite(ibi).all() else None,
            "start_s": round(float(self.times[s]), 4) if self.n else 0.0,
            "end_s": round(self.beat_pos_time(e) if e <= self.n else self.duration_s, 4),
            "first_bar": int(bars[s]) if self.n else 0, "last_bar": int(bars[e - 1]) if self.n else 0,
        }

    # ------------------------------------------------------------------- JSON
    def to_dict(self, source: dict | None = None) -> dict:
        bars = self.bars
        beats = [
            {"index": i, "time": round(float(t), 6), "bar": int(bars[i]), "beat_in_bar": int(self.beat_in_bar[i]),
             **({"strength": round(float(self.strength[i]), 3)} if self.strength is not None else {})}
            for i, t in enumerate(self.times)
        ]
        return {
            "schema": SCHEMA,
            "source": source or {},
            "conventions": CONVENTIONS,
            "tempo_bpm": round(self.tempo_bpm, 3),
            "time_signature": self.time_signature,
            "beats_per_bar": self.beats_per_bar,
            "grid_type": self.grid_type,
            "segments": self.tempo_segments(),
            "confidence": round(float(self.confidence), 3),
            "confidence_detail": self.confidence_detail,
            "n_beats": self.n,
            "n_bars": self.last_bar,
            "pickup_beats": self.pickup_beats,
            "duration_s": round(self.duration_s, 4),
            "sample_rate": self.sample_rate,
            "downbeats": [round(float(t), 6) for t in self.downbeat_times],
            "beats": beats,
            "analysis": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BeatGrid":
        beats = d["beats"]
        strength = [b["strength"] for b in beats] if beats and "strength" in beats[0] else None
        return cls(
            times=np.array([b["time"] for b in beats]), beat_in_bar=np.array([b["beat_in_bar"] for b in beats]),
            beats_per_bar=d["beats_per_bar"], time_signature=d["time_signature"], grid_type=d["grid_type"],
            duration_s=d["duration_s"], sample_rate=d["sample_rate"], confidence=d.get("confidence", 0.0),
            confidence_detail=d.get("confidence_detail", {}), strength=strength, meta=d.get("analysis", {}),
        )

    # ------------------------------------------------------------- LLM summary
    def summary(self) -> str:
        """Compact, factual description of the grid for the LLM's context."""
        if self.n == 0:
            return "The track has no beat grid (no beats detected)."
        segs = self.tempo_segments()
        lines = [
            f"Duration {self.duration_s:.3f} s. {self.n} beats (beat indices 0..{self.n - 1}; "
            f"beat position {self.n} = end of track). Time signature {self.time_signature}, "
            f"{self.beats_per_bar} beats per bar. Grid type: {self.grid_type}.",
            f"Bars 1..{self.last_bar}" + (f", plus a {self.pickup_beats}-beat pickup (bar 0, beats 0..{self.pickup_beats - 1})"
                                          if self.pickup_beats else "") + ".",
            f"Beat 0 is at {self.times[0]:.3f} s; audio before it is a pre-roll that beat ranges never touch.",
        ]
        if len(segs) == 1:
            s = segs[0]
            lines.append(f"Tempo {s['bpm']:.2f} BPM throughout" + (
                f" (local range {s['bpm_min']:.1f}-{s['bpm_max']:.1f})" if self.grid_type == "variable" else "") + ".")
        else:
            lines.append("Tempo segments:")
            for i, s in enumerate(segs):
                lines.append(f"  segment {i + 1}: beats {s['start_index']}-{s['end_index'] - 1}, bars {s['first_bar']}-"
                             f"{s['last_bar']}, {s['bpm']:.2f} BPM, {s['start_s']:.3f}-{s['end_s']:.3f} s")
            for i in range(1, len(segs)):
                a, b = segs[i - 1], segs[i]
                lines.append(f"  tempo change {i}: {a['bpm']:.2f} -> {b['bpm']:.2f} BPM at beat {b['start_index']} "
                             f"(bar {b['first_bar']}, {b['start_s']:.3f} s); last full bar before it is bar "
                             f"{self._last_full_bar_before(b['start_index'])}")
        odd = {k: v for k, v in self.bar_lengths().items() if k > 0 and v != self.beats_per_bar}
        if odd:
            shown = ", ".join(f"bar {k} has {v} beats" for k, v in list(odd.items())[:8])
            lines.append(f"Irregular bars: {shown}" + (" ..." if len(odd) > 8 else "") + ".")
        lines.append(f"Grid confidence {self.confidence:.2f}.")
        return "\n".join(lines)

    def _last_full_bar_before(self, beat_index: int) -> int | None:
        bars = self.bars
        lens = self.bar_lengths()
        for bar in range(int(bars[max(beat_index - 1, 0)]), 0, -1):
            s, e = self.bar_span(bar)
            if e <= beat_index and lens[bar] == self.beats_per_bar:
                return bar
        return None
