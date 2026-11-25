"""Audio loading and saving that preserves sample rate and bit depth."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

# Containers we can write losslessly in the same format as the input.
_LOSSLESS = {"WAV", "FLAC", "AIFF"}


@dataclass
class Audio:
    samples: np.ndarray  # float32, shape (n_samples, n_channels)
    sr: int
    subtype: str  # soundfile subtype of the source, e.g. PCM_16, PCM_24, FLOAT
    format: str  # soundfile container format of the source, e.g. WAV, FLAC, MP3
    path: Path | None = None

    @property
    def duration(self) -> float:
        return len(self.samples) / self.sr

    @property
    def channels(self) -> int:
        return self.samples.shape[1]

    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=1)

    def output_spec(self) -> tuple[str, str, str]:
        """(container, subtype, extension) used when writing edits of this audio.

        Lossless inputs keep their container and subtype. MP3 has no bit depth, so
        edits of an MP3 are written as 16-bit WAV (documented in DECISIONS.md).
        """
        if self.format in _LOSSLESS:
            return self.format, self.subtype, {"WAV": "wav", "FLAC": "flac", "AIFF": "aiff"}[self.format]
        return "WAV", "PCM_16", "wav"


def load(path: str | Path) -> Audio:
    path = Path(path)
    info = sf.info(str(path))
    # float32 holds 16- and 24-bit PCM exactly, so untouched samples round-trip bit-exactly.
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return Audio(x, sr, info.subtype, info.format, path)


def save(path: str | Path, samples: np.ndarray, sr: int, fmt: str = "WAV", subtype: str = "PCM_16") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = np.asarray(samples, dtype=np.float32)
    if subtype.startswith("PCM"):
        x = np.clip(x, -1.0, 1.0)
    if fmt == "MP3":  # listening copies only: ~90 kbps VBR keeps the repo small; masters stay WAV/FLAC
        sf.write(str(path), np.clip(x, -1.0, 1.0), sr, format=fmt, subtype=subtype, compression_level=0.85,
                 bitrate_mode="VARIABLE")
        return path
    sf.write(str(path), x, sr, format=fmt, subtype=subtype)
    return path


def save_like(path: str | Path, samples: np.ndarray, like: Audio) -> Path:
    """Write `samples` with the same sample rate and bit depth as `like`."""
    fmt, subtype, ext = like.output_spec()
    path = Path(path).with_suffix("." + ext)
    return save(path, samples, like.sr, fmt, subtype)


def resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    import soxr

    return soxr.resample(x, sr_in, sr_out, quality="VHQ").astype(np.float32)
