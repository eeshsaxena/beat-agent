"""Package the full-resolution eval artefacts for a GitHub release.

    python scripts/package_artifacts.py

The repository carries every grid, EDL, report, score and page, plus MP3 listening copies
of every audio file. Full-resolution audio (edited files at the input's sample rate and bit
depth, click auditions, sync stems) is too large for git: a 24-bit 6-minute edit is ~97 MB.
This script zips it, together with the stand-in test pack, for upload as release assets:

    gh release create v1.0 dist/*.zip --notes "Full-resolution eval artefacts"

Everything in these zips is also regenerated deterministically by `make evals`.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
AUDIO = {".wav", ".flac", ".mp3"}


def pack(name: str, files: list[Path]) -> Path:
    DIST.mkdir(exist_ok=True)
    out = DIST / name
    with zipfile.ZipFile(out, "w") as z:
        for f in sorted(files):
            # audio barely compresses; store it, deflate the rest
            z.write(f, f.relative_to(ROOT).as_posix(),
                    compress_type=zipfile.ZIP_STORED if f.suffix.lower() in AUDIO else zipfile.ZIP_DEFLATED)
    print(f"{out.relative_to(ROOT)}: {len(files)} files, {out.stat().st_size / 1e6:.0f} MB")
    return out


def keep(p: Path) -> bool:
    """Full-resolution audio worth shipping (GitHub release files must stay under 2 GB):
    * edited audio, sync stems and Part 3 renders at full resolution;
    * not the click-audition WAVs: their MP3 copies are in the repo, and they are only for listening;
    * not evals/out/part2/offline: those edits are byte-identical to evals/out/part2/reference
      (checked with sha256), so one copy is enough."""
    rel = p.relative_to(ROOT / "evals" / "out").parts
    if p.suffix.lower() not in {".wav", ".flac"} or "click_audition" in p.name:
        return False
    return not (rel[0] == "part2" and rel[1] == "offline")


def main():
    evals = [p for p in (ROOT / "evals" / "out").rglob("*") if p.is_file() and keep(p)]
    pack("beat-agent-eval-audio-fullres.zip", evals + list((ROOT / "evals" / "results").glob("*")))
    standin = [p for p in (ROOT / "testpack" / "stand-in").glob("*") if p.is_file()]
    pack("beat-agent-standin-testpack.zip", standin)


if __name__ == "__main__":
    main()
