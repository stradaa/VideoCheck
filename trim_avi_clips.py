#!/usr/bin/env python3
"""
Trim the first N seconds from every .avi in a folder and save the clipped files.

Requirements:
- ffmpeg must be installed and available on PATH.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save the first part of each .avi video to a destination folder."
    )
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Folder containing .avi files to clip.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Folder where clipped videos will be written.",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=20,
        help="Seconds to keep from the start of each video (default: 20).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping.",
    )
    return parser.parse_args()


def trim_video(src: Path, dst: Path, duration: int, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        str(src),
        "-t",
        str(duration),
        "-c",
        "copy",
        str(dst),
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {src}:\n{result.stderr.strip()}")


def main() -> int:
    args = parse_args()
    input_dir: Path = args.input_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        print(f"Input directory not found: {input_dir}", file=sys.stderr)
        return 1

    avi_files = sorted(p for p in input_dir.glob("*.avi") if p.is_file())
    if not avi_files:
        print(f"No .avi files found in {input_dir}")
        return 0

    for src in avi_files:
        dst = output_dir / src.name
        try:
            trim_video(src, dst, args.duration, args.overwrite)
            print(f"Saved first {args.duration}s of {src.name} -> {dst}")
        except Exception as exc:
            print(exc, file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
