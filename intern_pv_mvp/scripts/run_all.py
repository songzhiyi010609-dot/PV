#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run both MVP steps.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "malls_sample.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--crop-size-px", type=int, default=768)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd))
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    python = sys.executable
    step1 = [
        python,
        str(PROJECT_ROOT / "scripts" / "01_resolve_poi_and_fetch_imagery.py"),
        "--input",
        str(args.input),
        "--output-dir",
        str(args.output_dir),
        "--zoom",
        str(args.zoom),
        "--crop-size-px",
        str(args.crop_size_px),
    ]
    if args.limit:
        step1.extend(["--limit", str(args.limit)])
    run(step1)

    step2 = [
        python,
        str(PROJECT_ROOT / "scripts" / "02_detect_pv_and_potential.py"),
        "--input",
        str(args.output_dir / "poi_resolved.csv"),
        "--output-dir",
        str(args.output_dir),
    ]
    if args.limit:
        step2.extend(["--limit", str(args.limit)])
    run(step2)


if __name__ == "__main__":
    main()
