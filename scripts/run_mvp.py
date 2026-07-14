from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pv_mvp.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run mall-level PV coverage MVP.")
    parser.add_argument("--malls", default="data/malls.csv", help="Mall CSV path.")
    parser.add_argument("--output", default="outputs", help="Output directory.")
    parser.add_argument("--min-pv-pixels", type=int, default=600, help="Minimum suspected PV pixels.")
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0015,
        help="Minimum suspected PV area ratio in the image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = run_pipeline(
        args.malls,
        project_dir=PROJECT_DIR,
        output_dir=PROJECT_DIR / args.output,
        min_pv_pixels=args.min_pv_pixels,
        min_coverage=args.min_coverage,
    )
    suspected = int(results["has_pv"].sum())
    total = len(results)
    print(f"Complete: {suspected}/{total} malls suspected with PV.")
    print(f"Results: {PROJECT_DIR / args.output / 'mall_pv_results.csv'}")
    print(f"Review:  {PROJECT_DIR / args.output / 'review.html'}")


if __name__ == "__main__":
    main()
