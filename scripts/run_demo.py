from __future__ import annotations

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pv_mvp.pipeline import run_pipeline
from pv_mvp.synthetic import create_demo_dataset


def main() -> None:
    malls_csv = create_demo_dataset(PROJECT_DIR)
    results = run_pipeline(
        malls_csv,
        project_dir=PROJECT_DIR,
        output_dir=PROJECT_DIR / "outputs",
        min_pv_pixels=600,
        min_coverage=0.0015,
    )
    suspected = int(results["has_pv"].sum())
    total = len(results)
    print(f"Demo complete: {suspected}/{total} malls suspected with PV.")
    print(f"Results: {PROJECT_DIR / 'outputs' / 'mall_pv_results.csv'}")
    print(f"Review:  {PROJECT_DIR / 'outputs' / 'review.html'}")


if __name__ == "__main__":
    main()
