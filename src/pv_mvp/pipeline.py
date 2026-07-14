from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .detector import detect_pv
from .report import write_reports


REQUIRED_COLUMNS = {"mall_id", "name", "province", "city", "image_path"}
RESULT_COLUMNS = [
    "resolved_image_path",
    "image_path",
    "has_pv",
    "confidence",
    "pv_area_px",
    "image_area_px",
    "coverage",
    "component_count",
    "mean_panel_score",
    "reason",
    "mask_path",
    "overlay_path",
    "status",
    "error",
]


def _resolve_image_path(project_dir: Path, raw_path: str) -> Path:
    path = Path(str(raw_path).strip().strip('"'))
    if path.is_absolute():
        return path
    return project_dir / path


def run_pipeline(
    malls_csv: str | Path,
    *,
    project_dir: str | Path,
    output_dir: str | Path,
    min_pv_pixels: int = 600,
    min_coverage: float = 0.0015,
) -> pd.DataFrame:
    project_dir = Path(project_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mask_dir = output_dir / "masks"
    overlay_dir = output_dir / "overlays"

    malls_csv = Path(malls_csv)
    if not malls_csv.is_absolute():
        malls_csv = project_dir / malls_csv

    malls = pd.read_csv(malls_csv, encoding="utf-8-sig")
    missing = REQUIRED_COLUMNS - set(malls.columns)
    if missing:
        raise ValueError(f"Missing required columns in {malls_csv}: {sorted(missing)}")

    rows = []
    for _, mall in malls.iterrows():
        image_path = _resolve_image_path(project_dir, mall["image_path"])
        base = mall.to_dict()
        base["resolved_image_path"] = str(image_path)

        try:
            detection = detect_pv(
                image_path,
                min_pv_pixels=min_pv_pixels,
                min_coverage=min_coverage,
                mask_dir=mask_dir,
                overlay_dir=overlay_dir,
            )
            base.update(asdict(detection))
            base["status"] = "ok"
            base["error"] = ""
        except Exception as exc:  # Keep batch jobs inspectable.
            base.update(
                {
                    "has_pv": False,
                    "confidence": 0.0,
                    "pv_area_px": 0,
                    "image_area_px": 0,
                    "coverage": 0.0,
                    "component_count": 0,
                    "mean_panel_score": 0.0,
                    "reason": "error",
                    "mask_path": "",
                    "overlay_path": "",
                    "status": "error",
                    "error": str(exc),
                }
            )
        rows.append(base)

    if rows:
        results = pd.DataFrame(rows)
    else:
        results = malls.head(0).copy()
        for column in RESULT_COLUMNS:
            if column not in results.columns:
                results[column] = []
    results_csv = output_dir / "mall_pv_results.csv"
    results.to_csv(results_csv, index=False, encoding="utf-8-sig")

    write_reports(results, output_dir=output_dir, project_dir=project_dir)
    return results
