#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pv_mvp.detect import detect_image
from pv_mvp.io_utils import ensure_dir
from pv_mvp.report import write_review_html, write_summary


DETECTION_COLUMNS = [
    "mall_id",
    "name",
    "city",
    "address",
    "lon",
    "lat",
    "poi_status",
    "poi_source",
    "poi_name",
    "poi_address",
    "poi_score",
    "poi_reason",
    "image_status",
    "image_path",
    "imagery_source",
    "imagery_zoom",
    "crop_size_px",
    "estimated_mpp",
    "process_status",
    "process_error",
    "pv_status",
    "pv_area_px",
    "pv_ratio",
    "roof_candidate_area_px",
    "roof_candidate_ratio",
    "remaining_roof_proxy_px",
    "potential_level",
    "potential_reason",
    "mask_path",
    "pv_overlay_path",
    "roof_overlay_path",
]


def empty_detection(row: dict[str, object], status: str, error: str) -> dict[str, object]:
    result = dict(row)
    result.update(
        {
            "process_status": status,
            "process_error": error,
            "pv_status": "not_processed",
            "pv_area_px": 0,
            "pv_ratio": 0.0,
            "roof_candidate_area_px": 0,
            "roof_candidate_ratio": 0.0,
            "remaining_roof_proxy_px": 0,
            "potential_level": "not_processed",
            "potential_reason": error,
            "mask_path": "",
            "pv_overlay_path": "",
            "roof_overlay_path": "",
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2: detect suspected PV and rough roof potential.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "outputs" / "poi_resolved.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    poi_rows = pd.read_csv(args.input, encoding="utf-8-sig")
    if args.limit:
        poi_rows = poi_rows.head(args.limit)

    rows = []
    for _, row in tqdm(poi_rows.iterrows(), total=len(poi_rows), desc="detect pv + potential"):
        raw = row.to_dict()
        if row.get("image_status") != "ok":
            rows.append(empty_detection(raw, "skipped", f"image_status={row.get('image_status', '')}"))
            continue
        if not str(row.get("image_path") or "").strip():
            rows.append(empty_detection(raw, "skipped", "missing image_path"))
            continue
        try:
            detection = detect_image(
                mall_id=str(row.get("mall_id") or ""),
                name=str(row.get("name") or ""),
                image_path=str(row.get("image_path")),
                output_dir=output_dir,
            ).to_dict()
            merged = raw
            merged.update(detection)
            merged["process_status"] = "ok"
            merged["process_error"] = ""
            rows.append(merged)
        except Exception as exc:
            rows.append(empty_detection(raw, "error", str(exc)))

    results = pd.DataFrame(rows, columns=DETECTION_COLUMNS)
    output_path = output_dir / "mall_pv_potential_results.csv"
    results.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_summary(results, output_dir)
    write_review_html(results, output_dir)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
