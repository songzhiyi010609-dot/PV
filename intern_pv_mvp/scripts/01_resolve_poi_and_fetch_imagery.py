#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from pv_mvp.imagery import fetch_mall_image
from pv_mvp.io_utils import ensure_dir
from pv_mvp.poi import resolve_row


OUTPUT_COLUMNS = [
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 1: resolve mall POI and fetch satellite imagery.")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data" / "malls_sample.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--crop-size-px", type=int, default=768)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    image_dir = ensure_dir(output_dir / "imagery")
    cache_dir = ensure_dir(output_dir / "tile_cache" / "esri_world_imagery")

    malls = pd.read_csv(args.input, encoding="utf-8-sig")
    if args.limit:
        malls = malls.head(args.limit)

    rows = []
    for _, mall in tqdm(malls.iterrows(), total=len(malls), desc="resolve poi + imagery"):
        poi = resolve_row(mall.to_dict(), timeout=args.timeout, delay=args.delay).to_dict()
        image_path = ""
        image_status = "skipped"
        estimated_mpp = ""
        if poi["poi_status"] == "ok" and poi["lon"] is not None and poi["lat"] is not None:
            try:
                image_path, mpp = fetch_mall_image(
                    mall_id=str(poi["mall_id"]),
                    name=str(poi["name"]),
                    lon=float(poi["lon"]),
                    lat=float(poi["lat"]),
                    output_dir=image_dir,
                    cache_dir=cache_dir,
                    zoom=args.zoom,
                    crop_size_px=args.crop_size_px,
                    timeout=args.timeout,
                    delay=args.delay,
                )
                image_status = "ok"
                estimated_mpp = round(mpp, 3)
            except Exception as exc:
                image_status = f"error: {exc}"
        poi.update(
            {
                "image_status": image_status,
                "image_path": image_path,
                "imagery_source": "esri_world_imagery",
                "imagery_zoom": args.zoom,
                "crop_size_px": args.crop_size_px,
                "estimated_mpp": estimated_mpp,
            }
        )
        rows.append(poi)

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    output_path = output_dir / "poi_resolved.csv"
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
