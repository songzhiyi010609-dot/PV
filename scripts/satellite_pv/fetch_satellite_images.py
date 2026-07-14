from __future__ import annotations

import argparse
import csv
import math
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "satellite_experiment"
DEFAULT_INPUT = EXPERIMENT_ROOT / "data" / "shanghai_geocoded_sample.csv"
DEFAULT_IMAGE_DIR = EXPERIMENT_ROOT / "images"
DEFAULT_INDEX = EXPERIMENT_ROOT / "data" / "shanghai_imagery_index.csv"
DEFAULT_CACHE_DIR = EXPERIMENT_ROOT / "tile_cache" / "esri_world_imagery"
ESRI_TILE_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/tile/{z}/{y}/{x}"
)
TILE_SIZE = 256
MAX_MERCATOR_LAT = 85.05112878


def iter_limited(rows: Iterable[dict[str, str]], limit: int | None) -> Iterable[dict[str, str]]:
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        yield row


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().strip(".")
    return cleaned[:120] or "mall"


def lonlat_to_global_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
    scale = TILE_SIZE * (2**zoom)
    x = (lon + 180.0) / 360.0 * scale
    lat_rad = math.radians(lat)
    y = (
        0.5
        - math.log((1 + math.sin(lat_rad)) / (1 - math.sin(lat_rad))) / (4 * math.pi)
    ) * scale
    return x, y


def read_tile(z: int, x: int, y: int, cache_dir: Path, timeout: int) -> Image.Image:
    wrapped_x = x % (2**z)
    cache_path = cache_dir / str(z) / str(y) / f"{wrapped_x}.jpg"
    if cache_path.exists():
        return Image.open(cache_path).convert("RGB")

    url = ESRI_TILE_URL.format(z=z, x=wrapped_x, y=y)
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "pv-coverage-satellite-experiment/0.1"},
    )
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(cache_path, quality=92)
    return image


def build_crop(
    lon: float,
    lat: float,
    zoom: int,
    size: int,
    cache_dir: Path,
    timeout: int,
) -> Image.Image:
    center_x, center_y = lonlat_to_global_pixel(lon, lat, zoom)
    half = size / 2
    left = center_x - half
    top = center_y - half
    right = center_x + half
    bottom = center_y + half

    min_tile_x = math.floor(left / TILE_SIZE)
    max_tile_x = math.floor((right - 1) / TILE_SIZE)
    min_tile_y = math.floor(top / TILE_SIZE)
    max_tile_y = math.floor((bottom - 1) / TILE_SIZE)

    max_tile_index = (2**zoom) - 1
    min_tile_y = max(0, min_tile_y)
    max_tile_y = min(max_tile_index, max_tile_y)

    mosaic_width = (max_tile_x - min_tile_x + 1) * TILE_SIZE
    mosaic_height = (max_tile_y - min_tile_y + 1) * TILE_SIZE
    mosaic = Image.new("RGB", (mosaic_width, mosaic_height))

    for tile_y in range(min_tile_y, max_tile_y + 1):
        for tile_x in range(min_tile_x, max_tile_x + 1):
            tile = read_tile(zoom, tile_x, tile_y, cache_dir, timeout)
            px = (tile_x - min_tile_x) * TILE_SIZE
            py = (tile_y - min_tile_y) * TILE_SIZE
            mosaic.paste(tile, (px, py))

    crop_left = round(left - min_tile_x * TILE_SIZE)
    crop_top = round(top - min_tile_y * TILE_SIZE)
    return mosaic.crop((crop_left, crop_top, crop_left + size, crop_top + size))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Esri World Imagery crops around geocoded mall points."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    args.image_dir.mkdir(parents=True, exist_ok=True)
    args.index.parent.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str]] = []
    with args.input.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {args.input}")

        for row in iter_limited(reader, args.limit):
            status = "ok"
            image_path = ""
            error = ""
            try:
                lon = float(row.get("longitude") or "")
                lat = float(row.get("latitude") or "")
                name = row.get("name") or "mall"
                mall_id = row.get("id") or "unknown"
                filename = safe_filename(f"{mall_id}_{name}") + ".jpg"
                output_path = args.image_dir / filename

                if args.overwrite or not output_path.exists():
                    image = build_crop(
                        lon=lon,
                        lat=lat,
                        zoom=args.zoom,
                        size=args.size,
                        cache_dir=args.cache_dir,
                        timeout=args.timeout,
                    )
                    image.save(output_path, quality=95)

                image_path = str(output_path)
            except Exception as exc:
                status = "error"
                error = repr(exc)

            out_row = dict(row)
            out_row.update(
                {
                    "image_status": status,
                    "image_path": image_path,
                    "image_error": error,
                    "imagery_provider": "esri_world_imagery",
                    "imagery_zoom": str(args.zoom),
                    "imagery_size_px": str(args.size),
                }
            )
            index_rows.append(out_row)

            if args.delay > 0:
                time.sleep(args.delay)

    fieldnames = list(index_rows[0].keys()) if index_rows else []
    with args.index.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(index_rows)

    ok_count = sum(1 for row in index_rows if row["image_status"] == "ok")
    print(f"input={args.input}")
    print(f"image_dir={args.image_dir}")
    print(f"index={args.index}")
    print(f"rows={len(index_rows)}")
    print(f"images={ok_count}")


if __name__ == "__main__":
    main()
