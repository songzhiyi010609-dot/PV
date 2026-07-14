from __future__ import annotations

import argparse
import csv
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "satellite_experiment"
FULL_ROOT = EXPERIMENT_ROOT / "full_shanghai_dataset"
DEFAULT_INPUT = REPO_ROOT / "raw" / "\u4e0a\u6d77\u5e02.csv"
DEFAULT_GEOCODED = FULL_ROOT / "data" / "shanghai_geocoded.csv"
DEFAULT_IMAGE_INDEX = FULL_ROOT / "data" / "shanghai_imagery_index.csv"
DEFAULT_IMAGE_DIR = FULL_ROOT / "images"
ARCGIS_GEOCODE_URL = (
    "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/"
    "findAddressCandidates"
)
ESRI_EXPORT_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/export"
)
WEB_MERCATOR_RADIUS = 6378137.0


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().strip(".")
    return cleaned[:120] or "mall"


def iter_limited(rows: list[dict[str, str]], limit: int | None) -> Iterable[dict[str, str]]:
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        yield row


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def build_query(row: dict[str, str]) -> str:
    location = (row.get("location") or "").strip()
    city = (row.get("city") or "").strip()
    name = (row.get("name") or "").strip()
    return location or f"{city}{name}"


def geocode_one(query: str, timeout: int) -> dict[str, str]:
    params = {
        "SingleLine": query,
        "f": "json",
        "outFields": "Match_addr,Addr_type,PlaceName,City,Region",
        "maxLocations": 1,
        "sourceCountry": "CHN",
    }
    response = requests.get(ARCGIS_GEOCODE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    candidates = payload.get("candidates") or []
    if not candidates:
        return {
            "longitude": "",
            "latitude": "",
            "geocode_score": "",
            "geocode_address": "",
            "geocode_status": "not_found",
            "geocode_error": "",
        }

    candidate = candidates[0]
    location = candidate.get("location") or {}
    return {
        "longitude": str(location.get("x", "")),
        "latitude": str(location.get("y", "")),
        "geocode_score": str(candidate.get("score", "")),
        "geocode_address": str(candidate.get("address", "")),
        "geocode_status": "ok",
        "geocode_error": "",
    }


def run_geocoding(args: argparse.Namespace) -> None:
    input_fields, input_rows = read_csv(args.input)
    fields = input_fields + [
        "geocode_query",
        "longitude",
        "latitude",
        "geocode_score",
        "geocode_address",
        "geocode_status",
        "geocode_error",
        "geocode_provider",
    ]

    existing_by_id: dict[str, dict[str, str]] = {}
    if args.geocoded.exists() and not args.overwrite:
        _, existing_rows = read_csv(args.geocoded)
        existing_by_id = {row["id"]: row for row in existing_rows}

    output_rows: list[dict[str, str]] = []
    rows_to_process = list(iter_limited(input_rows, args.limit))
    total = len(rows_to_process)

    for index, source_row in enumerate(rows_to_process, start=1):
        mall_id = source_row["id"]
        existing = existing_by_id.get(mall_id)
        if existing and existing.get("geocode_status") == "ok":
            output_rows.append(existing)
            continue

        out_row = dict(source_row)
        query = build_query(source_row)
        try:
            result = geocode_one(query, timeout=args.timeout)
        except Exception as exc:
            result = {
                "longitude": "",
                "latitude": "",
                "geocode_score": "",
                "geocode_address": "",
                "geocode_status": "error",
                "geocode_error": repr(exc),
            }
        out_row.update(result)
        out_row["geocode_query"] = query
        out_row["geocode_provider"] = "arcgis_world_geocoding"
        output_rows.append(out_row)

        if index % args.checkpoint_every == 0:
            write_csv(args.geocoded, fields, output_rows)
            print(f"geocode_checkpoint={index}/{total}")
        if args.delay > 0:
            time.sleep(args.delay)

    write_csv(args.geocoded, fields, output_rows)
    ok_count = sum(1 for row in output_rows if row.get("geocode_status") == "ok")
    print(f"geocoded_output={args.geocoded}")
    print(f"geocode_rows={len(output_rows)}")
    print(f"geocode_ok={ok_count}")


def lonlat_to_web_mercator(lon: float, lat: float) -> tuple[float, float]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    x = WEB_MERCATOR_RADIUS * math.radians(lon)
    y = WEB_MERCATOR_RADIUS * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def export_image(row: dict[str, str], args: argparse.Namespace) -> dict[str, str]:
    out_row = dict(row)
    mall_id = row.get("id", "unknown")
    name = row.get("name", "mall")
    filename = safe_filename(f"{mall_id}_{name}") + ".jpg"
    image_path = args.image_dir / filename

    out_row.update(
        {
            "image_status": "ok" if image_path.exists() and not args.overwrite else "",
            "image_path": str(image_path),
            "image_error": "",
            "imagery_provider": "esri_world_imagery_export",
            "imagery_size_px": str(args.size),
            "imagery_radius_m": str(args.radius_m),
        }
    )
    if image_path.exists() and not args.overwrite:
        return out_row

    try:
        lon = float(row.get("longitude") or "")
        lat = float(row.get("latitude") or "")
        center_x, center_y = lonlat_to_web_mercator(lon, lat)
        bbox = (
            f"{center_x - args.radius_m},{center_y - args.radius_m},"
            f"{center_x + args.radius_m},{center_y + args.radius_m}"
        )
        params = {
            "bbox": bbox,
            "bboxSR": "3857",
            "imageSR": "3857",
            "size": f"{args.size},{args.size}",
            "format": "jpg",
            "f": "image",
        }
        response = requests.get(
            ESRI_EXPORT_URL,
            params=params,
            timeout=args.timeout,
            headers={"User-Agent": "pv-coverage-shanghai-full-experiment/0.1"},
        )
        response.raise_for_status()
        image_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = image_path.with_suffix(".tmp")
        tmp_path.write_bytes(response.content)
        with Image.open(tmp_path) as image:
            image.verify()
        tmp_path.replace(image_path)
        out_row["image_status"] = "ok"
    except Exception as exc:
        out_row["image_status"] = "error"
        out_row["image_error"] = repr(exc)

    return out_row


def run_image_export(args: argparse.Namespace) -> None:
    fields, rows = read_csv(args.geocoded)
    rows = [row for row in iter_limited(rows, args.limit)]
    image_fields = fields + [
        "image_status",
        "image_path",
        "image_error",
        "imagery_provider",
        "imagery_size_px",
        "imagery_radius_m",
    ]

    completed: list[dict[str, str]] = []
    total = len(rows)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(export_image, row, args) for row in rows]
        for index, future in enumerate(as_completed(futures), start=1):
            completed.append(future.result())
            if index % args.checkpoint_every == 0:
                completed_sorted = sorted(completed, key=lambda item: int(item.get("id") or 0))
                write_csv(args.image_index, image_fields, completed_sorted)
                print(f"image_checkpoint={index}/{total}")
            if args.delay > 0:
                time.sleep(args.delay)

    completed = sorted(completed, key=lambda item: int(item.get("id") or 0))
    write_csv(args.image_index, image_fields, completed)
    ok_count = sum(1 for row in completed if row.get("image_status") == "ok")
    print(f"imagery_index={args.image_index}")
    print(f"image_rows={len(completed)}")
    print(f"image_ok={ok_count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the full Shanghai mall satellite imagery dataset."
    )
    parser.add_argument("--stage", choices=["all", "geocode", "images"], default="all")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--geocoded", type=Path, default=DEFAULT_GEOCODED)
    parser.add_argument("--image-index", type=Path, default=DEFAULT_IMAGE_INDEX)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--radius-m", type=float, default=200.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--timeout", type=int, default=40)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.stage in ("all", "geocode"):
        run_geocoding(args)
    if args.stage in ("all", "images"):
        run_image_export(args)


if __name__ == "__main__":
    main()
