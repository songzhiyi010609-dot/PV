from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_ROOT = REPO_ROOT / "satellite_experiment" / "full_shanghai_dataset"
DEFAULT_GEOCODED = FULL_ROOT / "data" / "shanghai_geocoded.csv"
DEFAULT_IMAGE_DIR = FULL_ROOT / "images"
ARCGIS_GEOCODE_URL = (
    "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/"
    "findAddressCandidates"
)
SHANGHAI_BBOX = (120.85, 30.67, 122.15, 31.87)
SHANGHAI = "\u4e0a\u6d77\u5e02"


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_path.replace(path)


def is_in_shanghai(lon: float, lat: float) -> bool:
    west, south, east, north = SHANGHAI_BBOX
    return west <= lon <= east and south <= lat <= north


def row_needs_repair(row: dict[str, str]) -> bool:
    try:
        lon = float(row.get("longitude") or "")
        lat = float(row.get("latitude") or "")
    except Exception:
        return True
    return not is_in_shanghai(lon, lat)


def query_candidates(query: str, timeout: int) -> list[dict]:
    params = {
        "SingleLine": query,
        "f": "json",
        "outFields": "Match_addr,Addr_type,PlaceName,City,Region",
        "maxLocations": 8,
        "sourceCountry": "CHN",
    }
    response = requests.get(ARCGIS_GEOCODE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json().get("candidates") or []


def repair_one(row: dict[str, str], timeout: int) -> dict[str, str]:
    name = (row.get("name") or "").strip()
    location = (row.get("location") or "").strip()
    queries = [
        f"{SHANGHAI}{name}",
        f"{name} {SHANGHAI}",
        f"{SHANGHAI} {name}",
    ]
    if SHANGHAI in location:
        queries.insert(0, location)

    seen: set[str] = set()
    for query in queries:
        if not query or query in seen:
            continue
        seen.add(query)
        for candidate in query_candidates(query, timeout=timeout):
            point = candidate.get("location") or {}
            try:
                lon = float(point.get("x"))
                lat = float(point.get("y"))
            except Exception:
                continue
            if not is_in_shanghai(lon, lat):
                continue
            repaired = dict(row)
            repaired.update(
                {
                    "longitude": str(lon),
                    "latitude": str(lat),
                    "geocode_score": str(candidate.get("score", "")),
                    "geocode_address": str(candidate.get("address", "")),
                    "geocode_status": "ok",
                    "geocode_error": "",
                    "geocode_query": query,
                    "geocode_provider": "arcgis_world_geocoding_repaired_shanghai_bbox",
                    "geocode_repaired": "1",
                }
            )
            return repaired

    failed = dict(row)
    failed.update(
        {
            "longitude": "",
            "latitude": "",
            "geocode_score": "",
            "geocode_address": "",
            "geocode_status": "not_found",
            "geocode_error": "repair_no_candidate_in_shanghai_bbox",
            "geocode_repaired": "0",
        }
    )
    return failed


def remove_existing_image(row: dict[str, str], image_dir: Path) -> None:
    mall_id = row.get("id", "")
    for path in image_dir.glob(f"{mall_id}_*.jpg"):
        path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair Shanghai geocoding by enforcing a Shanghai bbox.")
    parser.add_argument("--geocoded", type=Path, default=DEFAULT_GEOCODED)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    fields, rows = read_csv(args.geocoded)
    if "geocode_repaired" not in fields:
        fields.append("geocode_repaired")

    repaired_count = 0
    still_missing = 0
    new_rows: list[dict[str, str]] = []
    for row in rows:
        if not row_needs_repair(row):
            row.setdefault("geocode_repaired", "0")
            new_rows.append(row)
            continue

        repaired = repair_one(row, timeout=args.timeout)
        remove_existing_image(row, args.image_dir)
        if repaired.get("geocode_status") == "ok":
            repaired_count += 1
        else:
            still_missing += 1
        new_rows.append(repaired)
        if args.delay > 0:
            time.sleep(args.delay)

    write_csv(args.geocoded, fields, new_rows)
    print(f"geocoded={args.geocoded}")
    print(f"repaired={repaired_count}")
    print(f"still_missing={still_missing}")


if __name__ == "__main__":
    main()
