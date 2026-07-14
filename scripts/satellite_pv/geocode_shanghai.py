from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable

import requests


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "raw" / "\u4e0a\u6d77\u5e02.csv"
DEFAULT_OUTPUT = (
    REPO_ROOT / "satellite_experiment" / "data" / "shanghai_geocoded_sample.csv"
)
ARCGIS_GEOCODE_URL = (
    "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/"
    "findAddressCandidates"
)


def iter_limited(rows: Iterable[dict[str, str]], limit: int | None) -> Iterable[dict[str, str]]:
    for index, row in enumerate(rows):
        if limit is not None and index >= limit:
            break
        yield row


def build_query(row: dict[str, str]) -> str:
    location = (row.get("location") or "").strip()
    city = (row.get("city") or "").strip()
    name = (row.get("name") or "").strip()

    if location:
        return location
    return f"{city}{name}"


def geocode(query: str, timeout: int) -> dict[str, str]:
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
        }

    candidate = candidates[0]
    location = candidate.get("location") or {}
    return {
        "longitude": str(location.get("x", "")),
        "latitude": str(location.get("y", "")),
        "geocode_score": str(candidate.get("score", "")),
        "geocode_address": str(candidate.get("address", "")),
        "geocode_status": "ok",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Geocode Shanghai mall addresses for satellite imagery fetching."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.input.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {args.input}")

        extra_fields = [
            "geocode_query",
            "longitude",
            "latitude",
            "geocode_score",
            "geocode_address",
            "geocode_status",
            "geocode_provider",
        ]
        fieldnames = list(reader.fieldnames) + extra_fields

        with args.output.open("w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()

            count = 0
            ok_count = 0
            for row in iter_limited(reader, args.limit):
                query = build_query(row)
                result = geocode(query, timeout=args.timeout)
                out_row = dict(row)
                out_row.update(result)
                out_row["geocode_query"] = query
                out_row["geocode_provider"] = "arcgis_world_geocoding"
                writer.writerow(out_row)

                count += 1
                if result["geocode_status"] == "ok":
                    ok_count += 1
                if args.delay > 0:
                    time.sleep(args.delay)

    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"rows={count}")
    print(f"geocoded={ok_count}")


if __name__ == "__main__":
    main()
