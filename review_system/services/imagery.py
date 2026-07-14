from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

from settings import PROJECT_ROOT, path_value


SATELLITE_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "satellite_pv"
if str(SATELLITE_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SATELLITE_SCRIPT_DIR))

from fetch_satellite_images import build_crop, lonlat_to_global_pixel, safe_filename  # noqa: E402


TILE_SIZE = 256
MAX_MERCATOR_LAT = 85.05112878


def global_pixel_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * (2**zoom)
    lon = x / scale * 360.0 - 180.0
    merc_y = 0.5 - y / scale
    lat = 90.0 - 360.0 * math.atan(math.exp(-merc_y * 2.0 * math.pi)) / math.pi
    return lon, max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)


def crop_filename(mall_id: int, name: str, lon: float, lat: float, size: int, zoom: int) -> str:
    stem = safe_filename(f"{mall_id}_{name}_{lon:.6f}_{lat:.6f}_z{zoom}_{size}")
    return f"{stem}.jpg"


def ensure_crop(config: dict[str, Any], mall: dict[str, Any]) -> Path | None:
    lon = mall.get("selected_lon_wgs84")
    lat = mall.get("selected_lat_wgs84")
    if lon is None or lat is None:
        return None
    lon = float(lon)
    lat = float(lat)
    imagery = config["imagery"]
    size = int(imagery["crop_size_px"])
    zoom = int(imagery["zoom"])
    crop_dir = path_value(config, "paths", "satellite_crop_dir")
    cache_dir = path_value(config, "paths", "tile_cache_dir")
    crop_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_path = crop_dir / crop_filename(int(mall["mall_id"]), str(mall["name"]), lon, lat, size, zoom)
    if output_path.exists():
        return output_path
    image = build_crop(
        lon=lon,
        lat=lat,
        zoom=zoom,
        size=size,
        cache_dir=cache_dir,
        timeout=int(imagery["timeout_seconds"]),
    )
    image.save(output_path, quality=int(imagery["jpeg_quality"]))
    return output_path


def marker_position(
    center_lon: float,
    center_lat: float,
    point_lon: float,
    point_lat: float,
    zoom: int,
    size: int,
    margin: int,
) -> tuple[float, float, bool]:
    center_x, center_y = lonlat_to_global_pixel(center_lon, center_lat, zoom)
    point_x, point_y = lonlat_to_global_pixel(point_lon, point_lat, zoom)
    x = size / 2 + (point_x - center_x)
    y = size / 2 + (point_y - center_y)
    visible = -margin <= x <= size + margin and -margin <= y <= size + margin
    return x, y, visible


def candidate_markers(config: dict[str, Any], mall: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if mall.get("selected_lon_wgs84") is None or mall.get("selected_lat_wgs84") is None:
        return []
    center_lon = float(mall["selected_lon_wgs84"])
    center_lat = float(mall["selected_lat_wgs84"])
    imagery = config["imagery"]
    zoom = int(imagery["zoom"])
    size = int(imagery["crop_size_px"])
    margin = int(imagery["marker_visible_margin_px"])
    markers: list[dict[str, Any]] = []
    for item in candidates:
        if item.get("wgs84_lon") is None or item.get("wgs84_lat") is None:
            continue
        x, y, visible = marker_position(
            center_lon=center_lon,
            center_lat=center_lat,
            point_lon=float(item["wgs84_lon"]),
            point_lat=float(item["wgs84_lat"]),
            zoom=zoom,
            size=size,
            margin=margin,
        )
        if visible:
            marker = dict(item)
            marker["x"] = x
            marker["y"] = y
            marker["x_pct"] = x / size * 100
            marker["y_pct"] = y / size * 100
            markers.append(marker)
    return markers


def pixel_to_lonlat(center_lon: float, center_lat: float, x: float, y: float, zoom: int, size: int) -> tuple[float, float]:
    center_x, center_y = lonlat_to_global_pixel(center_lon, center_lat, zoom)
    point_x = center_x + (x - size / 2)
    point_y = center_y + (y - size / 2)
    return global_pixel_to_lonlat(point_x, point_y, zoom)
