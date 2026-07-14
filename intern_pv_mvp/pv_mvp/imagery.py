from __future__ import annotations

import math
import time
from pathlib import Path

import requests
from PIL import Image

from .io_utils import ensure_dir, safe_filename


ESRI_TILE_URL = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_SIZE = 256
MAX_MERCATOR_LAT = 85.05112878


def lonlat_to_global_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
    scale = TILE_SIZE * (2**zoom)
    x = (lon + 180.0) / 360.0 * scale
    lat_rad = math.radians(lat)
    y = (0.5 - math.log((1 + math.sin(lat_rad)) / (1 - math.sin(lat_rad))) / (4 * math.pi)) * scale
    return x, y


def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392 * math.cos(math.radians(lat)) / (2**zoom)


def read_tile(z: int, x: int, y: int, cache_dir: Path, timeout: int = 30) -> Image.Image:
    cache_dir = ensure_dir(cache_dir / str(z) / str(x))
    tile_path = cache_dir / f"{y}.jpg"
    if tile_path.exists():
        return Image.open(tile_path).convert("RGB")

    url = ESRI_TILE_URL.format(z=z, x=x, y=y)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    tile_path.write_bytes(response.content)
    return Image.open(tile_path).convert("RGB")


def build_crop(
    lon: float,
    lat: float,
    *,
    zoom: int,
    crop_size_px: int,
    cache_dir: Path,
    timeout: int,
) -> Image.Image:
    center_x, center_y = lonlat_to_global_pixel(lon, lat, zoom)
    half = crop_size_px / 2
    left = int(math.floor(center_x - half))
    top = int(math.floor(center_y - half))
    right = left + crop_size_px
    bottom = top + crop_size_px

    tile_x_min = left // TILE_SIZE
    tile_y_min = top // TILE_SIZE
    tile_x_max = (right - 1) // TILE_SIZE
    tile_y_max = (bottom - 1) // TILE_SIZE

    canvas = Image.new("RGB", ((tile_x_max - tile_x_min + 1) * TILE_SIZE, (tile_y_max - tile_y_min + 1) * TILE_SIZE))
    for tx in range(tile_x_min, tile_x_max + 1):
        for ty in range(tile_y_min, tile_y_max + 1):
            tile = read_tile(zoom, tx, ty, cache_dir, timeout=timeout)
            canvas.paste(tile, ((tx - tile_x_min) * TILE_SIZE, (ty - tile_y_min) * TILE_SIZE))

    offset_x = left - tile_x_min * TILE_SIZE
    offset_y = top - tile_y_min * TILE_SIZE
    return canvas.crop((offset_x, offset_y, offset_x + crop_size_px, offset_y + crop_size_px))


def fetch_mall_image(
    *,
    mall_id: str,
    name: str,
    lon: float,
    lat: float,
    output_dir: Path,
    cache_dir: Path,
    zoom: int = 18,
    crop_size_px: int = 768,
    timeout: int = 30,
    delay: float = 0.05,
) -> tuple[str, float]:
    ensure_dir(output_dir)
    image = build_crop(lon, lat, zoom=zoom, crop_size_px=crop_size_px, cache_dir=cache_dir, timeout=timeout)
    filename = f"{safe_filename(mall_id, 'mall')}_{safe_filename(name, 'mall')}.jpg"
    image_path = output_dir / filename
    image.save(image_path, quality=94)
    time.sleep(max(delay, 0.0))
    return str(image_path), meters_per_pixel(lat, zoom)
