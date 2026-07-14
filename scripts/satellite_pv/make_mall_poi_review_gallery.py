#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Create visual POI review sheets for mall center relocation.

Each review image compares:
- left: the existing dataset satellite crop, which may be wrong;
- right: a fresh Esri World Imagery crop around the candidate POI coordinate;
- center marker: the candidate POI center to verify.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import re
import time
from io import BytesIO
from pathlib import Path
from typing import Iterable

import requests
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "20260708_relocate_mall_centers"
DEFAULT_REVIEW_CSV = DEFAULT_EXPERIMENT_DIR / "data" / "mall_center_review_needed.csv"
DEFAULT_OLD_IMAGE_DIR = PROJECT_ROOT / "datasets" / "shanghai_malls_satellite" / "images"
DEFAULT_OUTPUT_DIR = DEFAULT_EXPERIMENT_DIR / "poi_review_gallery"
DEFAULT_CACHE_DIR = DEFAULT_EXPERIMENT_DIR / "tile_cache" / "esri_world_imagery"

ESRI_TILE_URL = (
    "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
    "MapServer/tile/{z}/{y}/{x}"
)
TILE_SIZE = 256
MAX_MERCATOR_LAT = 85.05112878


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Make POI visual review gallery.")
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--old-image-dir", type=Path, default=DEFAULT_OLD_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows with candidate coordinates.")
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--panel-size", type=int, default=420)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip().strip(".")
    return cleaned[:150] or "mall"


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
        headers={"User-Agent": "pv-mall-poi-review/1.0"},
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

    mosaic = Image.new("RGB", ((max_tile_x - min_tile_x + 1) * TILE_SIZE, (max_tile_y - min_tile_y + 1) * TILE_SIZE))
    for tile_y in range(min_tile_y, max_tile_y + 1):
        for tile_x in range(min_tile_x, max_tile_x + 1):
            tile = read_tile(zoom, tile_x, tile_y, cache_dir, timeout)
            mosaic.paste(tile, ((tile_x - min_tile_x) * TILE_SIZE, (tile_y - min_tile_y) * TILE_SIZE))

    crop_left = round(left - min_tile_x * TILE_SIZE)
    crop_top = round(top - min_tile_y * TILE_SIZE)
    return mosaic.crop((crop_left, crop_top, crop_left + size, crop_top + size))


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in [
        Path(r"C:\Windows\Fonts\msyh.ttc"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\simsun.ttc"),
    ]:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def fit_image(image: Image.Image, size: int) -> Image.Image:
    image = image.convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (244, 244, 244))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    canvas.paste(image, (x, y))
    return canvas


def draw_crosshair(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int) -> None:
    red = (240, 40, 40)
    white = (255, 255, 255)
    for color, width in [(white, 5), (red, 2)]:
        draw.line((cx - size, cy, cx + size, cy), fill=color, width=width)
        draw.line((cx, cy - size, cx, cy + size), fill=color, width=width)
        draw.ellipse((cx - 11, cy - 11, cx + 11, cy + 11), outline=color, width=width)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = list(text)
    lines: list[str] = []
    current = ""
    for word in words:
        trial = current + word
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def find_old_image(image_dir: Path, mall_id: str) -> Path | None:
    matches = sorted(image_dir.glob(f"{mall_id}_*.jpg"))
    return matches[0] if matches else None


def iter_rows(path: Path, limit: int) -> Iterable[dict[str, str]]:
    yielded = 0
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if not row.get("candidate_lon_for_review") or not row.get("candidate_lat_for_review"):
                continue
            yield row
            yielded += 1
            if limit and yielded >= limit:
                break


def make_review_image(
    row: dict[str, str],
    old_image_dir: Path,
    output_path: Path,
    cache_dir: Path,
    crop_size: int,
    panel_size: int,
    zoom: int,
    timeout: int,
) -> dict[str, str]:
    mall_id = row.get("mall_id", "")
    name = row.get("name", "")
    lon = float(row.get("candidate_lon_for_review") or "")
    lat = float(row.get("candidate_lat_for_review") or "")
    old_path = find_old_image(old_image_dir, mall_id)
    if old_path and old_path.exists():
        old_panel = fit_image(Image.open(old_path), panel_size)
    else:
        old_panel = Image.new("RGB", (panel_size, panel_size), (225, 225, 225))

    poi_crop = build_crop(lon=lon, lat=lat, zoom=zoom, size=crop_size, cache_dir=cache_dir, timeout=timeout)
    poi_panel = fit_image(poi_crop, panel_size)
    poi_draw = ImageDraw.Draw(poi_panel)
    draw_crosshair(poi_draw, panel_size // 2, panel_size // 2, 34)

    margin = 24
    header_h = 150
    footer_h = 84
    gap = 18
    width = margin * 2 + panel_size * 2 + gap
    height = header_h + panel_size + footer_h
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(24)
    label_font = load_font(18)
    small_font = load_font(15)

    draw.rectangle((0, 0, width, header_h), fill=(32, 38, 46))
    title = f"{mall_id}  {name}"
    y = 18
    for line in wrap_text(draw, title, title_font, width - margin * 2)[:2]:
        draw.text((margin, y), line, fill=(255, 255, 255), font=title_font)
        y += 31
    meta = (
        f"候选POI: {row.get('place_name','')} | {row.get('provider','')} | "
        f"{row.get('confidence','')} | {lon:.8f}, {lat:.8f}"
    )
    for line in wrap_text(draw, meta, small_font, width - margin * 2)[:2]:
        draw.text((margin, y + 4), line, fill=(220, 226, 235), font=small_font)
        y += 22

    left_x = margin
    right_x = margin + panel_size + gap
    top_y = header_h
    canvas.paste(old_panel, (left_x, top_y))
    canvas.paste(poi_panel, (right_x, top_y))
    draw.rectangle((left_x, top_y, left_x + panel_size, top_y + panel_size), outline=(70, 70, 70), width=2)
    draw.rectangle((right_x, top_y, right_x + panel_size, top_y + panel_size), outline=(200, 50, 50), width=3)
    draw.rectangle((left_x, top_y, left_x + 142, top_y + 32), fill=(0, 0, 0))
    draw.text((left_x + 10, top_y + 6), "旧数据集图", fill=(255, 255, 255), font=label_font)
    draw.rectangle((right_x, top_y, right_x + 190, top_y + 32), fill=(150, 20, 20))
    draw.text((right_x + 10, top_y + 6), "候选POI中心图", fill=(255, 255, 255), font=label_font)

    footer_y = header_h + panel_size + 14
    reason = f"复核原因: {row.get('review_reason','')}"
    draw.text((margin, footer_y), reason[:90], fill=(40, 40, 40), font=small_font)
    draw.text((margin, footer_y + 26), "判断：右图十字是否落在商场主体/商业综合体屋顶中心，若落在道路、车站、住宅、店铺或泛城市点则拒绝。", fill=(40, 40, 40), font=small_font)
    if old_path:
        draw.text((margin, footer_y + 52), f"旧图: {old_path.name}", fill=(80, 80, 80), font=small_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)
    return {
        "mall_id": mall_id,
        "name": name,
        "review_image": str(output_path),
        "old_image": str(old_path or ""),
        "candidate_lon": f"{lon:.8f}",
        "candidate_lat": f"{lat:.8f}",
        "confidence": row.get("confidence", ""),
        "provider": row.get("provider", ""),
        "place_name": row.get("place_name", ""),
        "review_reason": row.get("review_reason", ""),
    }


def make_contact_sheets(image_paths: list[Path], out_dir: Path, columns: int = 3, rows: int = 4) -> list[Path]:
    page_paths: list[Path] = []
    thumb_w = 360
    thumb_h = 248
    pad = 12
    font = load_font(14)
    per_page = columns * rows
    for page_idx in range(0, len(image_paths), per_page):
        chunk = image_paths[page_idx : page_idx + per_page]
        page_no = page_idx // per_page + 1
        sheet = Image.new("RGB", (columns * thumb_w + (columns + 1) * pad, rows * thumb_h + (rows + 1) * pad), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for i, img_path in enumerate(chunk):
            img = Image.open(img_path).convert("RGB")
            img.thumbnail((thumb_w, thumb_h - 26), Image.Resampling.LANCZOS)
            col = i % columns
            row = i // columns
            x = pad + col * (thumb_w + pad)
            y = pad + row * (thumb_h + pad)
            sheet.paste(img, (x, y + 22))
            draw.text((x, y), img_path.stem[:42], fill=(20, 20, 20), font=font)
            draw.rectangle((x, y + 22, x + thumb_w, y + thumb_h), outline=(210, 210, 210), width=1)
        page_path = out_dir / f"poi_review_contact_sheet_{page_no:02d}.jpg"
        sheet.save(page_path, quality=90)
        page_paths.append(page_path)
    return page_paths


def write_gallery_html(rows: list[dict[str, str]], sheets: list[Path], html_path: Path) -> None:
    parts = [
        "<!doctype html>",
        "<html lang='zh-CN'><head><meta charset='utf-8'>",
        "<title>商场 POI 复核图集</title>",
        "<style>body{font-family:Microsoft YaHei,Arial,sans-serif;margin:24px;background:#f6f7f8;color:#20242a}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:18px}"
        ".card{background:#fff;border:1px solid #ddd;padding:12px}"
        "img{max-width:100%;height:auto;display:block}.meta{font-size:13px;color:#555;line-height:1.5}"
        "a{color:#1f5fbf}</style></head><body>",
        "<h1>商场 POI 复核图集</h1>",
        "<p>右侧图红色十字为候选 POI 中心。确认它是否落在商场主体/商业综合体屋顶中心。</p>",
        "<h2>分页联系图</h2><ul>",
    ]
    for sheet in sheets:
        parts.append(f"<li><a href='{html.escape(sheet.name)}'>{html.escape(sheet.name)}</a></li>")
    parts.append("</ul><h2>逐条复核</h2><div class='grid'>")
    for row in rows:
        img_name = Path(row["review_image"]).name
        parts.extend(
            [
                "<div class='card'>",
                f"<img src='review_images/{html.escape(img_name)}' alt='{html.escape(row['name'])}'>",
                "<div class='meta'>",
                f"<b>{html.escape(row['mall_id'])} {html.escape(row['name'])}</b><br>",
                f"POI: {html.escape(row['place_name'])} | {html.escape(row['provider'])} | {html.escape(row['confidence'])}<br>",
                f"坐标: {html.escape(row['candidate_lon'])}, {html.escape(row['candidate_lat'])}<br>",
                f"原因: {html.escape(row['review_reason'])}",
                "</div></div>",
            ]
        )
    parts.append("</div></body></html>")
    html_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    args = parse_args()
    review_image_dir = args.output_dir / "review_images"
    sheets_dir = args.output_dir / "contact_sheets"
    data_dir = args.output_dir / "data"
    review_image_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str]] = []
    image_paths: list[Path] = []
    for idx, row in enumerate(iter_rows(args.review_csv, args.limit), 1):
        mall_id = row.get("mall_id", "")
        name = row.get("name", "")
        out_path = review_image_dir / f"{safe_filename(f'{mall_id}_{name}')}.jpg"
        if args.overwrite or not out_path.exists():
            try:
                out_row = make_review_image(
                    row=row,
                    old_image_dir=args.old_image_dir,
                    output_path=out_path,
                    cache_dir=args.cache_dir,
                    crop_size=args.crop_size,
                    panel_size=args.panel_size,
                    zoom=args.zoom,
                    timeout=args.timeout,
                )
            except Exception as exc:  # noqa: BLE001 - keep batch running.
                out_row = {
                    "mall_id": mall_id,
                    "name": name,
                    "review_image": "",
                    "old_image": "",
                    "candidate_lon": row.get("candidate_lon_for_review", ""),
                    "candidate_lat": row.get("candidate_lat_for_review", ""),
                    "confidence": row.get("confidence", ""),
                    "provider": row.get("provider", ""),
                    "place_name": row.get("place_name", ""),
                    "review_reason": row.get("review_reason", ""),
                    "error": repr(exc),
                }
            if args.delay > 0:
                time.sleep(args.delay)
        else:
            out_row = {
                "mall_id": mall_id,
                "name": name,
                "review_image": str(out_path),
                "old_image": str(find_old_image(args.old_image_dir, mall_id) or ""),
                "candidate_lon": row.get("candidate_lon_for_review", ""),
                "candidate_lat": row.get("candidate_lat_for_review", ""),
                "confidence": row.get("confidence", ""),
                "provider": row.get("provider", ""),
                "place_name": row.get("place_name", ""),
                "review_reason": row.get("review_reason", ""),
                "error": "",
            }
        index_rows.append(out_row)
        if out_row.get("review_image"):
            image_paths.append(Path(out_row["review_image"]))
        if idx == 1 or idx % 20 == 0:
            print(f"[{idx}] {mall_id} {name} -> {out_row.get('review_image') or out_row.get('error')}")

    sheets = make_contact_sheets(image_paths, sheets_dir)
    index_path = data_dir / "poi_review_image_index.csv"
    fields = [
        "mall_id",
        "name",
        "review_image",
        "old_image",
        "candidate_lon",
        "candidate_lat",
        "confidence",
        "provider",
        "place_name",
        "review_reason",
        "error",
    ]
    with index_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in index_rows])

    html_path = args.output_dir / "poi_review_gallery.html"
    write_gallery_html(index_rows, sheets, html_path)
    print(f"rows={len(index_rows)} images={len(image_paths)} sheets={len(sheets)}")
    print(f"gallery={html_path}")
    print(f"index={index_path}")
    print(f"review_images={review_image_dir}")
    print(f"contact_sheets={sheets_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
