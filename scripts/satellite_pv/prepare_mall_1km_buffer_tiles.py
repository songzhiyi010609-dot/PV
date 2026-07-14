#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare 1km buffer imagery tiles around validated mall centers.

The script creates a timestamped experiment folder and writes tile manifests.
By default it only generates CSV indexes. Add --download to fetch Esri World
Imagery crops for the generated tile centers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fetch_satellite_images import build_crop, safe_filename


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CENTER_CSV = PROJECT_ROOT / "outputs" / "experiments" / "20260708_relocate_mall_centers" / "data" / "mall_center_precise_pass_index.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "20260708_mall_1km_factory_pv_buffers"
ESRI_SOURCE = "esri_world_imagery"
TILE_SIZE_WEB = 256
MAX_MERCATOR_LAT = 85.05112878
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 1km buffer tiles around mall centers.")
    parser.add_argument("--center-csv", type=Path, default=DEFAULT_CENTER_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Experiment root; timestamped run folder is created inside.")
    parser.add_argument("--run-id", default="", help="Run folder name. Default: current time YYYYMMDD_HHMMSS.")
    parser.add_argument("--no-timestamp", action="store_true", help="Use output-dir exactly.")
    parser.add_argument("--resume", action="store_true", help="Resume newest timestamped run under output-dir.")
    parser.add_argument("--ids", default="", help="Comma-separated mall ids.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-provisional-centers", action="store_true", help="Allow center CSVs without A/B confidence or can_enter_1km_analysis=1.")
    parser.add_argument("--radius-m", type=float, default=1000.0)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--tile-size-px", type=int, default=768)
    parser.add_argument("--overlap-px", type=int, default=128)
    parser.add_argument("--download", action="store_true", help="Download tile crops. Without this, only indexes are generated.")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--log-every", type=int, default=25)
    return parser.parse_args()


def current_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_timestamped(path: Path) -> bool:
    return bool(TIMESTAMP_PATTERN.match(path.name))


def latest_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir() and is_timestamped(path) and (path / "data" / "mall_1km_buffer_tiles.csv").exists()
    ]
    return sorted(candidates, key=lambda item: item.name, reverse=True)[0] if candidates else None


def resolve_output_dir(args: argparse.Namespace) -> None:
    base = args.output_dir
    if args.no_timestamp or is_timestamped(base):
        actual = base
        run_id = args.run_id or (base.name if is_timestamped(base) else "no_timestamp")
    elif args.resume:
        previous = latest_run(base)
        if previous:
            actual = previous
            run_id = previous.name
        else:
            run_id = args.run_id or current_run_id()
            actual = base / run_id
    else:
        run_id = args.run_id or current_run_id()
        actual = base / run_id
    args.output_base_dir = base
    args.output_dir = actual
    args.run_id = run_id
    if args.cache_dir is None:
        args.cache_dir = actual / "tile_cache" / ESRI_SOURCE


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "data": output_dir / "data",
        "images": output_dir / "images" / "buffer_tiles",
        "reports": output_dir / "reports",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(paths: dict[str, Path]) -> logging.Logger:
    logger = logging.getLogger("prepare_mall_1km_buffer_tiles")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(paths["logs"] / "prepare_mall_1km_buffer_tiles.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def first_value(row: dict[str, str], names: list[str]) -> str:
    for name in names:
        value = row.get(name, "")
        if value not in {"", None}:
            return str(value)
    return ""


def row_is_analysis_ready(row: dict[str, str], allow_provisional: bool) -> bool:
    if allow_provisional:
        return True
    can_enter = str(row.get("can_enter_1km_analysis", "")).strip().lower()
    if can_enter:
        return can_enter in {"1", "true", "yes", "y"}
    level = str(row.get("confidence_level", "")).strip().upper()
    if level:
        return level in {"A", "B"}
    return False


def load_centers(path: Path, ids: set[str], limit: int, allow_provisional: bool) -> list[dict[str, str]]:
    centers: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            if not row_is_analysis_ready(raw, allow_provisional):
                continue
            mall_id = first_value(raw, ["mall_id", "id"])
            if ids and mall_id not in ids:
                continue
            lon_text = first_value(raw, ["center_lon", "selected_lon_wgs84", "selected_lon", "longitude", "lon"])
            lat_text = first_value(raw, ["center_lat", "selected_lat_wgs84", "selected_lat", "latitude", "lat"])
            if not lon_text or not lat_text:
                continue
            row = dict(raw)
            row["_mall_id"] = mall_id
            row["_name"] = first_value(raw, ["name", "mall_name", "mall_name_raw"])
            row["_center_lon"] = lon_text
            row["_center_lat"] = lat_text
            row["_center_confidence"] = first_value(raw, ["confidence", "center_confidence", "confidence_level", "identity_status"])
            row["_center_source"] = first_value(raw, ["provider", "selected_provider", "resolution_method", "source"])
            centers.append(row)
            if limit and len(centers) >= limit:
                break
    return centers


def lonlat_to_global_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    lat = max(min(lat, MAX_MERCATOR_LAT), -MAX_MERCATOR_LAT)
    scale = TILE_SIZE_WEB * (2**zoom)
    x = (lon + 180.0) / 360.0 * scale
    lat_rad = math.radians(lat)
    y = (
        0.5
        - math.log((1 + math.sin(lat_rad)) / (1 - math.sin(lat_rad))) / (4 * math.pi)
    ) * scale
    return x, y


def global_pixel_to_lonlat(x: float, y: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE_WEB * (2**zoom)
    lon = x / scale * 360.0 - 180.0
    n = math.pi - 2.0 * math.pi * y / scale
    lat = math.degrees(math.atan(math.sinh(n)))
    return lon, lat


def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2**zoom)


def tile_offsets(radius_m: float, tile_size_px: int, overlap_px: int, mpp: float) -> list[tuple[float, float, float]]:
    stride_px = tile_size_px - overlap_px
    if stride_px <= 0:
        raise ValueError("tile-size-px must be greater than overlap-px")
    stride_m = stride_px * mpp
    half_tile_m = tile_size_px * mpp / 2
    max_offset = radius_m + half_tile_m
    steps = range(-math.ceil(max_offset / stride_m), math.ceil(max_offset / stride_m) + 1)
    offsets: list[tuple[float, float, float]] = []
    for ix in steps:
        for iy in steps:
            ox = ix * stride_m
            oy = iy * stride_m
            distance = math.hypot(ox, oy)
            if distance <= radius_m + half_tile_m * math.sqrt(2):
                offsets.append((ox, oy, distance))
    return sorted(offsets, key=lambda item: (item[2], item[0], item[1]))


TILE_FIELDS = [
    "run_id",
    "tile_id",
    "mall_id",
    "mall_name",
    "center_lon",
    "center_lat",
    "center_confidence",
    "center_source",
    "buffer_radius_m",
    "tile_center_lon",
    "tile_center_lat",
    "offset_x_m",
    "offset_y_m",
    "distance_to_center_m",
    "tile_size_px",
    "overlap_px",
    "estimated_mpp",
    "tile_ground_width_m",
    "imagery_source",
    "imagery_zoom",
    "image_status",
    "image_path",
    "image_error",
]

SUMMARY_FIELDS = [
    "run_id",
    "mall_id",
    "mall_name",
    "center_lon",
    "center_lat",
    "center_confidence",
    "center_source",
    "buffer_radius_m",
    "tile_count",
    "downloaded_tile_count",
    "estimated_mpp",
    "tile_ground_width_m",
    "imagery_source",
    "imagery_zoom",
]


def write_header_if_needed(path: Path, fields: list[str]) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()


def append_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    write_header_if_needed(path, fields)
    with path.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writerows([{field: row.get(field, "") for field in fields} for row in rows])


def completed_malls(summary_path: Path) -> set[str]:
    if not summary_path.exists():
        return set()
    with summary_path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row.get("mall_id", "") for row in csv.DictReader(fh) if row.get("mall_id")}


def prepare_one_mall(center: dict[str, str], args: argparse.Namespace, paths: dict[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mall_id = center["_mall_id"]
    mall_name = center["_name"]
    center_lon = float(center["_center_lon"])
    center_lat = float(center["_center_lat"])
    mpp = meters_per_pixel(center_lat, args.zoom)
    center_px, center_py = lonlat_to_global_pixel(center_lon, center_lat, args.zoom)
    offsets = tile_offsets(args.radius_m, args.tile_size_px, args.overlap_px, mpp)
    tile_rows: list[dict[str, Any]] = []
    downloaded = 0

    for index, (offset_x_m, offset_y_m, distance_m) in enumerate(offsets, 1):
        tile_px = center_px + offset_x_m / mpp
        tile_py = center_py - offset_y_m / mpp
        tile_lon, tile_lat = global_pixel_to_lonlat(tile_px, tile_py, args.zoom)
        tile_id = f"{mall_id}_{index:04d}"
        image_status = "not_downloaded"
        image_path = ""
        image_error = ""
        if args.download:
            filename = safe_filename(f"{tile_id}_{mall_name}") + ".jpg"
            output_path = paths["images"] / str(mall_id) / filename
            image_path = str(output_path)
            try:
                if args.overwrite or not output_path.exists():
                    image = build_crop(
                        lon=tile_lon,
                        lat=tile_lat,
                        zoom=args.zoom,
                        size=args.tile_size_px,
                        cache_dir=args.cache_dir,
                        timeout=args.timeout,
                    )
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    image.save(output_path, quality=94)
                    if args.delay > 0:
                        time.sleep(args.delay)
                image_status = "ok"
                downloaded += 1
            except Exception as exc:  # noqa: BLE001 - keep experiment running.
                image_status = "error"
                image_error = repr(exc)

        tile_rows.append(
            {
                "run_id": args.run_id,
                "tile_id": tile_id,
                "mall_id": mall_id,
                "mall_name": mall_name,
                "center_lon": f"{center_lon:.8f}",
                "center_lat": f"{center_lat:.8f}",
                "center_confidence": center["_center_confidence"],
                "center_source": center["_center_source"],
                "buffer_radius_m": f"{args.radius_m:.1f}",
                "tile_center_lon": f"{tile_lon:.8f}",
                "tile_center_lat": f"{tile_lat:.8f}",
                "offset_x_m": f"{offset_x_m:.2f}",
                "offset_y_m": f"{offset_y_m:.2f}",
                "distance_to_center_m": f"{distance_m:.2f}",
                "tile_size_px": str(args.tile_size_px),
                "overlap_px": str(args.overlap_px),
                "estimated_mpp": f"{mpp:.6f}",
                "tile_ground_width_m": f"{args.tile_size_px * mpp:.2f}",
                "imagery_source": ESRI_SOURCE,
                "imagery_zoom": str(args.zoom),
                "image_status": image_status,
                "image_path": image_path,
                "image_error": image_error,
            }
        )

    summary = {
        "run_id": args.run_id,
        "mall_id": mall_id,
        "mall_name": mall_name,
        "center_lon": f"{center_lon:.8f}",
        "center_lat": f"{center_lat:.8f}",
        "center_confidence": center["_center_confidence"],
        "center_source": center["_center_source"],
        "buffer_radius_m": f"{args.radius_m:.1f}",
        "tile_count": str(len(tile_rows)),
        "downloaded_tile_count": str(downloaded),
        "estimated_mpp": f"{mpp:.6f}",
        "tile_ground_width_m": f"{args.tile_size_px * mpp:.2f}",
        "imagery_source": ESRI_SOURCE,
        "imagery_zoom": str(args.zoom),
    }
    return tile_rows, summary


def write_config(args: argparse.Namespace, paths: dict[str, Path], total_centers: int) -> None:
    lines = [
        "# 商场 1km Buffer 厂房光伏潜力数据准备配置",
        "",
        f"- 运行 ID：`{args.run_id}`",
        f"- 输入中心点：`{args.center_csv}`",
        f"- 输出根目录：`{args.output_base_dir}`",
        f"- 实际输出目录：`{args.output_dir}`",
        f"- 中心点数量：{total_centers}",
        f"- 是否允许临时中心点：{args.allow_provisional_centers}",
        f"- buffer 半径：{args.radius_m} m",
        f"- 影像 zoom：{args.zoom}",
        f"- tile_size_px：{args.tile_size_px}",
        f"- overlap_px：{args.overlap_px}",
        f"- 是否下载影像：{args.download}",
        f"- 瓦片缓存：`{args.cache_dir}`",
        "",
        "## 输出",
        "",
        f"- tile 索引：`{paths['data'] / 'mall_1km_buffer_tiles.csv'}`",
        f"- 商场汇总：`{paths['data'] / 'mall_1km_buffer_summary.csv'}`",
        f"- 日志：`{paths['logs'] / 'prepare_mall_1km_buffer_tiles.log'}`",
    ]
    (paths["reports"] / "实验配置.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(paths: dict[str, Path], tile_path: Path, summary_path: Path, elapsed: float) -> None:
    summaries: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as fh:
            summaries = list(csv.DictReader(fh))
    total_tiles = sum(int(row.get("tile_count") or 0) for row in summaries)
    downloaded = sum(int(row.get("downloaded_tile_count") or 0) for row in summaries)
    lines = [
        "# 商场 1km Buffer 厂房光伏潜力数据准备报告",
        "",
        f"- 商场数：{len(summaries)}",
        f"- tile 总数：{total_tiles}",
        f"- 已下载 tile：{downloaded}",
        f"- 总耗时：{elapsed / 60:.2f} 分钟",
        f"- tile 索引：`{tile_path}`",
        f"- 商场汇总：`{summary_path}`",
        "",
        "## 下一步",
        "",
        "- 使用 Git-RSCLIP 对 tile 做厂房/仓库语义筛查。",
        "- 对高分厂房/仓库 tile 跑 BDAPPV / DeepPVMapper 光伏识别。",
        "- 对高潜力或低置信样本生成复核图。",
    ]
    (paths["reports"] / "1km厂房光伏潜力数据准备报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolve_output_dir(args)
    paths = ensure_dirs(args.output_dir)
    logger = setup_logger(paths)
    ids = {part.strip() for part in args.ids.split(",") if part.strip()}
    centers = load_centers(args.center_csv, ids, args.limit, args.allow_provisional_centers)
    tile_path = paths["data"] / "mall_1km_buffer_tiles.csv"
    summary_path = paths["data"] / "mall_1km_buffer_summary.csv"
    done = completed_malls(summary_path) if args.resume else set()
    if not args.resume:
        for path in [tile_path, summary_path]:
            if path.exists():
                path.unlink()
    pending = [center for center in centers if center["_mall_id"] not in done]
    write_config(args, paths, len(centers))

    logger.info("Run id=%s", args.run_id)
    logger.info("Output dir=%s", args.output_dir)
    logger.info("Centers=%s completed=%s pending=%s download=%s", len(centers), len(done), len(pending), args.download)

    start = time.time()
    for index, center in enumerate(pending, 1):
        item_start = time.time()
        tile_rows, summary = prepare_one_mall(center, args, paths)
        append_rows(tile_path, TILE_FIELDS, tile_rows)
        append_rows(summary_path, SUMMARY_FIELDS, [summary])
        if index == 1 or index % args.log_every == 0 or index == len(pending):
            elapsed = time.time() - start
            avg = elapsed / max(index, 1)
            eta = avg * (len(pending) - index)
            logger.info(
                "[%s/%s] id=%s %s | tiles=%s downloaded=%s | %.1fs/item | ETA %.1fmin",
                len(done) + index,
                len(centers),
                center["_mall_id"],
                center["_name"],
                summary["tile_count"],
                summary["downloaded_tile_count"],
                time.time() - item_start,
                eta / 60,
            )

    write_report(paths, tile_path, summary_path, time.time() - start)
    logger.info("Done. tiles=%s", tile_path)
    logger.info("Report=%s", paths["reports"] / "1km厂房光伏潜力数据准备报告.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
