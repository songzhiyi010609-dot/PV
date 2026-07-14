from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pv_mvp.detector import DetectionResult, detect_pv  # noqa: E402
from fetch_satellite_images import build_crop, safe_filename  # noqa: E402
from run_bdappv_inference import (  # noqa: E402
    classify_image as classify_bdappv_image,
    load_bdappv_helper,
    segment_image as segment_bdappv_image,
    status_from_score as bdappv_status_from_score,
)


DEFAULT_CENTER_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "experiments"
    / "mall_center_review_all"
    / "mall_center_review_approved.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "mall_pv_potential_screening"
DEFAULT_CACHE_DIR = PROJECT_ROOT / "satellite_experiment" / "tile_cache" / "esri_world_imagery"
DEFAULT_BDAPPV_MODEL_DIR = PROJECT_ROOT / "datasets" / "shanghai_malls_satellite" / "models" / "bdappv"
METHOD = "roof_heuristic_v0+pv_bdappv"
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]


class PvDetectionEngine:
    def __init__(
        self,
        *,
        detector: str,
        model_dir: Path,
        provider: str,
        device: str,
        batch_size: int,
        window_size: int,
        stride: int,
        possible_threshold: float,
        likely_threshold: float,
        segmentation_threshold: float,
        segment_score_threshold: float,
        min_component_pixels: int,
        allow_heuristic_fallback: bool,
    ) -> None:
        self.detector = detector
        self.provider = provider
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.batch_size = batch_size
        self.window_size = window_size
        self.stride = stride
        self.possible_threshold = possible_threshold
        self.likely_threshold = likely_threshold
        self.segmentation_threshold = segmentation_threshold
        self.segment_score_threshold = segment_score_threshold
        self.min_component_pixels = min_component_pixels
        self.allow_heuristic_fallback = allow_heuristic_fallback
        self.classifier = None
        self.segmenter = None
        self.load_error = ""

        if detector == "bdappv":
            try:
                module = load_bdappv_helper(model_dir)
                self.classifier = module.load_classification_model(provider, device=self.device)
                self.segmenter = module.load_segmentation_model(provider, device=self.device)
            except Exception as exc:
                self.load_error = repr(exc)
                if not allow_heuristic_fallback:
                    raise
                print(f"WARNING: BDAPPV load failed; using heuristic fallback: {self.load_error}")

    def _write_bdappv_outputs(
        self,
        *,
        image: Image.Image,
        binary_mask: np.ndarray,
        image_path: Path,
        mask_dir: Path | None,
        overlay_dir: Path | None,
    ) -> tuple[str, str]:
        mask_path = ""
        overlay_path = ""
        mask_u8 = binary_mask.astype(np.uint8) * 255

        if mask_dir is not None:
            mask_dir.mkdir(parents=True, exist_ok=True)
            output = mask_dir / f"{image_path.stem}_mask.png"
            cv2.imwrite(str(output), mask_u8)
            mask_path = str(output)

        if overlay_dir is not None:
            overlay_dir.mkdir(parents=True, exist_ok=True)
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            overlay = rgb.copy()
            if np.any(binary_mask):
                red = np.zeros_like(overlay)
                red[:, :, 0] = 255
                overlay[binary_mask] = (overlay[binary_mask] * 0.48 + red[binary_mask] * 0.52).astype(np.uint8)
                contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
                cv2.drawContours(overlay_bgr, contours, -1, (0, 0, 255), 2)
                overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
            output = overlay_dir / f"{image_path.stem}_overlay.png"
            Image.fromarray(overlay).save(output)
            overlay_path = str(output)

        return mask_path, overlay_path

    def _detect_bdappv(
        self,
        image_path: Path,
        *,
        mask_dir: Path | None,
        overlay_dir: Path | None,
    ) -> tuple[DetectionResult, str, str]:
        if self.classifier is None or self.segmenter is None:
            raise RuntimeError(self.load_error or "BDAPPV models are not loaded")

        image = Image.open(image_path).convert("RGB")
        score, best_window, scores, windows = classify_bdappv_image(
            image=image,
            model=self.classifier,
            device=self.device,
            batch_size=self.batch_size,
            window_size=self.window_size,
            stride=self.stride,
        )
        status = bdappv_status_from_score(score, self.possible_threshold, self.likely_threshold)
        probability_mask = np.zeros((image.height, image.width), dtype=np.float32)
        if score >= self.segment_score_threshold:
            probability_mask = segment_bdappv_image(
                image=image,
                model=self.segmenter,
                device=self.device,
                windows=windows,
                scores=scores,
                score_threshold=self.segment_score_threshold,
                batch_size=max(1, min(self.batch_size, 4)),
            )
        raw_binary_mask = probability_mask >= self.segmentation_threshold
        binary_mask = np.zeros_like(raw_binary_mask, dtype=bool)
        raw_component_count = 0
        rejected_component_count = 0
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_binary_mask.astype(np.uint8), 8)
        for label in range(1, num_labels):
            raw_component_count += 1
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_component_pixels:
                rejected_component_count += 1
                continue
            binary_mask[labels == label] = True
        pv_area_px = int(np.count_nonzero(binary_mask))
        image_area_px = int(image.width * image.height)
        coverage = pv_area_px / image_area_px if image_area_px else 0.0
        component_count = max(0, cv2.connectedComponents(binary_mask.astype(np.uint8), 8)[0] - 1)
        if status == "likely_pv" and component_count == 0:
            status = "possible_pv"
        selected_scores = [value for value in scores if value >= self.segment_score_threshold]
        mean_score = float(np.mean(selected_scores)) if selected_scores else 0.0
        mask_path, overlay_path = self._write_bdappv_outputs(
            image=image,
            binary_mask=binary_mask,
            image_path=image_path,
            mask_dir=mask_dir,
            overlay_dir=overlay_dir,
        )
        reason = (
            f"bdappv_score={score:.6f}; status={status}; best_window={','.join(map(str, best_window))}; "
            f"segmentation_pixels={pv_area_px}; coverage={coverage:.4%}; components={component_count}; "
            f"raw_components={raw_component_count}; rejected_below_{self.min_component_pixels}px={rejected_component_count}"
        )
        result = DetectionResult(
            image_path=str(image_path),
            has_pv=status != "no_clear_pv",
            confidence=round(float(score), 4),
            pv_area_px=pv_area_px,
            image_area_px=image_area_px,
            coverage=round(float(coverage), 6),
            component_count=component_count,
            mean_panel_score=round(mean_score, 4),
            reason=reason,
            mask_path=mask_path,
            overlay_path=overlay_path,
        )
        return result, status, f"bdappv_{self.provider}"

    def detect(
        self,
        image_path: str | Path,
        *,
        min_pv_pixels: int,
        min_coverage: float,
        mask_dir: Path | None,
        overlay_dir: Path | None,
    ) -> tuple[DetectionResult, str, str]:
        image_path = Path(image_path)
        if self.detector == "bdappv" and not self.load_error:
            try:
                return self._detect_bdappv(image_path, mask_dir=mask_dir, overlay_dir=overlay_dir)
            except Exception as exc:
                if not self.allow_heuristic_fallback:
                    raise
                fallback_error = repr(exc)
                print(f"WARNING: BDAPPV inference failed for {image_path.name}; using heuristic fallback: {fallback_error}")
        elif self.detector == "bdappv" and not self.allow_heuristic_fallback:
            raise RuntimeError(self.load_error)

        result = detect_pv(
            image_path,
            min_pv_pixels=min_pv_pixels,
            min_coverage=min_coverage,
            mask_dir=mask_dir,
            overlay_dir=overlay_dir,
        )
        status = classify_pv(result.has_pv, result.confidence, result.coverage)
        if self.detector == "bdappv":
            result = DetectionResult(**{**asdict(result), "reason": f"bdappv_fallback={self.load_error or 'inference_error'}; {result.reason}"})
            return result, status, "heuristic_fallback"
        return result, status, "heuristic_v0"


def meters_per_pixel(lat: float, zoom: int) -> float:
    return 156543.03392804097 * math.cos(math.radians(lat)) / (2**zoom)


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_centers(
    path: Path,
    limit: int | None,
    ids: set[str] | None,
    confidence_levels: set[str] | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found: {path}")
        for row in reader:
            mall_id = str(row.get("mall_id") or row.get("id") or "").strip()
            if ids and mall_id not in ids:
                continue
            confidence_level = str(row.get("confidence_level") or "").strip().upper()
            if confidence_levels and confidence_level not in confidence_levels:
                continue
            lon = parse_float(row.get("center_lon") or row.get("selected_lon_wgs84") or row.get("longitude"))
            lat = parse_float(row.get("center_lat") or row.get("selected_lat_wgs84") or row.get("latitude"))
            if lon is None or lat is None:
                continue
            row["mall_id"] = mall_id
            row["center_lon"] = f"{lon:.8f}"
            row["center_lat"] = f"{lat:.8f}"
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_detection_checkpoints(
    data_dir: Path,
    self_rows: list[dict[str, Any]],
    potential_rows: list[dict[str, Any]],
    tile_rows: list[dict[str, Any]],
) -> None:
    for filename, rows in (
        ("mall_self_pv_results_checkpoint.csv", self_rows),
        ("mall_1km_potential_summary_checkpoint.csv", potential_rows),
        ("mall_1km_potential_tiles_checkpoint.csv", tile_rows),
    ):
        if not rows:
            continue
        fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
        write_csv(data_dir / filename, rows, fieldnames)


def load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def image_from_row_path(row: dict[str, Any], key: str) -> Image.Image | None:
    value = str(row.get(key) or "").strip()
    if not value:
        return None
    path = Path(value)
    if not path.exists():
        return None
    return Image.open(path).convert("RGB")


def paste_thumbnail(
    sheet: Image.Image,
    image: Image.Image | None,
    box: tuple[int, int, int, int],
    *,
    fill: tuple[int, int, int],
) -> None:
    draw = ImageDraw.Draw(sheet)
    x1, y1, x2, y2 = box
    draw.rectangle(box, fill=fill)
    if image is None:
        return
    image.thumbnail((x2 - x1, y2 - y1))
    sheet.paste(image, (x1 + (x2 - x1 - image.width) // 2, y1 + (y2 - y1 - image.height) // 2))


def resize_to_fit(image: Image.Image, size: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((size, size))
    tile = Image.new("RGB", (size, size), (30, 41, 59))
    tile.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return tile


def merge_overlay_pixels(base: Image.Image, overlay: Image.Image | None, *, alpha: float) -> Image.Image:
    if overlay is None:
        return base
    overlay = overlay.resize(base.size).convert("RGB")
    base_arr = np.asarray(base.convert("RGB"), dtype=np.int16)
    overlay_arr = np.asarray(overlay, dtype=np.int16)
    diff = np.abs(overlay_arr - base_arr).sum(axis=2) > 36
    if not np.any(diff):
        return base
    out = base_arr.copy()
    blended = base_arr * (1 - alpha) + overlay_arr * alpha
    out[diff] = blended[diff]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def annotated_tile_from_row(row: dict[str, Any], size: int) -> Image.Image | None:
    base = image_from_row_path(row, "image_path")
    if base is None:
        return None
    base = resize_to_fit(base, size)
    roof = image_from_row_path(row, "roof_overlay_path")
    pv = image_from_row_path(row, "pv_overlay_path")
    roof = resize_to_fit(roof, size) if roof is not None else None
    pv = resize_to_fit(pv, size) if pv is not None else None
    annotated = merge_overlay_pixels(base, roof, alpha=0.78)
    return merge_overlay_pixels(annotated, pv, alpha=0.92)


def auto_review_map_size_px(lat: float, zoom: int, radius_m: int, max_px: int) -> int:
    size = int(math.ceil((radius_m * 2) / max(meters_per_pixel(lat, zoom), 1e-9)))
    return min(max(size, 1024), max_px)


def build_full_review_mosaics(
    centers: list[dict[str, str]],
    *,
    pv_engine: PvDetectionEngine,
    output_dir: Path,
    cache_dir: Path,
    timeout: int,
    overwrite: bool,
    zoom: int,
    radius_m: int,
    size_px: int,
    max_size_px: int,
    min_roof_pixels: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    annotated_dir = output_dir / "annotated"
    pv_overlay_dir = output_dir / "full_pv_overlays"
    roof_overlay_dir = output_dir / "full_roof_overlays"
    for directory in (raw_dir, annotated_dir, pv_overlay_dir, roof_overlay_dir):
        directory.mkdir(parents=True, exist_ok=True)

    title_font = load_font(24)
    panel_font = load_font(20)
    label_font = load_font(16)
    mosaic_paths: dict[str, str] = {}

    for row in centers:
        mall_id = str(row.get("mall_id") or "")
        if not mall_id:
            continue
        name = row.get("name") or row.get("selected_poi_name") or "mall"
        lon = float(row["center_lon"])
        lat = float(row["center_lat"])
        image_size = size_px or auto_review_map_size_px(lat, zoom, radius_m, max_size_px)
        stem = safe_filename(f"mall_{mall_id}_full_1km_z{zoom}")
        raw_path = raw_dir / f"{stem}.jpg"
        if overwrite or not raw_path.exists():
            image = build_crop(lon=lon, lat=lat, zoom=zoom, size=image_size, cache_dir=cache_dir, timeout=timeout)
            image.save(raw_path, quality=95)

        raw = Image.open(raw_path).convert("RGB")
        mpp = meters_per_pixel(lat, zoom)
        roof = score_roof_potential(
            raw_path,
            mpp=mpp,
            overlay_dir=roof_overlay_dir,
            min_roof_pixels=max(min_roof_pixels, int((image_size * image_size) * 0.0012)),
        )
        pv, _pv_status, _pv_method = pv_engine.detect(
            raw_path,
            min_pv_pixels=max(800, int((image_size * image_size) * 0.0008)),
            min_coverage=0.0008,
            mask_dir=None,
            overlay_dir=pv_overlay_dir,
        )
        roof_overlay = Image.open(roof["roof_overlay_path"]).convert("RGB") if roof["roof_overlay_path"] else None
        pv_overlay = Image.open(pv.overlay_path).convert("RGB") if pv.overlay_path else None
        annotated = merge_overlay_pixels(raw, roof_overlay, alpha=0.78)
        annotated = merge_overlay_pixels(annotated, pv_overlay, alpha=0.92)
        annotated_path = annotated_dir / f"{stem}_annotated.jpg"
        annotated.save(annotated_path, quality=94)

        display_max = min(1400, image_size)
        raw_display = raw.copy()
        annotated_display = annotated.copy()
        raw_display.thumbnail((display_max, display_max))
        annotated_display.thumbnail((display_max, display_max))
        panel_w = max(raw_display.width, annotated_display.width)
        panel_h = max(raw_display.height, annotated_display.height)
        margin = 24
        header_h = 96
        panel_title_h = 34
        panel_gap = 30
        sheet_w = margin * 2 + panel_w * 2 + panel_gap
        sheet_h = margin * 2 + header_h + panel_title_h + panel_h
        sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 247, 250))
        draw = ImageDraw.Draw(sheet)
        draw.text((margin, margin), f"{mall_id} {name}", fill=(17, 24, 39), font=title_font)
        draw.text(
            (margin, margin + 36),
            f"无缝 1km 复核图：左为原图，右为同一张图上的标注。zoom={zoom}, size={image_size}px",
            fill=(75, 85, 99),
            font=label_font,
        )

        left_x = margin
        right_x = margin + panel_w + panel_gap
        panel_y = margin + header_h
        image_y = panel_y + panel_title_h
        draw.text((left_x, panel_y), "原图", fill=(17, 24, 39), font=panel_font)
        draw.text((right_x, panel_y), "标注图", fill=(17, 24, 39), font=panel_font)
        sheet.paste(raw_display, (left_x + (panel_w - raw_display.width) // 2, image_y))
        sheet.paste(annotated_display, (right_x + (panel_w - annotated_display.width) // 2, image_y))

        sheet_path = output_dir / f"mall_{safe_filename(mall_id)}_review_mosaic.jpg"
        sheet.save(sheet_path, quality=92)
        mosaic_paths[mall_id] = str(sheet_path)

    return mosaic_paths


def build_review_mosaics(
    tile_rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    thumb_px: int,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in tile_rows:
        if str(row.get("image_status") or "") != "ok":
            continue
        grouped.setdefault(str(row.get("mall_id") or ""), []).append(row)

    title_font = load_font(24)
    panel_font = load_font(20)
    label_font = load_font(16)
    small_font = load_font(13)
    mosaic_paths: dict[str, str] = {}

    for mall_id, rows in grouped.items():
        rows = sorted(rows, key=lambda item: (float(item.get("offset_y_m") or 0), float(item.get("offset_x_m") or 0)))
        x_values = sorted({float(row.get("offset_x_m") or 0) for row in rows})
        y_values = sorted({float(row.get("offset_y_m") or 0) for row in rows}, reverse=True)
        x_index = {value: index for index, value in enumerate(x_values)}
        y_index = {value: index for index, value in enumerate(y_values)}

        cell_gap = 0
        margin = 24
        header_h = 98
        panel_title_h = 34
        panel_gap = 28
        label_h = 0
        tile_w = thumb_px
        tile_h = thumb_px
        cell_w = tile_w + cell_gap
        cell_h = tile_h + label_h + cell_gap
        panel_w = len(x_values) * cell_w - cell_gap
        panel_h = panel_title_h + len(y_values) * cell_h - cell_gap
        sheet_w = margin * 2 + panel_w * 2 + panel_gap
        sheet_h = margin * 2 + header_h + panel_h
        sheet = Image.new("RGB", (sheet_w, sheet_h), (245, 247, 250))
        draw = ImageDraw.Draw(sheet)

        name = str(rows[0].get("name") or "")
        draw.text((margin, margin), f"{mall_id} {name}", fill=(17, 24, 39), font=title_font)
        draw.text(
            (margin, margin + 36),
            "左：原始卫星图；右：带标注总图（红色疑似已有光伏，黄色疑似可铺设屋面/硬化面）。",
            fill=(75, 85, 99),
            font=label_font,
        )
        left_x = margin
        right_x = margin + panel_w + panel_gap
        panels_y = margin + header_h
        draw.text((left_x, panels_y), "原图", fill=(17, 24, 39), font=panel_font)
        draw.text((right_x, panels_y), "标注图", fill=(17, 24, 39), font=panel_font)

        for row in rows:
            dx = float(row.get("offset_x_m") or 0)
            dy = float(row.get("offset_y_m") or 0)
            col = x_index[dx]
            row_index = y_index[dy]
            y = panels_y + panel_title_h + row_index * cell_h
            for panel_name, panel_x, image in (
                ("raw", left_x, image_from_row_path(row, "image_path")),
                ("annotated", right_x, annotated_tile_from_row(row, thumb_px)),
            ):
                x = panel_x + col * cell_w
                tile = resize_to_fit(image, thumb_px) if image is not None else None
                paste_thumbnail(
                    sheet,
                    tile,
                    (x, y + label_h, x + tile_w, y + label_h + tile_h),
                    fill=(30, 41, 59),
                )
                if panel_name == "annotated" and abs(dx) < 1e-6 and abs(dy) < 1e-6:
                    draw.rectangle(
                        (x, y, x + tile_w, y + tile_h),
                        outline=(37, 99, 235),
                        width=4,
                    )

        output_path = output_dir / f"mall_{safe_filename(mall_id)}_review_mosaic.jpg"
        sheet.save(output_path, quality=92)
        mosaic_paths[mall_id] = str(output_path)

    return mosaic_paths


def offset_lonlat(lon: float, lat: float, dx_m: float, dy_m: float) -> tuple[float, float]:
    lat_delta = dy_m / 111_320.0
    lon_delta = dx_m / max(111_320.0 * math.cos(math.radians(lat)), 1e-9)
    return lon + lon_delta, lat + lat_delta


def potential_offsets(radius_m: int, max_tiles: int) -> list[tuple[float, float, float]]:
    if max_tiles <= 0:
        return []
    step = max(radius_m / 2, 250)
    values = [0.0]
    n = int(math.ceil(radius_m / step))
    for index in range(1, n + 1):
        values.extend([index * step, -index * step])

    offsets: list[tuple[float, float, float]] = []
    for dx in values:
        for dy in values:
            distance = math.hypot(dx, dy)
            if distance <= radius_m + 1e-6:
                offsets.append((dx, dy, distance))
    offsets.sort(key=lambda item: (item[2], abs(item[0]) + abs(item[1]), item[0], item[1]))
    return offsets[:max_tiles]


def classify_pv(has_pv: bool, confidence: float, coverage: float) -> str:
    if has_pv and confidence >= 0.65:
        return "likely_pv"
    if has_pv or confidence >= 0.48 or coverage >= 0.001:
        return "possible_pv"
    return "no_clear_pv"


def _roof_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    vegetation = (hue >= 35) & (hue <= 95) & (sat >= 35) & (val >= 45)
    water = (hue >= 88) & (hue <= 132) & (sat >= 35) & (val <= 170)
    very_dark = val <= 35
    roof_like = (
        (((sat <= 68) & (val >= 95) & (val <= 245)) | ((sat <= 95) & (val >= 155)))
        & ~vegetation
        & ~water
        & ~very_dark
    )

    mask = roof_like.astype(np.uint8) * 255
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    return mask


def score_roof_potential(
    image_path: Path,
    *,
    mpp: float,
    overlay_dir: Path | None,
    min_roof_pixels: int,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = _roof_mask(rgb)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    kept = np.zeros_like(mask)
    components = 0
    area_px = 0
    image_area = int(mask.shape[0] * mask.shape[1])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_roof_pixels or w < 28 or h < 28:
            continue
        if area > image_area * 0.18:
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        extent = area / max(w * h, 1)
        edge_density = float(np.count_nonzero(edges[y : y + h, x : x + w])) / max(w * h, 1)
        if aspect > 7.0 or extent < 0.28 or edge_density < 0.010:
            continue
        kept[labels == label] = 255
        components += 1
        area_px += int(area)

    roof_area_m2 = area_px * (mpp**2)
    score = min(1.0, (roof_area_m2 / 25_000.0) * 0.75 + min(components / 10.0, 1.0) * 0.25)
    overlay_path = ""
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        highlight = np.zeros_like(overlay)
        highlight[:, :, 0] = 255
        highlight[:, :, 1] = 224
        overlay = np.where(kept[:, :, None] > 0, (overlay * 0.55 + highlight * 0.45), overlay)
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        contours, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 215, 255), 2)
        output_path = overlay_dir / f"{image_path.stem}_roof_overlay.png"
        cv2.imwrite(str(output_path), overlay_bgr)
        overlay_path = str(output_path)

    return {
        "roof_score": round(float(score), 4),
        "roof_candidate_count": components,
        "roof_area_px": area_px,
        "roof_area_m2_est": round(float(roof_area_m2), 2),
        "roof_overlay_path": overlay_path,
    }


def condition_level(score: float, roof_area_m2: float, tile_count: int) -> str:
    if tile_count == 0:
        return "unknown"
    if score >= 0.60 and roof_area_m2 >= 20_000:
        return "high"
    if score >= 0.30 and roof_area_m2 >= 8_000:
        return "medium"
    if roof_area_m2 >= 3_000:
        return "low"
    return "not_obvious"


def save_crop(
    *,
    lon: float,
    lat: float,
    zoom: int,
    size: int,
    cache_dir: Path,
    timeout: int,
    output_path: Path,
    overwrite: bool,
) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not output_path.exists():
        image = build_crop(lon=lon, lat=lat, zoom=zoom, size=size, cache_dir=cache_dir, timeout=timeout)
        image.save(output_path, quality=95)
    return str(output_path)


def run_self_detection(
    row: dict[str, str],
    *,
    run_id: str,
    dirs: dict[str, Path],
    pv_engine: PvDetectionEngine,
    zoom: int,
    size: int,
    cache_dir: Path,
    timeout: int,
    overwrite: bool,
) -> dict[str, Any]:
    mall_id = row["mall_id"]
    name = row.get("name") or row.get("selected_poi_name") or "mall"
    lon = float(row["center_lon"])
    lat = float(row["center_lat"])
    stem = safe_filename(f"mall_{mall_id}_self_z{zoom}")
    image_path = dirs["self_images"] / f"{stem}.jpg"
    result: dict[str, Any] = {
        "run_id": run_id,
        "mall_id": mall_id,
        "name": name,
        "confidence_level": row.get("confidence_level", ""),
        "center_lon": f"{lon:.8f}",
        "center_lat": f"{lat:.8f}",
        "center_source": row.get("center_source", ""),
        "image_path": "",
        "image_status": "error",
        "pv_status": "not_checked",
        "pv_confidence": "",
        "pv_area_px": "",
        "pv_area_m2_est": "",
        "pv_coverage": "",
        "component_count": "",
        "mask_path": "",
        "overlay_path": "",
        "reason": "",
        "review_required": 1,
        "review_status": "pending",
        "error": "",
        "method": METHOD,
    }
    try:
        saved = save_crop(
            lon=lon,
            lat=lat,
            zoom=zoom,
            size=size,
            cache_dir=cache_dir,
            timeout=timeout,
            output_path=image_path,
            overwrite=overwrite,
        )
        detection, pv_status, pv_method = pv_engine.detect(
            saved,
            min_pv_pixels=600,
            min_coverage=0.0012,
            mask_dir=dirs["self_masks"],
            overlay_dir=dirs["self_overlays"],
        )
        mpp = meters_per_pixel(lat, zoom)
        result.update(
            {
                "image_path": saved,
                "image_status": "ok",
                "pv_status": pv_status,
                "pv_confidence": detection.confidence,
                "pv_area_px": detection.pv_area_px,
                "pv_area_m2_est": round(detection.pv_area_px * (mpp**2), 2),
                "pv_coverage": detection.coverage,
                "component_count": detection.component_count,
                "mask_path": detection.mask_path,
                "overlay_path": detection.overlay_path,
                "reason": detection.reason,
                "method": pv_method,
            }
        )
    except Exception as exc:
        result["error"] = repr(exc)
    return result


def run_surrounding_detection(
    row: dict[str, str],
    *,
    run_id: str,
    dirs: dict[str, Path],
    pv_engine: PvDetectionEngine,
    zoom: int,
    size: int,
    radius_m: int,
    max_tiles: int,
    cache_dir: Path,
    timeout: int,
    overwrite: bool,
    min_roof_pixels: int,
    delay: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    mall_id = row["mall_id"]
    name = row.get("name") or row.get("selected_poi_name") or "mall"
    lon = float(row["center_lon"])
    lat = float(row["center_lat"])
    tile_rows: list[dict[str, Any]] = []
    totals = {
        "downloaded": 0,
        "roof_tiles": 0,
        "roof_components": 0,
        "roof_area_m2": 0.0,
        "pv_tiles": 0,
        "pv_area_m2": 0.0,
        "pv_methods": set(),
    }

    for index, (dx, dy, distance) in enumerate(potential_offsets(radius_m, max_tiles), start=1):
        tile_lon, tile_lat = offset_lonlat(lon, lat, dx, dy)
        tile_id = f"{mall_id}_{index:03d}"
        stem = safe_filename(f"mall_{mall_id}_tile_{index:03d}_buffer_z{zoom}")
        image_path = dirs["buffer_images"] / f"{stem}.jpg"
        tile_row: dict[str, Any] = {
            "run_id": run_id,
            "tile_id": tile_id,
            "mall_id": mall_id,
            "name": name,
            "tile_center_lon": round(tile_lon, 8),
            "tile_center_lat": round(tile_lat, 8),
            "offset_x_m": round(dx, 2),
            "offset_y_m": round(dy, 2),
            "distance_to_center_m": round(distance, 2),
            "image_path": "",
            "image_status": "error",
            "roof_score": "",
            "roof_candidate_count": "",
            "roof_area_px": "",
            "roof_area_m2_est": "",
            "pv_status": "not_checked",
            "pv_confidence": "",
            "pv_area_m2_est": "",
            "roof_overlay_path": "",
            "pv_overlay_path": "",
            "error": "",
            "method": METHOD,
        }
        try:
            saved = save_crop(
                lon=tile_lon,
                lat=tile_lat,
                zoom=zoom,
                size=size,
                cache_dir=cache_dir,
                timeout=timeout,
                output_path=image_path,
                overwrite=overwrite,
            )
            totals["downloaded"] += 1
            mpp = meters_per_pixel(tile_lat, zoom)
            roof = score_roof_potential(
                Path(saved),
                mpp=mpp,
                overlay_dir=dirs["roof_overlays"],
                min_roof_pixels=min_roof_pixels,
            )
            detection, pv_status, pv_method = pv_engine.detect(
                saved,
                min_pv_pixels=500,
                min_coverage=0.0010,
                mask_dir=None,
                overlay_dir=dirs["buffer_pv_overlays"],
            )
            pv_area_m2 = detection.pv_area_px * (mpp**2)
            totals["pv_methods"].add(pv_method)
            if roof["roof_candidate_count"]:
                totals["roof_tiles"] += 1
            if pv_status in {"likely_pv", "possible_pv"}:
                totals["pv_tiles"] += 1
            totals["roof_components"] += int(roof["roof_candidate_count"])
            totals["roof_area_m2"] += float(roof["roof_area_m2_est"])
            totals["pv_area_m2"] += float(pv_area_m2)
            tile_row.update(
                {
                    "image_path": saved,
                    "image_status": "ok",
                    "roof_score": roof["roof_score"],
                    "roof_candidate_count": roof["roof_candidate_count"],
                    "roof_area_px": roof["roof_area_px"],
                    "roof_area_m2_est": roof["roof_area_m2_est"],
                    "pv_status": pv_status,
                    "pv_confidence": detection.confidence,
                    "pv_area_m2_est": round(float(pv_area_m2), 2),
                    "roof_overlay_path": roof["roof_overlay_path"],
                    "pv_overlay_path": detection.overlay_path,
                    "method": pv_method,
                }
            )
        except Exception as exc:
            tile_row["error"] = repr(exc)
        tile_rows.append(tile_row)
        if delay > 0:
            time.sleep(delay)

    tile_count = len(tile_rows)
    aggregate_score = min(
        1.0,
        (totals["roof_area_m2"] / 50_000.0) * 0.75
        + min(totals["roof_tiles"] / max(tile_count, 1), 1.0) * 0.25,
    )
    level = condition_level(aggregate_score, totals["roof_area_m2"], totals["downloaded"])
    summary = {
        "run_id": run_id,
        "mall_id": mall_id,
        "name": name,
        "center_lon": f"{lon:.8f}",
        "center_lat": f"{lat:.8f}",
        "buffer_radius_m": radius_m,
        "tile_count": tile_count,
        "downloaded_tile_count": totals["downloaded"],
        "roof_candidate_tile_count": totals["roof_tiles"],
        "roof_candidate_count": totals["roof_components"],
        "roof_area_m2_est": round(float(totals["roof_area_m2"]), 2),
        "existing_pv_tile_count": totals["pv_tiles"],
        "existing_pv_area_m2_est": round(float(totals["pv_area_m2"]), 2),
        "install_condition_score": round(float(aggregate_score), 4),
        "install_condition_level": level,
        "review_required": 1,
        "review_status": "pending",
        "evidence_dir": str(dirs["run_dir"]),
        "method": f"roof_heuristic_v0+pv_{'+'.join(sorted(totals['pv_methods'])) or pv_engine.detector}",
        "notes": "PV uses BDAPPV model inference; roof potential uses an OpenCV heuristic; manual review required",
    }
    return summary, tile_rows


def write_report(path: Path, *, run_id: str, self_rows: list[dict[str, Any]], potential_rows: list[dict[str, Any]]) -> None:
    pv_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    for row in self_rows:
        pv_counts[str(row.get("pv_status") or "unknown")] = pv_counts.get(str(row.get("pv_status") or "unknown"), 0) + 1
    for row in potential_rows:
        level_counts[str(row.get("install_condition_level") or "unknown")] = (
            level_counts.get(str(row.get("install_condition_level") or "unknown"), 0) + 1
        )
    lines = [
        f"# 商场光伏与周边铺设条件初筛报告",
        "",
        f"- run_id: `{run_id}`",
        f"- 方法: `{METHOD}`，卫星图像启发式初筛，结果必须人工复核后再用于结论。",
        f"- 商场本体检测数量: {len(self_rows)}",
        f"- 周边条件检测数量: {len(potential_rows)}",
        "",
        "## 商场本体疑似光伏",
    ]
    for key in sorted(pv_counts):
        lines.append(f"- {key}: {pv_counts[key]}")
    lines.extend(["", "## 周边铺设条件"])
    for key in sorted(level_counts):
        lines.append(f"- {key}: {level_counts[key]}")
    lines.extend(
        [
            "",
            "## 字段说明",
            "- `mall_self_pv_results.csv`: 商场中心切图上的疑似光伏检测结果和证据图路径。",
            "- `mall_1km_potential_summary.csv`: 1km 周边采样影像的可铺设条件汇总。",
            "- `mall_1km_potential_tiles.csv`: 每张周边采样影像的屋面候选、疑似已有光伏和证据图。",
            "- `images/review_mosaics/`: 每个商场一张无缝复核图，左侧为原图，右侧为同图标注版。",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen approved mall centers for existing PV and nearby PV installation potential."
    )
    parser.add_argument("--center-csv", type=Path, default=DEFAULT_CENTER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--no-timestamp", action="store_true")
    parser.add_argument("--ids", default="", help="Comma-separated mall_id values.")
    parser.add_argument("--confidence-levels", default="", help="Comma-separated center ratings, for example A or A,B.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--self-zoom", type=int, default=18)
    parser.add_argument("--self-size-px", type=int, default=1024)
    parser.add_argument("--buffer-zoom", type=int, default=18)
    parser.add_argument("--buffer-size-px", type=int, default=768)
    parser.add_argument("--radius-m", type=int, default=1000)
    parser.add_argument("--max-potential-tiles", type=int, default=13)
    parser.add_argument("--min-roof-pixels", type=int, default=900)
    parser.add_argument("--pv-detector", choices=["bdappv", "heuristic"], default="bdappv")
    parser.add_argument("--bdappv-model-dir", type=Path, default=DEFAULT_BDAPPV_MODEL_DIR)
    parser.add_argument("--bdappv-provider", choices=["google", "ign"], default="google")
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--bdappv-batch-size", type=int, default=12)
    parser.add_argument("--bdappv-window-size", type=int, default=400)
    parser.add_argument("--bdappv-stride", type=int, default=312)
    parser.add_argument("--bdappv-possible-threshold", type=float, default=0.45)
    parser.add_argument("--bdappv-likely-threshold", type=float, default=0.75)
    parser.add_argument("--bdappv-segmentation-threshold", type=float, default=0.50)
    parser.add_argument("--bdappv-segment-score-threshold", type=float, default=0.45)
    parser.add_argument("--bdappv-min-component-pixels", type=int, default=500)
    parser.add_argument("--no-heuristic-fallback", action="store_true")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--delay", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-review-mosaic", action="store_true")
    parser.add_argument("--mosaic-thumb-px", type=int, default=220)
    parser.add_argument("--review-map-zoom", type=int, default=18)
    parser.add_argument("--review-map-size-px", type=int, default=0, help="0 means auto size for the requested radius.")
    parser.add_argument("--review-map-max-size-px", type=int, default=4096)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id.strip() or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args.output_root if args.no_timestamp else args.output_root / run_id
    dirs = {
        "run_dir": run_dir,
        "data": run_dir / "data",
        "reports": run_dir / "reports",
        "self_images": run_dir / "images" / "mall_self_crops",
        "self_masks": run_dir / "images" / "pv_masks",
        "self_overlays": run_dir / "images" / "pv_overlays",
        "buffer_images": run_dir / "images" / "mall_1km_buffer_crops",
        "roof_overlays": run_dir / "images" / "roof_overlays",
        "buffer_pv_overlays": run_dir / "images" / "buffer_pv_overlays",
        "review_mosaics": run_dir / "images" / "review_mosaics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    ids = {item.strip() for item in args.ids.split(",") if item.strip()} or None
    confidence_levels = {item.strip().upper() for item in args.confidence_levels.split(",") if item.strip()} or None
    centers = load_centers(args.center_csv, args.limit, ids, confidence_levels)
    print(f"Loading PV detector={args.pv_detector} provider={args.bdappv_provider} device={args.device}")
    pv_engine = PvDetectionEngine(
        detector=args.pv_detector,
        model_dir=args.bdappv_model_dir,
        provider=args.bdappv_provider,
        device=args.device,
        batch_size=args.bdappv_batch_size,
        window_size=args.bdappv_window_size,
        stride=args.bdappv_stride,
        possible_threshold=args.bdappv_possible_threshold,
        likely_threshold=args.bdappv_likely_threshold,
        segmentation_threshold=args.bdappv_segmentation_threshold,
        segment_score_threshold=args.bdappv_segment_score_threshold,
        min_component_pixels=args.bdappv_min_component_pixels,
        allow_heuristic_fallback=not args.no_heuristic_fallback,
    )
    print(f"PV detector ready: requested={args.pv_detector} device={pv_engine.device}")
    self_rows: list[dict[str, Any]] = []
    potential_rows: list[dict[str, Any]] = []
    tile_rows: list[dict[str, Any]] = []

    for index, row in enumerate(centers, start=1):
        name = row.get("name") or row.get("selected_poi_name") or row.get("mall_id")
        print(f"[{index}/{len(centers)}] {row.get('mall_id')} {name}")
        self_rows.append(
            run_self_detection(
                row,
                run_id=run_id,
                dirs=dirs,
                pv_engine=pv_engine,
                zoom=args.self_zoom,
                size=args.self_size_px,
                cache_dir=args.cache_dir,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
        )
        summary, tiles = run_surrounding_detection(
            row,
            run_id=run_id,
            dirs=dirs,
            pv_engine=pv_engine,
            zoom=args.buffer_zoom,
            size=args.buffer_size_px,
            radius_m=args.radius_m,
            max_tiles=args.max_potential_tiles,
            cache_dir=args.cache_dir,
            timeout=args.timeout,
            overwrite=args.overwrite,
            min_roof_pixels=args.min_roof_pixels,
            delay=args.delay,
        )
        potential_rows.append(summary)
        tile_rows.extend(tiles)
        if args.checkpoint_every > 0 and index % args.checkpoint_every == 0:
            write_detection_checkpoints(dirs["data"], self_rows, potential_rows, tile_rows)
            print(f"checkpoint={index}/{len(centers)}")

    write_detection_checkpoints(dirs["data"], self_rows, potential_rows, tile_rows)

    if not args.skip_review_mosaic:
        mosaic_paths = build_full_review_mosaics(
            centers,
            pv_engine=pv_engine,
            output_dir=dirs["review_mosaics"],
            cache_dir=args.cache_dir,
            timeout=args.timeout,
            overwrite=args.overwrite,
            zoom=args.review_map_zoom,
            radius_m=args.radius_m,
            size_px=args.review_map_size_px,
            max_size_px=args.review_map_max_size_px,
            min_roof_pixels=args.min_roof_pixels,
        )
        for row in potential_rows:
            row["review_mosaic_path"] = mosaic_paths.get(str(row.get("mall_id") or ""), "")
    else:
        for row in potential_rows:
            row["review_mosaic_path"] = ""

    write_csv(
        dirs["data"] / "approved_centers_input.csv",
        centers,
        list(centers[0].keys()) if centers else ["mall_id", "name", "center_lon", "center_lat"],
    )
    write_csv(
        dirs["data"] / "mall_self_pv_results.csv",
        self_rows,
        [
            "run_id",
            "mall_id",
            "name",
            "confidence_level",
            "center_lon",
            "center_lat",
            "center_source",
            "image_path",
            "image_status",
            "pv_status",
            "pv_confidence",
            "pv_area_px",
            "pv_area_m2_est",
            "pv_coverage",
            "component_count",
            "mask_path",
            "overlay_path",
            "reason",
            "review_required",
            "review_status",
            "error",
            "method",
        ],
    )
    write_csv(
        dirs["data"] / "mall_1km_potential_summary.csv",
        potential_rows,
        [
            "run_id",
            "mall_id",
            "name",
            "center_lon",
            "center_lat",
            "buffer_radius_m",
            "tile_count",
            "downloaded_tile_count",
            "roof_candidate_tile_count",
            "roof_candidate_count",
            "roof_area_m2_est",
            "existing_pv_tile_count",
            "existing_pv_area_m2_est",
            "install_condition_score",
            "install_condition_level",
            "review_required",
            "review_status",
            "evidence_dir",
            "review_mosaic_path",
            "method",
            "notes",
        ],
    )
    write_csv(
        dirs["data"] / "mall_1km_potential_tiles.csv",
        tile_rows,
        [
            "run_id",
            "tile_id",
            "mall_id",
            "name",
            "tile_center_lon",
            "tile_center_lat",
            "offset_x_m",
            "offset_y_m",
            "distance_to_center_m",
            "image_path",
            "image_status",
            "roof_score",
            "roof_candidate_count",
            "roof_area_px",
            "roof_area_m2_est",
            "pv_status",
            "pv_confidence",
            "pv_area_m2_est",
            "roof_overlay_path",
            "pv_overlay_path",
            "error",
            "method",
        ],
    )
    write_report(
        dirs["reports"] / "pv_potential_screening_report.md",
        run_id=run_id,
        self_rows=self_rows,
        potential_rows=potential_rows,
    )

    print(f"run_dir={run_dir}")
    print(f"centers={len(centers)}")
    print(f"self_results={dirs['data'] / 'mall_self_pv_results.csv'}")
    print(f"potential_summary={dirs['data'] / 'mall_1km_potential_summary.csv'}")
    print(f"tile_results={dirs['data'] / 'mall_1km_potential_tiles.csv'}")
    print(f"review_mosaics={dirs['review_mosaics']}")


if __name__ == "__main__":
    main()
