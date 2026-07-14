from __future__ import annotations

import argparse
import csv
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image, ImageDraw

from make_mall_poi_review_gallery import (
    ESRI_TILE_URL,
    TILE_SIZE,
    lonlat_to_global_pixel,
    safe_filename,
)
from run_mall_pv_potential_screening import (
    DEFAULT_BDAPPV_MODEL_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_CENTER_CSV,
    DEFAULT_OUTPUT_ROOT,
    PvDetectionEngine,
    auto_review_map_size_px,
    condition_level,
    load_centers,
    load_font,
    meters_per_pixel,
    write_csv,
    _roof_mask,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GIT_RSCLIP_MODEL_DIR = PROJECT_ROOT / "Git-RSCLIP"
METHOD = "single_map_z18+roof_heuristic_v1+gitr_sclip_filter_v1+pv_bdappv"

ROOF_POSITIVE_PROMPTS = [
    "a remote sensing image of a factory roof",
    "a satellite image of a warehouse roof",
    "a satellite image of a logistics warehouse",
    "a top-down image of a large flat industrial roof",
    "a commercial or industrial rooftop suitable for solar panels",
]

ROOF_NEGATIVE_PROMPTS = [
    "a satellite image of a stadium or sports arena",
    "a satellite image of a sports field or running track",
    "a satellite image of a school campus",
    "a satellite image of a residential neighborhood",
    "a satellite image of red tile residential roofs",
    "a satellite image of roads and intersections",
    "a satellite image of a park or forest",
    "a satellite image of a lake or river",
    "a satellite image of a shopping mall plaza or atrium",
]

SELF_FIELDS = [
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
]

SUMMARY_FIELDS = [
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
    "roof_semantic_filter",
    "roof_semantic_pass_count",
    "roof_semantic_reject_count",
    "roof_semantic_review_count",
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
    "image_status",
    "error",
]

DETAIL_FIELDS = [
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
    "roof_semantic_filter",
    "roof_semantic_pass_count",
    "roof_semantic_reject_count",
    "roof_semantic_review_count",
    "roof_semantic_notes",
    "pv_status",
    "pv_confidence",
    "pv_area_m2_est",
    "roof_overlay_path",
    "pv_overlay_path",
    "error",
    "method",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Screen each mall from one complete high-resolution 1 km satellite map."
    )
    parser.add_argument("--center-csv", type=Path, default=DEFAULT_CENTER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--confidence-levels", default="A")
    parser.add_argument("--ids", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--radius-m", type=int, default=1000)
    parser.add_argument("--map-size-px", type=int, default=0, help="0 computes the pixel size from radius and zoom.")
    parser.add_argument("--map-max-size-px", type=int, default=4096)
    parser.add_argument("--self-size-px", type=int, default=1024)
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--download-attempts", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--min-roof-pixels", type=int, default=900)
    parser.add_argument("--bdappv-model-dir", type=Path, default=DEFAULT_BDAPPV_MODEL_DIR)
    parser.add_argument("--bdappv-provider", choices=["google", "ign"], default="google")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--bdappv-batch-size", type=int, default=12)
    parser.add_argument("--bdappv-window-size", type=int, default=400)
    parser.add_argument("--bdappv-stride", type=int, default=312)
    parser.add_argument("--bdappv-possible-threshold", type=float, default=0.45)
    parser.add_argument("--bdappv-likely-threshold", type=float, default=0.75)
    parser.add_argument("--bdappv-segmentation-threshold", type=float, default=0.50)
    parser.add_argument("--bdappv-segment-score-threshold", type=float, default=0.45)
    parser.add_argument("--bdappv-min-component-pixels", type=int, default=500)
    parser.add_argument("--roof-semantic-filter", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--roof-semantic-model-dir", type=Path, default=DEFAULT_GIT_RSCLIP_MODEL_DIR)
    parser.add_argument("--roof-semantic-batch-size", type=int, default=8)
    parser.add_argument(
        "--roof-semantic-min-positive",
        type=float,
        default=-4.0,
        help="Minimum best-positive Git-RSCLIP logit for a kept roof component.",
    )
    parser.add_argument(
        "--roof-semantic-pass-delta",
        type=float,
        default=2.0,
        help="Minimum positive-vs-negative Git-RSCLIP logit margin for a kept roof component.",
    )
    parser.add_argument(
        "--roof-semantic-review-delta",
        type=float,
        default=0.0,
        help="Components below pass but above this logit margin are marked review, not counted as area.",
    )
    parser.add_argument("--roof-semantic-crop-padding", type=float, default=0.15)
    return parser.parse_args()


def _tile_cache_path(cache_dir: Path, z: int, x: int, y: int) -> Path:
    return cache_dir / str(z) / str(y) / f"{x % (2**z)}.jpg"


def read_tile_resilient(
    z: int,
    x: int,
    y: int,
    *,
    cache_dir: Path,
    timeout: int,
    attempts: int,
) -> Image.Image:
    wrapped_x = x % (2**z)
    cache_path = _tile_cache_path(cache_dir, z, wrapped_x, y)
    if cache_path.exists():
        try:
            with Image.open(cache_path) as cached:
                return cached.convert("RGB")
        except OSError:
            cache_path.unlink(missing_ok=True)

    error: Exception | None = None
    for attempt in range(1, max(attempts, 1) + 1):
        try:
            response = requests.get(
                ESRI_TILE_URL.format(z=z, x=wrapped_x, y=y),
                timeout=(5, timeout),
                headers={"User-Agent": "pv-single-map-screening/2.0"},
            )
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_suffix(f".{time.time_ns()}.tmp")
            image.save(temp_path, format="JPEG", quality=92)
            temp_path.replace(cache_path)
            return image
        except (requests.RequestException, OSError) as exc:
            error = exc
            if attempt < attempts:
                time.sleep(min(0.5 * (2 ** (attempt - 1)), 4.0))
    raise RuntimeError(f"tile download failed z={z} x={wrapped_x} y={y}: {error!r}")


def build_crop_concurrent(
    lon: float,
    lat: float,
    zoom: int,
    size: int,
    *,
    cache_dir: Path,
    timeout: int,
    attempts: int,
    workers: int,
) -> Image.Image:
    center_x, center_y = lonlat_to_global_pixel(lon, lat, zoom)
    half = size / 2
    left = center_x - half
    top = center_y - half
    right = center_x + half
    bottom = center_y + half
    min_tile_x = math.floor(left / TILE_SIZE)
    max_tile_x = math.floor((right - 1) / TILE_SIZE)
    min_tile_y = max(0, math.floor(top / TILE_SIZE))
    max_tile_y = min((2**zoom) - 1, math.floor((bottom - 1) / TILE_SIZE))
    coordinates = [
        (tile_x, tile_y)
        for tile_y in range(min_tile_y, max_tile_y + 1)
        for tile_x in range(min_tile_x, max_tile_x + 1)
    ]
    mosaic = Image.new(
        "RGB",
        ((max_tile_x - min_tile_x + 1) * TILE_SIZE, (max_tile_y - min_tile_y + 1) * TILE_SIZE),
    )
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {
            executor.submit(
                read_tile_resilient,
                zoom,
                tile_x,
                tile_y,
                cache_dir=cache_dir,
                timeout=timeout,
                attempts=attempts,
            ): (tile_x, tile_y)
            for tile_x, tile_y in coordinates
        }
        for future in as_completed(futures):
            tile_x, tile_y = futures[future]
            tile = future.result()
            mosaic.paste(tile, ((tile_x - min_tile_x) * TILE_SIZE, (tile_y - min_tile_y) * TILE_SIZE))
    crop_left = round(left - min_tile_x * TILE_SIZE)
    crop_top = round(top - min_tile_y * TILE_SIZE)
    return mosaic.crop((crop_left, crop_top, crop_left + size, crop_top + size))


def center_box(width: int, height: int, size: int) -> tuple[int, int, int, int]:
    size = min(size, width, height)
    left = (width - size) // 2
    top = (height - size) // 2
    return left, top, left + size, top + size


def mask_metrics(mask_path: Path, box: tuple[int, int, int, int], mpp: float) -> dict[str, Any]:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"Unable to read PV mask: {mask_path}")
    left, top, right, bottom = box
    binary = mask[top:bottom, left:right] > 0
    area_px = int(np.count_nonzero(binary))
    image_area_px = int(binary.size)
    coverage = area_px / image_area_px if image_area_px else 0.0
    components = max(0, cv2.connectedComponents(binary.astype(np.uint8), 8)[0] - 1)
    if area_px >= 1000 and coverage >= 0.001:
        status = "likely_pv"
    elif area_px >= 500:
        status = "possible_pv"
    else:
        status = "no_clear_pv"
    return {
        "pv_status": status,
        "pv_area_px": area_px,
        "pv_area_m2_est": round(area_px * (mpp**2), 2),
        "pv_coverage": round(coverage, 6),
        "component_count": components,
    }


class RoofSemanticFilter:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: str,
        batch_size: int,
        min_positive: float,
        pass_delta: float,
        review_delta: float,
    ) -> None:
        self.model_dir = model_dir
        self.device = device
        self.batch_size = max(1, batch_size)
        self.min_positive = min_positive
        self.pass_delta = pass_delta
        self.review_delta = review_delta
        self.prompts = ROOF_POSITIVE_PROMPTS + ROOF_NEGATIVE_PROMPTS
        self.positive_count = len(ROOF_POSITIVE_PROMPTS)
        self.torch: Any | None = None
        self.processor: Any | None = None
        self.model: Any | None = None
        self.available = False
        self.error = ""
        self._load()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
            from transformers.utils import logging as hf_logging

            if not (self.model_dir / "model.safetensors").exists():
                raise FileNotFoundError(f"Missing model weights: {self.model_dir / 'model.safetensors'}")
            if self.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device.startswith("cuda") and not torch.cuda.is_available():
                self.device = "cpu"
            hf_logging.set_verbosity_error()
            self.processor = AutoProcessor.from_pretrained(str(self.model_dir), local_files_only=True)
            self.model = AutoModel.from_pretrained(str(self.model_dir), local_files_only=True).to(self.device).eval()
            self.torch = torch
            self.available = True
        except Exception as exc:
            self.error = repr(exc)
            self.available = False

    def score(self, crops: list[Image.Image]) -> list[dict[str, Any]]:
        if not crops or not self.available or self.processor is None or self.model is None or self.torch is None:
            return []
        results: list[dict[str, Any]] = []
        for batch_start in range(0, len(crops), self.batch_size):
            crop_batch = crops[batch_start : batch_start + self.batch_size]
            inputs = self.processor(text=self.prompts, images=crop_batch, padding="max_length", return_tensors="pt")
            inputs = {input_key: input_value.to(self.device) for input_key, input_value in inputs.items()}
            with self.torch.no_grad():
                logits = self.model(**inputs).logits_per_image.detach().float().cpu()
            for score_row in logits:
                positive_scores = score_row[: self.positive_count]
                negative_scores = score_row[self.positive_count :]
                best_positive_value, best_positive_index = self.torch.max(positive_scores, dim=0)
                best_negative_value, best_negative_index = self.torch.max(negative_scores, dim=0)
                positive_score = float(best_positive_value.item())
                negative_score = float(best_negative_value.item())
                delta = positive_score - negative_score
                if positive_score >= self.min_positive and delta >= self.pass_delta:
                    decision = "pass"
                elif delta >= self.review_delta:
                    decision = "review"
                else:
                    decision = "reject"
                results.append(
                    {
                        "decision": decision,
                        "best_positive_score": round(positive_score, 4),
                        "best_positive_label": ROOF_POSITIVE_PROMPTS[int(best_positive_index.item())],
                        "best_negative_score": round(negative_score, 4),
                        "best_negative_label": ROOF_NEGATIVE_PROMPTS[int(best_negative_index.item())],
                        "semantic_delta": round(delta, 4),
                    }
                )
        return results


def _padded_crop_box(
    *,
    left: int,
    top: int,
    width_px: int,
    height_px: int,
    image_width: int,
    image_height: int,
    padding_ratio: float,
) -> tuple[int, int, int, int]:
    pad_x = int(round(width_px * padding_ratio))
    pad_y = int(round(height_px * padding_ratio))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image_width, left + width_px + pad_x),
        min(image_height, top + height_px + pad_y),
    )


def _short_prompt(prompt: str) -> str:
    return (
        prompt.replace("a satellite image of ", "")
        .replace("a remote sensing image of ", "")
        .replace("a top-down image of ", "")
        .replace("a commercial or industrial ", "")
    )


def _roof_semantic_notes(components: list[dict[str, Any]], *, limit: int = 8) -> str:
    snippets: list[str] = []
    for component in components[:limit]:
        semantic = component.get("semantic", {})
        snippets.append(
            "{}:{} pos={}({}) neg={}({}) d={}".format(
                component.get("label_id"),
                semantic.get("decision", "pass"),
                _short_prompt(str(semantic.get("best_positive_label", ""))),
                semantic.get("best_positive_score", ""),
                _short_prompt(str(semantic.get("best_negative_label", ""))),
                semantic.get("best_negative_score", ""),
                semantic.get("semantic_delta", ""),
            )
        )
    if len(components) > limit:
        snippets.append(f"...+{len(components) - limit}")
    return "; ".join(snippets)


def score_roof_potential_semantic(
    image_path: Path,
    *,
    mpp: float,
    overlay_dir: Path | None,
    min_roof_pixels: int,
    semantic_filter: RoofSemanticFilter | None,
    crop_padding: float,
) -> dict[str, Any]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = _roof_mask(rgb)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)

    image_height, image_width = mask.shape
    image_area = int(image_height * image_width)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    components: list[dict[str, Any]] = []
    crops: list[Image.Image] = []
    pil_image = Image.fromarray(rgb)
    for label_id in range(1, num_labels):
        left, top, width_px, height_px, area = stats[label_id]
        if area < min_roof_pixels or width_px < 28 or height_px < 28:
            continue
        if area > image_area * 0.18:
            continue
        aspect = max(width_px / max(height_px, 1), height_px / max(width_px, 1))
        extent = area / max(width_px * height_px, 1)
        edge_density = float(np.count_nonzero(edges[top : top + height_px, left : left + width_px])) / max(
            width_px * height_px, 1
        )
        if aspect > 7.0 or extent < 0.28 or edge_density < 0.010:
            continue
        crop_box = _padded_crop_box(
            left=int(left),
            top=int(top),
            width_px=int(width_px),
            height_px=int(height_px),
            image_width=image_width,
            image_height=image_height,
            padding_ratio=crop_padding,
        )
        components.append(
            {
                "label_id": int(label_id),
                "area": int(area),
                "bbox": (int(left), int(top), int(width_px), int(height_px)),
                "crop_box": crop_box,
                "semantic": {"decision": "pass"},
            }
        )
        crops.append(pil_image.crop(crop_box))

    filter_status = "disabled"
    if semantic_filter is not None:
        filter_status = "gitr_sclip" if semantic_filter.available else f"unavailable:{semantic_filter.error}"
    if semantic_filter is not None and semantic_filter.available:
        semantic_scores = semantic_filter.score(crops)
        for component, semantic in zip(components, semantic_scores):
            component["semantic"] = semantic

    kept = np.zeros_like(mask)
    review_mask = np.zeros_like(mask)
    reject_mask = np.zeros_like(mask)
    pass_count = 0
    review_count = 0
    reject_count = 0
    area_px = 0
    for component in components:
        component_mask = labels == component["label_id"]
        decision = component.get("semantic", {}).get("decision", "pass")
        if decision == "pass":
            kept[component_mask] = 255
            pass_count += 1
            area_px += int(component["area"])
        elif decision == "review":
            review_mask[component_mask] = 255
            review_count += 1
        else:
            reject_mask[component_mask] = 255
            reject_count += 1

    roof_area_m2 = area_px * (mpp**2)
    score = min(1.0, (roof_area_m2 / 25_000.0) * 0.75 + min(pass_count / 10.0, 1.0) * 0.25)
    overlay_path = ""
    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        pass_highlight = np.zeros_like(overlay)
        pass_highlight[:, :, 0] = 255
        pass_highlight[:, :, 1] = 224
        overlay = np.where(kept[:, :, None] > 0, (overlay * 0.55 + pass_highlight * 0.45), overlay)
        overlay_bgr = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_RGB2BGR)
        pass_contours, _ = cv2.findContours(kept, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_bgr, pass_contours, -1, (0, 215, 255), 2)
        output_path = overlay_dir / f"{image_path.stem}_roof_overlay.png"
        cv2.imwrite(str(output_path), overlay_bgr)
        overlay_path = str(output_path)

    return {
        "roof_score": round(float(score), 4),
        "roof_candidate_count": pass_count,
        "roof_area_px": area_px,
        "roof_area_m2_est": round(float(roof_area_m2), 2),
        "roof_overlay_path": overlay_path,
        "roof_semantic_filter": filter_status,
        "roof_semantic_pass_count": pass_count,
        "roof_semantic_reject_count": reject_count,
        "roof_semantic_review_count": review_count,
        "roof_semantic_notes": _roof_semantic_notes(components),
    }


def combine_overlays(base: Image.Image, overlays: list[Image.Image]) -> Image.Image:
    base_arr = np.asarray(base.convert("RGB"), dtype=np.int16)
    output = base_arr.copy()
    for overlay in overlays:
        overlay_arr = np.asarray(overlay.resize(base.size).convert("RGB"), dtype=np.int16)
        changed = np.abs(overlay_arr - base_arr).sum(axis=2) > 36
        output[changed] = overlay_arr[changed]
    return Image.fromarray(np.clip(output, 0, 255).astype(np.uint8), mode="RGB")


def replace_with_jpeg(source_path: Path, destination_path: Path, *, quality: int = 90) -> str:
    with Image.open(source_path) as image:
        image.convert("RGB").save(destination_path, quality=quality, optimize=True)
    if source_path != destination_path:
        source_path.unlink(missing_ok=True)
    return str(destination_path)


def make_review_mosaic(
    *,
    mall_id: str,
    name: str,
    raw: Image.Image,
    annotated: Image.Image,
    zoom: int,
    radius_m: int,
    output_path: Path,
) -> None:
    display_max = 1400
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
    sheet = Image.new(
        "RGB",
        (margin * 2 + panel_w * 2 + panel_gap, margin * 2 + header_h + panel_title_h + panel_h),
        (245, 247, 250),
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), f"{mall_id} {name}", fill=(17, 24, 39), font=load_font(24))
    draw.text(
        (margin, margin + 38),
        f"周边 {radius_m}m：红色为疑似已有光伏，黄色为疑似可铺设区域；zoom={zoom}",
        fill=(75, 85, 99),
        font=load_font(16),
    )
    left_x = margin
    right_x = margin + panel_w + panel_gap
    panel_y = margin + header_h
    image_y = panel_y + panel_title_h
    draw.text((left_x, panel_y), "原图", fill=(17, 24, 39), font=load_font(20))
    draw.text((right_x, panel_y), "标注图", fill=(17, 24, 39), font=load_font(20))
    sheet.paste(raw_display, (left_x + (panel_w - raw_display.width) // 2, image_y))
    sheet.paste(annotated_display, (right_x + (panel_w - annotated_display.width) // 2, image_y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=92)


def load_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        return {str(row.get("mall_id") or ""): row for row in csv.DictReader(src) if row.get("mall_id")}


def save_results(
    data_dir: Path,
    self_by_id: dict[str, dict[str, Any]],
    summary_by_id: dict[str, dict[str, Any]],
    detail_by_id: dict[str, dict[str, Any]],
    *,
    final: bool,
) -> None:
    suffix = "" if final else "_checkpoint"
    write_csv(data_dir / f"mall_self_pv_results{suffix}.csv", list(self_by_id.values()), SELF_FIELDS)
    write_csv(data_dir / f"mall_1km_potential_summary{suffix}.csv", list(summary_by_id.values()), SUMMARY_FIELDS)
    write_csv(data_dir / f"mall_1km_potential_tiles{suffix}.csv", list(detail_by_id.values()), DETAIL_FIELDS)


def process_center(
    center: dict[str, str],
    *,
    args: argparse.Namespace,
    run_dir: Path,
    engine: PvDetectionEngine,
    roof_filter: RoofSemanticFilter | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mall_id = str(center["mall_id"])
    name = str(center.get("name") or center.get("selected_poi_name") or "mall")
    lon = float(center["center_lon"])
    lat = float(center["center_lat"])
    image_size = args.map_size_px or auto_review_map_size_px(lat, args.zoom, args.radius_m, args.map_max_size_px)
    stem = safe_filename(f"mall_{mall_id}_full_{args.radius_m}m_z{args.zoom}")
    dirs = {
        "raw": run_dir / "images" / "full_1km_raw",
        "masks": run_dir / "images" / "full_pv_masks",
        "pv": run_dir / "images" / "full_pv_overlays",
        "roof": run_dir / "images" / "full_roof_overlays",
        "annotated": run_dir / "images" / "full_annotated",
        "self": run_dir / "images" / "mall_self_crops",
        "self_masks": run_dir / "images" / "pv_masks",
        "self_pv": run_dir / "images" / "pv_overlays",
        "review": run_dir / "images" / "review_mosaics",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    raw_path = dirs["raw"] / f"{stem}.jpg"
    if args.overwrite or not raw_path.exists():
        image = build_crop_concurrent(
            lon,
            lat,
            args.zoom,
            image_size,
            cache_dir=args.cache_dir,
            timeout=args.timeout,
            attempts=args.download_attempts,
            workers=args.download_workers,
        )
        image.save(raw_path, quality=95)
    raw = Image.open(raw_path).convert("RGB")
    mpp = meters_per_pixel(lat, args.zoom)
    roof = score_roof_potential_semantic(
        raw_path,
        mpp=mpp,
        overlay_dir=dirs["roof"],
        min_roof_pixels=max(args.min_roof_pixels, int((image_size * image_size) * 0.0012)),
        semantic_filter=roof_filter if args.roof_semantic_filter else None,
        crop_padding=args.roof_semantic_crop_padding,
    )
    pv, pv_status, pv_method = engine.detect(
        raw_path,
        min_pv_pixels=max(800, int((image_size * image_size) * 0.0008)),
        min_coverage=0.0008,
        mask_dir=dirs["masks"],
        overlay_dir=dirs["pv"],
    )
    roof_png_path = Path(roof["roof_overlay_path"])
    pv_png_path = Path(pv.overlay_path)
    roof_overlay = Image.open(roof_png_path).convert("RGB")
    pv_overlay = Image.open(pv_png_path).convert("RGB")
    annotated = combine_overlays(raw, [roof_overlay, pv_overlay])
    roof_overlay_path = replace_with_jpeg(
        roof_png_path,
        dirs["roof"] / f"{stem}_roof_overlay.jpg",
        quality=90,
    )
    if pv.pv_area_px:
        pv_overlay_path = replace_with_jpeg(
            pv_png_path,
            dirs["pv"] / f"{stem}_pv_overlay.jpg",
            quality=90,
        )
    else:
        pv_png_path.unlink(missing_ok=True)
        pv_overlay_path = ""
    annotated_path = dirs["annotated"] / f"{stem}_annotated.jpg"
    annotated.save(annotated_path, quality=94)
    review_path = dirs["review"] / f"mall_{safe_filename(mall_id)}_review_mosaic.jpg"
    make_review_mosaic(
        mall_id=mall_id,
        name=name,
        raw=raw,
        annotated=annotated,
        zoom=args.zoom,
        radius_m=args.radius_m,
        output_path=review_path,
    )

    box = center_box(raw.width, raw.height, args.self_size_px)
    self_raw_path = dirs["self"] / f"mall_{safe_filename(mall_id)}_self_z{args.zoom}.jpg"
    self_mask_path = dirs["self_masks"] / f"mall_{safe_filename(mall_id)}_self_z{args.zoom}_mask.png"
    self_overlay_path = dirs["self_pv"] / f"mall_{safe_filename(mall_id)}_self_z{args.zoom}_overlay.jpg"
    raw.crop(box).save(self_raw_path, quality=95)
    Image.open(pv.mask_path).crop(box).save(self_mask_path)
    pv_overlay.crop(box).save(self_overlay_path, quality=92, optimize=True)
    self_metrics = mask_metrics(Path(pv.mask_path), box, mpp)
    self_row = {
        "run_id": args.run_id,
        "mall_id": mall_id,
        "name": name,
        "confidence_level": center.get("confidence_level", ""),
        "center_lon": center["center_lon"],
        "center_lat": center["center_lat"],
        "center_source": center.get("center_source", ""),
        "image_path": str(self_raw_path),
        "image_status": "ok",
        **self_metrics,
        "pv_confidence": pv.confidence,
        "mask_path": str(self_mask_path),
        "overlay_path": str(self_overlay_path),
        "reason": f"derived from the center {args.self_size_px}px of one full-map BDAPPV mask; {pv.reason}",
        "review_required": 1,
        "review_status": "pending",
        "error": "",
        "method": pv_method,
    }
    effective_pv_status = pv_status if pv.pv_area_px >= args.bdappv_min_component_pixels else "no_clear_pv"
    condition = condition_level(float(roof["roof_score"]), float(roof["roof_area_m2_est"]), 1)
    summary_row = {
        "run_id": args.run_id,
        "mall_id": mall_id,
        "name": name,
        "center_lon": center["center_lon"],
        "center_lat": center["center_lat"],
        "buffer_radius_m": args.radius_m,
        "tile_count": 1,
        "downloaded_tile_count": 1,
        "roof_candidate_tile_count": 1 if roof["roof_candidate_count"] else 0,
        "roof_candidate_count": roof["roof_candidate_count"],
        "roof_area_m2_est": roof["roof_area_m2_est"],
        "roof_semantic_filter": roof["roof_semantic_filter"],
        "roof_semantic_pass_count": roof["roof_semantic_pass_count"],
        "roof_semantic_reject_count": roof["roof_semantic_reject_count"],
        "roof_semantic_review_count": roof["roof_semantic_review_count"],
        "existing_pv_tile_count": 1 if effective_pv_status in {"possible_pv", "likely_pv"} else 0,
        "existing_pv_area_m2_est": round(pv.pv_area_px * (mpp**2), 2),
        "install_condition_score": roof["roof_score"],
        "install_condition_level": condition,
        "review_required": 1,
        "review_status": "pending",
        "evidence_dir": str(run_dir / "images"),
        "review_mosaic_path": str(review_path),
        "method": METHOD,
        "notes": "One complete high-resolution map; PV uses BDAPPV; roof potential uses OpenCV + Git-RSCLIP semantic filter; manual review required",
        "image_status": "ok",
        "error": "",
    }
    detail_row = {
        "run_id": args.run_id,
        "tile_id": f"mall_{mall_id}_full_{args.radius_m}m",
        "mall_id": mall_id,
        "name": name,
        "tile_center_lon": center["center_lon"],
        "tile_center_lat": center["center_lat"],
        "offset_x_m": 0,
        "offset_y_m": 0,
        "distance_to_center_m": 0,
        "image_path": str(raw_path),
        "image_status": "ok",
        "roof_score": roof["roof_score"],
        "roof_candidate_count": roof["roof_candidate_count"],
        "roof_area_px": roof["roof_area_px"],
        "roof_area_m2_est": roof["roof_area_m2_est"],
        "roof_semantic_filter": roof["roof_semantic_filter"],
        "roof_semantic_pass_count": roof["roof_semantic_pass_count"],
        "roof_semantic_reject_count": roof["roof_semantic_reject_count"],
        "roof_semantic_review_count": roof["roof_semantic_review_count"],
        "roof_semantic_notes": roof["roof_semantic_notes"],
        "pv_status": effective_pv_status,
        "pv_confidence": pv.confidence,
        "pv_area_m2_est": round(pv.pv_area_px * (mpp**2), 2),
        "roof_overlay_path": roof_overlay_path,
        "pv_overlay_path": pv_overlay_path,
        "error": "",
        "method": pv_method,
    }
    return self_row, summary_row, detail_row


def error_rows(center: dict[str, str], args: argparse.Namespace, error: Exception) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mall_id = str(center["mall_id"])
    name = str(center.get("name") or center.get("selected_poi_name") or "mall")
    message = repr(error)
    self_row = {
        "run_id": args.run_id,
        "mall_id": mall_id,
        "name": name,
        "confidence_level": center.get("confidence_level", ""),
        "center_lon": center["center_lon"],
        "center_lat": center["center_lat"],
        "image_status": "error",
        "pv_status": "not_checked",
        "review_required": 1,
        "review_status": "pending",
        "error": message,
        "method": METHOD,
    }
    summary_row = {
        "run_id": args.run_id,
        "mall_id": mall_id,
        "name": name,
        "center_lon": center["center_lon"],
        "center_lat": center["center_lat"],
        "buffer_radius_m": args.radius_m,
        "tile_count": 1,
        "downloaded_tile_count": 0,
        "install_condition_level": "unknown",
        "review_required": 1,
        "review_status": "pending",
        "method": METHOD,
        "image_status": "error",
        "error": message,
    }
    detail_row = {
        "run_id": args.run_id,
        "tile_id": f"mall_{mall_id}_full_{args.radius_m}m",
        "mall_id": mall_id,
        "name": name,
        "tile_center_lon": center["center_lon"],
        "tile_center_lat": center["center_lat"],
        "offset_x_m": 0,
        "offset_y_m": 0,
        "distance_to_center_m": 0,
        "image_status": "error",
        "pv_status": "not_checked",
        "error": message,
        "method": METHOD,
    }
    return self_row, summary_row, detail_row


def write_report(path: Path, *, total: int, ok: int, failed: int, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            [
                "# Shanghai mall PV single-map screening",
                "",
                f"- Run ID: `{args.run_id}`",
                f"- Centers: {total}",
                f"- Completed: {ok}",
                f"- Failed: {failed}",
                f"- Radius: {args.radius_m} m",
                f"- Zoom: {args.zoom}",
                "- Imagery: one complete map per mall, assembled from cached Esri tiles",
                "- PV: open-source BDAPPV classification and segmentation",
                "- Roof potential: OpenCV heuristic candidates + Git-RSCLIP semantic filtering",
                "- Review status: pending manual confirmation",
            ]
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    levels = {value.strip().upper() for value in args.confidence_levels.split(",") if value.strip()}
    ids = {value.strip() for value in args.ids.split(",") if value.strip()} or None
    centers = load_centers(args.center_csv, args.limit, ids, levels or None)
    run_dir = args.output_root / args.run_id
    data_dir = run_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    self_by_id = load_checkpoint(data_dir / "mall_self_pv_results_checkpoint.csv") if args.resume else {}
    summary_by_id = load_checkpoint(data_dir / "mall_1km_potential_summary_checkpoint.csv") if args.resume else {}
    detail_by_id = load_checkpoint(data_dir / "mall_1km_potential_tiles_checkpoint.csv") if args.resume else {}
    engine = PvDetectionEngine(
        detector="bdappv",
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
        allow_heuristic_fallback=False,
    )
    roof_filter = (
        RoofSemanticFilter(
            model_dir=args.roof_semantic_model_dir,
            device=args.device,
            batch_size=args.roof_semantic_batch_size,
            min_positive=args.roof_semantic_min_positive,
            pass_delta=args.roof_semantic_pass_delta,
            review_delta=args.roof_semantic_review_delta,
        )
        if args.roof_semantic_filter
        else None
    )
    if roof_filter is not None and not roof_filter.available:
        print(f"warning=roof_semantic_filter_unavailable {roof_filter.error}", flush=True)
    processed = 0
    for index, center in enumerate(centers, start=1):
        mall_id = str(center["mall_id"])
        previous = summary_by_id.get(mall_id, {})
        if args.resume and previous.get("image_status") == "ok" and Path(previous.get("review_mosaic_path", "")).exists():
            continue
        try:
            self_row, summary_row, detail_row = process_center(
                center,
                args=args,
                run_dir=run_dir,
                engine=engine,
                roof_filter=roof_filter,
            )
            print(f"ok={index}/{len(centers)} mall_id={mall_id}", flush=True)
        except Exception as exc:
            self_row, summary_row, detail_row = error_rows(center, args, exc)
            print(f"error={index}/{len(centers)} mall_id={mall_id} {exc!r}", flush=True)
        self_by_id[mall_id] = self_row
        summary_by_id[mall_id] = summary_row
        detail_by_id[mall_id] = detail_row
        processed += 1
        if processed % max(args.checkpoint_every, 1) == 0:
            save_results(data_dir, self_by_id, summary_by_id, detail_by_id, final=False)
            print(f"checkpoint={len(summary_by_id)}/{len(centers)}", flush=True)
    save_results(data_dir, self_by_id, summary_by_id, detail_by_id, final=False)
    save_results(data_dir, self_by_id, summary_by_id, detail_by_id, final=True)
    write_csv(data_dir / "approved_centers_input.csv", centers, list(centers[0].keys()) if centers else ["mall_id"])
    ok = sum(1 for row in summary_by_id.values() if row.get("image_status") == "ok")
    failed = len(summary_by_id) - ok
    write_report(run_dir / "reports" / "pv_single_map_screening_report.md", total=len(centers), ok=ok, failed=failed, args=args)
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
