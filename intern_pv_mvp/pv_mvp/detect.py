from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .io_utils import ensure_dir, safe_filename


@dataclass(frozen=True)
class DetectionResult:
    mall_id: str
    name: str
    image_path: str
    pv_status: str
    pv_area_px: int
    pv_ratio: float
    roof_candidate_area_px: int
    roof_candidate_ratio: float
    remaining_roof_proxy_px: int
    potential_level: str
    potential_reason: str
    mask_path: str
    pv_overlay_path: str
    roof_overlay_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def pv_candidate_mask(rgb: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    b_lab = lab[:, :, 2]
    rgb_i = rgb.astype(np.int16)
    blue_advantage = rgb_i[:, :, 2] - ((rgb_i[:, :, 0] + rgb_i[:, :, 1]) // 2)

    blue_panel = (
        (hue >= 82)
        & (hue <= 130)
        & (sat >= 22)
        & (val >= 35)
        & (val <= 185)
        & (blue_advantage >= 8)
    )
    dark_panel = (
        (val >= 25)
        & (val <= 115)
        & (sat >= 12)
        & (blue_advantage >= 8)
        & (b_lab <= 150)
    )
    mask = (blue_panel | dark_panel).astype("uint8") * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)), iterations=2)
    return filter_components(mask, min_area=80, min_w=14, min_h=6, min_extent=0.16)


def filter_components(mask: np.ndarray, *, min_area: int, min_w: int, min_h: int, min_extent: float) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < min_area or w < min_w or h < min_h:
            continue
        extent = area / max(w * h, 1)
        aspect = max(w / max(h, 1), h / max(w, 1))
        if extent >= min_extent and aspect <= 45:
            kept[labels == label] = 255
    return kept


def roof_candidate_mask(rgb: np.ndarray, pv_mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    green = (hue >= 35) & (hue <= 85) & (sat >= 45) & (val >= 45)
    water = (hue >= 88) & (hue <= 125) & (sat >= 45) & (val >= 35)
    very_dark = val < 30
    very_bright = val > 235

    neutral_roof = (sat <= 85) & (val >= 55) & (val <= 225)
    blue_gray_roof = (hue >= 82) & (hue <= 128) & (sat >= 15) & (sat <= 115) & (val >= 45) & (val <= 190)
    brown_or_red_roof = (hue <= 22) & (sat >= 25) & (sat <= 140) & (val >= 55) & (val <= 205)

    candidate = (neutral_roof | blue_gray_roof | brown_or_red_roof) & (~green) & (~water) & (~very_dark) & (~very_bright)
    candidate = candidate.astype("uint8") * 255
    candidate[pv_mask > 0] = 255

    candidate = cv2.medianBlur(candidate, 5)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)), iterations=1)

    edges = cv2.Canny(gray, 40, 120)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    kept = np.zeros_like(candidate)
    image_area = candidate.shape[0] * candidate.shape[1]
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < max(1800, int(image_area * 0.004)):
            continue
        if area > int(image_area * 0.18):
            continue
        aspect = max(w / max(h, 1), h / max(w, 1))
        extent = area / max(w * h, 1)
        edge_density = float(np.count_nonzero(edges[y : y + h, x : x + w])) / max(w * h, 1)
        if aspect <= 12 and extent >= 0.30 and edge_density >= 0.015:
            kept[labels == label] = 255
    return kept


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float = 0.42) -> Image.Image:
    base = rgb.astype(np.float32)
    tint = np.zeros_like(base)
    tint[:, :] = np.array(color, dtype=np.float32)
    blended = np.where(mask[:, :, None] > 0, base * (1 - alpha) + tint * alpha, base)
    return Image.fromarray(np.clip(blended, 0, 255).astype("uint8"))


def classify_pv(pv_ratio: float, pv_area_px: int) -> str:
    if pv_area_px >= 1200 and pv_ratio >= 0.003:
        return "likely_pv"
    if pv_area_px >= 400 and pv_ratio >= 0.001:
        return "possible_pv"
    return "no_clear_pv"


def classify_potential(roof_ratio: float, pv_ratio: float) -> tuple[str, str]:
    if roof_ratio >= 0.04 and pv_ratio < 0.02:
        return "high", "large regular roof candidates and limited existing PV detected"
    if roof_ratio >= 0.006:
        return "medium", "some roof candidates exist; manual review needed"
    return "low", "few large regular roof candidates in this crop"


def detect_image(
    *,
    mall_id: str,
    name: str,
    image_path: str | Path,
    output_dir: str | Path,
) -> DetectionResult:
    output_dir = ensure_dir(output_dir)
    mask_dir = ensure_dir(output_dir / "masks")
    overlay_dir = ensure_dir(output_dir / "overlays")
    rgb = load_rgb(image_path)
    image_area = int(rgb.shape[0] * rgb.shape[1])

    pv_mask = pv_candidate_mask(rgb)
    roof_mask = roof_candidate_mask(rgb, pv_mask)

    pv_area = int(np.count_nonzero(pv_mask))
    roof_area = int(np.count_nonzero(roof_mask))
    remaining = max(0, roof_area - pv_area)
    pv_ratio = pv_area / max(image_area, 1)
    roof_ratio = roof_area / max(image_area, 1)
    pv_status = classify_pv(pv_ratio, pv_area)
    potential_level, potential_reason = classify_potential(roof_ratio, pv_ratio)

    stem = f"{safe_filename(mall_id, 'mall')}_{safe_filename(name, 'mall')}"
    mask_path = mask_dir / f"{stem}_pv_mask.png"
    pv_overlay_path = overlay_dir / f"{stem}_pv_overlay.jpg"
    roof_overlay_path = overlay_dir / f"{stem}_roof_candidates.jpg"

    Image.fromarray(pv_mask).save(mask_path)
    overlay_mask(rgb, pv_mask, (255, 56, 32)).save(pv_overlay_path, quality=94)
    overlay_mask(rgb, roof_mask, (255, 190, 40)).save(roof_overlay_path, quality=94)

    return DetectionResult(
        mall_id=mall_id,
        name=name,
        image_path=str(image_path),
        pv_status=pv_status,
        pv_area_px=pv_area,
        pv_ratio=round(pv_ratio, 6),
        roof_candidate_area_px=roof_area,
        roof_candidate_ratio=round(roof_ratio, 6),
        remaining_roof_proxy_px=remaining,
        potential_level=potential_level,
        potential_reason=potential_reason,
        mask_path=str(mask_path),
        pv_overlay_path=str(pv_overlay_path),
        roof_overlay_path=str(roof_overlay_path),
    )
