from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectionResult:
    image_path: str
    has_pv: bool
    confidence: float
    pv_area_px: int
    image_area_px: int
    coverage: float
    component_count: int
    mean_panel_score: float
    reason: str
    mask_path: str = ""
    overlay_path: str = ""


def _read_rgb(image_path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def _candidate_mask(rgb: np.ndarray) -> np.ndarray:
    """Find dark blue-gray, panel-like regions in overhead imagery.

    This is only a baseline heuristic. It intentionally favors recall so the
    review page can surface suspicious rooftops for human checking.
    """

    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    b_lab = lab[:, :, 2]

    rgb_i = rgb.astype(np.int16)
    blue_advantage = rgb_i[:, :, 2] - ((rgb_i[:, :, 0] + rgb_i[:, :, 1]) // 2)

    blue_panel = (
        (hue >= 82)
        & (hue <= 128)
        & (sat >= 20)
        & (val >= 35)
        & (val <= 180)
        & (blue_advantage >= 8)
    )
    dark_panel = (
        (val >= 25)
        & (val <= 105)
        & (sat >= 15)
        & (blue_advantage >= 12)
        & (b_lab <= 145)
    )

    mask = (blue_panel | dark_panel).astype(np.uint8) * 255

    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_row = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_row, iterations=2)

    return mask


def _score_components(mask: np.ndarray, rgb: np.ndarray) -> tuple[np.ndarray, int, float]:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    kept = np.zeros_like(mask)
    scores: list[float] = []
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 40, 120)

    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if area < 70 or w < 16 or h < 7:
            continue

        aspect = max(w / max(h, 1), h / max(w, 1))
        extent = area / max(w * h, 1)
        edge_density = float(np.count_nonzero(edges[y : y + h, x : x + w])) / max(w * h, 1)

        # PV arrays in overhead imagery tend to form compact strips or blocks.
        shape_score = 0.0
        if 1.2 <= aspect <= 32.0:
            shape_score += 0.35
        if 0.22 <= extent <= 0.92:
            shape_score += 0.25
        if edge_density >= 0.035:
            shape_score += 0.25
        if area >= 90:
            shape_score += 0.15

        if shape_score >= 0.60:
            kept[labels == label] = 255
            scores.append(min(shape_score, 1.0))

    component_count = len(scores)
    mean_score = float(np.mean(scores)) if scores else 0.0
    return kept, component_count, mean_score


def _write_mask_and_overlay(
    rgb: np.ndarray,
    mask: np.ndarray,
    image_path: Path,
    mask_dir: Path | None,
    overlay_dir: Path | None,
) -> tuple[str, str]:
    mask_path = ""
    overlay_path = ""

    if mask_dir is not None:
        mask_dir.mkdir(parents=True, exist_ok=True)
        mask_file = mask_dir / f"{image_path.stem}_mask.png"
        cv2.imwrite(str(mask_file), mask)
        mask_path = str(mask_file)

    if overlay_dir is not None:
        overlay_dir.mkdir(parents=True, exist_ok=True)
        overlay = rgb.copy()
        highlight = np.zeros_like(overlay)
        highlight[:, :, 0] = 255
        alpha = 0.45
        overlay = np.where(mask[:, :, None] > 0, (overlay * (1 - alpha) + highlight * alpha), overlay)
        overlay = overlay.astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.drawContours(overlay_bgr, contours, -1, (0, 255, 255), 2)

        overlay_file = overlay_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_file), overlay_bgr)
        overlay_path = str(overlay_file)

    return mask_path, overlay_path


def detect_pv(
    image_path: str | Path,
    *,
    min_pv_pixels: int = 600,
    min_coverage: float = 0.0015,
    mask_dir: str | Path | None = None,
    overlay_dir: str | Path | None = None,
) -> DetectionResult:
    image_path = Path(image_path)
    rgb = _read_rgb(image_path)

    raw_mask = _candidate_mask(rgb)
    mask, component_count, mean_score = _score_components(raw_mask, rgb)

    pv_area_px = int(np.count_nonzero(mask))
    image_area_px = int(mask.shape[0] * mask.shape[1])
    coverage = pv_area_px / image_area_px if image_area_px else 0.0

    area_ok = pv_area_px >= min_pv_pixels
    coverage_ok = coverage >= min_coverage
    components_ok = component_count >= 2 or coverage >= (min_coverage * 3)
    has_pv = area_ok and coverage_ok and components_ok

    confidence = min(
        0.98,
        0.20
        + 0.35 * min(pv_area_px / max(min_pv_pixels, 1), 1.0)
        + 0.25 * min(coverage / max(min_coverage, 1e-9), 1.0)
        + 0.20 * mean_score,
    )
    if not has_pv:
        confidence = min(confidence, 0.49)

    reasons = []
    reasons.append(f"pv_pixels={pv_area_px}")
    reasons.append(f"coverage={coverage:.4%}")
    reasons.append(f"components={component_count}")
    if has_pv:
        reasons.append("decision=suspected_pv")
    else:
        reasons.append("decision=no_pv_or_below_threshold")

    mask_path, overlay_path = _write_mask_and_overlay(
        rgb,
        mask,
        image_path,
        Path(mask_dir) if mask_dir is not None else None,
        Path(overlay_dir) if overlay_dir is not None else None,
    )

    return DetectionResult(
        image_path=str(image_path),
        has_pv=has_pv,
        confidence=round(float(confidence), 4),
        pv_area_px=pv_area_px,
        image_area_px=image_area_px,
        coverage=round(float(coverage), 6),
        component_count=component_count,
        mean_panel_score=round(float(mean_score), 4),
        reason="; ".join(reasons),
        mask_path=mask_path,
        overlay_path=overlay_path,
    )
