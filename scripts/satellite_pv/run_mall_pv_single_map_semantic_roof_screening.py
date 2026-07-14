from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from make_mall_poi_review_gallery import safe_filename
from run_mall_pv_potential_screening import (
    DEFAULT_BDAPPV_MODEL_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_CENTER_CSV,
    DEFAULT_OUTPUT_ROOT,
    PvDetectionEngine,
    _roof_mask,
    auto_review_map_size_px,
    condition_level,
    load_centers,
    meters_per_pixel,
    write_csv,
)
from run_mall_pv_single_map_screening import (
    build_crop_concurrent,
    center_box,
    combine_overlays,
    load_checkpoint,
    make_review_mosaic,
    mask_metrics,
    replace_with_jpeg,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GIT_RSCLIP_MODEL_DIR = PROJECT_ROOT / "Git-RSCLIP"
METHOD = "single_map_z18+semantic_roof_v1+pv_bdappv"

POSITIVE_ROOF_PROMPTS = [
    "a satellite image of an industrial factory roof",
    "a satellite image of a warehouse roof",
    "a satellite image of a logistics warehouse building roof",
    "a top-down satellite view of a large flat roof suitable for solar panels",
    "a satellite image of a large commercial rooftop with open usable area",
]

NEGATIVE_ROOF_PROMPTS = [
    "a satellite image of a stadium or sports arena",
    "a satellite image of a school campus or sports field",
    "a satellite image of a residential neighborhood with small houses",
    "a satellite image of roads and intersections",
    "a satellite image of a park, trees, or green area",
    "a satellite image of water or a lake",
    "a satellite image of a shopping mall or entertainment complex",
    "a satellite image of red tile residential roofs",
    "a satellite image of a construction site or vacant land",
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
    "semantic_roof_candidate_count",
    "rejected_roof_candidate_count",
    "roof_area_m2_est",
    "raw_roof_area_m2_est",
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
    "semantic_roof_candidate_count",
    "rejected_roof_candidate_count",
    "roof_area_px",
    "raw_roof_area_px",
    "roof_area_m2_est",
    "raw_roof_area_m2_est",
    "pv_status",
    "pv_confidence",
    "pv_area_m2_est",
    "roof_overlay_path",
    "pv_overlay_path",
    "error",
    "method",
]

COMPONENT_FIELDS = [
    "run_id",
    "component_id",
    "mall_id",
    "name",
    "component_rank",
    "x",
    "y",
    "width",
    "height",
    "area_px",
    "area_m2_est",
    "aspect",
    "extent",
    "edge_density",
    "semantic_keep",
    "semantic_decision",
    "best_positive_label",
    "best_positive_score",
    "best_negative_label",
    "best_negative_score",
    "semantic_delta",
    "positive_share",
    "crop_path",
    "method",
]


@dataclass(frozen=True)
class RoofCandidate:
    rank: int
    label: int
    x: int
    y: int
    width: int
    height: int
    area_px: int
    aspect: float
    extent: float
    edge_density: float


class SemanticRoofFilter:
    def __init__(
        self,
        *,
        model_dir: Path,
        device: str,
        delta_threshold: float,
        positive_share_threshold: float,
        batch_size: int,
    ) -> None:
        if not (model_dir / "model.safetensors").exists():
            raise FileNotFoundError(f"Missing Git-RSCLIP weights: {model_dir / 'model.safetensors'}")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.delta_threshold = delta_threshold
        self.positive_share_threshold = positive_share_threshold
        self.batch_size = max(1, batch_size)
        self.prompts = POSITIVE_ROOF_PROMPTS + NEGATIVE_ROOF_PROMPTS
        self.positive_count = len(POSITIVE_ROOF_PROMPTS)
        self.processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()

    def score(self, crops: list[Image.Image]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for start in range(0, len(crops), self.batch_size):
            batch = crops[start : start + self.batch_size]
            inputs = self.processor(text=self.prompts, images=batch, padding="max_length", return_tensors="pt")
            inputs = {key: value.to(self.device) if hasattr(value, "to") else value for key, value in inputs.items()}
            with torch.no_grad():
                logits = self.model(**inputs).logits_per_image.detach().float().cpu()
            for row in logits:
                pos_logits = row[: self.positive_count]
                neg_logits = row[self.positive_count :]
                best_pos_value, best_pos_index = torch.max(pos_logits, dim=0)
                best_neg_value, best_neg_index = torch.max(neg_logits, dim=0)
                softmax = torch.softmax(row, dim=0)
                positive_share = float(softmax[: self.positive_count].sum().item())
                delta = float((best_pos_value - best_neg_value).item())
                keep = delta >= self.delta_threshold and positive_share >= self.positive_share_threshold
                results.append(
                    {
                        "semantic_keep": keep,
                        "semantic_decision": "semantic_keep" if keep else "semantic_reject",
                        "best_positive_label": POSITIVE_ROOF_PROMPTS[int(best_pos_index.item())],
                        "best_positive_score": float(best_pos_value.item()),
                        "best_negative_label": NEGATIVE_ROOF_PROMPTS[int(best_neg_index.item())],
                        "best_negative_score": float(best_neg_value.item()),
                        "semantic_delta": delta,
                        "positive_share": positive_share,
                    }
                )
        return results


def parse_args() -> argparse.Namespace:
    default_run_id = datetime.now().strftime("%Y%m%d_%H%M%S_semantic_roof_v1")
    parser = argparse.ArgumentParser(
        description="Screen mall 1km maps with BDAPPV PV detection and Git-RSCLIP semantic roof filtering."
    )
    parser.add_argument("--center-csv", type=Path, default=DEFAULT_CENTER_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id", default=default_run_id)
    parser.add_argument("--confidence-levels", default="A")
    parser.add_argument("--ids", default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--zoom", type=int, default=18)
    parser.add_argument("--radius-m", type=int, default=1000)
    parser.add_argument("--map-size-px", type=int, default=0)
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
    parser.add_argument("--max-roof-components", type=int, default=64)
    parser.add_argument("--roof-crop-padding", type=int, default=48)
    parser.add_argument("--clip-model-dir", type=Path, default=DEFAULT_GIT_RSCLIP_MODEL_DIR)
    parser.add_argument("--clip-device", default="auto")
    parser.add_argument("--semantic-batch-size", type=int, default=8)
    parser.add_argument("--semantic-delta-threshold", type=float, default=0.04)
    parser.add_argument("--semantic-positive-share-threshold", type=float, default=0.46)
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
    return parser.parse_args()


def extract_roof_candidates(image_path: Path, *, min_roof_pixels: int, max_components: int) -> tuple[np.ndarray, list[RoofCandidate]]:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = _roof_mask(rgb)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    image_area = int(mask.shape[0] * mask.shape[1])
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 45, 135)
    candidates: list[RoofCandidate] = []
    for label in range(1, num_labels):
        x, y, width, height, area = stats[label]
        if area < min_roof_pixels or width < 28 or height < 28:
            continue
        if area > image_area * 0.18:
            continue
        aspect = max(width / max(height, 1), height / max(width, 1))
        extent = area / max(width * height, 1)
        edge_density = float(np.count_nonzero(edges[y : y + height, x : x + width])) / max(width * height, 1)
        if aspect > 7.0 or extent < 0.28 or edge_density < 0.010:
            continue
        candidates.append(
            RoofCandidate(
                rank=0,
                label=label,
                x=int(x),
                y=int(y),
                width=int(width),
                height=int(height),
                area_px=int(area),
                aspect=round(float(aspect), 4),
                extent=round(float(extent), 4),
                edge_density=round(float(edge_density), 6),
            )
        )
    candidates.sort(key=lambda item: item.area_px, reverse=True)
    ranked = [
        RoofCandidate(
            rank=index,
            label=item.label,
            x=item.x,
            y=item.y,
            width=item.width,
            height=item.height,
            area_px=item.area_px,
            aspect=item.aspect,
            extent=item.extent,
            edge_density=item.edge_density,
        )
        for index, item in enumerate(candidates[:max_components], start=1)
    ]
    return labels, ranked


def crop_candidate(image: Image.Image, candidate: RoofCandidate, padding: int) -> Image.Image:
    left = max(0, candidate.x - padding)
    top = max(0, candidate.y - padding)
    right = min(image.width, candidate.x + candidate.width + padding)
    bottom = min(image.height, candidate.y + candidate.height + padding)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    crop.thumbnail((768, 768))
    return crop


def write_semantic_roof_overlay(
    *,
    raw: Image.Image,
    labels: np.ndarray,
    kept_labels: set[int],
    output_path: Path,
) -> str:
    rgb = np.asarray(raw.convert("RGB"), dtype=np.uint8)
    mask = np.isin(labels, list(kept_labels)).astype(np.uint8) * 255 if kept_labels else np.zeros(labels.shape, np.uint8)
    overlay = rgb.copy()
    highlight = np.zeros_like(overlay)
    highlight[:, :, 0] = 255
    highlight[:, :, 1] = 224
    overlay = np.where(mask[:, :, None] > 0, (overlay * 0.55 + highlight * 0.45), overlay).astype(np.uint8)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_bgr, contours, -1, (0, 215, 255), 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay_bgr)
    return str(output_path)


def score_semantic_roofs(
    *,
    raw_path: Path,
    raw: Image.Image,
    mall_id: str,
    name: str,
    args: argparse.Namespace,
    dirs: dict[str, Path],
    semantic_filter: SemanticRoofFilter,
    mpp: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    image_size = raw.width * raw.height
    min_roof_pixels = max(args.min_roof_pixels, int(image_size * 0.0012))
    labels, candidates = extract_roof_candidates(
        raw_path,
        min_roof_pixels=min_roof_pixels,
        max_components=args.max_roof_components,
    )
    crops = [crop_candidate(raw, candidate, args.roof_crop_padding) for candidate in candidates]
    scores = semantic_filter.score(crops) if crops else []
    kept_labels: set[int] = set()
    component_rows: list[dict[str, Any]] = []
    raw_area_px = sum(candidate.area_px for candidate in candidates)
    kept_area_px = 0
    for candidate, score in zip(candidates, scores):
        component_id = f"mall_{safe_filename(mall_id)}_roof_{candidate.rank:03d}"
        crop_path = dirs["roof_crops"] / f"{component_id}.jpg"
        crops[candidate.rank - 1].save(crop_path, quality=92, optimize=True)
        keep = bool(score["semantic_keep"])
        if keep:
            kept_labels.add(candidate.label)
            kept_area_px += candidate.area_px
        component_rows.append(
            {
                "run_id": args.run_id,
                "component_id": component_id,
                "mall_id": mall_id,
                "name": name,
                "component_rank": candidate.rank,
                "x": candidate.x,
                "y": candidate.y,
                "width": candidate.width,
                "height": candidate.height,
                "area_px": candidate.area_px,
                "area_m2_est": round(candidate.area_px * (mpp**2), 2),
                "aspect": candidate.aspect,
                "extent": candidate.extent,
                "edge_density": candidate.edge_density,
                "semantic_keep": int(keep),
                "semantic_decision": score["semantic_decision"],
                "best_positive_label": score["best_positive_label"],
                "best_positive_score": f"{score['best_positive_score']:.6f}",
                "best_negative_label": score["best_negative_label"],
                "best_negative_score": f"{score['best_negative_score']:.6f}",
                "semantic_delta": f"{score['semantic_delta']:.6f}",
                "positive_share": f"{score['positive_share']:.6f}",
                "crop_path": str(crop_path),
                "method": METHOD,
            }
        )
    roof_area_m2 = kept_area_px * (mpp**2)
    raw_roof_area_m2 = raw_area_px * (mpp**2)
    score = min(1.0, (roof_area_m2 / 25_000.0) * 0.75 + min(len(kept_labels) / 10.0, 1.0) * 0.25)
    overlay_path = write_semantic_roof_overlay(
        raw=raw,
        labels=labels,
        kept_labels=kept_labels,
        output_path=dirs["roof"] / f"{raw_path.stem}_semantic_roof_overlay.png",
    )
    return (
        {
            "roof_score": round(float(score), 4),
            "roof_candidate_count": len(candidates),
            "semantic_roof_candidate_count": len(kept_labels),
            "rejected_roof_candidate_count": len(candidates) - len(kept_labels),
            "roof_area_px": kept_area_px,
            "raw_roof_area_px": raw_area_px,
            "roof_area_m2_est": round(float(roof_area_m2), 2),
            "raw_roof_area_m2_est": round(float(raw_roof_area_m2), 2),
            "roof_overlay_path": overlay_path,
        },
        component_rows,
    )


def load_component_checkpoint(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        return {str(row.get("component_id") or ""): row for row in csv.DictReader(source) if row.get("component_id")}


def save_results(
    data_dir: Path,
    self_by_id: dict[str, dict[str, Any]],
    summary_by_id: dict[str, dict[str, Any]],
    detail_by_id: dict[str, dict[str, Any]],
    component_by_id: dict[str, dict[str, Any]],
    *,
    final: bool,
) -> None:
    suffix = "" if final else "_checkpoint"
    write_csv(data_dir / f"mall_self_pv_results{suffix}.csv", list(self_by_id.values()), SELF_FIELDS)
    write_csv(data_dir / f"mall_1km_potential_summary{suffix}.csv", list(summary_by_id.values()), SUMMARY_FIELDS)
    write_csv(data_dir / f"mall_1km_potential_tiles{suffix}.csv", list(detail_by_id.values()), DETAIL_FIELDS)
    write_csv(data_dir / f"roof_semantic_components{suffix}.csv", list(component_by_id.values()), COMPONENT_FIELDS)


def process_center(
    center: dict[str, str],
    *,
    args: argparse.Namespace,
    run_dir: Path,
    pv_engine: PvDetectionEngine,
    semantic_filter: SemanticRoofFilter,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    mall_id = str(center["mall_id"])
    name = str(center.get("name") or center.get("selected_poi_name") or "mall")
    lon = float(center["center_lon"])
    lat = float(center["center_lat"])
    image_size = args.map_size_px or auto_review_map_size_px(lat, args.zoom, args.radius_m, args.map_max_size_px)
    stem = safe_filename(f"mall_{mall_id}_full_{args.radius_m}m_z{args.zoom}")
    dirs = {
        "raw": run_dir / "images" / "full_1km_raw",
        "pv": run_dir / "images" / "full_pv_overlays",
        "masks": run_dir / "images" / "full_pv_masks",
        "roof": run_dir / "images" / "full_semantic_roof_overlays",
        "annotated": run_dir / "images" / "full_annotated",
        "review": run_dir / "images" / "review_mosaics",
        "self": run_dir / "images" / "mall_self_crops",
        "self_masks": run_dir / "images" / "pv_masks",
        "self_pv": run_dir / "images" / "pv_overlays",
        "roof_crops": run_dir / "images" / "roof_component_crops",
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
    roof, component_rows = score_semantic_roofs(
        raw_path=raw_path,
        raw=raw,
        mall_id=mall_id,
        name=name,
        args=args,
        dirs=dirs,
        semantic_filter=semantic_filter,
        mpp=mpp,
    )
    pv, pv_status, pv_method = pv_engine.detect(
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
        dirs["roof"] / f"{stem}_semantic_roof_overlay.jpg",
        quality=90,
    )
    if pv.pv_area_px:
        pv_overlay_path = replace_with_jpeg(pv_png_path, dirs["pv"] / f"{stem}_pv_overlay.jpg", quality=90)
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
        "roof_candidate_tile_count": 1 if roof["semantic_roof_candidate_count"] else 0,
        "roof_candidate_count": roof["roof_candidate_count"],
        "semantic_roof_candidate_count": roof["semantic_roof_candidate_count"],
        "rejected_roof_candidate_count": roof["rejected_roof_candidate_count"],
        "roof_area_m2_est": roof["roof_area_m2_est"],
        "raw_roof_area_m2_est": roof["raw_roof_area_m2_est"],
        "existing_pv_tile_count": 1 if effective_pv_status in {"possible_pv", "likely_pv"} else 0,
        "existing_pv_area_m2_est": round(pv.pv_area_px * (mpp**2), 2),
        "install_condition_score": roof["roof_score"],
        "install_condition_level": condition,
        "review_required": 1,
        "review_status": "pending",
        "evidence_dir": str(run_dir / "images"),
        "review_mosaic_path": str(review_path),
        "method": METHOD,
        "notes": "One complete high-resolution map; PV uses BDAPPV; roof potential uses OpenCV candidates plus Git-RSCLIP semantic filtering; manual review required",
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
        "semantic_roof_candidate_count": roof["semantic_roof_candidate_count"],
        "rejected_roof_candidate_count": roof["rejected_roof_candidate_count"],
        "roof_area_px": roof["roof_area_px"],
        "raw_roof_area_px": roof["raw_roof_area_px"],
        "roof_area_m2_est": roof["roof_area_m2_est"],
        "raw_roof_area_m2_est": roof["raw_roof_area_m2_est"],
        "pv_status": effective_pv_status,
        "pv_confidence": pv.confidence,
        "pv_area_m2_est": round(pv.pv_area_px * (mpp**2), 2),
        "roof_overlay_path": roof_overlay_path,
        "pv_overlay_path": pv_overlay_path,
        "error": "",
        "method": pv_method,
    }
    return self_row, summary_row, detail_row, component_rows


def error_rows(
    center: dict[str, str],
    args: argparse.Namespace,
    error: Exception,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
                "# Shanghai mall PV semantic-roof single-map screening",
                "",
                f"- Run ID: `{args.run_id}`",
                f"- Centers: {total}",
                f"- Completed: {ok}",
                f"- Failed: {failed}",
                f"- Radius: {args.radius_m} m",
                f"- Zoom: {args.zoom}",
                f"- Method: `{METHOD}`",
                "- Imagery: one complete map per mall, assembled from cached Esri tiles",
                "- PV: open-source BDAPPV classification and segmentation, unchanged from the previous route",
                "- Roof potential: OpenCV roof candidates + Git-RSCLIP semantic filtering",
                "- Component audit: `data/roof_semantic_components.csv`",
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
    component_by_id = (
        load_component_checkpoint(data_dir / "roof_semantic_components_checkpoint.csv") if args.resume else {}
    )

    print(f"Loading semantic roof model: {args.clip_model_dir}", flush=True)
    semantic_filter = SemanticRoofFilter(
        model_dir=args.clip_model_dir,
        device=args.clip_device,
        delta_threshold=args.semantic_delta_threshold,
        positive_share_threshold=args.semantic_positive_share_threshold,
        batch_size=args.semantic_batch_size,
    )
    print(f"Semantic roof model ready on {semantic_filter.device}", flush=True)
    print(f"Loading PV detector: BDAPPV provider={args.bdappv_provider}", flush=True)
    pv_engine = PvDetectionEngine(
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
    print(f"PV detector ready on {pv_engine.device}", flush=True)

    processed = 0
    for index, center in enumerate(centers, start=1):
        mall_id = str(center["mall_id"])
        previous = summary_by_id.get(mall_id, {})
        if args.resume and previous.get("image_status") == "ok" and Path(previous.get("review_mosaic_path", "")).exists():
            continue
        try:
            self_row, summary_row, detail_row, component_rows = process_center(
                center,
                args=args,
                run_dir=run_dir,
                pv_engine=pv_engine,
                semantic_filter=semantic_filter,
            )
            for row in component_rows:
                component_by_id[str(row["component_id"])] = row
            print(
                "ok={}/{} mall_id={} semantic_roofs={} rejected={}".format(
                    index,
                    len(centers),
                    mall_id,
                    summary_row.get("semantic_roof_candidate_count", ""),
                    summary_row.get("rejected_roof_candidate_count", ""),
                ),
                flush=True,
            )
        except Exception as exc:
            self_row, summary_row, detail_row = error_rows(center, args, exc)
            print(f"error={index}/{len(centers)} mall_id={mall_id} {exc!r}", flush=True)
        self_by_id[mall_id] = self_row
        summary_by_id[mall_id] = summary_row
        detail_by_id[mall_id] = detail_row
        processed += 1
        if processed % max(args.checkpoint_every, 1) == 0:
            save_results(data_dir, self_by_id, summary_by_id, detail_by_id, component_by_id, final=False)
            print(f"checkpoint={len(summary_by_id)}/{len(centers)}", flush=True)

    save_results(data_dir, self_by_id, summary_by_id, detail_by_id, component_by_id, final=False)
    save_results(data_dir, self_by_id, summary_by_id, detail_by_id, component_by_id, final=True)
    write_csv(data_dir / "approved_centers_input.csv", centers, list(centers[0].keys()) if centers else ["mall_id"])
    ok = sum(1 for row in summary_by_id.values() if row.get("image_status") == "ok")
    failed = len(summary_by_id) - ok
    write_report(
        run_dir / "reports" / "pv_single_map_semantic_roof_screening_report.md",
        total=len(centers),
        ok=ok,
        failed=failed,
        args=args,
    )
    print(f"run_dir={run_dir}", flush=True)


if __name__ == "__main__":
    main()
