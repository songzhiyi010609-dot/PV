#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run Git-RSCLIP visual review for relocated mall POI candidates.

The experiment compares two satellite views:
- old dataset crop: the image generated from the original database location;
- candidate POI crop: a fresh crop around the newly found candidate POI center.

The model is used as a zero-shot remote-sensing image classifier. Scores are
relative logits, not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoProcessor
from transformers.utils import logging as hf_logging

from make_mall_poi_review_gallery import (
    build_crop,
    draw_crosshair,
    find_old_image,
    fit_image,
    load_font,
    make_contact_sheets,
    safe_filename,
    wrap_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "Git-RSCLIP"
DEFAULT_RELOCATION_DIR = PROJECT_ROOT / "outputs" / "experiments" / "20260708_relocate_mall_centers"
DEFAULT_REVIEW_CSV = DEFAULT_RELOCATION_DIR / "data" / "mall_center_review_needed.csv"
DEFAULT_OLD_IMAGE_DIR = PROJECT_ROOT / "datasets" / "shanghai_malls_satellite" / "images"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "20260708_gitr_sclip_poi_review"
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")

POSITIVE_PROMPTS = [
    "a remote sensing image of a large shopping mall",
    "a remote sensing image of a shopping center",
    "a satellite image of a commercial complex",
    "a satellite image of a shopping plaza with large buildings",
    "a satellite image of a retail complex with parking lots",
]

NEGATIVE_PROMPTS = [
    "a remote sensing image of residential buildings",
    "a remote sensing image of roads and intersections",
    "a remote sensing image of a railway or metro station",
    "a remote sensing image of construction land or vacant land",
    "a remote sensing image of factories or warehouses",
    "a remote sensing image of a park or green space",
    "a remote sensing image of farmland or greenhouses",
    "a remote sensing image of a school campus or hospital",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Git-RSCLIP POI visual review.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--review-csv", type=Path, default=DEFAULT_REVIEW_CSV)
    parser.add_argument("--old-image-dir", type=Path, default=DEFAULT_OLD_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Experiment root. A timestamped run folder is created inside it by default.")
    parser.add_argument("--run-id", default="", help="Timestamp/run folder name. Default: current time in YYYYMMDD_HHMMSS.")
    parser.add_argument("--no-timestamp", action="store_true", help="Use --output-dir exactly, without appending a timestamp. Useful when resuming an exact folder.")
    parser.add_argument("--tile-cache-dir", type=Path, default=None, help="Tile cache directory. Default: <actual_output_dir>/tile_cache/esri_world_imagery.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=0, help="0 means all selected rows.")
    parser.add_argument("--include-no-candidate", action="store_true", help="Also score old images for rows without candidate POI coordinates.")
    parser.add_argument("--candidate-crop-size", type=int, default=768)
    parser.add_argument("--candidate-zoom", type=int, default=18)
    parser.add_argument("--tile-timeout", type=int, default=30)
    parser.add_argument("--tile-delay", type=float, default=0.03)
    parser.add_argument("--pass-threshold", type=float, default=0.50, help="Minimum mall_delta to count an image as mall-like.")
    parser.add_argument("--prefer-margin", type=float, default=0.35, help="Candidate/old delta margin required for a preference decision.")
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite-crops", action="store_true")
    parser.add_argument("--no-review-images", action="store_true")
    parser.add_argument("--no-contact-sheets", action="store_true")
    return parser.parse_args()


def current_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_timestamped_dir(path: Path) -> bool:
    return bool(TIMESTAMP_PATTERN.match(path.name))


def latest_timestamped_run(base_dir: Path) -> Path | None:
    if not base_dir.exists():
        return None
    candidates = [
        path
        for path in base_dir.iterdir()
        if path.is_dir()
        and is_timestamped_dir(path)
        and (path / "data" / "gitr_sclip_poi_review_scores.csv").exists()
    ]
    return sorted(candidates, key=lambda path: path.name, reverse=True)[0] if candidates else None


def resolve_output_paths(args: argparse.Namespace) -> None:
    base_output_dir = args.output_dir
    if args.no_timestamp or is_timestamped_dir(base_output_dir):
        actual_output_dir = base_output_dir
        run_id = args.run_id or (base_output_dir.name if is_timestamped_dir(base_output_dir) else "")
    elif args.resume:
        latest = latest_timestamped_run(base_output_dir)
        if latest is not None:
            actual_output_dir = latest
            run_id = latest.name
        else:
            run_id = args.run_id or current_run_id()
            actual_output_dir = base_output_dir / run_id
    else:
        run_id = args.run_id or current_run_id()
        actual_output_dir = base_output_dir / run_id

    args.output_base_dir = base_output_dir
    args.output_dir = actual_output_dir
    args.run_id = run_id
    if args.tile_cache_dir is None:
        args.tile_cache_dir = actual_output_dir / "tile_cache" / "esri_world_imagery"


def setup_logging(output_dir: Path) -> logging.Logger:
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("gitr_sclip_poi_review")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    file_handler = logging.FileHandler(log_dir / "gitr_sclip_poi_review.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "data": output_dir / "data",
        "reports": output_dir / "reports",
        "candidate_crops": output_dir / "images" / "candidate_poi_crops",
        "review_images": output_dir / "images" / "clip_review_images",
        "contact_sheets": output_dir / "contact_sheets",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def read_review_rows(path: Path, include_no_candidate: bool, limit: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            has_candidate = bool(row.get("candidate_lon_for_review")) and bool(row.get("candidate_lat_for_review"))
            if not has_candidate and not include_no_candidate:
                continue
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def read_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row.get("mall_id", "") for row in csv.DictReader(fh) if row.get("mall_id")}


def crop_center(image: Image.Image, ratio: float) -> Image.Image:
    width, height = image.size
    size = int(min(width, height) * ratio)
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def crop_box_ratio(image: Image.Image, left_ratio: float, top_ratio: float, size_ratio: float) -> Image.Image:
    width, height = image.size
    size = int(min(width, height) * size_ratio)
    left = int((width - size) * left_ratio)
    top = int((height - size) * top_ratio)
    return image.crop((left, top, left + size, top + size))


def make_views(image: Image.Image) -> list[tuple[str, Image.Image]]:
    image = image.convert("RGB")
    return [
        ("full", image),
        ("center_80", crop_center(image, 0.80)),
        ("center_60", crop_center(image, 0.60)),
        ("top_left", crop_box_ratio(image, 0.0, 0.0, 0.62)),
        ("top_right", crop_box_ratio(image, 1.0, 0.0, 0.62)),
        ("bottom_left", crop_box_ratio(image, 0.0, 1.0, 0.62)),
        ("bottom_right", crop_box_ratio(image, 1.0, 1.0, 0.62)),
    ]


def load_model(model_dir: Path, device: str, logger: logging.Logger) -> tuple[Any, Any, str]:
    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Missing model weights: {model_dir / 'model.safetensors'}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available; falling back to CPU.")
        device = "cpu"
    hf_logging.set_verbosity_error()
    logger.info("Loading Git-RSCLIP model from %s", model_dir)
    processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_dir), local_files_only=True)
    model = model.to(device).eval()
    logger.info("Model loaded on %s", device)
    return processor, model, device


def to_device(inputs: dict[str, Any], device: str) -> dict[str, Any]:
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def score_image(
    image: Image.Image,
    processor: Any,
    model: Any,
    device: str,
    prompts: list[str],
    positive_count: int,
) -> dict[str, Any]:
    views = make_views(image)
    images = [item[1] for item in views]
    inputs = processor(text=prompts, images=images, padding="max_length", return_tensors="pt")
    inputs = to_device(inputs, device)
    with torch.no_grad():
        logits = model(**inputs).logits_per_image.detach().float().cpu()

    view_scores: list[dict[str, Any]] = []
    for view_index, (view_name, _) in enumerate(views):
        row = logits[view_index]
        pos_logits = row[:positive_count]
        neg_logits = row[positive_count:]
        best_pos_value, best_pos_idx = torch.max(pos_logits, dim=0)
        best_neg_value, best_neg_idx = torch.max(neg_logits, dim=0)
        softmax = torch.softmax(row, dim=0)
        pos_share = float(softmax[:positive_count].sum().item())
        view_scores.append(
            {
                "view": view_name,
                "mall_score": float(best_pos_value.item()),
                "best_positive_label": prompts[int(best_pos_idx.item())],
                "best_negative_score": float(best_neg_value.item()),
                "best_negative_label": prompts[positive_count + int(best_neg_idx.item())],
                "mall_delta": float((best_pos_value - best_neg_value).item()),
                "mall_softmax_share": pos_share,
            }
        )

    best = max(view_scores, key=lambda item: item["mall_delta"])
    return {
        "best_view": best["view"],
        "mall_score": best["mall_score"],
        "best_positive_label": best["best_positive_label"],
        "best_negative_score": best["best_negative_score"],
        "best_negative_label": best["best_negative_label"],
        "mall_delta": best["mall_delta"],
        "mall_softmax_share": best["mall_softmax_share"],
        "mean_mall_delta": sum(item["mall_delta"] for item in view_scores) / len(view_scores),
    }


def format_float(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return str(value)


def decide(
    old_score: dict[str, Any] | None,
    candidate_score: dict[str, Any] | None,
    pass_threshold: float,
    prefer_margin: float,
) -> tuple[str, str]:
    old_delta = old_score["mall_delta"] if old_score else None
    candidate_delta = candidate_score["mall_delta"] if candidate_score else None

    if candidate_delta is None:
        if old_delta is not None and old_delta >= pass_threshold:
            return "old_image_mall_like_no_candidate", "medium"
        return "no_candidate_coordinate", "high"

    if old_delta is None:
        if candidate_delta >= pass_threshold:
            return "candidate_center_likely_mall", "high"
        return "candidate_unclear_no_old_image", "medium"

    if candidate_delta >= pass_threshold and candidate_delta >= old_delta + prefer_margin:
        return "candidate_center_likely_mall", "high"
    if old_delta >= pass_threshold and old_delta >= candidate_delta + prefer_margin:
        return "old_image_more_mall_like", "high"
    if candidate_delta < pass_threshold and old_delta < pass_threshold:
        return "both_unclear_or_non_mall", "medium"
    return "needs_manual_review", "medium"


def get_candidate_crop(
    row: dict[str, str],
    output_path: Path,
    cache_dir: Path,
    zoom: int,
    size: int,
    timeout: int,
    overwrite: bool,
) -> Path | None:
    lon_text = row.get("candidate_lon_for_review", "")
    lat_text = row.get("candidate_lat_for_review", "")
    if not lon_text or not lat_text:
        return None
    if output_path.exists() and not overwrite:
        return output_path
    lon = float(lon_text)
    lat = float(lat_text)
    image = build_crop(lon=lon, lat=lat, zoom=zoom, size=size, cache_dir=cache_dir, timeout=timeout)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path, quality=94)
    return output_path


def make_scored_review_image(
    row: dict[str, str],
    old_image_path: Path | None,
    candidate_image_path: Path | None,
    output_path: Path,
    decision: str,
    old_score: dict[str, Any] | None,
    candidate_score: dict[str, Any] | None,
) -> None:
    panel_size = 420
    margin = 24
    gap = 18
    header_h = 158
    footer_h = 94
    width = margin * 2 + panel_size * 2 + gap
    height = header_h + panel_size + footer_h
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    title_font = load_font(23)
    label_font = load_font(17)
    small_font = load_font(14)

    draw.rectangle((0, 0, width, header_h), fill=(28, 34, 42))
    title = f"{row.get('mall_id', '')}  {row.get('name', '')}"
    y = 16
    for line in wrap_text(draw, title, title_font, width - margin * 2)[:2]:
        draw.text((margin, y), line, fill=(255, 255, 255), font=title_font)
        y += 30
    meta = (
        f"CLIP决策: {decision} | POI: {row.get('place_name', '')} | "
        f"{row.get('provider', '')} | {row.get('confidence', '')}"
    )
    for line in wrap_text(draw, meta, small_font, width - margin * 2)[:2]:
        draw.text((margin, y + 2), line, fill=(218, 226, 235), font=small_font)
        y += 22

    left_x = margin
    right_x = margin + panel_size + gap
    top_y = header_h
    old_panel = Image.new("RGB", (panel_size, panel_size), (230, 230, 230))
    if old_image_path and old_image_path.exists():
        old_panel = fit_image(Image.open(old_image_path), panel_size)
    candidate_panel = Image.new("RGB", (panel_size, panel_size), (230, 230, 230))
    if candidate_image_path and candidate_image_path.exists():
        candidate_panel = fit_image(Image.open(candidate_image_path), panel_size)
        draw_candidate = ImageDraw.Draw(candidate_panel)
        draw_crosshair(draw_candidate, panel_size // 2, panel_size // 2, 34)
    canvas.paste(old_panel, (left_x, top_y))
    canvas.paste(candidate_panel, (right_x, top_y))
    draw.rectangle((left_x, top_y, left_x + panel_size, top_y + panel_size), outline=(70, 70, 70), width=2)
    draw.rectangle((right_x, top_y, right_x + panel_size, top_y + panel_size), outline=(200, 50, 50), width=3)
    draw.rectangle((left_x, top_y, left_x + 188, top_y + 32), fill=(0, 0, 0))
    draw.text((left_x + 10, top_y + 6), "旧数据集图", fill=(255, 255, 255), font=label_font)
    draw.rectangle((right_x, top_y, right_x + 210, top_y + 32), fill=(150, 20, 20))
    draw.text((right_x + 10, top_y + 6), "候选POI中心图", fill=(255, 255, 255), font=label_font)

    old_delta = format_float(old_score["mall_delta"] if old_score else "")
    candidate_delta = format_float(candidate_score["mall_delta"] if candidate_score else "")
    footer_y = header_h + panel_size + 14
    draw.text((margin, footer_y), f"old_delta={old_delta} | candidate_delta={candidate_delta}", fill=(38, 38, 38), font=small_font)
    draw.text((margin, footer_y + 24), f"old_best={old_score.get('best_positive_label','') if old_score else ''}", fill=(60, 60, 60), font=small_font)
    draw.text((margin, footer_y + 48), f"candidate_best={candidate_score.get('best_positive_label','') if candidate_score else ''}", fill=(60, 60, 60), font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


RESULT_FIELDS = [
    "mall_id",
    "name",
    "relocation_confidence",
    "provider",
    "place_name",
    "review_reason",
    "candidate_lon_for_review",
    "candidate_lat_for_review",
    "old_image_path",
    "candidate_image_path",
    "clip_review_image_path",
    "old_mall_delta",
    "old_mall_score",
    "old_best_negative_score",
    "old_mall_softmax_share",
    "old_best_view",
    "old_best_positive_label",
    "old_best_negative_label",
    "candidate_mall_delta",
    "candidate_mall_score",
    "candidate_best_negative_score",
    "candidate_mall_softmax_share",
    "candidate_best_view",
    "candidate_best_positive_label",
    "candidate_best_negative_label",
    "candidate_minus_old_delta",
    "clip_decision",
    "clip_priority",
    "error",
]


def build_result_row(
    row: dict[str, str],
    old_image_path: Path | None,
    candidate_image_path: Path | None,
    review_image_path: Path | None,
    old_score: dict[str, Any] | None,
    candidate_score: dict[str, Any] | None,
    decision: str,
    priority: str,
    error: str = "",
) -> dict[str, str]:
    old_delta = old_score["mall_delta"] if old_score else None
    candidate_delta = candidate_score["mall_delta"] if candidate_score else None
    gain = "" if old_delta is None or candidate_delta is None else candidate_delta - old_delta
    return {
        "mall_id": row.get("mall_id", ""),
        "name": row.get("name", ""),
        "relocation_confidence": row.get("confidence", ""),
        "provider": row.get("provider", ""),
        "place_name": row.get("place_name", ""),
        "review_reason": row.get("review_reason", ""),
        "candidate_lon_for_review": row.get("candidate_lon_for_review", ""),
        "candidate_lat_for_review": row.get("candidate_lat_for_review", ""),
        "old_image_path": str(old_image_path or ""),
        "candidate_image_path": str(candidate_image_path or ""),
        "clip_review_image_path": str(review_image_path or ""),
        "old_mall_delta": format_float(old_delta),
        "old_mall_score": format_float(old_score["mall_score"] if old_score else ""),
        "old_best_negative_score": format_float(old_score["best_negative_score"] if old_score else ""),
        "old_mall_softmax_share": format_float(old_score["mall_softmax_share"] if old_score else ""),
        "old_best_view": old_score["best_view"] if old_score else "",
        "old_best_positive_label": old_score["best_positive_label"] if old_score else "",
        "old_best_negative_label": old_score["best_negative_label"] if old_score else "",
        "candidate_mall_delta": format_float(candidate_delta),
        "candidate_mall_score": format_float(candidate_score["mall_score"] if candidate_score else ""),
        "candidate_best_negative_score": format_float(candidate_score["best_negative_score"] if candidate_score else ""),
        "candidate_mall_softmax_share": format_float(candidate_score["mall_softmax_share"] if candidate_score else ""),
        "candidate_best_view": candidate_score["best_view"] if candidate_score else "",
        "candidate_best_positive_label": candidate_score["best_positive_label"] if candidate_score else "",
        "candidate_best_negative_label": candidate_score["best_negative_label"] if candidate_score else "",
        "candidate_minus_old_delta": format_float(gain),
        "clip_decision": decision,
        "clip_priority": priority,
        "error": error,
    }


def append_result(path: Path, row: dict[str, str]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})


def write_config_report(args: argparse.Namespace, paths: dict[str, Path], total: int, device: str) -> None:
    lines = [
        "# Git-RSCLIP 商场 POI 复核实验配置",
        "",
        f"- 运行 ID：`{args.run_id or 'no_timestamp'}`",
        f"- 模型目录：`{args.model_dir}`",
        f"- 复核清单：`{args.review_csv}`",
        f"- 旧图目录：`{args.old_image_dir}`",
        f"- 输出根目录：`{args.output_base_dir}`",
        f"- 输出目录：`{args.output_dir}`",
        f"- 瓦片缓存目录：`{args.tile_cache_dir}`",
        f"- 设备：`{device}`",
        f"- 输入记录数：{total}",
        f"- 候选 POI 裁剪尺寸：{args.candidate_crop_size}px",
        f"- 候选 POI 裁剪 zoom：{args.candidate_zoom}",
        f"- pass_threshold：{args.pass_threshold}",
        f"- prefer_margin：{args.prefer_margin}",
        f"- include_no_candidate：{args.include_no_candidate}",
        "",
        "## 正向提示词",
        "",
    ]
    lines.extend([f"- `{prompt}`" for prompt in POSITIVE_PROMPTS])
    lines.extend(["", "## 负向提示词", ""])
    lines.extend([f"- `{prompt}`" for prompt in NEGATIVE_PROMPTS])
    lines.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 评分表：`{paths['data'] / 'gitr_sclip_poi_review_scores.csv'}`",
            f"- 中文报告：`{paths['reports'] / 'Git-RSCLIP商场POI复核报告.md'}`",
            f"- 日志：`{paths['logs'] / 'gitr_sclip_poi_review.log'}`",
            f"- 候选 POI 裁剪图：`{paths['candidate_crops']}`",
            f"- CLIP 复核对比图：`{paths['review_images']}`",
        ]
    )
    (paths["reports"] / "实验配置.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summary_report(output_dir: Path, result_path: Path, paths: dict[str, Path], elapsed: float) -> None:
    rows: list[dict[str, str]] = []
    if result_path.exists():
        with result_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    decision_counts = Counter(row.get("clip_decision", "") for row in rows)
    priority_counts = Counter(row.get("clip_priority", "") for row in rows)
    candidate_rows = [row for row in rows if row.get("candidate_image_path")]
    preferred_rows = [row for row in rows if row.get("clip_decision") == "candidate_center_likely_mall"]
    old_preferred_rows = [row for row in rows if row.get("clip_decision") == "old_image_more_mall_like"]

    lines = [
        "# Git-RSCLIP 商场 POI 复核报告",
        "",
        f"- 评分表：`{result_path}`",
        f"- 处理记录数：{len(rows)}",
        f"- 有候选 POI 图记录数：{len(candidate_rows)}",
        f"- 总耗时：{elapsed / 60:.2f} 分钟",
        "",
        "## 决策统计",
        "",
    ]
    for decision, count in decision_counts.most_common():
        lines.append(f"- {decision}: {count}")
    lines.extend(["", "## 优先级统计", ""])
    for priority, count in priority_counts.most_common():
        lines.append(f"- {priority}: {count}")
    lines.extend(["", "## 候选 POI 更像商场的样本 Top 40", ""])
    for row in sorted(preferred_rows, key=lambda item: float(item.get("candidate_minus_old_delta") or -999), reverse=True)[:40]:
        lines.append(
            f"- {row.get('mall_id')} {row.get('name')} | gain={row.get('candidate_minus_old_delta')} | "
            f"candidate_delta={row.get('candidate_mall_delta')} | old_delta={row.get('old_mall_delta')} | {row.get('place_name')}"
        )
    lines.extend(["", "## 旧图更像商场的样本 Top 40", ""])
    for row in sorted(old_preferred_rows, key=lambda item: float(item.get("candidate_minus_old_delta") or 999))[:40]:
        lines.append(
            f"- {row.get('mall_id')} {row.get('name')} | gain={row.get('candidate_minus_old_delta')} | "
            f"candidate_delta={row.get('candidate_mall_delta')} | old_delta={row.get('old_mall_delta')} | {row.get('place_name')}"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- `mall_delta = best_positive_logit - best_negative_logit`，值越大越像商场/商业综合体。",
            "- `candidate_minus_old_delta > 0` 表示候选 POI 图比旧图更像商场。",
            "- Git-RSCLIP 是复核辅助，不是最终真值；`needs_manual_review` 和边界样本仍需看图确认。",
        ]
    )
    report_path = paths["reports"] / "Git-RSCLIP商场POI复核报告.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolve_output_paths(args)
    paths = ensure_dirs(args.output_dir)
    logger = setup_logging(args.output_dir)
    result_path = paths["data"] / "gitr_sclip_poi_review_scores.csv"
    prompts = POSITIVE_PROMPTS + NEGATIVE_PROMPTS

    rows = read_review_rows(args.review_csv, args.include_no_candidate, args.limit)
    completed = read_completed_ids(result_path) if args.resume else set()
    if not args.resume and result_path.exists():
        result_path.unlink()
    pending = [row for row in rows if row.get("mall_id", "") not in completed]

    logger.info("Run id=%s", args.run_id or "no_timestamp")
    logger.info("Output dir=%s", args.output_dir)
    logger.info("Tile cache dir=%s", args.tile_cache_dir)
    logger.info("Input rows=%s, completed=%s, pending=%s", len(rows), len(completed), len(pending))
    processor, model, device = load_model(args.model_dir, args.device, logger)
    write_config_report(args, paths, len(rows), device)

    start = time.time()
    review_image_paths: list[Path] = []
    for index, row in enumerate(pending, 1):
        item_start = time.time()
        mall_id = row.get("mall_id", "")
        name = row.get("name", "")
        safe_name = safe_filename(f"{mall_id}_{name}")
        old_path = find_old_image(args.old_image_dir, mall_id)
        candidate_path = paths["candidate_crops"] / f"{safe_name}_candidate_poi.jpg"
        review_image_path = paths["review_images"] / f"{safe_name}_clip_review.jpg"
        old_score: dict[str, Any] | None = None
        candidate_score: dict[str, Any] | None = None
        error = ""

        try:
            if old_path and old_path.exists():
                old_score = score_image(Image.open(old_path).convert("RGB"), processor, model, device, prompts, len(POSITIVE_PROMPTS))

            actual_candidate_path = get_candidate_crop(
                row=row,
                output_path=candidate_path,
                cache_dir=args.tile_cache_dir,
                zoom=args.candidate_zoom,
                size=args.candidate_crop_size,
                timeout=args.tile_timeout,
                overwrite=args.overwrite_crops,
            )
            if actual_candidate_path:
                candidate_score = score_image(
                    Image.open(actual_candidate_path).convert("RGB"),
                    processor,
                    model,
                    device,
                    prompts,
                    len(POSITIVE_PROMPTS),
                )
                if args.tile_delay > 0:
                    time.sleep(args.tile_delay)

            decision, priority = decide(old_score, candidate_score, args.pass_threshold, args.prefer_margin)
            saved_review_image_path: Path | None = None
            if not args.no_review_images and (old_path or actual_candidate_path):
                make_scored_review_image(
                    row=row,
                    old_image_path=old_path,
                    candidate_image_path=actual_candidate_path,
                    output_path=review_image_path,
                    decision=decision,
                    old_score=old_score,
                    candidate_score=candidate_score,
                )
                saved_review_image_path = review_image_path
                review_image_paths.append(review_image_path)

            result = build_result_row(
                row=row,
                old_image_path=old_path,
                candidate_image_path=actual_candidate_path,
                review_image_path=saved_review_image_path,
                old_score=old_score,
                candidate_score=candidate_score,
                decision=decision,
                priority=priority,
            )
        except Exception as exc:  # noqa: BLE001 - keep experiment running and record failed samples.
            decision = "error"
            priority = "high"
            error = repr(exc)
            result = build_result_row(
                row=row,
                old_image_path=old_path,
                candidate_image_path=None,
                review_image_path=None,
                old_score=old_score,
                candidate_score=candidate_score,
                decision=decision,
                priority=priority,
                error=error,
            )

        append_result(result_path, result)

        done = len(completed) + index
        elapsed = time.time() - start
        avg = elapsed / max(index, 1)
        eta = avg * (len(pending) - index)
        if index == 1 or index % args.log_every == 0 or index == len(pending):
            logger.info(
                "[%s/%s] id=%s %s | decision=%s | old_delta=%s | candidate_delta=%s | %.1fs/item | ETA %.1fmin%s",
                done,
                len(rows),
                mall_id,
                name,
                decision,
                result.get("old_mall_delta", ""),
                result.get("candidate_mall_delta", ""),
                time.time() - item_start,
                eta / 60,
                f" | error={error}" if error else "",
            )

    if not args.no_contact_sheets and review_image_paths:
        logger.info("Creating contact sheets for %s review images", len(review_image_paths))
        make_contact_sheets(review_image_paths, paths["contact_sheets"])

    total_elapsed = time.time() - start
    write_summary_report(args.output_dir, result_path, paths, total_elapsed)
    logger.info("Done. results=%s", result_path)
    logger.info("Report=%s", paths["reports"] / "Git-RSCLIP商场POI复核报告.md")
    logger.info("Elapsed %.2f min", total_elapsed / 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
