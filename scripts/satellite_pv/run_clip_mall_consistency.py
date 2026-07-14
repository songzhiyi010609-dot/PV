from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


DATASET_ROOT = Path("C:/PV/datasets/shanghai_malls_satellite")
EXPERIMENT_ROOT = Path("C:/PV/outputs/experiments/20260708_clip_mall_pv_experiment")
DEFAULT_GIT_RSCLIP_MODEL = "lcybuaa/Git-RSCLIP-base"
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]

POSITIVE_PROMPTS = [
    ("large shopping mall", "a satellite image of a large shopping mall"),
    ("commercial shopping center", "a satellite image of a commercial shopping center"),
    ("urban retail complex", "a satellite image of an urban retail complex"),
    ("large commercial building with parking lots", "a satellite image of a large commercial building with parking lots"),
    ("shopping mall roof", "a top-down satellite view of a shopping mall roof"),
]

NEGATIVE_PROMPTS = [
    ("residential neighborhood", "a satellite image of a residential neighborhood"),
    ("industrial factory or warehouse", "a satellite image of an industrial factory or warehouse"),
    ("construction site", "a satellite image of a construction site"),
    ("vacant land", "a satellite image of vacant land"),
    ("park or green area", "a satellite image of a park or green area"),
    ("roads and intersections", "a satellite image of roads and intersections"),
    ("railway or metro station", "a satellite image of a railway station or metro station"),
    ("farmland or greenhouse", "a satellite image of farmland or greenhouse"),
]

NEGATIVE_LABEL_CN = {
    "residential neighborhood": "住宅区",
    "industrial factory or warehouse": "工厂/仓库",
    "construction site": "施工地",
    "vacant land": "空地",
    "park or green area": "公园/绿地",
    "roads and intersections": "道路/交叉口",
    "railway or metro station": "铁路/地铁站",
    "farmland or greenhouse": "农田/温室",
}


@dataclass
class CropRecord:
    row_index: int
    crop_name: str
    image: Image.Image


def load_font(size: int):
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as src:
        reader = csv.DictReader(src)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {path}")
        return list(reader.fieldnames), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def project_image_path(dataset_root: Path, row: dict[str, str]) -> Path:
    old = Path(row.get("image_path") or "")
    if old.name:
        candidate = dataset_root / "images" / old.name
        if candidate.exists():
            return candidate
    matches = sorted((dataset_root / "images").glob(f"{row.get('id', '')}_*.jpg"))
    if matches:
        return matches[0]
    return dataset_root / "images" / old.name


def load_existing_quality_status(dataset_root: Path) -> dict[str, str]:
    curated_root = dataset_root / "curated_v1"
    mapping: dict[str, str] = {}
    for filename, status in [
        ("shanghai_curated_pass_index.csv", "质量通过"),
        ("shanghai_review_needed_index.csv", "需复核"),
        ("shanghai_rejected_index.csv", "剔除候选"),
    ]:
        path = curated_root / filename
        if not path.exists():
            continue
        _fields, rows = read_csv(path)
        for row in rows:
            mapping[row["id"]] = status
    return mapping


def safe_float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def make_crops(image: Image.Image) -> list[tuple[str, Image.Image]]:
    image = image.convert("RGB")
    width, height = image.size
    crops: list[tuple[str, Image.Image]] = [("full", image)]

    def add_crop(name: str, left: int, top: int, right: int, bottom: int) -> None:
        left = max(0, min(left, width - 1))
        top = max(0, min(top, height - 1))
        right = max(left + 1, min(right, width))
        bottom = max(top + 1, min(bottom, height))
        crops.append((name, image.crop((left, top, right, bottom))))

    size70 = int(min(width, height) * 0.70)
    left70 = (width - size70) // 2
    top70 = (height - size70) // 2
    add_crop("center_70", left70, top70, left70 + size70, top70 + size70)

    size50 = int(min(width, height) * 0.50)
    positions = [
        ("top_left", 0, 0),
        ("top_right", width - size50, 0),
        ("center", (width - size50) // 2, (height - size50) // 2),
        ("bottom_left", 0, height - size50),
        ("bottom_right", width - size50, height - size50),
    ]
    for name, left, top in positions:
        add_crop(name, left, top, left + size50, top + size50)
    return crops


def load_git_rsclip(device: str, model_name: str):
    from transformers import AutoModel, AutoProcessor

    model = AutoModel.from_pretrained(model_name).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor, model_name


def score_batch(
    model,
    processor,
    device: str,
    crop_batch: list[CropRecord],
    prompts: list[str],
) -> np.ndarray:
    images = [record.image for record in crop_batch]
    inputs = processor(text=prompts, images=images, padding="max_length", return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits_per_image
        probs = torch.sigmoid(logits).detach().float().cpu().numpy()
    return probs


def classify_row(mall_score: float, negative_score: float, existing_quality: str) -> tuple[str, str]:
    delta = mall_score - negative_score
    if existing_quality == "剔除候选":
        return "明显不像商场", "剔除候选"
    if delta >= 0.08 and mall_score >= 0.55 and existing_quality == "质量通过":
        return "商场可能性高", "通过"
    if delta >= 0.02 and mall_score >= 0.50 and existing_quality != "剔除候选":
        return "疑似商场，需复核" if existing_quality != "质量通过" else "商场可能性高", (
            "需复核" if existing_quality != "质量通过" else "通过"
        )
    if delta >= -0.08 or mall_score >= 0.40:
        return "不像商场，需复核", "需复核"
    return "明显不像商场", "剔除候选"


def run_scoring(
    rows: list[dict[str, str]],
    dataset_root: Path,
    device: str,
    batch_size: int,
    limit: int | None,
    model_name: str,
) -> list[dict[str, str]]:
    if limit is not None:
        rows = rows[:limit]

    model, processor, model_name = load_git_rsclip(device, model_name)
    positive_labels = [label for label, _prompt in POSITIVE_PROMPTS]
    negative_labels = [label for label, _prompt in NEGATIVE_PROMPTS]
    prompts = [prompt for _label, prompt in POSITIVE_PROMPTS + NEGATIVE_PROMPTS]
    existing_quality_by_id = load_existing_quality_status(dataset_root)

    aggregates: list[dict[str, object]] = []
    crop_records: list[CropRecord] = []
    for row_index, row in enumerate(rows):
        image_path = project_image_path(dataset_root, row)
        try:
            with Image.open(image_path) as img:
                crops = make_crops(img)
        except Exception:
            crops = []
        aggregates.append(
            {
                "mall_score": -1.0,
                "best_positive_label": "",
                "best_positive_crop": "",
                "best_negative_score": -1.0,
                "best_negative_label": "",
                "best_negative_crop": "",
                "crop_count": len(crops),
                "model_name": model_name,
            }
        )
        for crop_name, crop in crops:
            crop_records.append(CropRecord(row_index=row_index, crop_name=crop_name, image=crop))

    for start in tqdm(range(0, len(crop_records), batch_size), desc="CLIP mall consistency"):
        batch = crop_records[start : start + batch_size]
        probs = score_batch(model, processor, device, batch, prompts)
        for record, score_vector in zip(batch, probs):
            agg = aggregates[record.row_index]
            pos_scores = score_vector[: len(positive_labels)]
            neg_scores = score_vector[len(positive_labels) :]
            pos_index = int(np.argmax(pos_scores))
            neg_index = int(np.argmax(neg_scores))
            pos_score = float(pos_scores[pos_index])
            neg_score = float(neg_scores[neg_index])
            if pos_score > float(agg["mall_score"]):
                agg["mall_score"] = pos_score
                agg["best_positive_label"] = positive_labels[pos_index]
                agg["best_positive_crop"] = record.crop_name
            if neg_score > float(agg["best_negative_score"]):
                agg["best_negative_score"] = neg_score
                agg["best_negative_label"] = negative_labels[neg_index]
                agg["best_negative_crop"] = record.crop_name

    scored_rows: list[dict[str, str]] = []
    for row, agg in zip(rows, aggregates):
        image_path = project_image_path(dataset_root, row)
        existing_quality = existing_quality_by_id.get(row["id"], "未评估")
        mall_score = float(agg["mall_score"])
        best_negative_score = float(agg["best_negative_score"])
        if mall_score < 0 or best_negative_score < 0:
            clip_status = "无法评估"
            curated_status = "需复核"
            delta = math.nan
        else:
            clip_status, curated_status = classify_row(mall_score, best_negative_score, existing_quality)
            delta = mall_score - best_negative_score
        out = dict(row)
        out.update(
            {
                "image_path": str(image_path),
                "clip_model": str(agg["model_name"]),
                "clip_crop_count": str(agg["crop_count"]),
                "clip_existing_quality_status": existing_quality,
                "mall_score": f"{mall_score:.6f}" if mall_score >= 0 else "",
                "best_positive_label": str(agg["best_positive_label"]),
                "best_positive_crop": str(agg["best_positive_crop"]),
                "best_negative_label": str(agg["best_negative_label"]),
                "best_negative_label_cn": NEGATIVE_LABEL_CN.get(str(agg["best_negative_label"]), ""),
                "best_negative_score": f"{best_negative_score:.6f}" if best_negative_score >= 0 else "",
                "best_negative_crop": str(agg["best_negative_crop"]),
                "mall_delta": f"{delta:.6f}" if not math.isnan(delta) else "",
                "clip_mall_status": clip_status,
                "clip_curated_status": curated_status,
            }
        )
        scored_rows.append(out)
    return scored_rows


def make_contact_sheet(rows: list[dict[str, str]], output_path: Path, title: str, subtitle_field: str, limit: int = 60) -> None:
    selected = rows[:limit]
    if not selected:
        return
    font = load_font(17)
    small_font = load_font(14)
    title_font = load_font(24)
    thumb = 220
    label_h = 98
    cols = 4
    sheet_rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, 52 + sheet_rows * (thumb + label_h)), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill=(24, 30, 40), font=title_font)

    for index, row in enumerate(selected):
        x = (index % cols) * thumb
        y = 52 + (index // cols) * (thumb + label_h)
        path = Path(row["image_path"])
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb, thumb))
            sheet.paste(image, (x + (thumb - image.width) // 2, y))
        name = f"{index + 1}. {row['name']}"[:23]
        draw.text((x + 6, y + thumb + 4), name, fill=(30, 35, 45), font=font)
        score_line = f"mall={row.get('mall_score','')} delta={row.get('mall_delta','')}"
        draw.text((x + 6, y + thumb + 31), score_line[:30], fill=(76, 86, 100), font=small_font)
        detail = row.get(subtitle_field, "") or row.get("clip_mall_status", "")
        draw.text((x + 6, y + thumb + 55), detail[:32], fill=(130, 76, 38), font=small_font)
        neg = row.get("best_negative_label_cn", "")
        if neg:
            draw.text((x + 6, y + thumb + 77), f"负类: {neg}"[:32], fill=(84, 95, 110), font=small_font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def write_reports(
    experiment_root: Path,
    dataset_root: Path,
    rows: list[dict[str, str]],
    model_name: str,
    batch_size: int,
    device: str,
) -> None:
    report_dir = experiment_root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    counts = Counter(row["clip_curated_status"] for row in rows)
    status_counts = Counter(row["clip_mall_status"] for row in rows)
    negative_counts = Counter(row["best_negative_label_cn"] for row in rows if row.get("best_negative_label_cn"))
    prompt_lines = ["正向提示词："] + [f"- {label}: {prompt}" for label, prompt in POSITIVE_PROMPTS]
    prompt_lines += ["", "负向提示词："] + [f"- {label}: {prompt}" for label, prompt in NEGATIVE_PROMPTS]

    config = {
        "experiment_name": "CLIP商场一致性筛查+屋顶光伏实验",
        "dataset_root": str(dataset_root),
        "model": model_name,
        "device": device,
        "batch_size": batch_size,
        "crop_strategy": "full + center_70 + five 50% windows",
        "non_destructive": True,
        "outputs": str(experiment_root),
    }
    (report_dir / "实验配置.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    (report_dir / "实验配置.md").write_text(
        "\n".join(
            [
                "# 实验配置",
                "",
                f"- 数据集：{dataset_root}",
                f"- 模型：{model_name}",
                f"- 设备：{device}",
                f"- batch size：{batch_size}",
                "- 裁剪策略：全图 + 中心裁剪 + 多窗口裁剪",
                "- 原始图片处理：不移动、不覆盖、不删除",
                "",
                "## 提示词",
                "",
                *prompt_lines,
            ]
        ),
        encoding="utf-8",
    )

    lines = [
        "# CLIP商场一致性评估报告",
        "",
        "## 总体结果",
        "",
        f"- 总样本数：{len(rows)}",
        f"- 通过：{counts.get('通过', 0)}",
        f"- 需复核：{counts.get('需复核', 0)}",
        f"- 剔除候选：{counts.get('剔除候选', 0)}",
        "",
        "## CLIP标签分布",
        "",
    ]
    for status, count in status_counts.most_common():
        lines.append(f"- {status}：{count}")
    lines.extend(["", "## 最常见负向类别", ""])
    for label, count in negative_counts.most_common():
        lines.append(f"- {label}：{count}")
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "CLIP 结果仅作为商场一致性筛查，不作为最终真值。所有剔除结论均保存为候选清单，原始数据集不被修改。",
            "正式屋顶光伏实验只使用 `clip_quality_pass_index.csv`，需复核样本单独保留。",
        ]
    )
    (report_dir / "CLIP商场一致性评估报告.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Git-RSCLIP mall consistency scoring on Shanghai mall imagery.")
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--model-name", default=DEFAULT_GIT_RSCLIP_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")

    dataset_root = args.dataset_root
    experiment_root = args.experiment_root
    input_path = args.input or (dataset_root / "data" / "shanghai_imagery_index.csv")
    fields, rows = read_csv(input_path)
    scored = run_scoring(
        rows=rows,
        dataset_root=dataset_root,
        device=args.device,
        batch_size=args.batch_size,
        limit=args.limit,
        model_name=args.model_name,
    )

    output_dir = experiment_root / "clip_mall_consistency"
    output_dir.mkdir(parents=True, exist_ok=True)
    extra_fields = [
        "clip_model",
        "clip_crop_count",
        "clip_existing_quality_status",
        "mall_score",
        "best_positive_label",
        "best_positive_crop",
        "best_negative_label",
        "best_negative_label_cn",
        "best_negative_score",
        "best_negative_crop",
        "mall_delta",
        "clip_mall_status",
        "clip_curated_status",
    ]
    out_fields = list(dict.fromkeys(fields + extra_fields))
    scores_csv = output_dir / "clip_mall_scores.csv"
    pass_csv = output_dir / "clip_quality_pass_index.csv"
    review_csv = output_dir / "clip_review_needed_index.csv"
    reject_csv = output_dir / "clip_rejected_candidate_index.csv"
    write_csv(scores_csv, out_fields, scored)
    write_csv(pass_csv, out_fields, [row for row in scored if row["clip_curated_status"] == "通过"])
    write_csv(review_csv, out_fields, [row for row in scored if row["clip_curated_status"] == "需复核"])
    write_csv(reject_csv, out_fields, [row for row in scored if row["clip_curated_status"] == "剔除候选"])

    sorted_nonmall = sorted(
        scored,
        key=lambda row: (
            row["clip_curated_status"] == "通过",
            safe_float(row.get("mall_delta", ""), 0),
            safe_float(row.get("mall_score", ""), 0),
        ),
    )
    sorted_mall = sorted(scored, key=lambda row: safe_float(row.get("mall_delta", ""), -99), reverse=True)
    make_contact_sheet(
        sorted_nonmall,
        output_dir / "疑似非商场前60张联系图.jpg",
        "疑似非商场/定位异常样本（前60）",
        "clip_mall_status",
        limit=60,
    )
    make_contact_sheet(
        sorted_mall,
        output_dir / "商场一致性高前60张联系图.jpg",
        "商场一致性高样本（前60）",
        "clip_mall_status",
        limit=60,
    )
    write_reports(
        experiment_root=experiment_root,
        dataset_root=dataset_root,
        rows=scored,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )

    manifest = {
        "scores_csv": str(scores_csv),
        "pass_csv": str(pass_csv),
        "review_csv": str(review_csv),
        "reject_csv": str(reject_csv),
        "count_total": len(scored),
        "count_pass": sum(1 for row in scored if row["clip_curated_status"] == "通过"),
        "count_review": sum(1 for row in scored if row["clip_curated_status"] == "需复核"),
        "count_reject": sum(1 for row in scored if row["clip_curated_status"] == "剔除候选"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
