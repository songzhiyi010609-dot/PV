from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SHANGHAI_BBOX = (120.85, 30.67, 122.15, 31.87)
NON_SHANGHAI_HINTS = [
    "浙江", "江苏", "舟山", "宿迁", "南宁", "宁波", "苏州", "杭州",
    "南京", "嘉兴", "湖州", "无锡", "常州",
]
GENERIC_GEOCODE_TERMS = [
    "中心", "浦东新区", "闵行区", "普陀区", "嘉定区", "上海市",
    "公交站", "地铁站", "道路", "路",
]
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]


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


def safe_float(value: str, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def in_shanghai_bbox(lon: float, lat: float) -> bool:
    west, south, east, north = SHANGHAI_BBOX
    return west <= lon <= east and south <= lat <= north


def project_image_path(dataset_root: Path, row: dict[str, str]) -> Path:
    old_path = Path(row.get("image_path") or "")
    if old_path.name:
        candidate = dataset_root / "images" / old_path.name
        if candidate.exists():
            return candidate
    mall_id = row.get("id", "")
    matches = sorted((dataset_root / "images").glob(f"{mall_id}_*.jpg"))
    if matches:
        return matches[0]
    return dataset_root / "images" / old_path.name


def image_metrics(path: Path) -> dict[str, str | float | int]:
    if not path.exists():
        return {
            "image_exists": "0",
            "image_width": "",
            "image_height": "",
            "image_mean": "",
            "image_std": "",
            "vegetation_ratio": "",
            "water_blue_ratio": "",
            "dark_roof_ratio": "",
            "edge_density": "",
            "center_non_green_ratio": "",
            "largest_rect_ratio": "",
        }

    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        arr = np.asarray(rgb, dtype=np.uint8)

    arr_f = arr.astype(np.float32)
    r = arr_f[:, :, 0]
    g = arr_f[:, :, 1]
    b = arr_f[:, :, 2]
    total = np.maximum(r + g + b, 1.0)
    green_share = g / total
    vegetation = (green_share > 0.38) & (g > r * 1.08) & (g > b * 1.05)
    blue_water = (b > r * 1.12) & (b > g * 1.05) & (b > 70)
    dark_roof = (arr_f.mean(axis=2) < 95) & ~vegetation & ~blue_water

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 70, 160)
    edge_density = float((edges > 0).mean())

    h0, h1 = int(height * 0.30), int(height * 0.70)
    w0, w1 = int(width * 0.30), int(width * 0.70)
    center_non_green = float((~vegetation[h0:h1, w0:w1]).mean())

    # Approximate large rectilinear roof evidence. It is a weak visual gate, not a
    # final mall detector.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 41, 2
    )
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_rect_ratio = 0.0
    image_area = width * height
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.004:
            continue
        rect = cv2.minAreaRect(contour)
        rect_area = rect[1][0] * rect[1][1]
        if rect_area <= 0:
            continue
        fill = area / rect_area
        if fill >= 0.45:
            largest_rect_ratio = max(largest_rect_ratio, rect_area / image_area)

    return {
        "image_exists": "1",
        "image_width": width,
        "image_height": height,
        "image_mean": f"{arr_f.mean():.3f}",
        "image_std": f"{arr_f.std():.3f}",
        "vegetation_ratio": f"{float(vegetation.mean()):.6f}",
        "water_blue_ratio": f"{float(blue_water.mean()):.6f}",
        "dark_roof_ratio": f"{float(dark_roof.mean()):.6f}",
        "edge_density": f"{edge_density:.6f}",
        "center_non_green_ratio": f"{center_non_green:.6f}",
        "largest_rect_ratio": f"{largest_rect_ratio:.6f}",
    }


def quality_flags(row: dict[str, str]) -> tuple[list[str], str]:
    flags: list[str] = []
    lon = safe_float(row.get("longitude", ""))
    lat = safe_float(row.get("latitude", ""))
    geocode_score = safe_float(row.get("geocode_score", ""))
    vegetation_ratio = safe_float(row.get("vegetation_ratio", ""))
    center_non_green_ratio = safe_float(row.get("center_non_green_ratio", ""))
    largest_rect_ratio = safe_float(row.get("largest_rect_ratio", ""))
    image_std = safe_float(row.get("image_std", ""))

    if row.get("image_exists") != "1":
        flags.append("图片缺失")
    if not in_shanghai_bbox(lon, lat):
        flags.append("坐标不在上海范围")
    if math.isnan(geocode_score) or geocode_score < 80:
        flags.append("地理编码分数低")
    if row.get("geocode_repaired") == "1":
        flags.append("坐标经过修复")

    location = row.get("location") or ""
    if "上海" not in location:
        flags.append("原始地址不含上海")
    if any(hint in location for hint in NON_SHANGHAI_HINTS):
        flags.append("原始地址疑似外地/异常")

    geocode_address = row.get("geocode_address") or ""
    if geocode_address in GENERIC_GEOCODE_TERMS:
        flags.append("地理编码结果过于泛化")

    if not math.isnan(image_std) and image_std < 18:
        flags.append("影像对比度异常低")
    if not math.isnan(vegetation_ratio) and vegetation_ratio > 0.55:
        flags.append("绿地占比过高")
    if not math.isnan(center_non_green_ratio) and center_non_green_ratio < 0.35:
        flags.append("中心区域非绿地占比低")
    if not math.isnan(largest_rect_ratio) and largest_rect_ratio < 0.015:
        flags.append("缺少大尺度矩形屋顶证据")

    # Hard exclusions are mostly geocode/data integrity problems. Visual-only
    # flags go to review instead of being deleted.
    if row.get("image_exists") != "1" or "坐标不在上海范围" in flags:
        return flags, "剔除"

    high_risk = {
        "地理编码分数低",
        "原始地址不含上海",
        "原始地址疑似外地/异常",
        "地理编码结果过于泛化",
    }
    visual_risk = {
        "绿地占比过高",
        "中心区域非绿地占比低",
        "缺少大尺度矩形屋顶证据",
    }
    if any(flag in high_risk for flag in flags):
        return flags, "需复核"
    if sum(1 for flag in flags if flag in visual_risk) >= 2:
        return flags, "需复核"
    return flags, "通过"


def make_contact_sheet(rows: list[dict[str, str]], output_path: Path, title: str, limit: int = 60) -> None:
    rows = rows[:limit]
    if not rows:
        return
    font = load_font(17)
    small = load_font(14)
    thumb = 220
    label_h = 88
    cols = 4
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, 48 + sheet_rows * (thumb + label_h)), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), title, fill=(25, 30, 38), font=load_font(24))
    for index, row in enumerate(rows):
        x = (index % cols) * thumb
        y = 48 + (index // cols) * (thumb + label_h)
        image_path = Path(row["project_image_path"])
        if image_path.exists():
            image = Image.open(image_path).convert("RGB")
            image.thumbnail((thumb, thumb))
            sheet.paste(image, (x + (thumb - image.width) // 2, y))
        label = f"{row['id']} {row['name']}"[:22]
        draw.text((x + 6, y + thumb + 5), label, fill=(30, 35, 45), font=font)
        draw.text((x + 6, y + thumb + 32), row["quality_status"], fill=(160, 70, 35), font=small)
        flag_text = row["quality_flags"][:28]
        draw.text((x + 6, y + thumb + 56), flag_text, fill=(84, 95, 110), font=small)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def main() -> None:
    parser = argparse.ArgumentParser(description="Non-destructive QA for Shanghai mall satellite dataset.")
    parser.add_argument("--dataset-root", type=Path, default=Path("C:/PV/datasets/shanghai_malls_satellite"))
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("C:/PV/outputs/experiments/20260708_shanghai_mall_quality_pv"),
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    experiment_root = args.experiment_root
    qa_dir = experiment_root / "01_dataset_quality"
    curated_dir = dataset_root / "curated_v1"
    index_path = dataset_root / "data" / "shanghai_imagery_index.csv"

    fields, rows = read_csv(index_path)
    output_rows: list[dict[str, str]] = []
    for row in rows:
        project_path = project_image_path(dataset_root, row)
        out = dict(row)
        out["original_image_path"] = row.get("image_path", "")
        out["project_image_path"] = str(project_path)
        out["image_path"] = str(project_path)
        out.update({key: str(value) for key, value in image_metrics(project_path).items()})
        flags, status = quality_flags(out)
        out["quality_flags"] = "；".join(flags)
        out["quality_status"] = status
        output_rows.append(out)

    extra_fields = [
        "original_image_path",
        "project_image_path",
        "image_exists",
        "image_width",
        "image_height",
        "image_mean",
        "image_std",
        "vegetation_ratio",
        "water_blue_ratio",
        "dark_roof_ratio",
        "edge_density",
        "center_non_green_ratio",
        "largest_rect_ratio",
        "quality_flags",
        "quality_status",
    ]
    output_fields = list(dict.fromkeys(fields + extra_fields))
    qa_csv = qa_dir / "shanghai_dataset_quality_assessment.csv"
    write_csv(qa_csv, output_fields, output_rows)

    pass_rows = [row for row in output_rows if row["quality_status"] == "通过"]
    review_rows = [row for row in output_rows if row["quality_status"] == "需复核"]
    reject_rows = [row for row in output_rows if row["quality_status"] == "剔除"]
    write_csv(curated_dir / "shanghai_curated_pass_index.csv", output_fields, pass_rows)
    write_csv(curated_dir / "shanghai_review_needed_index.csv", output_fields, review_rows)
    write_csv(curated_dir / "shanghai_rejected_index.csv", output_fields, reject_rows)

    # Copy no images. Keep a manifest only, so the original dataset is untouched.
    manifest = {
        "dataset_root": str(dataset_root),
        "source_index": str(index_path),
        "quality_assessment_csv": str(qa_csv),
        "curated_pass_index": str(curated_dir / "shanghai_curated_pass_index.csv"),
        "review_needed_index": str(curated_dir / "shanghai_review_needed_index.csv"),
        "rejected_index": str(curated_dir / "shanghai_rejected_index.csv"),
        "non_destructive": True,
        "image_copy_policy": "No image files are moved, overwritten, or deleted.",
    }
    (curated_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    by_status = Counter(row["quality_status"] for row in output_rows)
    by_flag = Counter()
    for row in output_rows:
        for flag in filter(None, row["quality_flags"].split("；")):
            by_flag[flag] += 1

    report = [
        "# 上海商场遥感数据集质量评估",
        "",
        "## 评估原则",
        "",
        "- 不移动、不覆盖、不删除原始图片。",
        "- 新增质量评估 CSV、复核清单和精选索引。",
        "- “需复核”不是删除结论，而是提示该图可能不是商场主体俯视图或定位不够可靠。",
        "",
        "## 总体结果",
        "",
        f"- 总记录数：{len(output_rows)}",
        f"- 通过：{by_status.get('通过', 0)}",
        f"- 需复核：{by_status.get('需复核', 0)}",
        f"- 剔除：{by_status.get('剔除', 0)}",
        "",
        "## 主要问题类型",
        "",
    ]
    for flag, count in by_flag.most_common():
        report.append(f"- {flag}：{count}")
    report.extend(
        [
            "",
            "## 输出文件",
            "",
            f"- 质量评估明细：{qa_csv}",
            f"- 精选通过索引：{curated_dir / 'shanghai_curated_pass_index.csv'}",
            f"- 需复核索引：{curated_dir / 'shanghai_review_needed_index.csv'}",
            f"- 剔除索引：{curated_dir / 'shanghai_rejected_index.csv'}",
            "",
            "## 后续实验使用建议",
            "",
            "屋顶光伏识别实验优先使用 `shanghai_curated_pass_index.csv`；",
            "对 `shanghai_review_needed_index.csv` 另做人工复核或二次定位，不直接混入正式结论。",
        ]
    )
    report_path = qa_dir / "数据集质量评估报告.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")

    risk_rows = sorted(
        review_rows + reject_rows,
        key=lambda row: (
            row["quality_status"] == "通过",
            safe_float(row.get("geocode_score", ""), 0),
            -safe_float(row.get("vegetation_ratio", ""), 0),
        ),
    )
    make_contact_sheet(
        risk_rows,
        qa_dir / "疑似非商场或定位异常样本联系图.jpg",
        "疑似非商场/定位异常样本（前60）",
        limit=60,
    )
    make_contact_sheet(
        pass_rows[:60],
        qa_dir / "通过样本抽查联系图.jpg",
        "质量通过样本抽查（前60）",
        limit=60,
    )

    print(f"quality_csv={qa_csv}")
    print(f"report={report_path}")
    print(f"curated_pass={curated_dir / 'shanghai_curated_pass_index.csv'}")
    print(f"review_needed={curated_dir / 'shanghai_review_needed_index.csv'}")
    print(f"rejected={curated_dir / 'shanghai_rejected_index.csv'}")
    print(f"status_counts={dict(by_status)}")


if __name__ == "__main__":
    main()
