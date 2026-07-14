from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_ROOT = Path("C:/PV/outputs/experiments/20260708_raw_bdappv_test")
PREDICTIONS = EXPERIMENT_ROOT / "results" / "bdappv_predictions_raw_706.csv"
SUMMARY_CSV = EXPERIMENT_ROOT / "results" / "bdappv_summary_raw_706.csv"
REPORT_MD = EXPERIMENT_ROOT / "reports" / "屋顶光伏识别实验报告.md"
CONFIG_MD = EXPERIMENT_ROOT / "reports" / "实验配置.md"
TOP_CANDIDATES = EXPERIMENT_ROOT / "results" / "光伏高分候选联系图.jpg"
TOP24_INPUT = EXPERIMENT_ROOT / "input" / "bdappv_top24_for_segmentation.csv"
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


def read_rows() -> list[dict[str, str]]:
    with PREDICTIONS.open("r", encoding="utf-8-sig", newline="") as src:
        return list(csv.DictReader(src))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def make_contact_sheet(rows: list[dict[str, str]], output_path: Path, top_n: int = 40) -> None:
    selected = rows[:top_n]
    font = load_font(17)
    small = load_font(14)
    title_font = load_font(24)
    thumb = 230
    label_h = 82
    cols = 4
    sheet_rows = (len(selected) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, 52 + sheet_rows * (thumb + label_h)), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "BDAPPV 屋顶光伏高分候选（原始706张）", fill=(24, 30, 40), font=title_font)

    for index, row in enumerate(selected):
        x = (index % cols) * thumb
        y = 52 + (index // cols) * (thumb + label_h)
        path = Path(row["image_path"])
        if path.exists():
            image = Image.open(path).convert("RGB")
            image.thumbnail((thumb, thumb))
            sheet.paste(image, (x + (thumb - image.width) // 2, y))
        draw.text((x + 6, y + thumb + 5), f"{index + 1}. {row['name']}"[:23], fill=(30, 35, 45), font=font)
        score = float(row.get("pv_score_max") or 0)
        draw.text((x + 6, y + thumb + 33), f"{row['pv_status_model']}  score={score:.3f}", fill=(76, 86, 100), font=small)
        draw.text((x + 6, y + thumb + 56), f"id={row['id']}  {row.get('area_wan_sqm','')}万㎡"[:32], fill=(84, 95, 110), font=small)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=94)


def main() -> None:
    rows = read_rows()
    sorted_rows = sorted(rows, key=lambda row: float(row.get("pv_score_max") or 0), reverse=True)
    status_counts = Counter(row["pv_status_model"] for row in rows)

    summary_fields = [
        "rank",
        "id",
        "name",
        "location",
        "area_wan_sqm",
        "longitude",
        "latitude",
        "geocode_score",
        "pv_score_max",
        "pv_status_model",
        "best_window",
        "image_path",
    ]
    summary_rows = []
    for rank, row in enumerate(sorted_rows, start=1):
        summary_rows.append({field: row.get(field, "") for field in summary_fields})
        summary_rows[-1]["rank"] = str(rank)
    write_csv(SUMMARY_CSV, summary_fields, summary_rows)

    top24 = sorted_rows[:24]
    write_csv(TOP24_INPUT, list(rows[0].keys()), top24)
    make_contact_sheet(sorted_rows, TOP_CANDIDATES, top_n=40)

    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_MD.write_text(
        "\n".join(
            [
                "# 实验配置",
                "",
                "- 实验名称：原始上海商场 706 张遥感图 BDAPPV 屋顶光伏识别测试",
                "- 输入索引：C:/PV/outputs/experiments/20260708_raw_bdappv_test/input/raw_706_project_paths.csv",
                "- 原始图片目录：C:/PV/datasets/shanghai_malls_satellite/images",
                "- 模型：gabrielkasmi/bdappv-models, google 版 InceptionV3 分类模型",
                "- 设备：cuda:0, NVIDIA GeForce RTX 4090",
                "- batch size：64",
                "- 输入裁剪：每张 1024x1024 图切为多窗口，窗口 400px，步长 312px",
                "- 阈值：possible >= 0.45, likely >= 0.75",
                "- 原始数据处理：不移动、不覆盖、不删除图片",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "# 屋顶光伏识别实验报告",
        "",
        "## 总体结果",
        "",
        f"- 输入图片：{len(rows)} 张",
        f"- likely_pv：{status_counts.get('likely_pv', 0)}",
        f"- possible_pv：{status_counts.get('possible_pv', 0)}",
        f"- no_clear_pv：{status_counts.get('no_clear_pv', 0)}",
        "",
        "## Top 20 候选",
        "",
    ]
    for row in sorted_rows[:20]:
        lines.append(
            f"- {row['id']} {row['name']}：score={float(row.get('pv_score_max') or 0):.3f}，"
            f"状态={row['pv_status_model']}"
        )
    lines.extend(
        [
            "",
            "## 说明",
            "",
            "本次实验直接使用原始 706 张上海商场遥感图，不经过 CLIP 商场一致性筛选。",
            "BDAPPV 为开源屋顶光伏识别模型，当前使用 google 版分类权重。",
            "高分候选需要结合原图和后续分割叠加图人工复核，尤其要注意温室、蓝色屋顶、工厂屋顶和非商场定位带来的误判。",
        ]
    )
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")

    manifest = {
        "predictions": str(PREDICTIONS),
        "summary": str(SUMMARY_CSV),
        "top_candidates": str(TOP_CANDIDATES),
        "top24_input": str(TOP24_INPUT),
        "status_counts": dict(status_counts),
        "total": len(rows),
    }
    (EXPERIMENT_ROOT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
