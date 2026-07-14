from __future__ import annotations

import csv
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = REPO_ROOT / "satellite_experiment"
IMAGE_DIR = EXPERIMENT_ROOT / "images"
OUTPUT_DIR = EXPERIMENT_ROOT / "visual_verification"
INDEX_PATH = EXPERIMENT_ROOT / "data" / "shanghai_imagery_index.csv"
CONTACT_SHEET_PATH = OUTPUT_DIR / "shanghai_pv_visual_verification.jpg"
REPORT_CSV_PATH = OUTPUT_DIR / "shanghai_pv_visual_verification.csv"
REPORT_MD_PATH = OUTPUT_DIR / "shanghai_pv_visual_verification.md"
FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/NotoSansSC-VF.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
]


MANUAL_FINDINGS = {
    "36": {
        "pv_status": "\u7591\u4f3c\u6709\u5149\u4f0f",
        "confidence": "\u4e2d",
        "evidence": (
            "\u753b\u9762\u53f3\u4e0a\u5c4b\u9876\u6709\u89c4\u6574\u6df1\u8272\u6761\u5e26\u77e9\u9635\uff0c"
            "\u5f62\u6001\u7c7b\u4f3c\u5c4b\u9876\u5149\u4f0f\u9635\u5217\u3002"
        ),
        "review_note": (
            "\u9700\u8fdb\u4e00\u6b65\u6838\u5bf9\u5546\u573a\u5efa\u7b51\u8fb9\u754c\uff0c"
            "\u5f53\u524d\u53ea\u80fd\u5224\u65ad\u4e3a\u5468\u8fb9/\u76ee\u6807\u533a\u57df\u7591\u4f3c\u3002"
        ),
    },
    "42": {
        "pv_status": "\u672a\u89c1\u660e\u786e\u5149\u4f0f",
        "confidence": "\u4f4e",
        "evidence": (
            "\u5f71\u50cf\u4e2d\u5fc3\u4e3a\u7a7a\u5730\u548c\u9053\u8def\uff0c"
            "\u672a\u770b\u5230\u5546\u573a\u5c4b\u9876\u6216\u8fde\u7eed\u5149\u4f0f\u9635\u5217\u3002"
        ),
        "review_note": (
            "\u5750\u6807\u53ef\u80fd\u504f\u79bb\u5b9e\u9645\u5546\u573a\uff0c\u5efa\u8bae\u5148\u91cd\u65b0\u6838\u51c6\u5b9a\u4f4d\u3002"
        ),
    },
    "52": {
        "pv_status": "\u672a\u89c1\u660e\u786e\u5149\u4f0f",
        "confidence": "\u4e2d",
        "evidence": (
            "\u4e3b\u4f53\u5c4b\u9762\u53ef\u89c1\u6761\u7eb9\u548c\u5929\u7a97/\u6784\u67b6\uff0c"
            "\u4f46\u4e0d\u5177\u5907\u5178\u578b\u5149\u4f0f\u677f\u77e9\u9635\u7279\u5f81\u3002"
        ),
        "review_note": "\u5de6\u4fa7\u5c0f\u5efa\u7b51\u6709\u84dd\u8272\u683c\u72b6\u7269\uff0c\u4f46\u4e0d\u5c5e\u4e8e\u4e3b\u4f53\u5546\u573a\u5c4b\u9876\u3002",
    },
    "57": {
        "pv_status": "\u672a\u89c1\u660e\u786e\u5149\u4f0f",
        "confidence": "\u4e2d",
        "evidence": (
            "\u753b\u9762\u5185\u5efa\u7b51\u5c4b\u9876\u591a\u4e3a\u6d45\u8272\u5c4b\u9762\u3001"
            "\u8bbe\u5907\u548c\u9634\u5f71\uff0c\u672a\u89c1\u89c4\u6574\u5149\u4f0f\u9635\u5217\u3002"
        ),
        "review_note": "\u53ef\u89c6\u8303\u56f4\u8db3\u591f\u505a\u521d\u7b5b\uff0c\u4f46\u9700\u914d\u5408\u5546\u573a\u8fb9\u754c\u590d\u6838\u3002",
    },
    "73": {
        "pv_status": "\u672a\u89c1\u660e\u786e\u5149\u4f0f",
        "confidence": "\u4e2d",
        "evidence": (
            "\u591a\u5904\u5c4b\u9876\u6709\u6761\u7eb9\u6784\u4ef6/\u767e\u53f6/\u5929\u7a97\uff0c"
            "\u4f46\u7f3a\u5c11\u5149\u4f0f\u9635\u5217\u7684\u8fde\u7eed\u6df1\u8272\u677f\u9762\u7279\u5f81\u3002"
        ),
        "review_note": "\u6682\u4e0d\u5efa\u8bae\u5224\u4e3a\u6709\u5149\u4f0f\u3002",
    },
}


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def wrap_text(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for part in text.splitlines():
        lines.extend(textwrap.wrap(part, width=width) or [""])
    return lines


def status_color(status: str) -> tuple[int, int, int]:
    if "\u7591\u4f3c" in status:
        return (232, 146, 38)
    if "\u6709\u5149\u4f0f" in status and "\u672a\u89c1" not in status:
        return (38, 148, 83)
    return (104, 113, 126)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INDEX_PATH.open("r", encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))

    report_rows = []
    for row in rows:
        finding = MANUAL_FINDINGS.get(row["id"], {})
        report_rows.append(
            {
                "id": row["id"],
                "name": row["name"],
                "location": row["location"],
                "longitude": row["longitude"],
                "latitude": row["latitude"],
                "geocode_score": row["geocode_score"],
                "image_path": row["image_path"],
                "pv_status": finding.get("pv_status", "\u672a\u5ba1\u6838"),
                "confidence": finding.get("confidence", ""),
                "evidence": finding.get("evidence", ""),
                "review_note": finding.get("review_note", ""),
            }
        )

    with REPORT_CSV_PATH.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=list(report_rows[0].keys()))
        writer.writeheader()
        writer.writerows(report_rows)

    title_font = load_font(28)
    body_font = load_font(20)
    small_font = load_font(17)

    card_w = 440
    img_h = 330
    text_h = 190
    margin = 28
    gap = 22
    cols = 2
    rows_count = (len(report_rows) + cols - 1) // cols
    sheet_w = margin * 2 + cols * card_w + (cols - 1) * gap
    sheet_h = margin * 2 + rows_count * (img_h + text_h + gap) - gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(report_rows):
        col = index % cols
        row_index = index // cols
        x = margin + col * (card_w + gap)
        y = margin + row_index * (img_h + text_h + gap)

        draw.rounded_rectangle(
            (x, y, x + card_w, y + img_h + text_h),
            radius=8,
            fill=(255, 255, 255),
            outline=(210, 214, 220),
            width=1,
        )

        image_path = Path(row["image_path"])
        if not image_path.exists():
            image_path = IMAGE_DIR / Path(row["image_path"]).name
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((card_w, img_h))
        img_x = x + (card_w - image.width) // 2
        img_y = y + (img_h - image.height) // 2
        sheet.paste(image, (img_x, img_y))

        baseline = y + img_h + 14
        title = f"{row['id']}  {row['name']}"
        for line in wrap_text(title, width=18)[:2]:
            draw.text((x + 16, baseline), line, fill=(23, 28, 37), font=body_font)
            baseline += 26

        status = row["pv_status"]
        badge_color = status_color(status)
        badge_text = f"{status} / \u7f6e\u4fe1\u5ea6\uff1a{row['confidence']}"
        draw.rounded_rectangle(
            (x + 16, baseline + 4, x + card_w - 16, baseline + 36),
            radius=6,
            fill=badge_color,
        )
        draw.text((x + 28, baseline + 8), badge_text, fill=(255, 255, 255), font=small_font)
        baseline += 48

        evidence = f"\u4f9d\u636e\uff1a{row['evidence']}"
        for line in wrap_text(evidence, width=24)[:3]:
            draw.text((x + 16, baseline), line, fill=(48, 55, 66), font=small_font)
            baseline += 24

    sheet.save(CONTACT_SHEET_PATH, quality=95)

    lines = [
        "# \u4e0a\u6d77\u5546\u573a\u5149\u4f0f\u89c6\u89c9\u9a8c\u8bc1\u6837\u672c",
        "",
        "\u6837\u672c\uff1a5 \u4e2a\u4e0a\u6d77\u5546\u573a\u536b\u661f\u5f71\u50cf\u88c1\u526a\u56fe\u3002",
        "",
        "## \u7ed3\u8bba",
        "",
    ]
    for row in report_rows:
        lines.append(
            f"- {row['id']} {row['name']}: {row['pv_status']} "
            f"(\u7f6e\u4fe1\u5ea6\uff1a{row['confidence']})"
        )
    lines.extend(
        [
            "",
            "## \u6ce8\u610f",
            "",
            (
                "\u8fd9\u662f\u57fa\u4e8e\u536b\u661f\u56fe\u7684\u4eba\u5de5\u76ee\u89c6\u521d\u7b5b\uff0c"
                "\u5e76\u672a\u5f15\u5165\u5546\u573a\u5efa\u7b51\u8f6e\u5ed3\u3002"
                "\u7591\u4f3c\u7ed3\u679c\u9700\u8981\u7ed3\u5408\u5efa\u7b51\u8fb9\u754c\u3001"
                "\u66f4\u9ad8\u5206\u8fa8\u7387\u5f71\u50cf\u6216\u73b0\u573a\u4fe1\u606f\u590d\u6838\u3002"
            ),
        ]
    )
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"csv={REPORT_CSV_PATH}")
    print(f"report={REPORT_MD_PATH}")
    print(f"contact_sheet={CONTACT_SHEET_PATH}")


if __name__ == "__main__":
    main()
