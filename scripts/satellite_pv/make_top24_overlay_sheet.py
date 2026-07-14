from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EXPERIMENT_ROOT = Path("C:/PV/outputs/experiments/20260708_raw_bdappv_test")
SEGMENTED = EXPERIMENT_ROOT / "results" / "bdappv_top24_segmented.csv"
OUT = EXPERIMENT_ROOT / "results" / "光伏Top24分割叠加图.jpg"
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


def main() -> None:
    with SEGMENTED.open("r", encoding="utf-8-sig", newline="") as src:
        rows = list(csv.DictReader(src))

    rows = sorted(rows, key=lambda row: float(row.get("pv_score_max") or 0), reverse=True)
    thumb = 250
    label_h = 92
    cols = 4
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, 54 + sheet_rows * (thumb + label_h)), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)
    draw.text((12, 10), "BDAPPV Top24 光伏分割叠加图", fill=(24, 30, 40), font=load_font(24))
    font = load_font(16)
    small = load_font(13)
    for index, row in enumerate(rows):
        x = (index % cols) * thumb
        y = 54 + (index // cols) * (thumb + label_h)
        path = Path(row.get("overlay_path") or row["image_path"])
        if not path.exists():
            path = Path(row["image_path"])
        image = Image.open(path).convert("RGB")
        image.thumbnail((thumb, thumb))
        sheet.paste(image, (x + (thumb - image.width) // 2, y))
        draw.text((x + 6, y + thumb + 5), f"{index + 1}. {row['name']}"[:23], fill=(30, 35, 45), font=font)
        score = float(row.get("pv_score_max") or 0)
        mask = float(row.get("mask_ratio") or 0)
        draw.text((x + 6, y + thumb + 31), f"score={score:.3f} mask={mask:.4f}", fill=(76, 86, 100), font=small)
        draw.text((x + 6, y + thumb + 53), f"id={row['id']} {row.get('area_wan_sqm','')}万㎡"[:32], fill=(84, 95, 110), font=small)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUT, quality=94)
    print(OUT)


if __name__ == "__main__":
    main()
