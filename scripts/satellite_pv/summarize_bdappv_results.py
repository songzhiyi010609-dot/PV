from __future__ import annotations

import csv
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]
FULL_ROOT = REPO_ROOT / "satellite_experiment" / "full_shanghai_dataset"
PREDICTIONS = FULL_ROOT / "results" / "bdappv_predictions.csv"
SUMMARY_CSV = FULL_ROOT / "results" / "bdappv_summary.csv"
SUMMARY_MD = FULL_ROOT / "results" / "bdappv_summary.md"
CONTACT_SHEET = FULL_ROOT / "results" / "bdappv_top_candidates.jpg"
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


def read_predictions() -> list[dict[str, str]]:
    with PREDICTIONS.open("r", encoding="utf-8-sig", newline="") as src:
        return list(csv.DictReader(src))


def write_summary(rows: list[dict[str, str]]) -> None:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = row["pv_status_model"]
        status_counts[status] = status_counts.get(status, 0) + 1

    sorted_rows = sorted(rows, key=lambda row: float(row.get("pv_score_max") or 0), reverse=True)
    with SUMMARY_CSV.open("w", encoding="utf-8-sig", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=[
                "rank",
                "id",
                "name",
                "location",
                "pv_score_max",
                "pv_status_model",
                "geocode_score",
                "image_path",
                "overlay_path",
                "mask_ratio",
            ],
        )
        writer.writeheader()
        for rank, row in enumerate(sorted_rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "id": row["id"],
                    "name": row["name"],
                    "location": row["location"],
                    "pv_score_max": row["pv_score_max"],
                    "pv_status_model": row["pv_status_model"],
                    "geocode_score": row["geocode_score"],
                    "image_path": row["image_path"],
                    "overlay_path": row["overlay_path"],
                    "mask_ratio": row["mask_ratio"],
                }
            )

    lines = [
        "# BDAPPV Shanghai mall PV recognition summary",
        "",
        f"Total images: {len(rows)}",
        "",
        "## Status counts",
        "",
    ]
    for status, count in sorted(status_counts.items()):
        lines.append(f"- {status}: {count}")
    lines.extend(["", "## Top candidates", ""])
    for row in sorted_rows[:20]:
        lines.append(
            f"- {row['id']} {row['name']}: score={row['pv_score_max']} "
            f"status={row['pv_status_model']}"
        )
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def make_contact_sheet(rows: list[dict[str, str]], top_n: int = 24) -> None:
    rows = sorted(rows, key=lambda row: float(row.get("pv_score_max") or 0), reverse=True)[:top_n]
    font = load_font(18)
    small_font = load_font(15)
    thumb = 230
    label_h = 82
    cols = 4
    sheet_rows = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * thumb, sheet_rows * (thumb + label_h)), (246, 247, 249))
    draw = ImageDraw.Draw(sheet)

    for index, row in enumerate(rows):
        x = (index % cols) * thumb
        y = (index // cols) * (thumb + label_h)
        image_path = Path(row.get("overlay_path") or row["image_path"])
        if not image_path.exists():
            image_path = Path(row["image_path"])
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((thumb, thumb))
        sheet.paste(image, (x + (thumb - image.width) // 2, y))
        title = f"{index + 1}. {row['name']}"[:20]
        draw.text((x + 8, y + thumb + 5), title, fill=(30, 35, 45), font=font)
        score_text = f"{row['pv_status_model']}  {float(row['pv_score_max'] or 0):.3f}"
        draw.text((x + 8, y + thumb + 34), score_text, fill=(76, 86, 100), font=small_font)

    CONTACT_SHEET.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(CONTACT_SHEET, quality=94)


def main() -> None:
    rows = read_predictions()
    SUMMARY_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_summary(rows)
    make_contact_sheet(rows)
    print(f"summary_csv={SUMMARY_CSV}")
    print(f"summary_md={SUMMARY_MD}")
    print(f"contact_sheet={CONTACT_SHEET}")


if __name__ == "__main__":
    main()
