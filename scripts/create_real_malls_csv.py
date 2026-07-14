from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
REQUIRED_COLUMNS = ["mall_id", "name", "province", "city", "lat", "lon", "image_path"]


def _safe_id(text: str, idx: int) -> str:
    slug = re.sub(r"[^0-9a-zA-Z_\-\u4e00-\u9fff]+", "_", text).strip("_")
    return slug or f"mall_{idx:04d}"


def _to_project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(path)


def _scan_images(image_dir: Path) -> list[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    images = [p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images, key=lambda p: str(p).lower())


def _from_images(args: argparse.Namespace) -> pd.DataFrame:
    image_dir = Path(args.image_dir)
    if not image_dir.is_absolute():
        image_dir = PROJECT_DIR / image_dir

    rows = []
    for idx, image_path in enumerate(_scan_images(image_dir), start=1):
        stem = image_path.stem
        rows.append(
            {
                "mall_id": _safe_id(stem, idx),
                "name": stem,
                "province": args.province,
                "city": args.city,
                "lat": "",
                "lon": "",
                "image_path": _to_project_path(image_path),
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def _from_metadata(args: argparse.Namespace) -> pd.DataFrame:
    metadata = Path(args.metadata)
    if not metadata.is_absolute():
        metadata = PROJECT_DIR / metadata
    df = pd.read_csv(metadata, encoding="utf-8-sig")

    image_col = None
    for candidate in ["image_path", "filename", "file", "image", "image_file"]:
        if candidate in df.columns:
            image_col = candidate
            break
    if image_col is None:
        raise ValueError("Metadata must contain image_path, filename, file, image, or image_file column.")

    image_base = Path(args.image_dir)
    if not image_base.is_absolute():
        image_base = PROJECT_DIR / image_base

    rows = []
    for idx, row in df.iterrows():
        raw_image = Path(str(row[image_col]).strip().strip('"'))
        image_path = raw_image if raw_image.is_absolute() else image_base / raw_image
        name = str(row.get("name", image_path.stem))
        rows.append(
            {
                "mall_id": str(row.get("mall_id", _safe_id(name, idx + 1))),
                "name": name,
                "province": str(row.get("province", args.province)),
                "city": str(row.get("city", args.city)),
                "lat": row.get("lat", ""),
                "lon": row.get("lon", ""),
                "image_path": _to_project_path(image_path),
            }
        )
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create data/malls.csv from real mall satellite image chips.")
    parser.add_argument("--image-dir", default="data/real_malls/images", help="Folder containing real mall images.")
    parser.add_argument("--metadata", default="", help="Optional CSV with mall metadata and image filename/path.")
    parser.add_argument("--output", default="data/malls.csv", help="Output CSV path.")
    parser.add_argument("--province", default="", help="Default province if metadata is absent.")
    parser.add_argument("--city", default="", help="Default city if metadata is absent.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.metadata:
        df = _from_metadata(args)
    else:
        df = _from_images(args)

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(df)} real mall rows to {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
