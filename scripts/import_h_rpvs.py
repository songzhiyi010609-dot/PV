from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pv_mvp.pipeline import run_pipeline


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
MASK_WORDS = {"label", "labels", "mask", "masks", "gt", "annotation", "annotations"}


def _is_mask_like(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    stem = path.stem.lower()
    return bool(parts & MASK_WORDS) or any(word in stem for word in MASK_WORDS)


def _to_project_path(path: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_DIR).as_posix()
    except ValueError:
        return str(path)


def _scan_images(source: Path) -> list[Path]:
    files = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted([p for p in files if not _is_mask_like(p)], key=lambda p: str(p).lower())


def _scan_masks(source: Path) -> dict[str, Path]:
    masks = [p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS and _is_mask_like(p)]
    by_stem: dict[str, Path] = {}
    for mask in sorted(masks, key=lambda p: str(p).lower()):
        stem = mask.stem.lower()
        for suffix in ["_label", "-label", "_mask", "-mask", "_gt", "-gt"]:
            stem = stem.removesuffix(suffix)
        by_stem.setdefault(stem, mask)
    return by_stem


def _mask_has_pv(mask_path: Path | None) -> str:
    if mask_path is None:
        return ""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return ""
    return str(bool(np.count_nonzero(mask) > 0))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import downloaded H-RPVS samples into the MVP CSV format.")
    parser.add_argument("--source", default="data/H-RPVS", help="Unzipped H-RPVS dataset folder.")
    parser.add_argument("--output", default="data/h_rpvs_malls.csv", help="Output CSV path.")
    parser.add_argument("--limit", type=int, default=200, help="Maximum number of samples to import.")
    parser.add_argument("--run", action="store_true", help="Run MVP pipeline after creating CSV.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.is_absolute():
        source = PROJECT_DIR / source
    if not source.exists():
        raise FileNotFoundError(f"H-RPVS folder not found: {source}")

    images = _scan_images(source)
    if args.limit > 0:
        images = images[: args.limit]
    masks = _scan_masks(source)

    rows = []
    for idx, image in enumerate(images, start=1):
        mask = masks.get(image.stem.lower())
        rows.append(
            {
                "mall_id": f"h_rpvs_{idx:05d}",
                "name": f"H-RPVS真实屋顶样本{idx:05d}",
                "province": "Germany",
                "city": "Heilbronn",
                "lat": "",
                "lon": "",
                "image_path": _to_project_path(image),
                "source_dataset": "H-RPVS",
                "label_path": _to_project_path(mask) if mask else "",
                "demo_label_has_pv": _mask_has_pv(mask),
            }
        )

    output = Path(args.output)
    if not output.is_absolute():
        output = PROJECT_DIR / output
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Wrote {len(rows)} H-RPVS rows to {output}")

    if args.run:
        results = run_pipeline(output, project_dir=PROJECT_DIR, output_dir=PROJECT_DIR / "outputs" / "h_rpvs")
        suspected = int(results["has_pv"].sum())
        print(f"MVP result on H-RPVS import: {suspected}/{len(results)} suspected with PV")
        print(f"Review: {PROJECT_DIR / 'outputs' / 'h_rpvs' / 'review.html'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
