#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Finalize mall center relocation outputs.

This creates a conservative coordinate table:
- high candidates with usable coordinates keep center_lon/center_lat.
- medium candidates are written to a separate candidate table for review.
- low/unresolved candidates keep their raw candidate coordinates for review,
  but official center_lon/center_lat are blank to avoid accidental misuse.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXPERIMENT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "20260708_relocate_mall_centers"
SEVERE_FLAGS = {
    "outside_shanghai_bbox",
    "bad_place_hint",
    "city_not_shanghai",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize mall center coordinate relocation tables.")
    parser.add_argument("--experiment-dir", type=Path, default=DEFAULT_EXPERIMENT_DIR)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def has_severe_flag(flags: str) -> bool:
    parts = {part.strip() for part in (flags or "").split("|") if part.strip()}
    return bool(parts & SEVERE_FLAGS)


def make_review_reason(row: dict[str, str], usable: bool) -> str:
    reasons: list[str] = []
    confidence = row.get("confidence", "")
    flags = row.get("flags", "")
    if usable:
        if confidence == "medium":
            reasons.append("medium_confidence_manual_spot_check")
        return "|".join(reasons)
    if confidence in {"low", "unresolved"}:
        reasons.append(f"confidence_{confidence}")
    if has_severe_flag(flags):
        reasons.append("severe_flag")
    if "weak_name_match" in flags:
        reasons.append("weak_name_match")
    if "generic_candidate" in flags:
        reasons.append("generic_candidate")
    if not row.get("selected_lon") or not row.get("selected_lat"):
        reasons.append("no_usable_coordinate")
    return "|".join(dict.fromkeys(reasons))


def main() -> int:
    args = parse_args()
    data_dir = args.experiment_dir / "data"
    report_dir = args.experiment_dir / "reports"
    raw_path = data_dir / "mall_center_coordinates_v1.csv"
    clean_path = data_dir / "mall_center_coordinates_clean.csv"
    pass_path = data_dir / "mall_center_precise_pass_index.csv"
    medium_path = data_dir / "mall_center_medium_candidate_index.csv"
    review_path = data_dir / "mall_center_review_needed.csv"
    report_path = report_dir / "\u5546\u573a\u4e2d\u5fc3\u91cd\u5b9a\u4f4d\u6e05\u6d17\u62a5\u544a.md"

    rows = read_rows(raw_path)
    clean_rows: list[dict[str, str]] = []
    pass_rows: list[dict[str, str]] = []
    medium_rows: list[dict[str, str]] = []
    review_rows: list[dict[str, str]] = []

    for row in rows:
        confidence = row.get("confidence", "")
        raw_lon = row.get("selected_lon", "")
        raw_lat = row.get("selected_lat", "")
        high_usable = (
            confidence == "high"
            and bool(raw_lon)
            and bool(raw_lat)
            and not has_severe_flag(row.get("flags", ""))
        )
        medium_candidate = (
            confidence == "medium"
            and bool(raw_lon)
            and bool(raw_lat)
            and not has_severe_flag(row.get("flags", ""))
        )
        clean = {
            "mall_id": row.get("mall_id", ""),
            "name": row.get("name", ""),
            "center_lon": raw_lon if high_usable else "",
            "center_lat": raw_lat if high_usable else "",
            "usable_for_imagery": "1" if high_usable else "0",
            "confidence": confidence,
            "center_score": row.get("center_score", ""),
            "provider": row.get("provider", ""),
            "place_name": row.get("place_name", ""),
            "match_addr": row.get("match_addr", ""),
            "addr_type": row.get("addr_type", ""),
            "place_type": row.get("place_type", ""),
            "name_similarity": row.get("name_similarity", ""),
            "provider_score": row.get("provider_score", ""),
            "flags": row.get("flags", ""),
            "review_reason": make_review_reason(row, high_usable),
            "candidate_lon_for_review": raw_lon,
            "candidate_lat_for_review": raw_lat,
            "old_location_unused": row.get("old_location_unused", ""),
        }
        clean_rows.append(clean)
        if high_usable:
            pass_rows.append(clean)
        elif medium_candidate:
            medium_row = clean.copy()
            medium_row["center_lon"] = raw_lon
            medium_row["center_lat"] = raw_lat
            medium_row["review_reason"] = "medium_confidence_manual_spot_check"
            medium_rows.append(medium_row)
            review_rows.append(medium_row)
        else:
            review_rows.append(clean)

    fields = [
        "mall_id",
        "name",
        "center_lon",
        "center_lat",
        "usable_for_imagery",
        "confidence",
        "center_score",
        "provider",
        "place_name",
        "match_addr",
        "addr_type",
        "place_type",
        "name_similarity",
        "provider_score",
        "flags",
        "review_reason",
        "candidate_lon_for_review",
        "candidate_lat_for_review",
        "old_location_unused",
    ]
    write_rows(clean_path, fields, clean_rows)
    write_rows(pass_path, fields, pass_rows)
    write_rows(medium_path, fields, medium_rows)
    write_rows(review_path, fields, review_rows)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("confidence", "unknown")] = counts.get(row.get("confidence", "unknown"), 0) + 1
    provider_counts: dict[str, int] = {}
    for row in pass_rows:
        provider_counts[row.get("provider", "unknown")] = provider_counts.get(row.get("provider", "unknown"), 0) + 1

    report_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# 商场中心重定位清洗报告",
        "",
        f"- 原始重定位结果：`{raw_path}`",
        f"- 清洗后完整表：`{clean_path}`",
        f"- 高置信可用于后续影像裁剪：`{pass_path}`",
        f"- 中置信候选，建议人工抽查：`{medium_path}`",
        f"- 需要复核：`{review_path}`",
        f"- 总记录数：{len(rows)}",
        f"- 高置信可用中心点：{len(pass_rows)}",
        f"- 中置信候选中心点：{len(medium_rows)}",
        f"- 需复核/未解析：{len(review_rows)}",
        "",
        "## 原始置信度统计",
        "",
    ]
    for key in ["high", "medium", "low", "unresolved"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.extend(["", "## 可用中心点来源", ""])
    for provider, count in sorted(provider_counts.items(), key=lambda item: item[0]):
        lines.append(f"- {provider}: {count}")
    lines.extend(
        [
            "",
            "## 使用规则",
            "",
            "- `center_lon` / `center_lat` 在完整清洗表中只对高置信且无严重风险标记的样本保留。",
            "- `medium` 另存为候选中心点，不进入高置信通过表；人工确认后可再加入影像裁剪。",
            "- `low` 和 `unresolved` 的正式中心点置空；如有候选坐标，仅保留在 `candidate_lon_for_review` / `candidate_lat_for_review` 供复核。",
            "- 后续重新生成遥感图时只应使用 `mall_center_precise_pass_index.csv`，不要直接使用 raw 表。",
            "- 原数据库 `location` 字段没有参与定位，仅作为 `old_location_unused` 留痕。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"rows={len(rows)} high_usable={len(pass_rows)} medium_candidates={len(medium_rows)} review={len(review_rows)}")
    print(f"clean={clean_path}")
    print(f"pass={pass_path}")
    print(f"medium={medium_path}")
    print(f"review={review_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
