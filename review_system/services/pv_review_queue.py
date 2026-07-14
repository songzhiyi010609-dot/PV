from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from settings import path_value


def read_csv_index(path: Path, key: str = "mall_id") -> dict[int, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: dict[int, dict[str, str]] = {}
        for row in reader:
            raw_key = str(row.get(key, "")).strip()
            if not raw_key:
                continue
            try:
                rows[int(raw_key)] = row
            except ValueError:
                continue
        return rows


def optional_float(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def optional_int(value: Any) -> int | None:
    number = optional_float(value)
    return int(number) if number is not None else None


def configured_run_dir(config: dict[str, Any]) -> Path | None:
    raw_path = config.get("pv_review", {}).get("source_run_dir")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return path_value(config, "paths", "project_root") / path


def image_path(run_dir: Path | None, folder: str, mall_id: int, suffix: str) -> str:
    if run_dir is None:
        return ""
    path = run_dir / "images" / folder / f"mall_{mall_id}_{suffix}"
    return str(path) if path.exists() else ""


def load_pv_outputs(config: dict[str, Any]) -> tuple[Path | None, dict[int, dict[str, str]], dict[int, dict[str, str]]]:
    run_dir = configured_run_dir(config)
    if run_dir is None:
        return None, {}, {}
    data_dir = run_dir / "data"
    self_rows = read_csv_index(data_dir / "mall_self_pv_results.csv")
    if not self_rows:
        self_rows = read_csv_index(data_dir / "mall_self_pv_results_checkpoint.csv")
    potential_rows = read_csv_index(data_dir / "mall_1km_potential_summary.csv")
    if not potential_rows:
        potential_rows = read_csv_index(data_dir / "mall_1km_potential_summary_checkpoint.csv")
    return run_dir, self_rows, potential_rows


def import_pv_review_inputs(conn: sqlite3.Connection, config: dict[str, Any]) -> dict[str, int]:
    run_dir, self_rows, potential_rows = load_pv_outputs(config)
    manual_a_rows = conn.execute(
        """
        select
            m.mall_id, m.name, m.province, m.city,
            r.final_lon as center_lon, r.final_lat as center_lat,
            r.reviewer as poi_reviewer, r.reviewed_at as poi_reviewed_at
        from reviews r
        join malls m on m.mall_id = r.mall_id
        where r.decision_level = 'A'
          and r.final_lon is not null
          and r.final_lat is not null
        order by m.mall_id
        """
    ).fetchall()

    imported = 0
    with_pv_output = 0
    for row in manual_a_rows:
        mall_id = int(row["mall_id"])
        self_row = self_rows.get(mall_id, {})
        potential_row = potential_rows.get(mall_id, {})
        if self_row or potential_row:
            with_pv_output += 1
        image_status = self_row.get("image_status") or potential_row.get("image_status") or "missing_pv_output"
        run_id = self_row.get("run_id") or potential_row.get("run_id") or ""
        full_annotated_path = image_path(run_dir, "full_annotated", mall_id, "full_1000m_z18_annotated.jpg")
        full_raw_path = image_path(run_dir, "full_1km_raw", mall_id, "full_1000m_z18.jpg")
        full_pv_overlay_path = image_path(run_dir, "full_pv_overlays", mall_id, "full_1000m_z18_pv_overlay.jpg")
        full_roof_overlay_path = image_path(run_dir, "full_roof_overlays", mall_id, "full_1000m_z18_roof_overlay.jpg")
        review_mosaic_path = image_path(run_dir, "review_mosaics", mall_id, "review_mosaic.jpg")
        conn.execute(
            """
            insert into pv_review_items(
                mall_id, name, province, city, center_lon, center_lat,
                poi_reviewer, poi_reviewed_at, pv_run_id, pv_run_dir, image_status,
                self_image_path, self_overlay_path, full_raw_path, full_annotated_path,
                full_pv_overlay_path, full_roof_overlay_path, review_mosaic_path,
                pv_status, pv_confidence, pv_area_m2_est,
                roof_candidate_count, roof_area_m2_est, install_condition_level,
                notes, updated_at
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
            on conflict(mall_id) do update set
                name=excluded.name,
                province=excluded.province,
                city=excluded.city,
                center_lon=excluded.center_lon,
                center_lat=excluded.center_lat,
                poi_reviewer=excluded.poi_reviewer,
                poi_reviewed_at=excluded.poi_reviewed_at,
                pv_run_id=excluded.pv_run_id,
                pv_run_dir=excluded.pv_run_dir,
                image_status=excluded.image_status,
                self_image_path=excluded.self_image_path,
                self_overlay_path=excluded.self_overlay_path,
                full_raw_path=excluded.full_raw_path,
                full_annotated_path=excluded.full_annotated_path,
                full_pv_overlay_path=excluded.full_pv_overlay_path,
                full_roof_overlay_path=excluded.full_roof_overlay_path,
                review_mosaic_path=excluded.review_mosaic_path,
                pv_status=excluded.pv_status,
                pv_confidence=excluded.pv_confidence,
                pv_area_m2_est=excluded.pv_area_m2_est,
                roof_candidate_count=excluded.roof_candidate_count,
                roof_area_m2_est=excluded.roof_area_m2_est,
                install_condition_level=excluded.install_condition_level,
                notes=excluded.notes,
                updated_at=current_timestamp
            """,
            (
                mall_id,
                row["name"],
                row["province"],
                row["city"],
                row["center_lon"],
                row["center_lat"],
                row["poi_reviewer"],
                row["poi_reviewed_at"],
                run_id,
                str(run_dir) if run_dir else "",
                image_status,
                self_row.get("image_path", ""),
                self_row.get("overlay_path", ""),
                full_raw_path,
                full_annotated_path,
                full_pv_overlay_path,
                full_roof_overlay_path,
                review_mosaic_path,
                self_row.get("pv_status", ""),
                optional_float(self_row.get("pv_confidence")),
                optional_float(self_row.get("pv_area_m2_est")),
                optional_int(potential_row.get("roof_candidate_count")),
                optional_float(potential_row.get("roof_area_m2_est")),
                potential_row.get("install_condition_level", ""),
                potential_row.get("notes", ""),
            ),
        )
        imported += 1
    conn.commit()
    return {"manual_a": imported, "with_pv_output": with_pv_output}


def export_pv_review_results(conn: sqlite3.Connection, config: dict[str, Any]) -> int:
    output_path = path_value(config, "paths", "pv_review_results_csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = conn.execute(
        """
        select
            p.mall_id, p.name, p.province, p.city, p.center_lon, p.center_lat,
            p.poi_reviewer, p.poi_reviewed_at, p.pv_run_id, p.image_status,
            p.pv_status, p.pv_confidence, p.pv_area_m2_est,
            p.roof_candidate_count, p.roof_area_m2_est, p.install_condition_level,
            r.reviewer, r.decision, r.corrected_pv_status, r.corrected_potential_level,
            r.reason, r.notes, r.reviewed_at
        from pv_review_items p
        join pv_reviews r on r.mall_id = p.mall_id
        order by r.reviewed_at desc, p.mall_id
        """
    ).fetchall()
    fieldnames = [
        "mall_id",
        "name",
        "province",
        "city",
        "center_lon",
        "center_lat",
        "poi_reviewer",
        "poi_reviewed_at",
        "pv_run_id",
        "image_status",
        "pv_status",
        "pv_confidence",
        "pv_area_m2_est",
        "roof_candidate_count",
        "roof_area_m2_est",
        "install_condition_level",
        "reviewer",
        "decision",
        "corrected_pv_status",
        "corrected_potential_level",
        "reason",
        "notes",
        "reviewed_at",
    ]
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
    return len(rows)
