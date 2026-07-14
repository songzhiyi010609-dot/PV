from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any

from settings import path_value


MALL_FIELDS = [
    "mall_id",
    "name",
    "province",
    "city",
    "source_run_id",
    "source_run_dir",
    "confidence_level",
    "can_enter_1km_analysis",
    "review_required",
    "review_status",
    "identity_status",
    "identity_reasons",
    "selected_lon_wgs84",
    "selected_lat_wgs84",
    "approved_center_lon",
    "approved_center_lat",
    "selected_provider",
    "selected_poi_name",
    "selected_address",
    "selected_poi_type",
    "selected_name_similarity",
    "official_provider_count",
    "agreement_provider_count",
    "agreement_radius_m",
    "nearest_provider_distance_m",
    "best_identity_score",
    "name_evidence",
    "official_coord_evidence",
    "image_evidence",
    "clip_decision",
    "needs_review",
    "suspicious_reason",
    "source_csv",
]


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except (TypeError, ValueError):
        return default


def as_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value))
    except (TypeError, ValueError):
        return None


def first_value(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        value = str(row.get(key, "") or "").strip()
        if value:
            return value
    return ""


def has_suspicious_keyword(row: dict[str, str], keywords: list[str]) -> str:
    text = " ".join(
        str(row.get(key, ""))
        for key in ["selected_poi_name", "selected_address", "selected_poi_type", "identity_reasons", "official_coord_evidence"]
    )
    hits = [keyword for keyword in keywords if keyword and keyword in text]
    return "|".join(hits)


def should_review(row: dict[str, str], config: dict[str, Any]) -> tuple[bool, str]:
    review_config = config["review"]
    levels = set(review_config.get("include_confidence_levels", ["C"]))
    level = str(row.get("confidence_level", "")).strip()
    source_review_required = as_int(row.get("review_required"))
    review_status = str(row.get("review_status", "")).strip()
    suspicious = has_suspicious_keyword(row, list(review_config.get("suspicious_keywords", [])))
    reasons: list[str] = []
    if source_review_required:
        reasons.append("review_required=1")
    if level in levels:
        reasons.append(f"confidence_level={level}")
    if review_status in {"needs_manual_review", "blocked_unresolved"}:
        reasons.append(f"review_status={review_status}")
    if suspicious:
        reasons.append(f"suspicious={suspicious}")
    return bool(reasons), "; ".join(reasons)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def load_mall_region_index(config: dict[str, Any]) -> dict[int, tuple[str, str]]:
    project_root = path_value(config, "paths", "project_root")
    db_path = project_root / "malls_new.db"
    if not db_path.exists():
        return {}
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("select id, province, city from malls").fetchall()
    return {
        int(row["id"]): (
            str(row["province"] or "").strip(),
            str(row["city"] or "").strip(),
        )
        for row in rows
    }


def candidate_csv_for_row(row: dict[str, str], input_path: Path) -> Path | None:
    source_run_dir = str(row.get("source_run_dir", "")).strip()
    if not source_run_dir:
        candidates = [
            input_path.parent / "official_poi_candidates.csv",
            input_path.parent / "data" / "official_poi_candidates.csv",
            input_path.parent.parent / "data" / "official_poi_candidates.csv",
        ]
    else:
        candidates = [Path(source_run_dir) / "data" / "official_poi_candidates.csv"]
    for path in candidates:
        if path.exists():
            return path
    return None


def upsert_mall(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
    values = [row.get(field) for field in MALL_FIELDS]
    placeholders = ", ".join("?" for _ in MALL_FIELDS)
    updates = ", ".join(f"{field}=excluded.{field}" for field in MALL_FIELDS if field != "mall_id")
    conn.execute(
        f"""
        insert into malls({", ".join(MALL_FIELDS)})
        values ({placeholders})
        on conflict(mall_id) do update set {updates}, updated_at=current_timestamp
        """,
        values,
    )


def import_candidates(conn: sqlite3.Connection, candidate_csv: Path, source_run_id: str) -> int:
    rows = load_csv_rows(candidate_csv)
    count = 0
    for row in rows:
        mall_id = as_int(row.get("mall_id"))
        if mall_id <= 0:
            continue
        conn.execute(
            """
            insert or ignore into poi_candidates(
                mall_id, provider, query, rank, poi_name, address, province, city, district,
                poi_type, wgs84_lon, wgs84_lat, identity_score, name_similarity, flags, source_run_id
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                mall_id,
                row.get("provider", ""),
                row.get("query", ""),
                as_int(row.get("rank")),
                row.get("poi_name", ""),
                row.get("address", ""),
                row.get("province", ""),
                row.get("city", ""),
                row.get("district", ""),
                row.get("poi_type", ""),
                as_float(row.get("wgs84_lon")),
                as_float(row.get("wgs84_lat")),
                as_float(row.get("identity_score")),
                as_float(row.get("name_similarity")),
                row.get("flags", ""),
                source_run_id,
            ),
        )
        count += 1
    return count


def backfill_missing_selected_coords(conn: sqlite3.Connection) -> int:
    rows = conn.execute(
        """
        select mall_id, selected_provider, selected_poi_name, selected_address
        from malls
        where (selected_lon_wgs84 is null or selected_lat_wgs84 is null)
          and (
            coalesce(selected_provider, '') <> ''
            or coalesce(selected_poi_name, '') <> ''
            or coalesce(selected_address, '') <> ''
          )
        """
    ).fetchall()
    count = 0
    for row in rows:
        candidate = conn.execute(
            """
            select wgs84_lon, wgs84_lat
            from poi_candidates
            where mall_id = ?
              and wgs84_lon is not null
              and wgs84_lat is not null
            order by
              case when provider = ? then 0 else 1 end,
              case when poi_name = ? then 0 else 1 end,
              case when address = ? then 0 else 1 end,
              rank
            limit 1
            """,
            (
                row["mall_id"],
                row["selected_provider"] or "",
                row["selected_poi_name"] or "",
                row["selected_address"] or "",
            ),
        ).fetchone()
        if candidate is None:
            continue
        conn.execute(
            """
            update malls
            set selected_lon_wgs84 = ?,
                selected_lat_wgs84 = ?,
                updated_at = current_timestamp
            where mall_id = ?
              and (selected_lon_wgs84 is null or selected_lat_wgs84 is null)
            """,
            (candidate["wgs84_lon"], candidate["wgs84_lat"], row["mall_id"]),
        )
        count += 1
    return count


def reset_second_review_queue(conn: sqlite3.Connection, config: dict[str, Any]) -> int:
    review_config = config.get("review", {})
    if not review_config.get("second_review_enabled", False):
        return 0
    levels = tuple(str(item).strip().upper() for item in review_config.get("second_review_levels", ["B", "C", "D"]) if item)
    if not levels:
        return 0
    placeholders = ", ".join("?" for _ in levels)
    suspicious_reason_sql = """
                case
                    when coalesce(suspicious_reason, '') like '%second_review=B/C/D needs map-picked A center%'
                        then suspicious_reason
                    when coalesce(suspicious_reason, '') = ''
                        then 'second_review=B/C/D needs map-picked A center'
                    else trim(coalesce(suspicious_reason, '') || '; second_review=B/C/D needs map-picked A center')
                end
    """
    rows = conn.execute(
        f"""
        select r.*
        from reviews r
        join malls m on m.mall_id = r.mall_id
        where r.decision_level in ({placeholders})
          and not exists (
            select 1
            from review_history h
            where h.mall_id = r.mall_id
              and h.reviewed_at = r.reviewed_at
              and h.decision_level = r.decision_level
              and h.archive_reason = 'second_review_requeue'
          )
        """,
        levels,
    ).fetchall()
    if not rows:
        conn.execute(
            f"""
            update malls
            set needs_review = 1,
                review_required = 1,
                review_status = 'second_review_required',
                suspicious_reason = {suspicious_reason_sql},
                updated_at = current_timestamp
            where mall_id in (
                select mall_id
                from review_history
                where archive_reason = 'second_review_requeue'
                  and decision_level in ({placeholders})
            )
            """,
            levels,
        )
        return 0

    for row in rows:
        conn.execute(
            """
            insert into review_history(
                mall_id, reviewer, decision_level, final_lon, final_lat, final_polygon_geojson,
                reason, notes, review_round, reviewed_at, archive_reason
            )
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'second_review_requeue')
            """,
            (
                row["mall_id"],
                row["reviewer"],
                row["decision_level"],
                row["final_lon"],
                row["final_lat"],
                row["final_polygon_geojson"],
                row["reason"],
                row["notes"],
                row["review_round"] or 1,
                row["reviewed_at"],
            ),
        )
    conn.execute(
        f"""
        update malls
        set needs_review = 1,
            review_required = 1,
            review_status = 'second_review_required',
            suspicious_reason = {suspicious_reason_sql},
            updated_at = current_timestamp
        where mall_id in (
            select mall_id
            from reviews
            where decision_level in ({placeholders})
        )
        """,
        levels,
    )
    conn.execute(f"delete from reviews where decision_level in ({placeholders})", levels)
    return len(rows)


def import_review_inputs(conn: sqlite3.Connection, config: dict[str, Any]) -> dict[str, int]:
    input_files = [Path(item) for item in config["paths"].get("input_result_files", [])]
    imported = 0
    queued = 0
    candidate_files_seen: set[Path] = set()
    candidate_rows = 0
    region_index = load_mall_region_index(config)
    for input_path in input_files:
        rows = load_csv_rows(input_path)
        for src_row in rows:
            mall_id = as_int(src_row.get("mall_id"))
            if mall_id <= 0:
                continue
            province, city = region_index.get(mall_id, ("", ""))
            province = first_value(src_row, "province") or province
            city = first_value(src_row, "city") or city
            needs_review, suspicious_reason = should_review(src_row, config)
            source_run_dir = first_value(src_row, "source_run_dir")
            if not source_run_dir and input_path.parent.name == "data":
                source_run_dir = str(input_path.parent.parent)
            source_run_id = first_value(src_row, "source_run_id", "run_id")
            selected_lon = as_float(first_value(src_row, "selected_lon_wgs84", "center_lon"))
            selected_lat = as_float(first_value(src_row, "selected_lat_wgs84", "center_lat"))
            can_enter_1km = as_int(src_row.get("can_enter_1km_analysis"))
            row = {
                "mall_id": mall_id,
                "name": src_row.get("name", ""),
                "province": province,
                "city": city,
                "source_run_id": source_run_id,
                "source_run_dir": source_run_dir,
                "confidence_level": src_row.get("confidence_level", ""),
                "can_enter_1km_analysis": can_enter_1km,
                "review_required": as_int(src_row.get("review_required")),
                "review_status": src_row.get("review_status", ""),
                "identity_status": src_row.get("identity_status", ""),
                "identity_reasons": src_row.get("identity_reasons", ""),
                "selected_lon_wgs84": selected_lon,
                "selected_lat_wgs84": selected_lat,
                "approved_center_lon": as_float(src_row.get("approved_center_lon")) or (selected_lon if can_enter_1km else None),
                "approved_center_lat": as_float(src_row.get("approved_center_lat")) or (selected_lat if can_enter_1km else None),
                "selected_provider": first_value(src_row, "selected_provider", "center_source"),
                "selected_poi_name": first_value(src_row, "selected_poi_name", "center_poi_name"),
                "selected_address": first_value(src_row, "selected_address", "center_address"),
                "selected_poi_type": first_value(src_row, "selected_poi_type", "center_poi_type", "center_type"),
                "selected_name_similarity": as_float(src_row.get("selected_name_similarity")),
                "official_provider_count": as_int(src_row.get("official_provider_count")),
                "agreement_provider_count": as_int(src_row.get("agreement_provider_count")),
                "agreement_radius_m": as_float(src_row.get("agreement_radius_m")),
                "nearest_provider_distance_m": as_float(src_row.get("nearest_provider_distance_m")),
                "best_identity_score": as_float(src_row.get("best_identity_score")),
                "name_evidence": src_row.get("name_evidence", ""),
                "official_coord_evidence": src_row.get("official_coord_evidence", ""),
                "image_evidence": src_row.get("image_evidence", ""),
                "clip_decision": src_row.get("clip_decision", ""),
                "needs_review": 1 if needs_review else 0,
                "suspicious_reason": suspicious_reason,
                "source_csv": str(input_path),
            }
            upsert_mall(conn, row)
            imported += 1
            queued += 1 if needs_review else 0
            candidate_csv = candidate_csv_for_row(src_row, input_path)
            if candidate_csv is not None and candidate_csv not in candidate_files_seen:
                candidate_rows += import_candidates(conn, candidate_csv, source_run_id)
                candidate_files_seen.add(candidate_csv)
    coord_backfilled = backfill_missing_selected_coords(conn)
    second_review_requeued = reset_second_review_queue(conn, config)
    conn.commit()
    export_review_queue(conn, path_value(config, "paths", "review_queue_csv"))
    experiment_root = config["paths"].get("experiment_output_root")
    if experiment_root:
        experiment_path = Path(str(experiment_root))
        if not experiment_path.is_absolute():
            experiment_path = path_value(config, "paths", "project_root") / experiment_path
        export_review_queue(conn, experiment_path / "review_queue.csv")
    return {
        "imported": imported,
        "queued": queued,
        "candidate_rows": candidate_rows,
        "coord_backfilled": coord_backfilled,
        "second_review_requeued": second_review_requeued,
    }


def export_review_queue(conn: sqlite3.Connection, output_path: Path) -> None:
    rows = conn.execute(
        """
        select mall_id, name, province, city, confidence_level, identity_status, selected_poi_name,
               selected_address, selected_poi_type, selected_lon_wgs84, selected_lat_wgs84,
               official_coord_evidence, identity_reasons, suspicious_reason
        from malls
        where needs_review = 1
        order by mall_id
        """
    ).fetchall()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as fh:
        fieldnames = list(rows[0].keys()) if rows else [
            "mall_id",
            "name",
            "province",
            "city",
            "confidence_level",
            "identity_status",
            "selected_poi_name",
            "selected_address",
            "selected_poi_type",
            "selected_lon_wgs84",
            "selected_lat_wgs84",
            "official_coord_evidence",
            "identity_reasons",
            "suspicious_reason",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([dict(row) for row in rows])
