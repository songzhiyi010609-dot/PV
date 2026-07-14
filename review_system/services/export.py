from __future__ import annotations

import csv
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from settings import path_value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def experiment_output_path(config: dict[str, Any], filename: str) -> Path | None:
    root = config["paths"].get("experiment_output_root")
    if not root:
        return None
    root_path = Path(str(root))
    if not root_path.is_absolute():
        root_path = path_value(config, "paths", "project_root") / root_path
    return root_path / filename


def export_review_results(conn: sqlite3.Connection, config: dict[str, Any]) -> int:
    rows = conn.execute(
        """
        select
            m.mall_id, m.name, m.province, m.city, m.confidence_level as old_confidence_level,
            m.identity_status, m.selected_poi_name, m.selected_address, m.selected_poi_type,
            m.selected_lon_wgs84, m.selected_lat_wgs84,
            r.reviewer, r.decision_level, r.final_lon, r.final_lat,
            r.final_polygon_geojson, r.reason, r.notes, r.review_round, r.reviewed_at
        from reviews r
        join malls m on m.mall_id = r.mall_id
        order by r.reviewed_at desc, m.mall_id
        """
    ).fetchall()
    fieldnames = [
        "mall_id",
        "name",
        "province",
        "city",
        "old_confidence_level",
        "identity_status",
        "selected_poi_name",
        "selected_address",
        "selected_poi_type",
        "selected_lon_wgs84",
        "selected_lat_wgs84",
        "reviewer",
        "decision_level",
        "final_lon",
        "final_lat",
        "final_polygon_geojson",
        "reason",
        "notes",
        "review_round",
        "reviewed_at",
    ]
    result_rows = [dict(row) for row in rows]
    write_csv(path_value(config, "paths", "review_results_csv"), result_rows, fieldnames)
    experiment_path = experiment_output_path(config, "review_results.csv")
    if experiment_path is not None:
        write_csv(experiment_path, result_rows, fieldnames)
    return len(rows)


def export_approved_centers(conn: sqlite3.Connection, config: dict[str, Any]) -> int:
    rows = conn.execute(
        """
        select
            m.mall_id, m.name, m.province, m.city,
            coalesce(r.decision_level, m.confidence_level) as confidence_level,
            case
                when r.decision_level in ('A', 'B') then r.final_lon
                else m.approved_center_lon
            end as center_lon,
            case
                when r.decision_level in ('A', 'B') then r.final_lat
                else m.approved_center_lat
            end as center_lat,
            case
                when r.decision_level in ('A', 'B') then 'manual_review'
                else 'official_poi_auto'
            end as center_source,
            r.reviewer,
            r.reason,
            r.final_polygon_geojson,
            r.review_round,
            r.reviewed_at,
            m.selected_provider,
            m.selected_poi_name,
            m.official_coord_evidence
        from malls m
        left join reviews r on r.mall_id = m.mall_id
        where
            (r.decision_level in ('A', 'B') and r.final_lon is not null and r.final_lat is not null)
            or (
                r.mall_id is null
                and m.confidence_level in ('A', 'B')
                and m.approved_center_lon is not null
                and m.approved_center_lat is not null
            )
        order by m.mall_id
        """
    ).fetchall()
    fieldnames = [
        "mall_id",
        "name",
        "province",
        "city",
        "confidence_level",
        "center_lon",
        "center_lat",
        "center_source",
        "reviewer",
        "reason",
        "final_polygon_geojson",
        "review_round",
        "reviewed_at",
        "selected_provider",
        "selected_poi_name",
        "official_coord_evidence",
    ]
    approved_rows = [dict(row) for row in rows]
    write_csv(path_value(config, "paths", "approved_centers_csv"), approved_rows, fieldnames)
    experiment_path = experiment_output_path(config, "mall_center_review_approved.csv")
    if experiment_path is not None:
        write_csv(experiment_path, approved_rows, fieldnames)
    return len(rows)


def export_summary(conn: sqlite3.Connection, config: dict[str, Any], reviewed_count: int, approved_count: int) -> None:
    total = conn.execute("select count(*) as n from malls").fetchone()["n"]
    queued = conn.execute("select count(*) as n from malls where needs_review = 1").fetchone()["n"]
    pending = conn.execute(
        """
        select count(*) as n
        from malls m
        left join reviews r on r.mall_id = m.mall_id
        where m.needs_review = 1 and r.mall_id is null
        """
    ).fetchone()["n"]
    level_rows = conn.execute(
        """
        select coalesce(r.decision_level, m.confidence_level) as level, count(*) as n
        from malls m
        left join reviews r on r.mall_id = m.mall_id
        group by coalesce(r.decision_level, m.confidence_level)
        order by level
        """
    ).fetchall()
    lines = [
        "# 商场中心人工复核汇总",
        "",
        f"- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- 总记录数：{total}",
        f"- 进入复核队列：{queued}",
        f"- 已人工复核：{reviewed_count}",
        f"- 待人工复核：{pending}",
        f"- 可进入 1km 分析中心点：{approved_count}",
        f"- 复核结果 CSV：`{path_value(config, 'paths', 'review_results_csv')}`",
        f"- 通过中心点 CSV：`{path_value(config, 'paths', 'approved_centers_csv')}`",
        "",
        "## 当前等级统计",
        "",
    ]
    for row in level_rows:
        lines.append(f"- {row['level'] or '空'}：{row['n']}")
    content = "\n".join(lines) + "\n"
    path_value(config, "paths", "summary_markdown").write_text(content, encoding="utf-8")
    experiment_path = experiment_output_path(config, "review_summary.md")
    if experiment_path is not None:
        experiment_path.parent.mkdir(parents=True, exist_ok=True)
        experiment_path.write_text(content, encoding="utf-8")


def export_all(conn: sqlite3.Connection, config: dict[str, Any]) -> dict[str, int]:
    reviewed_count = export_review_results(conn, config)
    approved_count = export_approved_centers(conn, config)
    export_summary(conn, config, reviewed_count, approved_count)
    return {"reviewed": reviewed_count, "approved": approved_count}
