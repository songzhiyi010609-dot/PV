from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from settings import path_value


def connect(config: dict[str, Any]) -> sqlite3.Connection:
    db_path = path_value(config, "paths", "database")
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma foreign_keys=on")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        create table if not exists malls (
            mall_id integer primary key,
            name text not null,
            province text,
            city text,
            source_run_id text,
            source_run_dir text,
            confidence_level text,
            can_enter_1km_analysis integer default 0,
            review_required integer default 0,
            review_status text,
            identity_status text,
            identity_reasons text,
            selected_lon_wgs84 real,
            selected_lat_wgs84 real,
            approved_center_lon real,
            approved_center_lat real,
            selected_provider text,
            selected_poi_name text,
            selected_address text,
            selected_poi_type text,
            selected_name_similarity real,
            official_provider_count integer,
            agreement_provider_count integer,
            agreement_radius_m real,
            nearest_provider_distance_m real,
            best_identity_score real,
            name_evidence text,
            official_coord_evidence text,
            image_evidence text,
            clip_decision text,
            needs_review integer default 0,
            suspicious_reason text,
            source_csv text,
            updated_at text default current_timestamp
        );

        create table if not exists poi_candidates (
            id integer primary key autoincrement,
            mall_id integer not null,
            provider text,
            query text,
            rank integer,
            poi_name text,
            address text,
            province text,
            city text,
            district text,
            poi_type text,
            wgs84_lon real,
            wgs84_lat real,
            identity_score real,
            name_similarity real,
            flags text,
            source_run_id text,
            unique(mall_id, provider, rank, poi_name, wgs84_lon, wgs84_lat)
        );

        create table if not exists reviews (
            mall_id integer primary key,
            reviewer text not null,
            decision_level text not null check(decision_level in ('A', 'B', 'C', 'D')),
            final_lon real,
            final_lat real,
            final_polygon_geojson text,
            reason text,
            notes text,
            review_round integer default 1,
            reviewed_at text default current_timestamp,
            foreign key(mall_id) references malls(mall_id)
        );

        create table if not exists review_history (
            id integer primary key autoincrement,
            mall_id integer not null,
            reviewer text,
            decision_level text,
            final_lon real,
            final_lat real,
            final_polygon_geojson text,
            reason text,
            notes text,
            review_round integer,
            reviewed_at text,
            archived_at text default current_timestamp,
            archive_reason text,
            foreign key(mall_id) references malls(mall_id)
        );

        create table if not exists pv_review_items (
            mall_id integer primary key,
            name text not null,
            province text,
            city text,
            center_lon real,
            center_lat real,
            poi_reviewer text,
            poi_reviewed_at text,
            pv_run_id text,
            pv_run_dir text,
            image_status text,
            self_image_path text,
            self_overlay_path text,
            full_raw_path text,
            full_annotated_path text,
            full_pv_overlay_path text,
            full_roof_overlay_path text,
            review_mosaic_path text,
            pv_status text,
            pv_confidence real,
            pv_area_m2_est real,
            roof_candidate_count integer,
            roof_area_m2_est real,
            install_condition_level text,
            notes text,
            updated_at text default current_timestamp,
            foreign key(mall_id) references malls(mall_id)
        );

        create table if not exists pv_reviews (
            mall_id integer primary key,
            reviewer text not null,
            decision text not null,
            corrected_pv_status text,
            corrected_potential_level text,
            reason text,
            notes text,
            reviewed_at text default current_timestamp,
            foreign key(mall_id) references pv_review_items(mall_id)
        );

        create table if not exists sessions (
            token text primary key,
            username text not null,
            expires_at text not null,
            created_at text default current_timestamp
        );

        create index if not exists idx_malls_needs_review on malls(needs_review);
        create index if not exists idx_candidates_mall on poi_candidates(mall_id);
        create index if not exists idx_reviews_level on reviews(decision_level);
        create index if not exists idx_pv_items_province on pv_review_items(province);
        create index if not exists idx_pv_reviews_decision on pv_reviews(decision);
        """
    )
    columns = {row["name"] for row in conn.execute("pragma table_info(reviews)").fetchall()}
    if "final_polygon_geojson" not in columns:
        conn.execute("alter table reviews add column final_polygon_geojson text")
    if "review_round" not in columns:
        conn.execute("alter table reviews add column review_round integer default 1")
    mall_columns = {row["name"] for row in conn.execute("pragma table_info(malls)").fetchall()}
    if "province" not in mall_columns:
        conn.execute("alter table malls add column province text")
    if "city" not in mall_columns:
        conn.execute("alter table malls add column city text")
    conn.commit()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def first_existing(paths: list[str]) -> Path | None:
    for item in paths:
        path = Path(item)
        if path.exists():
            return path
    return None
