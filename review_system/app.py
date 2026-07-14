from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import uvicorn
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from db import connect, init_db, row_to_dict
from services.auth import create_session, delete_session, get_session_user, load_users, verify_user
from services.export import export_all
from services.imagery import candidate_markers, ensure_crop
from services.pv_review_queue import export_pv_review_results, import_pv_review_inputs
from services.review_queue import import_review_inputs
from settings import SYSTEM_ROOT, ensure_output_dirs, ensure_users_file, load_config, path_value


app = FastAPI(title="PV Mall Center Review")
templates = Jinja2Templates(directory=str(SYSTEM_ROOT / "templates"))
app.mount("/static", StaticFiles(directory=str(SYSTEM_ROOT / "static")), name="static")


def app_config(request: Request) -> dict[str, Any]:
    return request.app.state.config


def db_conn(request: Request) -> sqlite3.Connection:
    return connect(app_config(request))


def current_user(request: Request) -> str | None:
    config = app_config(request)
    token = request.cookies.get(str(config["auth"]["session_cookie_name"]))
    with connect(config) as conn:
        return get_session_user(conn, token)


def require_user(request: Request) -> str:
    user = current_user(request)
    if not user:
        raise HTTPException(status_code=303, headers={"Location": "/login"})
    return user


def redirect(path: str) -> RedirectResponse:
    return RedirectResponse(path, status_code=303)


@app.on_event("startup")
def startup() -> None:
    config = load_config(getattr(app.state, "config_path", None) or Path(SYSTEM_ROOT / "config" / "review_config.json"))
    ensure_output_dirs(config)
    users_path, users_created = ensure_users_file(config)
    users, setup_needed = load_users(users_path)
    with connect(config) as conn:
        init_db(conn)
        import_stats = import_review_inputs(conn, config)
        export_stats = export_all(conn, config)
        pv_import_stats = import_pv_review_inputs(conn, config)
        pv_export_count = export_pv_review_results(conn, config)
    app.state.config = config
    app.state.users_path = users_path
    app.state.users_created = users_created
    app.state.users = users
    app.state.setup_needed = setup_needed
    app.state.import_stats = import_stats
    app.state.export_stats = export_stats
    app.state.pv_import_stats = pv_import_stats
    app.state.pv_export_count = pv_export_count


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = "") -> HTMLResponse:
    if current_user(request):
        return redirect("/")
    users_path = Path(str(app_config(request)["auth"]["users_file"]))
    users, setup_needed = load_users(users_path)
    request.app.state.users = users
    request.app.state.setup_needed = setup_needed
    return templates.TemplateResponse(
        request,
        "login.html",
        {
            "request": request,
            "error": error,
            "users_path": getattr(request.app.state, "users_path", None),
            "users_created": getattr(request.app.state, "users_created", False),
            "setup_needed": setup_needed,
        },
    )


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)) -> RedirectResponse:
    users_path = Path(str(app_config(request)["auth"]["users_file"]))
    users, setup_needed = load_users(users_path)
    request.app.state.users = users
    request.app.state.setup_needed = setup_needed
    if setup_needed:
        return redirect("/login?error=请先修改 users.local.json 里的示例密码")
    if not verify_user(users, username, password):
        return redirect("/login?error=账号或密码不正确")
    with db_conn(request) as conn:
        token = create_session(conn, username, int(app_config(request)["auth"]["session_ttl_hours"]))
    response = redirect("/")
    response.set_cookie(
        str(app_config(request)["auth"]["session_cookie_name"]),
        token,
        httponly=True,
        samesite="lax",
        max_age=int(app_config(request)["auth"]["session_ttl_hours"]) * 3600,
    )
    return response


@app.get("/logout")
def logout(request: Request) -> RedirectResponse:
    token = request.cookies.get(str(app_config(request)["auth"]["session_cookie_name"]))
    with db_conn(request) as conn:
        delete_session(conn, token)
    response = redirect("/login")
    response.delete_cookie(str(app_config(request)["auth"]["session_cookie_name"]))
    return response


def province_clause(province: str) -> tuple[str, list[Any]]:
    province = (province or "").strip()
    if not province:
        return "", []
    return " and coalesce(m.province, '') = ?", [province]


def province_options(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        select distinct coalesce(province, '') as province
        from malls
        where coalesce(province, '') <> ''
        order by province
        """
    ).fetchall()
    return [str(row["province"]) for row in rows]


def dashboard_rows(conn: sqlite3.Connection, view: str, province: str = "") -> list[sqlite3.Row]:
    where = "m.needs_review = 1"
    if view == "pending":
        where = "m.needs_review = 1 and r.mall_id is null"
    elif view == "reviewed":
        where = "r.mall_id is not null"
    elif view == "all":
        where = "1 = 1"
    province_sql, province_params = province_clause(province)
    return conn.execute(
        f"""
        select m.mall_id, m.name, m.province, m.city, m.confidence_level, m.identity_status, m.selected_poi_name,
               m.official_coord_evidence, m.suspicious_reason, r.decision_level, r.reviewer, r.reviewed_at,
               h.decision_level as previous_decision_level
        from malls m
        left join reviews r on r.mall_id = m.mall_id
        left join review_history h on h.id = (
            select h2.id
            from review_history h2
            where h2.mall_id = m.mall_id
              and h2.archive_reason = 'second_review_requeue'
            order by h2.archived_at desc, h2.id desc
            limit 1
        )
        where {where}{province_sql}
        order by case when r.mall_id is null then 0 else 1 end, m.mall_id
        limit 500
        """,
        province_params,
    ).fetchall()


def dashboard_stats(conn: sqlite3.Connection, province: str = "") -> dict[str, int]:
    province_sql, province_params = province_clause(province)

    def scalar(sql: str, params: list[Any] | None = None) -> int:
        return int(conn.execute(sql, params or []).fetchone()["n"])

    return {
        "total": scalar(f"select count(*) as n from malls m where 1=1{province_sql}", province_params),
        "queue": scalar(f"select count(*) as n from malls m where needs_review = 1{province_sql}", province_params),
        "pending": scalar(
            f"select count(*) as n from malls m left join reviews r on r.mall_id=m.mall_id where m.needs_review=1 and r.mall_id is null{province_sql}",
            province_params,
        ),
        "reviewed": scalar(
            f"select count(*) as n from reviews r join malls m on m.mall_id = r.mall_id where 1=1{province_sql}",
            province_params,
        ),
        "approved": scalar(
            "select count(*) as n from malls m left join reviews r on r.mall_id=m.mall_id "
            f"where (r.decision_level in ('A','B') or (r.mall_id is null and m.confidence_level in ('A','B'))){province_sql}",
            province_params,
        ),
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, view: str = "pending", province: str = "") -> HTMLResponse:
    user = require_user(request)
    with db_conn(request) as conn:
        rows = dashboard_rows(conn, view, province)
        stats = dashboard_stats(conn, province)
        provinces = province_options(conn)
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "request": request,
            "user": user,
            "rows": rows,
            "stats": stats,
            "view": view,
            "province_filter": province,
            "provinces": provinces,
            "config": app_config(request),
        },
    )


def pv_clause(province: str) -> tuple[str, list[Any]]:
    province = (province or "").strip()
    if not province:
        return "", []
    return " and coalesce(p.province, '') = ?", [province]


def pv_province_options(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        """
        select distinct coalesce(province, '') as province
        from pv_review_items
        where coalesce(province, '') <> ''
        order by province
        """
    ).fetchall()
    return [str(row["province"]) for row in rows]


def pv_dashboard_rows(conn: sqlite3.Connection, view: str, province: str = "") -> list[sqlite3.Row]:
    where = "1 = 1"
    if view == "pending":
        where = "r.mall_id is null"
    elif view == "reviewed":
        where = "r.mall_id is not null"
    elif view == "with_image":
        where = "coalesce(p.full_annotated_path, '') <> ''"
    elif view == "missing_image":
        where = "coalesce(p.full_annotated_path, '') = ''"
    province_sql, province_params = pv_clause(province)
    return conn.execute(
        f"""
        select
            p.mall_id, p.name, p.province, p.city, p.image_status,
            p.pv_status, p.pv_confidence, p.pv_area_m2_est,
            p.roof_candidate_count, p.roof_area_m2_est, p.install_condition_level,
            p.full_annotated_path, p.poi_reviewed_at,
            r.decision, r.corrected_pv_status, r.corrected_potential_level, r.reviewed_at
        from pv_review_items p
        left join pv_reviews r on r.mall_id = p.mall_id
        where {where}{province_sql}
        order by case when r.mall_id is null then 0 else 1 end, p.mall_id
        limit 500
        """,
        province_params,
    ).fetchall()


def pv_dashboard_stats(conn: sqlite3.Connection, province: str = "") -> dict[str, int]:
    province_sql, province_params = pv_clause(province)

    def scalar(sql: str, params: list[Any] | None = None) -> int:
        return int(conn.execute(sql, params or []).fetchone()["n"])

    return {
        "total": scalar(f"select count(*) as n from pv_review_items p where 1=1{province_sql}", province_params),
        "pending": scalar(
            f"select count(*) as n from pv_review_items p left join pv_reviews r on r.mall_id=p.mall_id where r.mall_id is null{province_sql}",
            province_params,
        ),
        "reviewed": scalar(
            f"select count(*) as n from pv_reviews r join pv_review_items p on p.mall_id = r.mall_id where 1=1{province_sql}",
            province_params,
        ),
        "with_image": scalar(
            f"select count(*) as n from pv_review_items p where coalesce(p.full_annotated_path, '') <> ''{province_sql}",
            province_params,
        ),
        "missing_image": scalar(
            f"select count(*) as n from pv_review_items p where coalesce(p.full_annotated_path, '') = ''{province_sql}",
            province_params,
        ),
    }


@app.get("/pv", response_class=HTMLResponse)
def pv_dashboard(request: Request, view: str = "pending", province: str = "") -> HTMLResponse:
    user = require_user(request)
    with db_conn(request) as conn:
        rows = pv_dashboard_rows(conn, view, province)
        stats = pv_dashboard_stats(conn, province)
        provinces = pv_province_options(conn)
    return templates.TemplateResponse(
        request,
        "pv_dashboard.html",
        {
            "request": request,
            "user": user,
            "rows": rows,
            "stats": stats,
            "view": view,
            "province_filter": province,
            "provinces": provinces,
            "config": app_config(request),
        },
    )


def fetch_pv_item(conn: sqlite3.Connection, mall_id: int) -> dict[str, Any]:
    row = conn.execute("select * from pv_review_items where mall_id = ?", (mall_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="PV review item not found")
    return dict(row)


def fetch_pv_review(conn: sqlite3.Connection, mall_id: int) -> dict[str, Any] | None:
    row = conn.execute("select * from pv_reviews where mall_id = ?", (mall_id,)).fetchone()
    return row_to_dict(row)


def next_pv_pending_id(conn: sqlite3.Connection, current_id: int, province: str = "") -> int | None:
    province_sql, province_params = pv_clause(province)
    row = conn.execute(
        f"""
        select p.mall_id
        from pv_review_items p
        left join pv_reviews r on r.mall_id = p.mall_id
        where r.mall_id is null and p.mall_id > ?{province_sql}
        order by p.mall_id
        limit 1
        """,
        [current_id] + province_params,
    ).fetchone()
    if row is not None:
        return int(row["mall_id"])
    row = conn.execute(
        f"""
        select p.mall_id
        from pv_review_items p
        left join pv_reviews r on r.mall_id = p.mall_id
        where r.mall_id is null{province_sql}
        order by p.mall_id
        limit 1
        """,
        province_params,
    ).fetchone()
    return int(row["mall_id"]) if row is not None else None


def previous_pv_queue_id(conn: sqlite3.Connection, current_id: int, province: str = "") -> int | None:
    province_sql, province_params = pv_clause(province)
    row = conn.execute(
        f"""
        select p.mall_id
        from pv_review_items p
        where p.mall_id < ?{province_sql}
        order by p.mall_id desc
        limit 1
        """,
        [current_id] + province_params,
    ).fetchone()
    return int(row["mall_id"]) if row is not None else None


@app.get("/pv/review/{mall_id}", response_class=HTMLResponse)
def pv_review_page(request: Request, mall_id: int, province: str = "") -> HTMLResponse:
    user = require_user(request)
    with db_conn(request) as conn:
        item = fetch_pv_item(conn, mall_id)
        province_filter = province or str(item.get("province") or "")
        review = fetch_pv_review(conn, mall_id)
        prev_id = previous_pv_queue_id(conn, mall_id, province_filter)
        next_id = next_pv_pending_id(conn, mall_id, province_filter)
    return templates.TemplateResponse(
        request,
        "pv_review.html",
        {
            "request": request,
            "user": user,
            "item": item,
            "review": review,
            "prev_id": prev_id,
            "next_id": next_id,
            "province_filter": province_filter,
            "config": app_config(request),
        },
    )


@app.get("/pv/image/{mall_id}/{kind}")
def pv_image(request: Request, mall_id: int, kind: str) -> FileResponse:
    require_user(request)
    field_by_kind = {
        "annotated": "full_annotated_path",
        "raw": "full_raw_path",
        "pv": "full_pv_overlay_path",
        "roof": "full_roof_overlay_path",
        "self": "self_image_path",
        "self_overlay": "self_overlay_path",
        "mosaic": "review_mosaic_path",
    }
    field = field_by_kind.get(kind)
    if field is None:
        raise HTTPException(status_code=404, detail="Unknown PV image kind")
    with db_conn(request) as conn:
        item = fetch_pv_item(conn, mall_id)
    raw_path = str(item.get(field) or "").strip()
    if not raw_path:
        raise HTTPException(status_code=404, detail="PV image unavailable")
    path = Path(raw_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="PV image file missing")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/pv/review/{mall_id}")
def save_pv_review(
    request: Request,
    mall_id: int,
    decision: str = Form(...),
    corrected_pv_status: str = Form(""),
    corrected_potential_level: str = Form(""),
    reason: str = Form(""),
    notes: str = Form(""),
    province: str = Form(""),
) -> RedirectResponse:
    user = require_user(request)
    decision = decision.strip()
    config = app_config(request)
    if decision not in set(config.get("pv_review", {}).get("decisions", [])):
        raise HTTPException(status_code=400, detail="Invalid PV decision")
    with db_conn(request) as conn:
        fetch_pv_item(conn, mall_id)
        conn.execute(
            """
            insert into pv_reviews(
                mall_id, reviewer, decision, corrected_pv_status,
                corrected_potential_level, reason, notes, reviewed_at
            )
            values (?, ?, ?, ?, ?, ?, ?, current_timestamp)
            on conflict(mall_id) do update set
                reviewer=excluded.reviewer,
                decision=excluded.decision,
                corrected_pv_status=excluded.corrected_pv_status,
                corrected_potential_level=excluded.corrected_potential_level,
                reason=excluded.reason,
                notes=excluded.notes,
                reviewed_at=current_timestamp
            """,
            (
                mall_id,
                user,
                decision,
                corrected_pv_status,
                corrected_potential_level,
                reason,
                notes,
            ),
        )
        conn.commit()
        export_pv_review_results(conn, config)
        next_id = next_pv_pending_id(conn, mall_id, province)
    suffix = f"?{urlencode({'province': province})}" if province else ""
    reviewed_suffix = f"&{urlencode({'province': province})}" if province else ""
    return redirect(f"/pv/review/{next_id}{suffix}" if next_id is not None else f"/pv?view=reviewed{reviewed_suffix}")


def fetch_mall(conn: sqlite3.Connection, mall_id: int) -> dict[str, Any]:
    row = conn.execute("select * from malls where mall_id = ?", (mall_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Mall not found")
    return dict(row)


def fetch_candidates(conn: sqlite3.Connection, mall_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        select provider, rank, poi_name, address, district, poi_type, wgs84_lon, wgs84_lat,
               identity_score, name_similarity, flags
        from poi_candidates
        where mall_id = ?
        order by provider, rank
        """,
        (mall_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def image_center_mall(mall: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[dict[str, Any], bool]:
    image_mall = dict(mall)
    if image_mall.get("selected_lon_wgs84") is not None and image_mall.get("selected_lat_wgs84") is not None:
        return image_mall, False
    for item in candidates:
        if item.get("wgs84_lon") is None or item.get("wgs84_lat") is None:
            continue
        image_mall["selected_lon_wgs84"] = item["wgs84_lon"]
        image_mall["selected_lat_wgs84"] = item["wgs84_lat"]
        return image_mall, True
    return image_mall, False


def fetch_review(conn: sqlite3.Connection, mall_id: int) -> dict[str, Any] | None:
    row = conn.execute("select * from reviews where mall_id = ?", (mall_id,)).fetchone()
    return row_to_dict(row)


def fetch_previous_review(conn: sqlite3.Connection, mall_id: int) -> dict[str, Any] | None:
    row = conn.execute(
        """
        select *
        from review_history
        where mall_id = ?
          and archive_reason = 'second_review_requeue'
        order by archived_at desc, id desc
        limit 1
        """,
        (mall_id,),
    ).fetchone()
    return row_to_dict(row)


def next_pending_id(conn: sqlite3.Connection, current_id: int, province: str = "") -> int | None:
    province_sql, province_params = province_clause(province)
    row = conn.execute(
        f"""
        select m.mall_id
        from malls m
        left join reviews r on r.mall_id = m.mall_id
        where m.needs_review = 1 and r.mall_id is null and m.mall_id > ?{province_sql}
        order by m.mall_id
        limit 1
        """,
        [current_id] + province_params,
    ).fetchone()
    if row is not None:
        return int(row["mall_id"])
    row = conn.execute(
        f"""
        select m.mall_id
        from malls m
        left join reviews r on r.mall_id = m.mall_id
        where m.needs_review = 1 and r.mall_id is null{province_sql}
        order by m.mall_id
        limit 1
        """,
        province_params,
    ).fetchone()
    return int(row["mall_id"]) if row is not None else None


def previous_queue_id(conn: sqlite3.Connection, current_id: int, province: str = "") -> int | None:
    province_sql, province_params = province_clause(province)
    row = conn.execute(
        f"""
        select m.mall_id
        from malls m
        where m.needs_review = 1 and m.mall_id < ?{province_sql}
        order by m.mall_id desc
        limit 1
        """,
        [current_id] + province_params,
    ).fetchone()
    return int(row["mall_id"]) if row is not None else None


@app.get("/review/{mall_id}", response_class=HTMLResponse)
def review_page(request: Request, mall_id: int, province: str = "") -> HTMLResponse:
    user = require_user(request)
    with db_conn(request) as conn:
        mall = fetch_mall(conn, mall_id)
        province_filter = province or str(mall.get("province") or "")
        candidates = fetch_candidates(conn, mall_id)
        review = fetch_review(conn, mall_id)
        previous_review = fetch_previous_review(conn, mall_id)
        prev_id = previous_queue_id(conn, mall_id, province_filter)
        next_id = next_pending_id(conn, mall_id, province_filter)
    image_mall, image_center_is_fallback = image_center_mall(mall, candidates)
    crop_available = image_mall.get("selected_lon_wgs84") is not None and image_mall.get("selected_lat_wgs84") is not None
    markers = candidate_markers(app_config(request), image_mall, candidates)
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "request": request,
            "user": user,
            "mall": mall,
            "image_mall": image_mall,
            "image_center_is_fallback": image_center_is_fallback,
            "candidates": candidates,
            "markers": markers,
            "review": review,
            "previous_review": previous_review,
            "prev_id": prev_id,
            "next_id": next_id,
            "province_filter": province_filter,
            "crop_available": crop_available,
            "config": app_config(request),
        },
    )


@app.get("/satellite/{mall_id}")
def satellite_image(request: Request, mall_id: int) -> FileResponse:
    require_user(request)
    with db_conn(request) as conn:
        mall = fetch_mall(conn, mall_id)
        candidates = fetch_candidates(conn, mall_id)
    image_mall, _ = image_center_mall(mall, candidates)
    crop_path = ensure_crop(app_config(request), image_mall)
    if crop_path is None or not crop_path.exists():
        raise HTTPException(status_code=404, detail="Satellite image unavailable")
    return FileResponse(crop_path, media_type="image/jpeg")


@app.get("/api/map_config")
def map_config(request: Request) -> JSONResponse:
    require_user(request)
    online_map = app_config(request).get("online_map", {})
    return JSONResponse(
        {
            "enabled": bool(online_map.get("enabled", True)),
            "provider": online_map.get("provider", "amap"),
            "amapKey": online_map.get("amap_key", ""),
            "defaultZoom": int(online_map.get("default_zoom", 18)),
        }
    )


@app.post("/review/{mall_id}")
def save_review(
    request: Request,
    mall_id: int,
    decision_level: str = Form(...),
    final_lon: str = Form(""),
    final_lat: str = Form(""),
    final_polygon_geojson: str = Form(""),
    reason: str = Form(""),
    notes: str = Form(""),
    province: str = Form(""),
) -> RedirectResponse:
    user = require_user(request)
    decision_level = decision_level.strip().upper()
    if decision_level not in set(app_config(request)["review"]["decision_levels"]):
        raise HTTPException(status_code=400, detail="Invalid decision level")
    lon_value = float(final_lon) if final_lon.strip() else None
    lat_value = float(final_lat) if final_lat.strip() else None
    if decision_level in {"A", "B"} and (lon_value is None or lat_value is None):
        raise HTTPException(status_code=400, detail="A/B must provide final lon/lat")
    with db_conn(request) as conn:
        fetch_mall(conn, mall_id)
        conn.execute(
            """
            insert into reviews(mall_id, reviewer, decision_level, final_lon, final_lat, final_polygon_geojson, reason, notes, review_round, reviewed_at)
            values (?, ?, ?, ?, ?, ?, ?, ?, ?, current_timestamp)
            on conflict(mall_id) do update set
                reviewer=excluded.reviewer,
                decision_level=excluded.decision_level,
                final_lon=excluded.final_lon,
                final_lat=excluded.final_lat,
                final_polygon_geojson=excluded.final_polygon_geojson,
                reason=excluded.reason,
                notes=excluded.notes,
                review_round=excluded.review_round,
                reviewed_at=current_timestamp
            """,
            (
                mall_id,
                user,
                decision_level,
                lon_value,
                lat_value,
                final_polygon_geojson.strip(),
                reason,
                notes,
                2 if app_config(request).get("review", {}).get("second_review_enabled", False) else 1,
            ),
        )
        conn.commit()
        export_all(conn, app_config(request))
        import_pv_review_inputs(conn, app_config(request))
        export_pv_review_results(conn, app_config(request))
        next_id = next_pending_id(conn, mall_id, province)
    suffix = f"?{urlencode({'province': province})}" if province else ""
    reviewed_suffix = f"&{urlencode({'province': province})}" if province else ""
    return redirect(f"/review/{next_id}{suffix}" if next_id is not None else f"/?view=reviewed{reviewed_suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the mall center review web system.")
    parser.add_argument("--config", type=Path, default=SYSTEM_ROOT / "config" / "review_config.json")
    parser.add_argument("--host", default="")
    parser.add_argument("--port", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    app.state.config_path = args.config
    config = load_config(args.config)
    host = args.host or str(config["server"]["host"])
    port = args.port or int(config["server"]["port"])
    uvicorn.run(app, host=host, port=port, reload=bool(config["server"].get("reload", False)))
