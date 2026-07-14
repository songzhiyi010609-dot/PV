from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = SYSTEM_ROOT / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "review_config.json"
USERS_EXAMPLE_PATH = CONFIG_DIR / "users.local.example.json"
DEFAULT_USERS_PATH = CONFIG_DIR / "users.local.json"


DEFAULT_CONFIG: dict[str, Any] = {
    "server": {"host": "0.0.0.0", "port": 8787, "reload": False},
    "auth": {
        "users_file": str(DEFAULT_USERS_PATH),
        "session_cookie_name": "pv_review_session",
        "session_ttl_hours": 12,
    },
    "paths": {
        "project_root": str(PROJECT_ROOT),
        "output_root": str(PROJECT_ROOT / "outputs" / "review" / "mall_center_review"),
        "database": str(PROJECT_ROOT / "outputs" / "review" / "mall_center_review" / "review.db"),
        "review_queue_csv": str(PROJECT_ROOT / "outputs" / "review" / "mall_center_review" / "review_queue.csv"),
        "review_results_csv": str(PROJECT_ROOT / "outputs" / "review" / "mall_center_review" / "review_results.csv"),
        "approved_centers_csv": str(
            PROJECT_ROOT / "outputs" / "review" / "mall_center_review" / "mall_center_review_approved.csv"
        ),
        "pv_review_results_csv": str(PROJECT_ROOT / "outputs" / "review" / "mall_center_review" / "pv_review_results.csv"),
        "summary_markdown": str(PROJECT_ROOT / "outputs" / "review" / "mall_center_review" / "review_summary.md"),
        "experiment_output_root": str(PROJECT_ROOT / "outputs" / "experiments" / "mall_center_review_all"),
        "satellite_crop_dir": str(PROJECT_ROOT / "outputs" / "review" / "mall_center_review" / "satellite_crops"),
        "tile_cache_dir": str(
            PROJECT_ROOT / "outputs" / "review" / "mall_center_review" / "tile_cache" / "esri_world_imagery"
        ),
        "input_result_files": [
            str(PROJECT_ROOT / "outputs" / "experiments" / "poi_completed_100" / "completed_mall_poi_results_100.csv"),
            str(PROJECT_ROOT / "outputs" / "experiments" / "poi_remaining_after100" / "data" / "mall_center_resolved.csv"),
            str(
                PROJECT_ROOT
                / "outputs"
                / "experiments"
                / "poi_validate_zhejiang"
                / "20260713_zhejiang_full"
                / "data"
                / "mall_center_resolved.csv"
            ),
        ],
    },
    "imagery": {
        "provider": "esri_world_imagery",
        "zoom": 18,
        "crop_size_px": 1536,
        "timeout_seconds": 30,
        "jpeg_quality": 94,
        "marker_visible_margin_px": 120,
    },
    "online_map": {
        "enabled": True,
        "provider": "amap",
        "amap_key": "",
        "default_zoom": 18,
    },
    "pv_review": {
        "source_run_dir": str(
            PROJECT_ROOT
            / "outputs"
            / "experiments"
            / "mall_pv_potential_screening"
            / "20260713_shanghai_A_bdappv_singlemap_v2"
        ),
        "decisions": [
            "通过",
            "光伏误检",
            "潜力屋顶误检",
            "边界需修正",
            "影像缺失/异常",
            "暂缓",
        ],
        "default_decision": "通过",
        "pv_status_options": ["沿用算法", "有光伏", "无明显光伏", "不确定"],
        "potential_level_options": ["沿用算法", "高", "中", "低", "无", "不确定"],
    },
    "review": {
        "include_confidence_levels": ["C"],
        "second_review_enabled": True,
        "second_review_levels": ["B", "C", "D"],
        "suspicious_keywords": [
            "入口",
            "出口",
            "停车场",
            "服务台",
            "办公区",
            "展厅",
            "南门",
            "北门",
            "东门",
            "西门",
            "营地",
            "地铁",
            "公交",
            "项目",
            "地块",
        ],
        "decision_levels": ["A", "B", "C", "D"],
        "default_decision_level": "C",
        "reasons": [
            "确认商场主体中心",
            "坐标修正后通过",
            "入口/门点不是中心",
            "停车场/内部店铺误匹配",
            "项目地块/建设中",
            "办公区/展厅误匹配",
            "多源坐标不一致",
            "影像不清楚",
            "不是商场",
            "不在上海",
            "暂缓",
        ],
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not config_path.exists():
        config_path.write_text(json.dumps(DEFAULT_CONFIG, ensure_ascii=False, indent=2), encoding="utf-8")
    user_config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    config = deep_merge(DEFAULT_CONFIG, user_config)
    online_map = config.setdefault("online_map", {})
    if not online_map.get("amap_key"):
        api_key_path = PROJECT_ROOT / "config" / "api_keys.local.json"
        if api_key_path.exists():
            try:
                api_keys = json.loads(api_key_path.read_text(encoding="utf-8-sig"))
                online_map["amap_key"] = api_keys.get("AMAP_WEB_KEY", "")
            except json.JSONDecodeError:
                online_map["amap_key"] = ""
    config["_config_path"] = str(config_path)
    return config


def path_value(config: dict[str, Any], section: str, key: str) -> Path:
    value = Path(str(config[section][key]))
    if value.is_absolute():
        return value
    return PROJECT_ROOT / value


def ensure_output_dirs(config: dict[str, Any]) -> None:
    for key in [
        "output_root",
        "review_queue_csv",
        "review_results_csv",
        "approved_centers_csv",
        "pv_review_results_csv",
        "summary_markdown",
        "experiment_output_root",
        "satellite_crop_dir",
        "tile_cache_dir",
        "database",
    ]:
        path = path_value(config, "paths", key)
        target = path if key in {"output_root", "experiment_output_root", "satellite_crop_dir", "tile_cache_dir"} else path.parent
        target.mkdir(parents=True, exist_ok=True)


def ensure_users_file(config: dict[str, Any]) -> tuple[Path, bool]:
    users_path = Path(str(config["auth"]["users_file"]))
    if not users_path.is_absolute():
        users_path = PROJECT_ROOT / users_path
    users_path.parent.mkdir(parents=True, exist_ok=True)
    created = False
    if not users_path.exists():
        if USERS_EXAMPLE_PATH.exists():
            users_path.write_text(USERS_EXAMPLE_PATH.read_text(encoding="utf-8-sig"), encoding="utf-8")
        else:
            users_path.write_text(
                json.dumps({"users": {"admin": {"password": "replace_with_password", "display_name": "管理员"}}}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        created = True
    config["auth"]["users_file"] = str(users_path)
    return users_path, created
