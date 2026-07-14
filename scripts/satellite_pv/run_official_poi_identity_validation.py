#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate mall identity with official POI APIs.

This script queries AMap, Baidu Map, and Tencent Map official WebService POI
APIs when keys are available. It does not modify the original database or
image dataset. Results are written to a timestamped experiment directory.

Environment variables:
- AMAP_WEB_KEY
- BAIDU_MAP_AK
- TENCENT_MAP_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "malls_new.db"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "poi_validate"
DEFAULT_CLIP_ROOT = PROJECT_ROOT / "outputs" / "experiments" / "20260708_gitr_sclip_poi_review"
DEFAULT_CLIP_SCORES = DEFAULT_CLIP_ROOT / "data" / "gitr_sclip_poi_review_scores.csv"
DEFAULT_API_KEY_CONFIG = PROJECT_ROOT / "config" / "api_keys.local.json"
SHANGHAI = "\u4e0a\u6d77\u5e02"
SHANGHAI_SHORT = "\u4e0a\u6d77"
SHANGHAI_BBOX = (120.85, 30.67, 122.15, 31.87)
TARGET_BBOXES = {
    "\u4e0a\u6d77\u5e02": SHANGHAI_BBOX,
    "\u4e0a\u6d77": SHANGHAI_BBOX,
    "\u6d59\u6c5f\u7701": (118.0, 27.0, 123.2, 31.6),
    "\u6d59\u6c5f": (118.0, 27.0, 123.2, 31.6),
}
TIMESTAMP_PATTERN = re.compile(r"^\d{8}_\d{6}$")

AMAP_TEXT_URL = "https://restapi.amap.com/v3/place/text"
BAIDU_PLACE_URL = "https://api.map.baidu.com/place/v2/search"
TENCENT_PLACE_URL = "https://apis.map.qq.com/ws/place/v1/search"
PROVIDER_ENDPOINTS = {
    "amap": AMAP_TEXT_URL,
    "baidu": BAIDU_PLACE_URL,
    "tencent": TENCENT_PLACE_URL,
}
PROVIDER_DISPLAY = {
    "amap": "AMap/Gaode place text search",
    "baidu": "Baidu place search",
    "tencent": "Tencent place search (/ws/place/v1/search)",
}
SENSITIVE_PARAM_KEYS = {"key", "ak", "sk", "sig", "signature"}

MALL_HINTS = (
    "mall",
    "\u8d2d\u7269\u4e2d\u5fc3",
    "\u5546\u573a",
    "\u5546\u4e1a",
    "\u5546\u4e1a\u4e2d\u5fc3",
    "\u5546\u4e1a\u7efc\u5408\u4f53",
    "\u5e7f\u573a",
    "\u767e\u8d27",
    "\u5927\u60a6\u57ce",
    "\u5370\u8c61\u57ce",
    "\u4e07\u8c61",
    "\u5929\u8857",
    "\u5408\u751f\u6c47",
    "\u6765\u798f\u58eb",
    "\u5965\u7279\u83b1\u65af",
)

BAD_HINTS = (
    "\u5730\u94c1",
    "\u5730\u94c1\u7ad9",
    "\u516c\u4ea4",
    "\u516c\u4ea4\u7ad9",
    "\u706b\u8f66\u7ad9",
    "\u9ad8\u94c1\u7ad9",
    "\u8f66\u7ad9",
    "\u505c\u8f66\u573a",
    "\u8def\u53e3",
    "\u4ea4\u53c9\u53e3",
    "\u9053\u8def",
    "\u5b66\u6821",
    "\u533b\u9662",
    "\u9152\u5e97",
    "\u516c\u5bd3",
    "\u5c0f\u533a",
    "\u9910\u5385",
    "\u5496\u5561",
    "\u9ea6\u5f53\u52b3",
    "\u80af\u5fb7\u57fa",
    "\u661f\u5df4\u514b",
    "\u9879\u76ee\u90e8",
    "\u65bd\u5de5",
    "\u5efa\u8bbe\u4e2d",
    "\u5730\u5757",
)

GENERIC_TOKENS = (
    SHANGHAI,
    SHANGHAI_SHORT,
    "\u8d2d\u7269\u4e2d\u5fc3",
    "\u8d2d\u7269\u516c\u56ed",
    "\u5546\u573a",
    "\u5546\u4e1a\u5e7f\u573a",
    "\u5546\u4e1a\u4e2d\u5fc3",
    "\u5546\u4e1a",
    "\u7efc\u5408\u4f53",
    "\u5e7f\u573a",
    "\u4e2d\u5fc3",
    "\u9879\u76ee",
    "\u4e00\u671f",
    "\u4e8c\u671f",
    "\u4e09\u671f",
    "mall",
    "MALL",
    "Mall",
    "MAX",
    "MEGA",
)

POSITIVE_TYPE_HINTS = (
    "\u8d2d\u7269",
    "\u5546\u573a",
    "\u5546\u4e1a",
    "\u767e\u8d27",
    "\u8d2d\u7269\u670d\u52a1",
    "\u7efc\u5408\u5546\u573a",
)

MAIN_MALL_TYPE_HINTS = (
    "\u5546\u573a",
    "\u8d2d\u7269\u4e2d\u5fc3",
    "\u7efc\u5408\u5546\u573a",
    "\u767e\u8d27",
    "\u5546\u4e1a\u7efc\u5408\u4f53",
    "\u5965\u7279\u83b1\u65af",
    "\u8d2d\u7269\u516c\u56ed",
)

INTERNAL_POI_HINTS = (
    "\u505c\u8f66\u573a",
    "\u51fa\u53e3",
    "\u5165\u53e3",
    "\u5e97)",
    "\u5e97\uff09",
    "\u6253\u5361\u70b9",
    "\u8b66\u52a1\u5ba4",
    "\u670d\u52a1\u53f0",
    "\u5ba2\u670d\u4e2d\u5fc3",
    "\u536b\u751f\u95f4",
    "\u5395\u6240",
    "\u7535\u68af",
    "\u6276\u68af",
    "\u957f\u68af",
    "\u697c",
    "\u5c42",
    "B1",
    "B2",
    "B3",
    "F1",
    "F2",
    "F3",
    "L1",
    "L2",
    "L3",
)

INTERNAL_TYPE_HINTS = (
    "\u505c\u8f66\u573a",
    "\u505c\u8f66\u573a\u51fa\u5165\u53e3",
    "\u9910\u996e\u670d\u52a1",
    "\u5feb\u9910\u5385",
    "\u4e2d\u9910\u5385",
    "\u4e13\u5356\u5e97",
    "\u670d\u9970\u978b\u5305",
    "\u949f\u8868\u5e97",
    "\u4f53\u80b2\u4f11\u95f2\u670d\u52a1",
    "\u653f\u5e9c\u673a\u6784",
    "\u516c\u5b89\u8b66\u5bdf",
    "\u98ce\u666f\u540d\u80dc",
    "\u89c2\u666f\u70b9",
)


@dataclass
class MallRecord:
    mall_id: int
    name: str
    province: str
    city: str


@dataclass
class PoiCandidate:
    mall_id: int
    mall_name: str
    provider: str
    query: str
    rank: int
    poi_id: str
    poi_name: str
    address: str
    province: str
    city: str
    district: str
    poi_type: str
    raw_lon: float | None
    raw_lat: float | None
    raw_coord_type: str
    wgs84_lon: float | None
    wgs84_lat: float | None
    in_shanghai: bool
    name_similarity: float
    provider_score: float
    identity_score: float
    flags: str
    raw_json: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Shanghai mall identity with official POI APIs.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--city", default=SHANGHAI)
    parser.add_argument("--province", default="", help="Optional province filter. If set, records are loaded by province.")
    parser.add_argument("--target-region", default="", help="POI API region and bbox target. Defaults to province, then city.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Experiment root; timestamped run folder is created inside.")
    parser.add_argument("--run-id", default="", help="Run folder name. Default: current time YYYYMMDD_HHMMSS.")
    parser.add_argument("--no-timestamp", action="store_true", help="Use output-dir exactly.")
    parser.add_argument("--flat-timestamp", action="store_true", help="Append timestamp to output-dir name instead of creating a nested timestamp folder.")
    parser.add_argument("--resume", action="store_true", help="Resume the newest timestamped run under output-dir.")
    parser.add_argument("--ids", default="", help="Comma-separated mall ids.")
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N records after city/id filtering.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--api-key-config", type=Path, default=DEFAULT_API_KEY_CONFIG)
    parser.add_argument("--amap-key", default=os.getenv("AMAP_WEB_KEY", ""))
    parser.add_argument("--baidu-ak", default=os.getenv("BAIDU_MAP_AK", ""))
    parser.add_argument("--tencent-key", default=os.getenv("TENCENT_MAP_KEY", ""))
    parser.add_argument("--clip-scores-csv", type=Path, default=DEFAULT_CLIP_SCORES)
    parser.add_argument("--ignore-clip", action="store_true")
    parser.add_argument("--agreement-radius-m", type=float, default=150.0)
    parser.add_argument("--strong-name-threshold", type=float, default=0.52)
    parser.add_argument("--medium-name-threshold", type=float, default=0.34)
    parser.add_argument("--image-mall-threshold", type=float, default=0.50)
    parser.add_argument("--delay", type=float, default=0.15)
    parser.add_argument("--rate-limit-retries", type=int, default=2)
    parser.add_argument("--rate-limit-sleep", type=float, default=1.2)
    parser.add_argument("--disable-provider-after-rate-limit", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--max-queries", type=int, default=4)
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


def current_run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def is_timestamped(path: Path) -> bool:
    return bool(TIMESTAMP_PATTERN.match(path.name))


def latest_run(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = [
        path
        for path in base.iterdir()
        if path.is_dir() and is_timestamped(path) and (path / "data" / "official_poi_identity_summary.csv").exists()
    ]
    return sorted(candidates, key=lambda item: item.name, reverse=True)[0] if candidates else None


def latest_clip_scores(base: Path) -> Path | None:
    if not base.exists():
        return None
    candidates = [
        path / "data" / "gitr_sclip_poi_review_scores.csv"
        for path in base.iterdir()
        if path.is_dir() and is_timestamped(path) and (path / "data" / "gitr_sclip_poi_review_scores.csv").exists()
    ]
    return sorted(candidates, key=lambda path: path.parent.parent.name, reverse=True)[0] if candidates else None


def resolve_output_dir(args: argparse.Namespace) -> None:
    base = args.output_dir
    if args.no_timestamp or is_timestamped(base):
        actual = base
        run_id = args.run_id or (base.name if is_timestamped(base) else "no_timestamp")
    elif args.resume:
        previous = latest_run(base)
        if previous:
            actual = previous
            run_id = previous.name
        else:
            run_id = args.run_id or current_run_id()
            actual = base / run_id
    else:
        run_id = args.run_id or current_run_id()
        if args.flat_timestamp:
            actual = base.parent / f"{base.name}_{run_id}"
        else:
            actual = base / run_id
    args.output_base_dir = base.parent if args.flat_timestamp else base
    args.output_dir = actual
    args.run_id = run_id


def resolve_clip_scores(args: argparse.Namespace) -> None:
    if args.ignore_clip:
        return
    if args.clip_scores_csv.exists():
        return
    if args.clip_scores_csv == DEFAULT_CLIP_SCORES:
        latest = latest_clip_scores(DEFAULT_CLIP_ROOT)
        if latest:
            args.clip_scores_csv = latest


def ensure_dirs(output_dir: Path) -> dict[str, Path]:
    paths = {
        "root": output_dir,
        "data": output_dir / "data",
        "reports": output_dir / "reports",
        "logs": output_dir / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(paths: dict[str, Path]) -> logging.Logger:
    logger = logging.getLogger("official_poi_identity_validation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    file_handler = logging.FileHandler(paths["logs"] / "official_poi_identity_validation.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def redact_params(params: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in params.items():
        if key.lower() in SENSITIVE_PARAM_KEYS:
            redacted[key] = "***"
        else:
            redacted[key] = value
    return redacted


def first_config_value(config: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = config.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def apply_api_key_config(args: argparse.Namespace, logger: logging.Logger | None = None) -> None:
    sources = {
        "amap": "argument_or_environment" if args.amap_key else "none",
        "tencent": "argument_or_environment" if args.tencent_key else "none",
        "baidu": "argument_or_environment" if args.baidu_ak else "none",
    }
    config_has = {"amap": False, "tencent": False, "baidu": False}
    config_path = args.api_key_config
    if not config_path.exists():
        args.api_key_sources = sources
        args.api_key_config_has = config_has
        return
    try:
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001 - keep batch runnable with env/CLI keys.
        if logger:
            logger.warning("Failed to read API key config %s: %s", config_path, repr(exc))
        args.api_key_sources = sources
        args.api_key_config_has = config_has
        return
    if not isinstance(config, dict):
        if logger:
            logger.warning("API key config is not a JSON object: %s", config_path)
        args.api_key_sources = sources
        args.api_key_config_has = config_has
        return
    amap_config = first_config_value(
        config,
        ("AMAP_WEB_SERVICE_KEY", "AMAP_REST_KEY", "AMAP_WEB_KEY", "amap_web_key", "amapKey", "amap_key"),
    )
    tencent_config = first_config_value(
        config,
        ("TENCENT_MAP_KEY", "tencent_map_key", "tencentKey", "tencent_key"),
    )
    baidu_config = first_config_value(config, ("BAIDU_MAP_AK", "baidu_map_ak", "baiduAk", "baidu_ak"))
    config_has = {"amap": bool(amap_config), "tencent": bool(tencent_config), "baidu": bool(baidu_config)}
    if not args.amap_key and amap_config:
        args.amap_key = amap_config
        sources["amap"] = "config"
    if not args.tencent_key and tencent_config:
        args.tencent_key = tencent_config
        sources["tencent"] = "config"
    if not args.baidu_ak and baidu_config:
        args.baidu_ak = baidu_config
        sources["baidu"] = "config"
    args.api_key_sources = sources
    args.api_key_config_has = config_has


def load_records(db_path: Path, province: str, city: str, ids: set[int], offset: int, limit: int) -> list[MallRecord]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if province:
            rows = conn.execute("select id, name, province, city from malls where province = ? order by id", (province,)).fetchall()
        else:
            rows = conn.execute("select id, name, province, city from malls where city = ? order by id", (city,)).fetchall()
    records = [
        MallRecord(
            int(row["id"]),
            str(row["name"] or "").strip(),
            str(row["province"] or "").strip(),
            str(row["city"] or "").strip(),
        )
        for row in rows
    ]
    if ids:
        records = [record for record in records if record.mall_id in ids]
    if offset > 0:
        records = records[offset:]
    if limit > 0:
        records = records[:limit]
    return records


def normalize_text(text: str) -> str:
    value = (text or "").lower()
    value = re.sub(r"[\s\u3000·•,，.。:：;；!！?？'\"“”‘’()\[\]{}（）【】<>《》/\-|_+&＆]", "", value)
    for token in GENERIC_TOKENS:
        value = value.replace(token.lower(), "")
    return value


def compact_text(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", "", (text or "").lower())


def core_name(name: str) -> str:
    value = re.sub(r"[（(].*?[）)]", "", name or "")
    value = value.replace(SHANGHAI, "").replace(SHANGHAI_SHORT, "")
    return re.sub(r"\s+", "", value).strip()


def char_bigrams(text: str) -> set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    if len(text) == 1:
        return {text}
    return {text[index : index + 2] for index in range(len(text) - 1)}


def name_similarity(name: str, candidate_text: str) -> float:
    n1 = normalize_text(name)
    n2 = normalize_text(candidate_text)
    if not n1 or not n2:
        return 0.0
    if n1 in n2 or n2 in n1:
        return min(1.0, 0.72 + 0.28 * min(len(n1), len(n2)) / max(len(n1), len(n2), 1))
    b1 = char_bigrams(n1)
    b2 = char_bigrams(n2)
    if not b1 or not b2:
        return 0.0
    dice = 2.0 * len(b1 & b2) / (len(b1) + len(b2))
    chars = len(set(n1) & set(n2)) / max(len(set(n1)), 1)
    return max(0.0, min(1.0, 0.78 * dice + 0.22 * chars))


def stable_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        value = re.sub(r"\s+", " ", (value or "").strip())
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def make_queries(name: str, max_queries: int) -> list[str]:
    stripped = re.sub(r"[（(].*?[）)]", "", name).strip()
    without_shanghai = stripped.replace(SHANGHAI, "").replace(SHANGHAI_SHORT, "").strip()
    queries = [
        name,
        stripped,
        without_shanghai,
        f"{without_shanghai} \u8d2d\u7269\u4e2d\u5fc3",
        f"{without_shanghai} \u5546\u573a",
        f"{without_shanghai} \u5e7f\u573a",
        f"{without_shanghai} mall",
    ]
    return stable_unique(queries)[:max_queries]


def has_any(text: str, hints: tuple[str, ...]) -> bool:
    lower = (text or "").lower()
    return any(hint.lower() in lower for hint in hints)


def target_region(args: argparse.Namespace) -> str:
    return str(getattr(args, "target_region", "") or getattr(args, "province", "") or getattr(args, "city", "") or SHANGHAI).strip()


def target_region_short(args: argparse.Namespace) -> str:
    region = target_region(args)
    return region.removesuffix("\u7701").removesuffix("\u5e02")


def in_target_region(args: argparse.Namespace, lon: float | None, lat: float | None) -> bool:
    if lon is None or lat is None:
        return False
    region = target_region(args)
    bbox = TARGET_BBOXES.get(region) or TARGET_BBOXES.get(target_region_short(args))
    if bbox is None:
        return True
    west, south, east, north = bbox
    return west <= lon <= east and south <= lat <= north


def in_shanghai(lon: float | None, lat: float | None) -> bool:
    if lon is None or lat is None:
        return False
    west, south, east, north = SHANGHAI_BBOX
    return west <= lon <= east and south <= lat <= north


def out_of_china(lon: float, lat: float) -> bool:
    return not (73.66 < lon < 135.05 and 3.86 < lat < 53.55)


def transform_lat(x: float, y: float) -> float:
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def transform_lon(x: float, y: float) -> float:
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def gcj02_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    if out_of_china(lon, lat):
        return lon, lat
    a = 6378245.0
    ee = 0.00669342162296594323
    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlon = transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrt_magic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrt_magic * math.cos(radlat) * math.pi)
    mg_lat = lat + dlat
    mg_lon = lon + dlon
    return lon * 2 - mg_lon, lat * 2 - mg_lat


def bd09_to_gcj02(lon: float, lat: float) -> tuple[float, float]:
    x = lon - 0.0065
    y = lat - 0.006
    z = math.sqrt(x * x + y * y) - 0.00002 * math.sin(y * math.pi)
    theta = math.atan2(y, x) - 0.000003 * math.cos(x * math.pi)
    return z * math.cos(theta), z * math.sin(theta)


def bd09_to_wgs84(lon: float, lat: float) -> tuple[float, float]:
    gcj_lon, gcj_lat = bd09_to_gcj02(lon, lat)
    return gcj02_to_wgs84(gcj_lon, gcj_lat)


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius = 6371008.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def safe_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class OfficialPoiClient:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger) -> None:
        self.args = args
        self.logger = logger
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "PV-official-poi-identity-validation/1.0"})
        self.disabled_providers: set[str] = set()
        self.rate_limit_failures: Counter[str] = Counter()

    def _get(self, url: str, params: dict[str, Any], provider: str) -> dict[str, Any] | None:
        if self.args.delay > 0:
            time.sleep(self.args.delay)
        try:
            response = self.session.get(url, params=params, timeout=self.args.timeout)
            response.raise_for_status()
            return response.json()
        except Exception as exc:  # noqa: BLE001 - keep batch running.
            self.logger.warning("%s request failed: %s | params=%s", provider, repr(exc), redact_params(params))
            return None

    def amap_search(self, record: MallRecord, query: str) -> list[PoiCandidate]:
        if not self.args.amap_key:
            return []
        params = {
            "key": self.args.amap_key,
            "keywords": query,
            "city": record.city or target_region(self.args),
            "citylimit": "true",
            "offset": self.args.max_results,
            "page": 1,
            "extensions": "all",
            "output": "json",
        }
        payload = self._get(AMAP_TEXT_URL, params, "amap")
        if not payload or str(payload.get("status")) != "1":
            self.logger.warning("amap returned non-ok status for %s: %s", record.mall_id, payload)
            return []
        out: list[PoiCandidate] = []
        for rank, item in enumerate(payload.get("pois") or [], 1):
            loc = str(item.get("location") or "")
            lon = lat = None
            if "," in loc:
                lon, lat = [safe_float(part) for part in loc.split(",", 1)]
            wgs_lon = wgs_lat = None
            if lon is not None and lat is not None:
                wgs_lon, wgs_lat = gcj02_to_wgs84(lon, lat)
            out.append(make_candidate(self.args, record, "amap", query, rank, item, lon, lat, "gcj02", wgs_lon, wgs_lat))
        return out

    def baidu_search(self, record: MallRecord, query: str) -> list[PoiCandidate]:
        if not self.args.baidu_ak:
            return []
        params = {
            "query": query,
            "region": record.city or target_region(self.args),
            "city_limit": "true",
            "output": "json",
            "ak": self.args.baidu_ak,
            "scope": 2,
            "page_size": self.args.max_results,
            "page_num": 0,
        }
        payload = self._get(BAIDU_PLACE_URL, params, "baidu")
        if not payload or int(payload.get("status", -1)) != 0:
            self.logger.warning("baidu returned non-ok status for %s: %s", record.mall_id, payload)
            return []
        out: list[PoiCandidate] = []
        for rank, item in enumerate(payload.get("results") or [], 1):
            loc = item.get("location") or {}
            lon = safe_float(loc.get("lng"))
            lat = safe_float(loc.get("lat"))
            wgs_lon = wgs_lat = None
            if lon is not None and lat is not None:
                wgs_lon, wgs_lat = bd09_to_wgs84(lon, lat)
            out.append(make_candidate(self.args, record, "baidu", query, rank, item, lon, lat, "bd09", wgs_lon, wgs_lat))
        return out

    def tencent_search(self, record: MallRecord, query: str) -> list[PoiCandidate]:
        if not self.args.tencent_key or "tencent" in self.disabled_providers:
            return []
        params = {
            "keyword": query,
            "boundary": f"region({record.city or target_region_short(self.args)},0)",
            "page_size": self.args.max_results,
            "page_index": 1,
            "key": self.args.tencent_key,
        }
        payload = None
        for attempt in range(self.args.rate_limit_retries + 1):
            payload = self._get(TENCENT_PLACE_URL, params, "tencent")
            if payload and int(payload.get("status", -1)) == 120 and attempt < self.args.rate_limit_retries:
                self.logger.warning(
                    "tencent hit QPS limit for id=%s query=%s; sleep %.1fs then retry (%s/%s)",
                    record.mall_id,
                    query,
                    self.args.rate_limit_sleep,
                    attempt + 1,
                    self.args.rate_limit_retries,
                )
                time.sleep(self.args.rate_limit_sleep)
                continue
            break
        if not payload or int(payload.get("status", -1)) != 0:
            self.logger.warning("tencent returned non-ok status for %s: %s", record.mall_id, payload)
            if payload and int(payload.get("status", -1)) == 120:
                self.rate_limit_failures["tencent"] += 1
                if self.rate_limit_failures["tencent"] >= self.args.disable_provider_after_rate_limit:
                    self.disabled_providers.add("tencent")
                    self.logger.warning(
                        "tencent disabled for this run after %s consecutive rate-limit failures. "
                        "Check Tencent quota/QPS allocation or run AMap-only.",
                        self.rate_limit_failures["tencent"],
                    )
            return []
        self.rate_limit_failures["tencent"] = 0
        out: list[PoiCandidate] = []
        for rank, item in enumerate(payload.get("data") or [], 1):
            loc = item.get("location") or {}
            lon = safe_float(loc.get("lng"))
            lat = safe_float(loc.get("lat"))
            wgs_lon = wgs_lat = None
            if lon is not None and lat is not None:
                wgs_lon, wgs_lat = gcj02_to_wgs84(lon, lat)
            out.append(make_candidate(self.args, record, "tencent", query, rank, item, lon, lat, "gcj02", wgs_lon, wgs_lat))
        return out


def extract_poi_fields(provider: str, raw: dict[str, Any]) -> dict[str, str]:
    if provider == "amap":
        return {
            "poi_id": str(raw.get("id") or ""),
            "poi_name": str(raw.get("name") or ""),
            "address": str(raw.get("address") or ""),
            "province": str(raw.get("pname") or ""),
            "city": str(raw.get("cityname") or ""),
            "district": str(raw.get("adname") or ""),
            "poi_type": str(raw.get("type") or ""),
        }
    if provider == "baidu":
        detail = raw.get("detail_info") or {}
        return {
            "poi_id": str(raw.get("uid") or ""),
            "poi_name": str(raw.get("name") or ""),
            "address": str(raw.get("address") or ""),
            "province": str(raw.get("province") or ""),
            "city": str(raw.get("city") or ""),
            "district": str(raw.get("area") or ""),
            "poi_type": str(raw.get("tag") or detail.get("tag") or ""),
        }
    return {
        "poi_id": str(raw.get("id") or ""),
        "poi_name": str(raw.get("title") or ""),
        "address": str(raw.get("address") or ""),
        "province": str(raw.get("province") or ""),
        "city": str(raw.get("city") or ""),
        "district": str(raw.get("district") or ""),
        "poi_type": str(raw.get("category") or ""),
    }


def score_candidate(
    args: argparse.Namespace,
    record: MallRecord,
    fields: dict[str, str],
    provider_score: float,
    wgs_lon: float | None,
    wgs_lat: float | None,
) -> tuple[float, float, str]:
    compare = " ".join([fields["poi_name"], fields["address"], fields["poi_type"], fields["city"], fields["district"]])
    sim = name_similarity(record.name, compare)
    flags: list[str] = []
    score = provider_score
    score += sim * 45.0
    if in_target_region(args, wgs_lon, wgs_lat):
        score += 15.0
    else:
        score -= 45.0
        flags.append("outside_target_bbox")

    core = normalize_text(core_name(record.name))
    mall_norm = normalize_text(record.name)
    name_norm = normalize_text(fields["poi_name"])
    mall_compact = compact_text(record.name)
    core_compact = compact_text(core_name(record.name))
    name_compact = compact_text(fields["poi_name"])
    type_text = fields["poi_type"]
    name_type_text = " ".join([fields["poi_name"], type_text])
    candidate_norm = normalize_text(compare)
    exact_name_match = bool(
        (name_norm and name_norm in {mall_norm, core})
        or (name_compact and name_compact in {mall_compact, core_compact})
    )
    main_mall_type = has_any(type_text, MAIN_MALL_TYPE_HINTS)
    internal_poi = has_any(fields["poi_name"], INTERNAL_POI_HINTS) or has_any(type_text, INTERNAL_TYPE_HINTS)

    if exact_name_match:
        score += 30.0
    elif core and core in candidate_norm:
        score += 18.0
    elif sim < 0.25:
        score -= 18.0
        flags.append("weak_name_match")

    if main_mall_type:
        score += 22.0
        flags.append("main_mall_type")
    elif has_any(name_type_text, MALL_HINTS) or has_any(type_text, POSITIVE_TYPE_HINTS):
        score += 10.0
    else:
        flags.append("no_mall_hint")

    if internal_poi and not (exact_name_match and main_mall_type):
        score -= 36.0
        flags.append("internal_sub_poi")

    if has_any(name_type_text, BAD_HINTS):
        score -= 28.0
        flags.append("bad_place_hint")

    if len(normalize_text(fields["poi_name"])) <= 2:
        score -= 8.0
        flags.append("generic_candidate")

    return round(score, 3), sim, "|".join(dict.fromkeys(flags))


def make_candidate(
    args: argparse.Namespace,
    record: MallRecord,
    provider: str,
    query: str,
    rank: int,
    raw: dict[str, Any],
    raw_lon: float | None,
    raw_lat: float | None,
    raw_coord_type: str,
    wgs_lon: float | None,
    wgs_lat: float | None,
) -> PoiCandidate:
    fields = extract_poi_fields(provider, raw)
    provider_score = max(0.0, 25.0 - (rank - 1) * 2.0)
    identity_score, sim, flags = score_candidate(args, record, fields, provider_score, wgs_lon, wgs_lat)
    return PoiCandidate(
        mall_id=record.mall_id,
        mall_name=record.name,
        provider=provider,
        query=query,
        rank=rank,
        poi_id=fields["poi_id"],
        poi_name=fields["poi_name"],
        address=fields["address"],
        province=fields["province"],
        city=fields["city"],
        district=fields["district"],
        poi_type=fields["poi_type"],
        raw_lon=raw_lon,
        raw_lat=raw_lat,
        raw_coord_type=raw_coord_type,
        wgs84_lon=wgs_lon,
        wgs84_lat=wgs_lat,
        in_shanghai=in_target_region(args, wgs_lon, wgs_lat),
        name_similarity=sim,
        provider_score=provider_score,
        identity_score=identity_score,
        flags=flags,
        raw_json=json.dumps(raw, ensure_ascii=False),
    )


CANDIDATE_FIELDS = [
    "mall_id",
    "mall_name",
    "provider",
    "query",
    "rank",
    "poi_id",
    "poi_name",
    "address",
    "province",
    "city",
    "district",
    "poi_type",
    "raw_lon",
    "raw_lat",
    "raw_coord_type",
    "wgs84_lon",
    "wgs84_lat",
    "in_shanghai",
    "name_similarity",
    "provider_score",
    "identity_score",
    "flags",
    "raw_json",
]

SUMMARY_FIELDS = [
    "mall_id",
    "name",
    "selected_lon_wgs84",
    "selected_lat_wgs84",
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
    "identity_status",
    "identity_reasons",
    "clip_decision",
    "clip_candidate_mall_delta",
    "clip_old_mall_delta",
]

RESOLVED_FIELDS = [
    "run_id",
    "mall_id",
    "name",
    "center_lon",
    "center_lat",
    "confidence_level",
    "can_enter_1km_analysis",
    "identity_status",
    "center_type",
    "center_source",
    "center_poi_name",
    "center_address",
    "center_poi_type",
    "name_evidence",
    "official_coord_evidence",
    "image_evidence",
    "identity_reasons",
    "review_status",
    "review_required",
]


def candidate_to_row(candidate: PoiCandidate) -> dict[str, str]:
    row = candidate.__dict__.copy()
    for key in ["raw_lon", "raw_lat", "wgs84_lon", "wgs84_lat", "name_similarity", "provider_score", "identity_score"]:
        value = row.get(key)
        row[key] = "" if value is None else f"{float(value):.8f}" if key.endswith(("lon", "lat")) else f"{float(value):.6f}"
    row["in_shanghai"] = "1" if candidate.in_shanghai else "0"
    return {field: str(row.get(field, "")) for field in CANDIDATE_FIELDS}


def append_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def read_completed_ids(path: Path) -> set[int]:
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            try:
                done.add(int(row.get("mall_id") or 0))
            except ValueError:
                continue
    return done


def load_clip_scores(path: Path, ignore_clip: bool) -> dict[str, dict[str, str]]:
    if ignore_clip or not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row.get("mall_id", ""): row for row in csv.DictReader(fh) if row.get("mall_id")}


def collect_candidates(
    client: OfficialPoiClient,
    record: MallRecord,
    max_queries: int,
    provider_names: list[str],
    strong_name_threshold: float,
) -> list[PoiCandidate]:
    out: list[PoiCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for query in make_queries(record.name, max_queries):
        provider_results = []
        provider_results.extend(client.amap_search(record, query))
        provider_results.extend(client.baidu_search(record, query))
        provider_results.extend(client.tencent_search(record, query))
        for candidate in provider_results:
            key = (candidate.provider, normalize_text(candidate.poi_name), f"{candidate.wgs84_lon:.6f},{candidate.wgs84_lat:.6f}")
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)
        if len(provider_names) == 1 and has_strong_single_provider_candidate(out, strong_name_threshold):
            break
        if enough_provider_agreement(out, radius_m=client.args.agreement_radius_m):
            break
    return out


def has_strong_single_provider_candidate(candidates: list[PoiCandidate], threshold: float) -> bool:
    return any(
        item.in_shanghai
        and item.wgs84_lon is not None
        and item.wgs84_lat is not None
        and item.name_similarity >= threshold
        and "bad_place_hint" not in item.flags
        for item in candidates
    )


def format_provider_hits(candidates: list[PoiCandidate], provider_names: list[str]) -> str:
    if not provider_names:
        return "none"
    counts = Counter(candidate.provider for candidate in candidates)
    return ",".join(f"{provider}:{counts.get(provider, 0)}" for provider in provider_names)


def enough_provider_agreement(candidates: list[PoiCandidate], radius_m: float) -> bool:
    good = [item for item in candidates if item.in_shanghai and item.wgs84_lon is not None and item.wgs84_lat is not None and item.name_similarity >= 0.45 and "bad_place_hint" not in item.flags]
    for candidate in good:
        providers = {candidate.provider}
        for other in good:
            if candidate.provider == other.provider:
                continue
            distance = haversine_m(candidate.wgs84_lon, candidate.wgs84_lat, other.wgs84_lon, other.wgs84_lat)
            if distance <= radius_m:
                providers.add(other.provider)
        if len(providers) >= 2:
            return True
    return False


def summarize_record(record: MallRecord, candidates: list[PoiCandidate], clip_row: dict[str, str] | None, args: argparse.Namespace) -> dict[str, str]:
    viable = [
        item
        for item in candidates
        if item.in_shanghai
        and item.wgs84_lon is not None
        and item.wgs84_lat is not None
        and "bad_place_hint" not in item.flags
    ]
    if not viable:
        return empty_summary(record, clip_row, "reject_candidate", "no_viable_official_poi")

    selected = max(viable, key=lambda item: (item.identity_score, item.name_similarity, -item.rank))
    agreeing: list[PoiCandidate] = []
    distances: list[float] = []
    for item in viable:
        if item.wgs84_lon is None or item.wgs84_lat is None or selected.wgs84_lon is None or selected.wgs84_lat is None:
            continue
        distance = haversine_m(selected.wgs84_lon, selected.wgs84_lat, item.wgs84_lon, item.wgs84_lat)
        if distance <= args.agreement_radius_m:
            agreeing.append(item)
            if item.provider != selected.provider:
                distances.append(distance)

    official_provider_count = len({item.provider for item in viable})
    agreement_provider_count = len({item.provider for item in agreeing})
    nearest_provider_distance = min(distances) if distances else ""
    selected_lon = sum(item.wgs84_lon for item in agreeing if item.wgs84_lon is not None) / max(len(agreeing), 1)
    selected_lat = sum(item.wgs84_lat for item in agreeing if item.wgs84_lat is not None) / max(len(agreeing), 1)

    if selected.name_similarity >= args.strong_name_threshold:
        name_evidence = "strong"
    elif selected.name_similarity >= args.medium_name_threshold:
        name_evidence = "medium"
    else:
        name_evidence = "weak"

    if agreement_provider_count >= 2:
        official_coord_evidence = "multi_provider_agree"
    elif official_provider_count >= 1 and name_evidence == "strong" and selected.identity_score >= 62:
        official_coord_evidence = "single_provider_strong"
    else:
        official_coord_evidence = "weak_or_disputed"

    image_evidence, clip_decision, clip_candidate_delta, clip_old_delta = classify_image_evidence(clip_row, args.image_mall_threshold)
    status, reasons = decide_identity_status(name_evidence, official_coord_evidence, image_evidence, selected, agreement_provider_count)

    return {
        "mall_id": str(record.mall_id),
        "name": record.name,
        "selected_lon_wgs84": f"{selected_lon:.8f}",
        "selected_lat_wgs84": f"{selected_lat:.8f}",
        "selected_provider": selected.provider,
        "selected_poi_name": selected.poi_name,
        "selected_address": selected.address,
        "selected_poi_type": selected.poi_type,
        "selected_name_similarity": f"{selected.name_similarity:.6f}",
        "official_provider_count": str(official_provider_count),
        "agreement_provider_count": str(agreement_provider_count),
        "agreement_radius_m": f"{args.agreement_radius_m:.1f}",
        "nearest_provider_distance_m": "" if nearest_provider_distance == "" else f"{nearest_provider_distance:.2f}",
        "best_identity_score": f"{selected.identity_score:.3f}",
        "name_evidence": name_evidence,
        "official_coord_evidence": official_coord_evidence,
        "image_evidence": image_evidence,
        "identity_status": status,
        "identity_reasons": "|".join(reasons),
        "clip_decision": clip_decision,
        "clip_candidate_mall_delta": clip_candidate_delta,
        "clip_old_mall_delta": clip_old_delta,
    }


def classify_image_evidence(clip_row: dict[str, str] | None, image_threshold: float) -> tuple[str, str, str, str]:
    if not clip_row:
        return "not_available", "", "", ""
    decision = clip_row.get("clip_decision", "")
    candidate_delta = clip_row.get("candidate_mall_delta", "")
    old_delta = clip_row.get("old_mall_delta", "")
    try:
        candidate_value = float(candidate_delta) if candidate_delta else None
    except ValueError:
        candidate_value = None
    if decision == "candidate_center_likely_mall" and candidate_value is not None and candidate_value >= image_threshold:
        return "candidate_image_mall_like", decision, candidate_delta, old_delta
    if decision == "old_image_more_mall_like":
        return "old_image_more_mall_like", decision, candidate_delta, old_delta
    if decision in {"both_unclear_or_non_mall", "candidate_unclear_no_old_image"}:
        return "not_mall_like_or_unclear", decision, candidate_delta, old_delta
    return "manual_review_or_not_available", decision, candidate_delta, old_delta


def decide_identity_status(
    name_evidence: str,
    coord_evidence: str,
    image_evidence: str,
    selected: PoiCandidate,
    agreement_provider_count: int,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if "bad_place_hint" in selected.flags or not selected.in_shanghai:
        return "reject_candidate", ["bad_place_or_outside_target_region"]
    if name_evidence == "weak":
        reasons.append("name_match_weak")
    else:
        reasons.append(f"name_match_{name_evidence}")
    reasons.append(coord_evidence)
    reasons.append(image_evidence)

    if name_evidence == "strong" and coord_evidence == "multi_provider_agree" and image_evidence in {"candidate_image_mall_like", "not_available"}:
        return "auto_pass", reasons
    if name_evidence == "strong" and coord_evidence == "single_provider_strong" and image_evidence == "candidate_image_mall_like":
        return "auto_pass", reasons
    if image_evidence == "old_image_more_mall_like" and agreement_provider_count < 2:
        return "manual_review", reasons + ["old_image_conflicts_with_official_candidate"]
    if name_evidence == "weak" or coord_evidence == "weak_or_disputed":
        return "manual_review", reasons
    return "manual_review", reasons


def empty_summary(record: MallRecord, clip_row: dict[str, str] | None, status: str, reason: str) -> dict[str, str]:
    image_evidence, clip_decision, clip_candidate_delta, clip_old_delta = classify_image_evidence(clip_row, 0.50)
    return {
        "mall_id": str(record.mall_id),
        "name": record.name,
        "selected_lon_wgs84": "",
        "selected_lat_wgs84": "",
        "selected_provider": "",
        "selected_poi_name": "",
        "selected_address": "",
        "selected_poi_type": "",
        "selected_name_similarity": "",
        "official_provider_count": "0",
        "agreement_provider_count": "0",
        "agreement_radius_m": "",
        "nearest_provider_distance_m": "",
        "best_identity_score": "",
        "name_evidence": "none",
        "official_coord_evidence": "none",
        "image_evidence": image_evidence,
        "identity_status": status,
        "identity_reasons": reason,
        "clip_decision": clip_decision,
        "clip_candidate_mall_delta": clip_candidate_delta,
        "clip_old_mall_delta": clip_old_delta,
    }


def confidence_level_from_summary(summary: dict[str, str]) -> tuple[str, str, str]:
    identity_status = summary.get("identity_status", "")
    name_evidence = summary.get("name_evidence", "")
    coord_evidence = summary.get("official_coord_evidence", "")
    image_evidence = summary.get("image_evidence", "")

    if identity_status == "auto_pass":
        if (
            name_evidence == "strong"
            and coord_evidence == "multi_provider_agree"
            and image_evidence in {"candidate_image_mall_like", "not_available"}
        ):
            return "A", "1", "approved_auto"
        if (
            name_evidence == "strong"
            and coord_evidence in {"single_provider_strong", "multi_provider_agree"}
            and image_evidence == "candidate_image_mall_like"
        ):
            return "B", "1", "approved_auto_sample_check"
        return "B", "1", "approved_auto_sample_check"
    if identity_status == "manual_review":
        return "C", "0", "needs_manual_review"
    return "D", "0", "blocked_unresolved"


def summary_to_resolved_row(summary: dict[str, str], run_id: str) -> dict[str, str]:
    level, can_enter, review_status = confidence_level_from_summary(summary)
    review_required = "0" if level == "A" else "1"
    if level in {"C", "D"}:
        review_required = "1"
    return {
        "run_id": run_id,
        "mall_id": summary.get("mall_id", ""),
        "name": summary.get("name", ""),
        "center_lon": summary.get("selected_lon_wgs84", "") if can_enter == "1" else "",
        "center_lat": summary.get("selected_lat_wgs84", "") if can_enter == "1" else "",
        "confidence_level": level,
        "can_enter_1km_analysis": can_enter,
        "identity_status": summary.get("identity_status", ""),
        "center_type": "official_poi_point" if can_enter == "1" else "",
        "center_source": summary.get("selected_provider", ""),
        "center_poi_name": summary.get("selected_poi_name", ""),
        "center_address": summary.get("selected_address", ""),
        "center_poi_type": summary.get("selected_poi_type", ""),
        "name_evidence": summary.get("name_evidence", ""),
        "official_coord_evidence": summary.get("official_coord_evidence", ""),
        "image_evidence": summary.get("image_evidence", ""),
        "identity_reasons": summary.get("identity_reasons", ""),
        "review_status": review_status,
        "review_required": review_required,
    }


def write_config(args: argparse.Namespace, paths: dict[str, Path], provider_names: list[str], total: int) -> None:
    lines = [
        "# 官方 POI 商场身份一致性验证配置",
        "",
        f"- 运行 ID：`{args.run_id}`",
        f"- 数据库：`{args.db}`",
        f"- 输出根目录：`{args.output_base_dir}`",
        f"- 实际输出目录：`{args.output_dir}`",
        f"- 记录数：{total}",
        f"- 启用 provider：{', '.join(provider_names) if provider_names else '无'}",
        f"- 多源一致半径：{args.agreement_radius_m} m",
        f"- 强名称匹配阈值：{args.strong_name_threshold}",
        f"- 中名称匹配阈值：{args.medium_name_threshold}",
        f"- 图像商场阈值：{args.image_mall_threshold}",
        f"- CLIP 评分表：`{args.clip_scores_csv}`",
        f"- 中心点分级输出：`{paths['data'] / 'mall_center_resolved.csv'}`",
        "",
        "## 证据规则",
        "",
        "- `name_evidence`：官方 POI 名称/地址/类型和数据库商场名的文本一致性。",
        "- `official_coord_evidence`：多 provider 坐标是否在指定半径内一致。",
        "- `image_evidence`：可选合并 Git-RSCLIP 视觉复核结果。",
        "- `identity_status=auto_pass` 仍代表高置信自动通过，不代表人工真值。",
        "- `confidence_level=A/B` 才允许进入 1km 厂房/光伏潜力分析。",
        "- `confidence_level=C/D` 必须人工复核或暂不进入后续流程。",
    ]
    (paths["reports"] / "实验配置.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(paths: dict[str, Path], summary_path: Path, candidate_path: Path, elapsed: float) -> None:
    rows: list[dict[str, str]] = []
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))
    status_counts = Counter(row.get("identity_status", "") for row in rows)
    coord_counts = Counter(row.get("official_coord_evidence", "") for row in rows)
    image_counts = Counter(row.get("image_evidence", "") for row in rows)
    resolved_path = paths["data"] / "mall_center_resolved.csv"
    resolved_rows: list[dict[str, str]] = []
    if resolved_path.exists():
        with resolved_path.open("r", encoding="utf-8-sig", newline="") as fh:
            resolved_rows = list(csv.DictReader(fh))
    level_counts = Counter(row.get("confidence_level", "") for row in resolved_rows)
    lines = [
        "# 官方 POI 商场身份一致性验证报告",
        "",
        f"- 汇总结果：`{summary_path}`",
        f"- 候选明细：`{candidate_path}`",
        f"- 已解析中心点：`{resolved_path}`",
        f"- 处理记录数：{len(rows)}",
        f"- 总耗时：{elapsed / 60:.2f} 分钟",
        "",
        "## 中心点置信等级统计",
        "",
    ]
    for key in ["A", "B", "C", "D"]:
        lines.append(f"- {key}: {level_counts.get(key, 0)}")
    lines.extend([
        "",
        "## identity_status 统计",
        "",
    ])
    for key, count in status_counts.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## 坐标证据统计", ""])
    for key, count in coord_counts.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## 图像证据统计", ""])
    for key, count in image_counts.most_common():
        lines.append(f"- {key}: {count}")
    lines.extend(["", "## 自动通过样本 Top 50", ""])
    auto_rows = [row for row in rows if row.get("identity_status") == "auto_pass"]
    for row in sorted(auto_rows, key=lambda item: float(item.get("best_identity_score") or 0), reverse=True)[:50]:
        lines.append(
            f"- {row.get('mall_id')} {row.get('name')} | {row.get('selected_poi_name')} | "
            f"{row.get('selected_lon_wgs84')},{row.get('selected_lat_wgs84')} | "
            f"{row.get('official_coord_evidence')} | {row.get('image_evidence')}"
        )
    lines.extend(["", "## 需要人工复核样本 Top 80", ""])
    review_rows = [row for row in rows if row.get("identity_status") == "manual_review"]
    for row in sorted(review_rows, key=lambda item: float(item.get("best_identity_score") or 0), reverse=True)[:80]:
        lines.append(
            f"- {row.get('mall_id')} {row.get('name')} | {row.get('selected_poi_name')} | "
            f"{row.get('identity_reasons')} | {row.get('selected_lon_wgs84')},{row.get('selected_lat_wgs84')}"
        )
    (paths["reports"] / "官方POI商场身份一致性验证报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    resolve_output_dir(args)
    resolve_clip_scores(args)
    paths = ensure_dirs(args.output_dir)
    logger = setup_logger(paths)
    apply_api_key_config(args, logger)
    provider_names = []
    if args.amap_key:
        provider_names.append("amap")
    if args.baidu_ak:
        provider_names.append("baidu")
    if args.tencent_key:
        provider_names.append("tencent")

    ids = {int(part.strip()) for part in args.ids.split(",") if part.strip().isdigit()}
    records = load_records(args.db, args.province, args.city, ids, args.offset, args.limit)
    summary_path = paths["data"] / "official_poi_identity_summary.csv"
    candidate_path = paths["data"] / "official_poi_candidates.csv"
    resolved_path = paths["data"] / "mall_center_resolved.csv"
    completed = read_completed_ids(summary_path) if args.resume else set()
    if not args.resume:
        for path in [summary_path, candidate_path, resolved_path]:
            if path.exists():
                path.unlink()
    pending = [record for record in records if record.mall_id not in completed]
    clip_scores = load_clip_scores(args.clip_scores_csv, args.ignore_clip)

    logger.info("Run id=%s", args.run_id)
    logger.info("Output dir=%s", args.output_dir)
    logger.info("API key config=%s | exists=%s", args.api_key_config, args.api_key_config.exists())
    logger.info(
        "API key sources: amap=%s(config_has=%s), tencent=%s(config_has=%s), baidu=%s(config_has=%s)",
        args.api_key_sources["amap"],
        args.api_key_config_has["amap"],
        args.api_key_sources["tencent"],
        args.api_key_config_has["tencent"],
        args.api_key_sources["baidu"],
        args.api_key_config_has["baidu"],
    )
    logger.info("Providers enabled=%s", ",".join(provider_names) if provider_names else "none")
    for provider in ["amap", "baidu", "tencent"]:
        logger.info(
            "Provider %s: %s | service=%s | endpoint=%s",
            provider,
            "enabled" if provider in provider_names else "disabled",
            PROVIDER_DISPLAY[provider],
            PROVIDER_ENDPOINTS[provider],
        )
    logger.info("Records=%s completed=%s pending=%s", len(records), len(completed), len(pending))
    write_config(args, paths, provider_names, len(records))
    if not provider_names:
        logger.warning(
            "No official API keys provided. Fill %s or set AMAP_WEB_KEY, BAIDU_MAP_AK, or TENCENT_MAP_KEY.",
            args.api_key_config,
        )

    client = OfficialPoiClient(args, logger)
    start = time.time()
    for index, record in enumerate(pending, 1):
        item_start = time.time()
        candidates = (
            collect_candidates(client, record, args.max_queries, provider_names, args.strong_name_threshold)
            if provider_names
            else []
        )
        append_rows(candidate_path, CANDIDATE_FIELDS, [candidate_to_row(candidate) for candidate in candidates])
        summary = summarize_record(record, candidates, clip_scores.get(str(record.mall_id)), args)
        append_rows(summary_path, SUMMARY_FIELDS, [summary])
        resolved = summary_to_resolved_row(summary, args.run_id)
        append_rows(resolved_path, RESOLVED_FIELDS, [resolved])

        elapsed = time.time() - start
        avg = elapsed / max(index, 1)
        eta = avg * (len(pending) - index)
        if index == 1 or index % args.log_every == 0 or index == len(pending):
            logger.info(
                "[%s/%s] id=%s %s | status=%s level=%s | api_hits=%s | official_providers=%s agree=%s | selected=%s | %.1fs/item | ETA %.1fmin",
                len(completed) + index,
                len(records),
                record.mall_id,
                record.name,
                summary["identity_status"],
                resolved["confidence_level"],
                format_provider_hits(candidates, provider_names),
                summary["official_provider_count"],
                summary["agreement_provider_count"],
                summary["selected_poi_name"],
                time.time() - item_start,
                eta / 60,
            )

    write_report(paths, summary_path, candidate_path, time.time() - start)
    logger.info("Done. summary=%s", summary_path)
    logger.info("Resolved centers=%s", resolved_path)
    logger.info("Report=%s", paths["reports"] / "官方POI商场身份一致性验证报告.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
