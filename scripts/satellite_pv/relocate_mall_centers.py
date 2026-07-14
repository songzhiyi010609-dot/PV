#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Relocate Shanghai mall centers from mall names only.

The script intentionally ignores the original `location` column because many
records point to roads, stations, nearby projects, or even the wrong city.
It writes new non-destructive CSVs with provider candidates, selected center
coordinates, confidence labels, and review flags.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = PROJECT_ROOT / "malls_new.db"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments" / "20260708_relocate_mall_centers"
SHANGHAI = "\u4e0a\u6d77\u5e02"
SHANGHAI_BBOX = (120.85, 30.67, 122.15, 31.87)
ARCGIS_SUGGEST_URL = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/suggest"
ARCGIS_FIND_URL = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"
BAIDU_SUGGEST_URL = "https://map.baidu.com/su"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"

MALL_HINTS = (
    "mall",
    "\u8d2d\u7269\u4e2d\u5fc3",
    "\u5546\u573a",
    "\u5546\u4e1a",
    "\u5546\u4e1a\u4f53",
    "\u5546\u4e1a\u4e2d\u5fc3",
    "\u5e7f\u573a",
    "\u767e\u8d27",
    "\u5370\u8c61\u57ce",
    "\u4e07\u8c61",
    "\u9f99\u6e56",
    "\u5929\u8857",
    "\u5927\u60a6\u57ce",
    "\u73af\u5b87\u57ce",
    "\u5408\u751f\u6c47",
    "\u4e45\u5149",
    "\u6c47\u91d1",
    "\u6e2f\u6c47",
    "\u6765\u798f\u58eb",
    "\u5927\u878d\u57ce",
    "\u5357\u4e30\u57ce",
    "\u5965\u7279\u83b1\u65af",
)

BAD_PLACE_HINTS = (
    "\u5730\u94c1\u7ad9",
    "\u5730\u94c1",
    "\u516c\u4ea4\u7ad9",
    "\u516c\u4ea4",
    "\u706b\u8f66\u7ad9",
    "\u9ad8\u94c1\u7ad9",
    "\u6c7d\u8f66\u7ad9",
    "\u8f66\u7ad9",
    "\u8f66\u5e93",
    "\u505c\u8f66\u573a",
    "\u9053\u8def",
    "\u8def\u53e3",
    "\u4ea4\u53c9\u53e3",
    "\u516c\u56ed",
    "\u5c0f\u533a",
    "\u516c\u5bd3",
    "\u9152\u5e97",
    "\u5b66\u6821",
    "\u533b\u9662",
    "\u519c\u573a",
    "\u5efa\u8bbe\u4e2d",
    "\u65bd\u5de5",
    "\u5730\u5757",
    "\u9879\u76ee\u90e8",
)

GENERIC_NAME_TOKENS = (
    "\u4e0a\u6d77",
    SHANGHAI,
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


@dataclass
class MallRecord:
    mall_id: int
    name: str
    city: str
    old_location: str


@dataclass
class Candidate:
    mall_id: int
    mall_name: str
    query: str
    provider: str
    rank: int
    place_name: str
    match_addr: str
    address: str
    city: str
    region: str
    addr_type: str
    place_type: str
    lon: float | None
    lat: float | None
    provider_score: float
    name_similarity: float
    center_score: float
    confidence: str
    flags: str
    raw: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Relocate mall center coordinates from mall names only.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite database path.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Experiment output directory.")
    parser.add_argument("--city", default=SHANGHAI, help="City filter in the database.")
    parser.add_argument("--ids", default="", help="Comma-separated mall ids for a focused run.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of malls, mainly for smoke tests.")
    parser.add_argument("--delay", type=float, default=0.12, help="Delay between provider calls.")
    parser.add_argument("--nominatim-delay", type=float, default=1.1, help="Minimum delay between Nominatim calls.")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds.")
    parser.add_argument("--max-queries", type=int, default=8, help="Max query variants per mall.")
    parser.add_argument("--max-suggestions", type=int, default=5, help="Max suggestions per query.")
    parser.add_argument("--osm-first", action="store_true", help="Try one OSM/Nominatim name query before ArcGIS fallback.")
    parser.add_argument("--osm-extra-queries", type=int, default=3, help="Extra OSM queries after the first one when no high-confidence hit is found.")
    parser.add_argument("--skip-arcgis", action="store_true", help="Only use OSM/Nominatim and Baidu suggestion text; mainly for diagnostics.")
    parser.add_argument("--resume", action="store_true", help="Skip ids already present in the selected output CSV.")
    parser.add_argument("--no-baidu-suggest", action="store_true", help="Do not use Baidu suggestion text as extra query hints.")
    return parser.parse_args()


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "data": root / "data",
        "reports": root / "reports",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def load_records(db_path: Path, city: str, ids: set[int], limit: int) -> list[MallRecord]:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        sql = "select id, name, city, coalesce(location, '') as location from malls where city = ? order by id"
        rows = conn.execute(sql, (city,)).fetchall()

    records = [
        MallRecord(
            mall_id=int(row["id"]),
            name=str(row["name"] or "").strip(),
            city=str(row["city"] or "").strip(),
            old_location=str(row["location"] or "").strip(),
        )
        for row in rows
    ]
    if ids:
        records = [row for row in records if row.mall_id in ids]
    if limit > 0:
        records = records[:limit]
    return records


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[\s\u3000·•,，.。:：;；!！?？'\"“”‘’()\[\]{}（）【】<>《》/\-|_+&＆]", "", text)
    for token in GENERIC_NAME_TOKENS:
        text = text.replace(token.lower(), "")
    return text


def core_name(name: str) -> str:
    value = re.sub(r"[（(].*?[）)]", "", name or "")
    value = value.replace(SHANGHAI, "").replace("\u4e0a\u6d77", "")
    value = re.sub(r"\s+", "", value)
    return value.strip()


def char_bigrams(text: str) -> set[str]:
    text = normalize_text(text)
    if not text:
        return set()
    if len(text) == 1:
        return {text}
    return {text[i : i + 2] for i in range(len(text) - 1)}


def name_similarity(name: str, candidate_text: str) -> float:
    n1 = normalize_text(name)
    n2 = normalize_text(candidate_text)
    if not n1 or not n2:
        return 0.0
    if n1 in n2 or n2 in n1:
        short = min(len(n1), len(n2))
        long = max(len(n1), len(n2))
        return min(1.0, 0.72 + 0.28 * short / max(long, 1))
    b1 = char_bigrams(n1)
    b2 = char_bigrams(n2)
    if not b1 or not b2:
        return 0.0
    overlap = len(b1 & b2)
    dice = 2.0 * overlap / (len(b1) + len(b2))
    chars = len(set(n1) & set(n2)) / max(len(set(n1)), 1)
    return max(0.0, min(1.0, 0.78 * dice + 0.22 * chars))


def is_in_shanghai(lon: float | None, lat: float | None) -> bool:
    if lon is None or lat is None or math.isnan(lon) or math.isnan(lat):
        return False
    west, south, east, north = SHANGHAI_BBOX
    return west <= lon <= east and south <= lat <= north


def has_any(text: str, hints: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(hint.lower() in low for hint in hints)


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


def make_query_variants(name: str) -> list[str]:
    stripped = re.sub(r"[（(].*?[）)]", "", name).strip()
    without_shanghai = stripped.replace(SHANGHAI, "").replace("\u4e0a\u6d77", "").strip()
    variants = [
        f"{name} {SHANGHAI}",
        f"{stripped} {SHANGHAI}",
        f"{without_shanghai} {SHANGHAI}",
        f"{without_shanghai} \u8d2d\u7269\u4e2d\u5fc3 {SHANGHAI}",
        f"{without_shanghai} \u5546\u573a {SHANGHAI}",
        f"{without_shanghai} \u5e7f\u573a {SHANGHAI}",
        f"{without_shanghai} mall {SHANGHAI}",
        name,
        stripped,
    ]
    return stable_unique(variants)


def json_fingerprint(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(data.encode("utf-8")).hexdigest()[:12]


class GeoClient:
    def __init__(self, timeout: float, delay: float, nominatim_delay: float, log_path: Path) -> None:
        self.timeout = timeout
        self.delay = delay
        self.nominatim_delay = nominatim_delay
        self.last_nominatim_at = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "PV-Mall-Relocator/1.0 (+local research; contact: local)",
                "Accept": "application/json,text/javascript,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh,en;q=0.7",
            }
        )
        self.log_path = log_path

    def _sleep(self) -> None:
        if self.delay > 0:
            time.sleep(self.delay)

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        self._sleep()
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            text = resp.text.strip()
            if text.startswith("window.baidu.sug(") and text.endswith(");"):
                text = text[len("window.baidu.sug(") : -2]
            return json.loads(text)
        except Exception as exc:  # noqa: BLE001 - keep batch running and log provider failures.
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"url": url, "params": params, "error": repr(exc)}, ensure_ascii=False) + "\n")
            return None

    def _get_nominatim_json(self, params: dict[str, Any]) -> Any:
        wait = self.nominatim_delay - (time.monotonic() - self.last_nominatim_at)
        if wait > 0:
            time.sleep(wait)
        self.last_nominatim_at = time.monotonic()
        try:
            resp = self.session.get(NOMINATIM_SEARCH_URL, params=params, timeout=self.timeout)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:  # noqa: BLE001 - keep batch running and log provider failures.
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"url": NOMINATIM_SEARCH_URL, "params": params, "error": repr(exc)}, ensure_ascii=False) + "\n")
            return None

    def baidu_suggestions(self, name: str, max_items: int) -> list[str]:
        params = {
            "wd": name,
            "cid": 289,
            "type": 0,
            "newmap": 1,
            "pc_ver": 2,
        }
        payload = self._get_json(BAIDU_SUGGEST_URL, params)
        if not isinstance(payload, dict):
            return []
        values: list[str] = []
        for raw in payload.get("s", []) or []:
            if not isinstance(raw, str):
                continue
            parts = [part.strip() for part in raw.split("$") if part.strip()]
            for part in parts:
                if not part or part == SHANGHAI or part.endswith("\u533a"):
                    continue
                if len(part) < 2:
                    continue
                values.append(f"{part} {SHANGHAI}")
        return stable_unique(values)[:max_items]

    def arcgis_suggest(self, query: str, max_items: int) -> list[dict[str, Any]]:
        params = {
            "f": "json",
            "text": query,
            "countryCode": "CHN",
            "searchExtent": ",".join(str(x) for x in SHANGHAI_BBOX),
            "maxSuggestions": max_items,
        }
        payload = self._get_json(ARCGIS_SUGGEST_URL, params)
        if not isinstance(payload, dict):
            return []
        suggestions = payload.get("suggestions") or []
        return [item for item in suggestions if isinstance(item, dict)]

    def arcgis_find(self, query: str, magic_key: str | None, max_items: int) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "f": "json",
            "SingleLine": query,
            "sourceCountry": "CHN",
            "searchExtent": ",".join(str(x) for x in SHANGHAI_BBOX),
            "maxLocations": max_items,
            "outFields": "Match_addr,Addr_type,Type,PlaceName,Place_addr,City,Region,Country,SourceID",
        }
        if magic_key:
            params["magicKey"] = magic_key
        payload = self._get_json(ARCGIS_FIND_URL, params)
        if not isinstance(payload, dict):
            return []
        candidates = payload.get("candidates") or []
        return [item for item in candidates if isinstance(item, dict)]

    def nominatim_search(self, query: str, max_items: int) -> list[dict[str, Any]]:
        params = {
            "format": "jsonv2",
            "q": query,
            "countrycodes": "cn",
            "bounded": 1,
            "viewbox": "120.85,31.87,122.15,30.67",
            "limit": max_items,
            "addressdetails": 1,
            "namedetails": 1,
            "extratags": 1,
        }
        payload = self._get_nominatim_json(params)
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]


def candidate_from_arcgis(
    record: MallRecord,
    query: str,
    raw: dict[str, Any],
    rank: int,
    source: str,
) -> Candidate:
    attrs = raw.get("attributes") or {}
    loc = raw.get("location") or {}
    lon = loc.get("x")
    lat = loc.get("y")
    try:
        lon_f = float(lon) if lon is not None else None
        lat_f = float(lat) if lat is not None else None
    except (TypeError, ValueError):
        lon_f = None
        lat_f = None

    place_name = str(attrs.get("PlaceName") or "")
    match_addr = str(attrs.get("Match_addr") or raw.get("address") or "")
    address = str(attrs.get("Place_addr") or "")
    city = str(attrs.get("City") or "")
    region = str(attrs.get("Region") or "")
    addr_type = str(attrs.get("Addr_type") or "")
    place_type = str(attrs.get("Type") or "")
    provider_score = float(raw.get("score") or 0.0)
    compare_text = " ".join([place_name, match_addr, address])
    sim = name_similarity(record.name, compare_text)
    center_score, confidence, flags = score_candidate(
        record.name,
        provider_score,
        sim,
        place_name,
        match_addr,
        address,
        city,
        region,
        addr_type,
        place_type,
        lon_f,
        lat_f,
    )
    return Candidate(
        mall_id=record.mall_id,
        mall_name=record.name,
        query=query,
        provider=source,
        rank=rank,
        place_name=place_name,
        match_addr=match_addr,
        address=address,
        city=city,
        region=region,
        addr_type=addr_type,
        place_type=place_type,
        lon=lon_f,
        lat=lat_f,
        provider_score=provider_score,
        name_similarity=sim,
        center_score=center_score,
        confidence=confidence,
        flags=flags,
        raw=json.dumps(raw, ensure_ascii=False),
    )


def candidate_from_nominatim(
    record: MallRecord,
    query: str,
    raw: dict[str, Any],
    rank: int,
) -> Candidate:
    address = raw.get("address") if isinstance(raw.get("address"), dict) else {}
    namedetails = raw.get("namedetails") if isinstance(raw.get("namedetails"), dict) else {}
    display_name = str(raw.get("display_name") or "")
    place_name = str(
        raw.get("name")
        or namedetails.get("name")
        or namedetails.get("name:zh")
        or namedetails.get("name:en")
        or display_name.split(",")[0].strip()
    )
    city = str(address.get("city") or address.get("state") or "")
    region = str(address.get("state") or "")
    osm_class = str(raw.get("class") or "")
    osm_type = str(raw.get("type") or "")
    try:
        lon = float(raw.get("lon")) if raw.get("lon") is not None else None
        lat = float(raw.get("lat")) if raw.get("lat") is not None else None
    except (TypeError, ValueError):
        lon = None
        lat = None

    importance = raw.get("importance")
    try:
        importance_f = float(importance or 0.0)
    except (TypeError, ValueError):
        importance_f = 0.0

    poi_like_types = {
        "mall",
        "commercial",
        "retail",
        "department_store",
        "supermarket",
        "shopping_centre",
        "marketplace",
    }
    addr_type = "POI" if osm_type in poi_like_types or osm_class in {"shop", "amenity", "building"} else "OSM"
    provider_score = 45.0 + min(max(importance_f, 0.0), 1.0) * 45.0
    if osm_type in poi_like_types:
        provider_score += 18.0
    if osm_type in {"station", "subway_entrance", "bus_stop", "parking"}:
        provider_score -= 22.0
    provider_score = max(0.0, min(provider_score, 100.0))

    compare_text = " ".join([place_name, display_name])
    sim = name_similarity(record.name, compare_text)
    center_score, confidence, flags = score_candidate(
        record.name,
        provider_score,
        sim,
        place_name,
        display_name,
        display_name,
        city,
        region,
        addr_type,
        f"osm:{osm_class}/{osm_type}",
        lon,
        lat,
    )
    return Candidate(
        mall_id=record.mall_id,
        mall_name=record.name,
        query=query,
        provider="osm_nominatim",
        rank=rank,
        place_name=place_name,
        match_addr=display_name,
        address=display_name,
        city=city,
        region=region,
        addr_type=addr_type,
        place_type=f"osm:{osm_class}/{osm_type}",
        lon=lon,
        lat=lat,
        provider_score=provider_score,
        name_similarity=sim,
        center_score=center_score,
        confidence=confidence,
        flags=flags,
        raw=json.dumps(raw, ensure_ascii=False),
    )


def score_candidate(
    mall_name: str,
    provider_score: float,
    sim: float,
    place_name: str,
    match_addr: str,
    address: str,
    city: str,
    region: str,
    addr_type: str,
    place_type: str,
    lon: float | None,
    lat: float | None,
) -> tuple[float, str, str]:
    flags: list[str] = []
    all_text = " ".join([place_name, match_addr, address, city, region, addr_type, place_type])
    mall_core = core_name(mall_name)
    candidate_norm = normalize_text(all_text)
    mall_core_norm = normalize_text(mall_core)

    score = 0.0
    score += min(max(provider_score, 0.0), 100.0) / 100.0 * 28.0
    score += sim * 42.0

    if is_in_shanghai(lon, lat):
        score += 12.0
    else:
        score -= 45.0
        flags.append("outside_shanghai_bbox")

    if "POI" in addr_type.upper():
        score += 15.0
    else:
        if addr_type:
            flags.append(f"addr_type_{addr_type}")
        if addr_type.lower() in {"locality", "streetname", "pointaddress", "streetaddress"}:
            score -= 18.0

    city_region_text = " ".join([city, region, address, match_addr])
    if city and city != SHANGHAI and "\u4e0a\u6d77" not in city_region_text:
        score -= 25.0
        flags.append("city_not_shanghai")

    if has_any(all_text, MALL_HINTS):
        score += 8.0
    else:
        flags.append("no_mall_hint")

    if mall_core_norm and mall_core_norm in candidate_norm:
        score += 15.0
    elif sim < 0.28:
        score -= 16.0
        flags.append("weak_name_match")

    if has_any(all_text, BAD_PLACE_HINTS):
        score -= 20.0
        flags.append("bad_place_hint")

    if not place_name and not match_addr:
        score -= 15.0
        flags.append("blank_name")

    if len(normalize_text(place_name or match_addr)) <= 2:
        score -= 10.0
        flags.append("generic_candidate")

    if score >= 78 and sim >= 0.38 and is_in_shanghai(lon, lat) and "POI" in addr_type.upper():
        confidence = "high"
    elif score >= 58 and sim >= 0.24 and is_in_shanghai(lon, lat):
        confidence = "medium"
    elif score >= 38 and is_in_shanghai(lon, lat):
        confidence = "low"
    else:
        confidence = "unresolved"

    return round(score, 3), confidence, "|".join(stable_unique(flags))


def collect_candidates(
    client: GeoClient,
    record: MallRecord,
    use_baidu_suggest: bool,
    max_queries: int,
    max_suggestions: int,
    osm_first: bool,
    osm_extra_queries: int,
    skip_arcgis: bool,
) -> list[Candidate]:
    base_queries = make_query_variants(record.name)
    queries = base_queries[:max_queries]

    candidates: list[Candidate] = []
    seen_candidate_keys: set[tuple[str, str, str]] = set()
    rank = 1

    def add_osm_results(osm_query_list: list[str]) -> None:
        nonlocal rank
        for osm_query in osm_query_list:
            raw_candidates = client.nominatim_search(osm_query, max_suggestions)
            for raw in raw_candidates:
                cand = candidate_from_nominatim(record, osm_query, raw, rank)
                key = (
                    normalize_text(cand.place_name or cand.match_addr),
                    f"{cand.lon:.7f}" if cand.lon is not None else "",
                    f"{cand.lat:.7f}" if cand.lat is not None else "",
                )
                if key in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(key)
                candidates.append(cand)
                rank += 1
            if any(c.confidence == "high" and c.center_score >= 88 for c in candidates):
                break

    if osm_first:
        add_osm_results(stable_unique([core_name(record.name) + f" {SHANGHAI}", record.name])[:1])

    if skip_arcgis:
        if osm_extra_queries > 0 and not any(c.confidence == "high" and c.center_score >= 88 for c in candidates):
            add_osm_results(stable_unique([core_name(record.name) + f" {SHANGHAI}", record.name, *queries[:2]])[:4])
        return candidates

    if use_baidu_suggest and not any(c.confidence == "high" and c.center_score >= 88 for c in candidates):
        baidu_queries = client.baidu_suggestions(record.name, max_suggestions)
        queries = stable_unique(baidu_queries + base_queries)[:max_queries]

    for query in queries:
        if any(c.confidence == "high" and c.center_score >= 88 for c in candidates):
            break
        suggestions = client.arcgis_suggest(query, max_suggestions)
        suggestion_queries: list[tuple[str, str | None, str]] = []
        for item in suggestions:
            text = str(item.get("text") or "").strip()
            magic_key = str(item.get("magicKey") or "").strip() or None
            if text:
                suggestion_queries.append((text, magic_key, "arcgis_suggest_find"))

        if not suggestion_queries:
            suggestion_queries.append((query, None, "arcgis_direct_find"))
        else:
            suggestion_queries.append((query, None, "arcgis_direct_find"))

        for single_line, magic_key, source in suggestion_queries[: max_suggestions + 1]:
            raw_candidates = client.arcgis_find(single_line, magic_key, max_suggestions)
            for raw in raw_candidates:
                cand = candidate_from_arcgis(record, query, raw, rank, source)
                key = (
                    normalize_text(cand.place_name or cand.match_addr),
                    f"{cand.lon:.7f}" if cand.lon is not None else "",
                    f"{cand.lat:.7f}" if cand.lat is not None else "",
                )
                if key in seen_candidate_keys:
                    continue
                seen_candidate_keys.add(key)
                candidates.append(cand)
                rank += 1
        if any(c.confidence == "high" and c.center_score >= 88 for c in candidates):
            break

    if not any(c.confidence == "high" and c.center_score >= 88 for c in candidates):
        osm_queries = stable_unique([core_name(record.name) + f" {SHANGHAI}", record.name, *queries[:3]])
        add_osm_results(osm_queries[: max(0, osm_extra_queries) + 1])
    return candidates


def pick_best(record: MallRecord, candidates: list[Candidate]) -> dict[str, Any]:
    acceptable = [cand for cand in candidates if cand.confidence != "unresolved"]
    pool = acceptable or candidates
    if not pool:
        return {
            "mall_id": record.mall_id,
            "name": record.name,
            "selected_lon": "",
            "selected_lat": "",
            "confidence": "unresolved",
            "center_score": 0.0,
            "provider": "",
            "place_name": "",
            "match_addr": "",
            "address": "",
            "addr_type": "",
            "place_type": "",
            "name_similarity": 0.0,
            "provider_score": 0.0,
            "flags": "no_candidate",
            "old_location_unused": record.old_location,
        }
    best = sorted(pool, key=lambda c: (c.center_score, c.name_similarity, c.provider_score), reverse=True)[0]
    usable_lon = "" if best.lon is None or best.confidence == "unresolved" else f"{best.lon:.8f}"
    usable_lat = "" if best.lat is None or best.confidence == "unresolved" else f"{best.lat:.8f}"
    return {
        "mall_id": best.mall_id,
        "name": best.mall_name,
        "selected_lon": usable_lon,
        "selected_lat": usable_lat,
        "confidence": best.confidence,
        "center_score": f"{best.center_score:.3f}",
        "provider": best.provider,
        "place_name": best.place_name,
        "match_addr": best.match_addr,
        "address": best.address,
        "addr_type": best.addr_type,
        "place_type": best.place_type,
        "name_similarity": f"{best.name_similarity:.4f}",
        "provider_score": f"{best.provider_score:.3f}",
        "flags": best.flags,
        "old_location_unused": record.old_location,
    }


CANDIDATE_FIELDS = [
    "mall_id",
    "mall_name",
    "query",
    "provider",
    "rank",
    "place_name",
    "match_addr",
    "address",
    "city",
    "region",
    "addr_type",
    "place_type",
    "lon",
    "lat",
    "provider_score",
    "name_similarity",
    "center_score",
    "confidence",
    "flags",
    "raw",
]

SELECTED_FIELDS = [
    "mall_id",
    "name",
    "selected_lon",
    "selected_lat",
    "confidence",
    "center_score",
    "provider",
    "place_name",
    "match_addr",
    "address",
    "addr_type",
    "place_type",
    "name_similarity",
    "provider_score",
    "flags",
    "old_location_unused",
]


def candidate_to_row(cand: Candidate) -> dict[str, Any]:
    row = cand.__dict__.copy()
    row["lon"] = "" if cand.lon is None else f"{cand.lon:.8f}"
    row["lat"] = "" if cand.lat is None else f"{cand.lat:.8f}"
    row["provider_score"] = f"{cand.provider_score:.3f}"
    row["name_similarity"] = f"{cand.name_similarity:.4f}"
    row["center_score"] = f"{cand.center_score:.3f}"
    return row


def append_rows(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
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
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                done.add(int(row.get("mall_id") or 0))
            except ValueError:
                continue
    return done


def write_report(
    report_path: Path,
    selected_path: Path,
    candidates_path: Path,
    total_requested: int,
    processed: int,
) -> None:
    rows: list[dict[str, str]] = []
    if selected_path.exists():
        with selected_path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = list(csv.DictReader(fh))

    counts: dict[str, int] = {}
    for row in rows:
        counts[row.get("confidence", "unknown")] = counts.get(row.get("confidence", "unknown"), 0) + 1
    review_rows = [
        row
        for row in rows
        if row.get("confidence") in {"low", "unresolved"}
        or "bad_place_hint" in row.get("flags", "")
        or "weak_name_match" in row.get("flags", "")
    ]

    lines = [
        "# 商场中心重定位报告",
        "",
        f"- 数据库：`{DEFAULT_DB}`",
        "- 查询原则：只使用 `name + 上海市` 及商场名派生词，不使用原 `location` 字段参与定位。",
        f"- 请求记录数：{total_requested}",
        f"- 本次处理数：{processed}",
        f"- 已输出选择结果：{len(rows)}",
        f"- 候选明细：`{candidates_path}`",
        f"- 中心点结果：`{selected_path}`",
        "",
        "## 置信度统计",
        "",
    ]
    for key in ["high", "medium", "low", "unresolved"]:
        lines.append(f"- {key}: {counts.get(key, 0)}")
    lines.extend(
        [
            "",
            "## 重要说明",
            "",
            "- `selected_lon` / `selected_lat` 为 WGS84 经纬度候选中心点。",
            "- `confidence=high` 表示命中了上海范围内的 POI，且名称相似度和商场语义较强。",
            "- `confidence=medium` 可用于后续遥感图裁剪的候选中心，但建议抽查。",
            "- `confidence=low` 和 `unresolved` 不建议直接用于生成最终数据集，应进入人工复核或接入高德/百度/腾讯官方 POI API 进一步校验。",
            "- 原数据库、原图片、原索引均未修改。",
            "",
            "## 需要优先复核的样本",
            "",
        ]
    )
    for row in review_rows[:80]:
        lines.append(
            f"- {row.get('mall_id')} {row.get('name')} | {row.get('confidence')} | "
            f"{row.get('selected_lon')},{row.get('selected_lat')} | {row.get('place_name') or row.get('match_addr')} | {row.get('flags')}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    paths = ensure_dirs(args.output_dir)
    candidates_path = paths["data"] / "mall_center_candidates_name_only.csv"
    selected_path = paths["data"] / "mall_center_coordinates_v1.csv"
    report_path = paths["reports"] / "\u5546\u573a\u4e2d\u5fc3\u91cd\u5b9a\u4f4d\u62a5\u544a.md"
    log_path = paths["logs"] / "provider_errors.jsonl"

    ids = {int(x.strip()) for x in args.ids.split(",") if x.strip().isdigit()}
    records = load_records(args.db, args.city, ids, args.limit)
    if not records:
        print("No records matched.", file=sys.stderr)
        return 2

    completed = read_completed_ids(selected_path) if args.resume else set()
    pending = [record for record in records if record.mall_id not in completed]
    client = GeoClient(timeout=args.timeout, delay=args.delay, nominatim_delay=args.nominatim_delay, log_path=log_path)

    print(f"records={len(records)} pending={len(pending)} output={args.output_dir}")
    processed = 0
    for idx, record in enumerate(pending, 1):
        candidates = collect_candidates(
            client=client,
            record=record,
            use_baidu_suggest=not args.no_baidu_suggest,
            max_queries=args.max_queries,
            max_suggestions=args.max_suggestions,
            osm_first=args.osm_first,
            osm_extra_queries=args.osm_extra_queries,
            skip_arcgis=args.skip_arcgis,
        )
        selected = pick_best(record, candidates)
        append_rows(candidates_path, CANDIDATE_FIELDS, [candidate_to_row(c) for c in candidates])
        append_rows(selected_path, SELECTED_FIELDS, [selected])
        processed += 1
        if idx == 1 or idx % 10 == 0:
            print(
                f"[{idx}/{len(pending)}] id={record.mall_id} {record.name} -> "
                f"{selected['confidence']} {selected['selected_lon']},{selected['selected_lat']} "
                f"{selected['place_name'] or selected['match_addr']}"
            )

    write_report(report_path, selected_path, candidates_path, len(records), processed)
    print(f"done processed={processed}")
    print(f"selected={selected_path}")
    print(f"candidates={candidates_path}")
    print(f"report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
