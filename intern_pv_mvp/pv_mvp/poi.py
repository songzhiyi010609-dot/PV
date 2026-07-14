from __future__ import annotations

from dataclasses import dataclass, asdict
import time
from typing import Any

import requests

from .io_utils import to_float


ARCGIS_GEOCODE_URL = "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates"


@dataclass(frozen=True)
class PoiResult:
    mall_id: str
    name: str
    city: str
    address: str
    lon: float | None
    lat: float | None
    poi_status: str
    poi_source: str
    poi_name: str
    poi_address: str
    poi_score: float | None
    poi_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_query(name: str, city: str, address: str) -> str:
    parts = [city.strip(), address.strip(), name.strip()]
    return " ".join(part for part in parts if part)


def geocode_with_arcgis(query: str, timeout: int = 20) -> dict[str, Any] | None:
    params = {
        "f": "json",
        "SingleLine": query,
        "maxLocations": 3,
        "outFields": "Match_addr,Addr_type,Score",
        "sourceCountry": "CHN",
        "langCode": "zh-CN",
    }
    response = requests.get(ARCGIS_GEOCODE_URL, params=params, timeout=timeout)
    response.raise_for_status()
    candidates = response.json().get("candidates") or []
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: float(item.get("score") or 0), reverse=True)
    return candidates[0]


def resolve_row(row: dict[str, object], *, timeout: int = 20, delay: float = 0.2) -> PoiResult:
    mall_id = str(row.get("mall_id") or row.get("id") or "").strip()
    name = str(row.get("name") or "").strip()
    city = str(row.get("city") or "").strip()
    address = str(row.get("address") or row.get("location") or "").strip()
    lat = to_float(row.get("lat") or row.get("latitude"))
    lon = to_float(row.get("lon") or row.get("longitude"))

    if lon is not None and lat is not None:
        return PoiResult(
            mall_id=mall_id,
            name=name,
            city=city,
            address=address,
            lon=lon,
            lat=lat,
            poi_status="ok",
            poi_source="input_coordinates",
            poi_name=name,
            poi_address=address,
            poi_score=None,
            poi_reason="used lat/lon from input csv",
        )

    query = build_query(name, city, address)
    if not query:
        return PoiResult(mall_id, name, city, address, None, None, "error", "none", "", "", None, "missing query and coordinates")

    try:
        candidate = geocode_with_arcgis(query, timeout=timeout)
        time.sleep(max(delay, 0.0))
    except Exception as exc:
        return PoiResult(mall_id, name, city, address, None, None, "error", "arcgis", "", "", None, f"geocode failed: {exc}")

    if not candidate:
        return PoiResult(mall_id, name, city, address, None, None, "not_found", "arcgis", "", "", None, "no geocode candidate")

    location = candidate.get("location") or {}
    lon = to_float(location.get("x"))
    lat = to_float(location.get("y"))
    score = to_float(candidate.get("score"))
    poi_address = str(candidate.get("address") or "")
    status = "ok" if lon is not None and lat is not None else "error"
    reason = "geocoded by ArcGIS World Geocoding"
    if score is not None and score < 75:
        reason += "; low score, needs manual review"
    return PoiResult(mall_id, name, city, address, lon, lat, status, "arcgis", poi_address, poi_address, score, reason)
