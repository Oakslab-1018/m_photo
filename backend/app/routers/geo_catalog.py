"""GeoJSON feature collections for the imported geographic catalog.

Read-only HTTP surface over
:class:`backend.app.services.geo_catalog_store.GeoCatalogStore`. The JSON
catalogs are never read here: the store (database) is the single source, and a
deployment that has not imported the catalog answers 503 instead of serving
stale/partial geography.

Contract (see ``docs/geo-catalog.md``)::

    GET /api/geo/catalog/countries.geojson
    GET /api/geo/catalog/admin1.geojson?country=CN
    GET /api/geo/catalog/cities.geojson?admin1_id=CN.02&limit=100&offset=0

Every emitted feature carries ``geometry.coordinates = [lng, lat]`` plus
numeric ``properties.lat``/``properties.lng``. Rows without coordinates are
never emitted (they are reported through ``meta.missing_coordinates``), and
``(0, 0)`` is rejected by the importer, the store and this serializer.
Unscoped city queries are refused unless they are bounded by ``country``,
``admin1_id`` or ``bbox``; free text ``q`` alone is never enough.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ..config import Settings
from ..deps import get_db_session, get_settings
from ..domain.geolocation.country_policy import (
    canonical_country_iso,
    is_country_member,
    normalize_iso,
)
from ..services.geo_catalog_store import GeoCatalogStore

router = APIRouter(prefix="/geo/catalog", tags=["geo-catalog"])

GEOJSON_MEDIA_TYPE = "application/geo+json"
DEFAULT_LIMIT = 500
MAX_LIMIT = 2000
_ADMIN1_ID_RE = re.compile(r"^[A-Z]{2}\.[A-Z0-9]+$")


def get_catalog_store(session: Session = Depends(get_db_session)) -> GeoCatalogStore:
    """Build the read-only store on the request-scoped session."""

    return GeoCatalogStore(session)


def _bad_request(detail: str) -> HTTPException:
    return HTTPException(status_code=422, detail=detail)


def _available_revisions(store: GeoCatalogStore, settings: Settings) -> dict[str, Any]:
    """Return the imported revisions or fail closed with a clear 503."""

    try:
        revisions = store.revisions()
    except SQLAlchemyError as exc:
        raise HTTPException(
            status_code=503,
            detail="地理目录数据库不可用，请确认迁移已执行且目录已导入。",
        ) from exc
    missing = [
        name for name in ("countries", "admin1", "cities") if not revisions.get(name)
    ]
    if missing:
        raise HTTPException(
            status_code=503,
            detail="地理目录尚未导入数据库（缺少国家/行政区/城市指纹），请先完成启动导入。",
        )
    return revisions


def _parse_bbox(raw: str) -> tuple[float, float, float, float]:
    parts = raw.split(",")
    if len(parts) != 4:
        raise _bad_request("bbox 必须为 minLng,minLat,maxLng,maxLat 四个数值。")
    try:
        values = [float(part) for part in parts]
    except (TypeError, ValueError) as exc:
        raise _bad_request("bbox 必须是数值。") from exc
    if not all(math.isfinite(value) for value in values):
        raise _bad_request("bbox 必须是有限数值。")
    min_lng, min_lat, max_lng, max_lat = values
    if min_lng > max_lng or min_lat > max_lat:
        raise _bad_request("bbox 顺序必须为 minLng,minLat,maxLng,maxLat，且不支持跨日期变更线。")
    if not (-180.0 <= min_lng <= 180.0 and -180.0 <= max_lng <= 180.0):
        raise _bad_request("bbox 经度必须在 [-180, 180] 范围内。")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise _bad_request("bbox 纬度必须在 [-90, 90] 范围内。")
    return min_lng, min_lat, max_lng, max_lat


def _iso_param(store: GeoCatalogStore, value: str, field: str) -> str:
    code = normalize_iso(value)
    if not code:
        raise _bad_request(f"{field} 必须是两位 ISO 代码。")
    if store.get_place("country", code) is None:
        raise _bad_request(f"未知的国家/来源代码：{code}。")
    return code


def _resolve_scope(
    store: GeoCatalogStore,
    *,
    level: str,
    country: str | None,
    source_iso: str | None,
    admin1_id: str | None,
) -> dict[str, Any]:
    """Normalize user-supplied IDs and enforce parent consistency (422)."""

    scope: dict[str, Any] = {}
    canonical: str | None = None
    raw_source: str | None = None

    if country is not None:
        code = _iso_param(store, country, "country")
        scope["country"] = code
        canonical = canonical_country_iso(code)
        scope["canonical_country"] = canonical
    if source_iso is not None:
        raw_source = _iso_param(store, source_iso, "source_iso")
        scope["source_iso"] = raw_source
    if admin1_id is not None:
        if level == "country":
            raise _bad_request("国家层级不支持 admin1_id 参数。")
        candidate = admin1_id.strip().upper()
        if not _ADMIN1_ID_RE.fullmatch(candidate):
            raise _bad_request("admin1_id 必须形如 ISO.CODE。")
        row = store.get_place("admin1", candidate)
        if row is None:
            raise _bad_request(f"未知的一级行政区 ID：{candidate}。")
        if raw_source is not None and row.source_iso != raw_source:
            raise _bad_request("admin1_id 与 source_iso 不属于同一来源范围。")
        if canonical is not None and canonical_country_iso(row.source_iso) != canonical:
            raise _bad_request("admin1_id 与 country 不属于同一国家。")
        raw_source = raw_source or row.source_iso
        scope["admin1_id"] = candidate
    if canonical is not None and raw_source is not None and not is_country_member(
        canonical, raw_source
    ):
        raise _bad_request("source_iso 不属于所选国家。")
    return scope


def _feature(place: Any) -> dict[str, Any]:
    """Serialize one GeoPlace; invalid/placeholder geometry is never emitted."""

    lat = place.lat
    lng = place.lng
    if (
        lat is None
        or lng is None
        or not math.isfinite(float(lat))
        or not math.isfinite(float(lng))
        or not (-90.0 <= float(lat) <= 90.0)
        or not (-180.0 <= float(lng) <= 180.0)
        or (float(lat) == 0.0 and float(lng) == 0.0)
    ):
        raise HTTPException(status_code=500, detail="目录坐标不合法，已拒绝输出该要素。")
    lat_value = float(lat)
    lng_value = float(lng)
    return {
        "type": "Feature",
        "id": place.place_id,
        "geometry": {"type": "Point", "coordinates": [lng_value, lat_value]},
        "properties": {
            "id": place.place_id,
            "level": place.level,
            "label": place.label,
            "name": place.name,
            "parent_id": place.parent_id,
            "country_iso": place.country_iso,
            "source_iso": place.source_iso,
            "admin1_id": place.admin1_id,
            "city_id": place.place_id if place.level == "city" else None,
            "coordinate_source": place.coordinate_source,
            "coordinate_quality": place.coordinate_quality,
            "coordinate_city_id": place.coordinate_city_id,
            "population": place.population,
            "lat": lat_value,
            "lng": lng_value,
        },
    }


def _feature_collection(
    *,
    store: GeoCatalogStore,
    settings: Settings,
    level: str,
    country: str | None,
    source_iso: str | None,
    admin1_id: str | None,
    q: str | None,
    bbox_raw: str | None,
    limit: int,
    offset: int,
) -> Response:
    revisions = _available_revisions(store, settings)
    bbox = _parse_bbox(bbox_raw) if bbox_raw is not None else None
    scope = _resolve_scope(
        store, level=level, country=country, source_iso=source_iso, admin1_id=admin1_id
    )
    if level == "city" and not (
        scope.get("country") or scope.get("admin1_id") or bbox is not None
    ):
        raise _bad_request(
            "城市查询必须用 country、admin1_id 或 bbox 限定范围，不能只按 q 检索。"
        )

    try:
        page = store.query_places(
            level,
            country=scope.get("country"),
            source_iso=scope.get("source_iso"),
            admin1_id=scope.get("admin1_id"),
            q=q,
            bbox=bbox,
            limit=limit,
            offset=offset,
            coordinates_only=True,
        )
    except ValueError as exc:
        raise _bad_request(f"查询参数无效：{exc}") from exc

    features = [_feature(place) for place in page["rows"]]
    returned = len(features)
    total = int(page["total"])
    payload = {
        "type": "FeatureCollection",
        "features": features,
        "meta": {
            "level": level,
            "scope": {**scope, "q": q, "bbox": list(bbox) if bbox is not None else None},
            "revisions": revisions,
            "total": total,
            "returned": returned,
            "missing_coordinates": int(page["missing_coordinates"]),
            "limit": limit,
            "offset": offset,
            "truncated": offset + returned < total,
        },
    }
    try:
        body = json.dumps(payload, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:  # defensive: never emit non-finite JSON
        raise HTTPException(
            status_code=500, detail="目录坐标包含非法数值，已拒绝输出。"
        ) from exc
    return Response(content=body, media_type=GEOJSON_MEDIA_TYPE)


@router.get("/countries.geojson")
def countries_geojson(
    country: str | None = Query(default=None, description="ISO 国家代码（可选）"),
    source_iso: str | None = Query(default=None, description="原始来源范围（可选）"),
    q: str | None = Query(default=None, description="名称/别名子串（Unicode 归一化）"),
    bbox: str | None = Query(default=None, description="minLng,minLat,maxLng,maxLat"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(get_settings),
    store: GeoCatalogStore = Depends(get_catalog_store),
) -> Response:
    """Country/territory points (canonical ``country_iso`` grouping preserved)."""

    return _feature_collection(
        store=store,
        settings=settings,
        level="country",
        country=country,
        source_iso=source_iso,
        admin1_id=None,
        q=q,
        bbox_raw=bbox,
        limit=limit,
        offset=offset,
    )


@router.get("/admin1.geojson")
def admin1_geojson(
    country: str | None = Query(default=None, description="ISO 国家代码（可选，CN 归并 HK/MO/TW）"),
    source_iso: str | None = Query(default=None, description="原始来源范围（可选）"),
    admin1_id: str | None = Query(default=None, description="一级行政区 ID，如 CN.02"),
    q: str | None = Query(default=None, description="名称/别名子串（Unicode 归一化）"),
    bbox: str | None = Query(default=None, description="minLng,minLat,maxLng,maxLat"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(get_settings),
    store: GeoCatalogStore = Depends(get_catalog_store),
) -> Response:
    """Admin1 directory points; rows without a matching city are reported, not plotted."""

    return _feature_collection(
        store=store,
        settings=settings,
        level="admin1",
        country=country,
        source_iso=source_iso,
        admin1_id=admin1_id,
        q=q,
        bbox_raw=bbox,
        limit=limit,
        offset=offset,
    )


@router.get("/cities.geojson")
def cities_geojson(
    country: str | None = Query(default=None, description="ISO 国家代码（CN 归并 HK/MO/TW）"),
    source_iso: str | None = Query(default=None, description="原始来源范围（可选）"),
    admin1_id: str | None = Query(default=None, description="一级行政区 ID，如 CN.02"),
    q: str | None = Query(default=None, description="名称/别名子串（Unicode 归一化）"),
    bbox: str | None = Query(default=None, description="minLng,minLat,maxLng,maxLat"),
    limit: int = Query(default=DEFAULT_LIMIT, ge=1, le=MAX_LIMIT),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(get_settings),
    store: GeoCatalogStore = Depends(get_catalog_store),
) -> Response:
    """City points; the query must be bounded by country, admin1_id or bbox."""

    return _feature_collection(
        store=store,
        settings=settings,
        level="city",
        country=country,
        source_iso=source_iso,
        admin1_id=admin1_id,
        q=q,
        bbox_raw=bbox,
        limit=limit,
        offset=offset,
    )


__all__ = ["router", "get_catalog_store"]
