"""Read-only database access to the imported geographic catalog.

This is the frozen stage API that later routers/adapters consume; it never
reads the JSON files and never writes. ``GeoCatalogStore`` wraps one
SQLAlchemy session (typically the request-scoped session from
:mod:`backend.app.deps`) and exposes:

* :meth:`GeoCatalogStore.get_place` - exact lookup by ``(level, place_id)``
* :meth:`GeoCatalogStore.get_places` - bounded batch lookup by stable IDs
* :meth:`GeoCatalogStore.revisions` - source fingerprints + import version
* :meth:`GeoCatalogStore.query_places` - bounded, scoped, paginated queries
* :meth:`GeoCatalogStore.admin1_rows` / :meth:`GeoCatalogStore.city_rows` -
  the existing ``Admin1`` / ``CityRow`` catalogs for the guided adapter
* :meth:`GeoCatalogStore.countries` - original country payloads

Scoping follows the existing CN policy: ``country='CN'`` groups the raw
``CN``/``HK``/``MO``/``TW`` source scopes, while a raw ``TW``/``HK``/``MO``
``country`` without ``source_iso`` is interpreted as that raw scope. Free
text search is Unicode-normalized (NFKC + casefold) and executed in Python
over already-scoped rows, so SQL stays fully parameterized. Ordering is
stable per scope; city rows order by population (desc) then numeric id.
"""
from __future__ import annotations

import unicodedata
from typing import Any, Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..domain.geolocation.country_policy import (
    canonical_country_iso,
    is_region_iso,
    normalize_iso,
)
from .admin1_catalog import Admin1
from .city_catalog import CityRow
from .geo_catalog_models import LEVELS, GeoCatalogSource, GeoPlace

#: Default page size and hard cap for ``query_places`` (the later GeoJSON
#: router must never expose an unbounded city query).
QUERY_DEFAULT_LIMIT = 500
QUERY_MAX_LIMIT = 2000
BATCH_PLACE_IDS_MAX = 500

BBox = tuple[float, float, float, float]


def _check_level(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS!r}; got {level!r}")
    return level


def _normalize_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _validate_page(limit: int, offset: int) -> tuple[int, int]:
    if isinstance(limit, bool) or not isinstance(limit, int) or not (
        1 <= limit <= QUERY_MAX_LIMIT
    ):
        raise ValueError(f"limit must be an integer between 1 and {QUERY_MAX_LIMIT}")
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError("offset must be a non-negative integer")
    return limit, offset


def _validate_bbox(bbox: BBox) -> BBox:
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise ValueError("bbox must be (minLng, minLat, maxLng, maxLat)")
    values = tuple(float(value) for value in bbox)
    min_lng, min_lat, max_lng, max_lat = values
    if min_lng > max_lng or min_lat > max_lat:
        raise ValueError("bbox must not be inverted (antimeridian wrapping unsupported)")
    if not (-180.0 <= min_lng <= 180.0 and -180.0 <= max_lng <= 180.0):
        raise ValueError("bbox longitudes must be within [-180, 180]")
    if not (-90.0 <= min_lat <= 90.0 and -90.0 <= max_lat <= 90.0):
        raise ValueError("bbox latitudes must be within [-90, 90]")
    return min_lng, min_lat, max_lng, max_lat


def _order_for(level: str) -> tuple[Any, ...]:
    if level == "city":
        # Stable scope ordering: scope first, then the catalog's own ranking
        # signal (population) and finally the numeric GeoNames id.
        return (
            GeoPlace.country_iso.asc(),
            GeoPlace.source_iso.asc(),
            GeoPlace.population.desc(),
            GeoPlace.geoname_id.asc(),
            GeoPlace.place_id.asc(),
        )
    return (
        GeoPlace.country_iso.asc(),
        GeoPlace.source_iso.asc(),
        GeoPlace.place_id.asc(),
    )


def _matches(place: GeoPlace, needle: str) -> bool:
    candidates = (
        place.place_id,
        place.code or "",
        place.name,
        place.ascii_name or "",
        place.name_zh or "",
        *place.aliases,
    )
    return any(needle in _normalize_text(candidate) for candidate in candidates)


class GeoCatalogStore:
    """Read-only catalog queries over one SQLAlchemy session."""

    def __init__(self, session: Session) -> None:
        self._session = session

    # ----- exact identity -------------------------------------------------
    def get_place(self, level: str, place_id: str) -> GeoPlace | None:
        """Return one place by unified identity, or ``None`` when unknown."""

        _check_level(level)
        if not isinstance(place_id, str) or not place_id:
            raise ValueError("place_id must be a non-empty string")
        return self._session.execute(
            select(GeoPlace).where(
                GeoPlace.level == level, GeoPlace.place_id == place_id
            )
        ).scalar_one_or_none()

    def get_places(self, level: str, place_ids: Sequence[str]) -> tuple[GeoPlace, ...]:
        """Return known rows for stable IDs, preserving request order.

        The batch is bounded by :data:`BATCH_PLACE_IDS_MAX`; unknown IDs are
        omitted rather than fabricated. This is used by local alias indexes
        whose IDs are stable but whose rows may lag behind the label snapshot.
        """

        _check_level(level)
        if isinstance(place_ids, (str, bytes)):
            raise ValueError("place_ids must be a sequence of strings")
        normalized: list[str] = []
        seen: set[str] = set()
        for place_id in place_ids:
            if not isinstance(place_id, str) or not place_id:
                raise ValueError("place_ids must contain non-empty strings")
            if place_id not in seen:
                normalized.append(place_id)
                seen.add(place_id)
        if len(normalized) > BATCH_PLACE_IDS_MAX:
            raise ValueError(f"place_ids must contain at most {BATCH_PLACE_IDS_MAX} IDs")
        if not normalized:
            return ()
        rows = self._session.execute(
            select(GeoPlace).where(
                GeoPlace.level == level, GeoPlace.place_id.in_(normalized)
            )
        ).scalars()
        by_id = {place.place_id: place for place in rows}
        return tuple(by_id[place_id] for place_id in normalized if place_id in by_id)

    def revisions(self) -> dict[str, str | None]:
        """Return ``countries``/``admin1``/``cities`` fingerprints + version.

        Values are the raw JSON SHA-256 digests recorded at import time (the
        inference-service revision contract keeps using file digests, never a
        database hash). Missing sources yield ``None``.
        """

        stored = {
            row.source: row
            for row in self._session.execute(select(GeoCatalogSource)).scalars()
        }

        def fingerprint(name: str) -> str | None:
            row = stored.get(name)
            return row.fingerprint if row is not None else None

        version_row = stored.get("countries") or stored.get("admin1") or stored.get("cities")
        return {
            "countries": fingerprint("countries"),
            "admin1": fingerprint("admin1"),
            "cities": fingerprint("cities"),
            "import_version": version_row.import_version if version_row is not None else None,
        }

    # ----- scoped queries -------------------------------------------------
    def query_places(
        self,
        level: str,
        *,
        country: str | None = None,
        source_iso: str | None = None,
        admin1_id: str | None = None,
        q: str | None = None,
        bbox: BBox | None = None,
        limit: int = QUERY_DEFAULT_LIMIT,
        offset: int = 0,
        coordinates_only: bool = True,
    ) -> dict[str, Any]:
        """Return ``{'rows', 'total', 'missing_coordinates'}`` for one scope.

        ``total`` counts matching rows after ``coordinates_only``; 
        ``missing_coordinates`` counts matching rows (before that filter) whose
        coordinate pair is NULL, so callers can report unavailable admin1 rows
        without inventing points. Callers validate ancestry with
        :meth:`get_place`; this method never fabricates parent rows.
        """

        _check_level(level)
        limit, offset = _validate_page(limit, offset)

        conditions: list[Any] = [GeoPlace.level == level]

        raw_source = normalize_iso(source_iso) if source_iso is not None else ""
        if source_iso is not None and not raw_source:
            raise ValueError("source_iso must be a two-letter ISO code")
        if country is not None:
            country_code = normalize_iso(country)
            if not country_code:
                raise ValueError("country must be a two-letter ISO code")
            if is_region_iso(country_code) and not raw_source:
                # Raw TW/HK/MO without an explicit source scope keeps its raw
                # identity instead of silently regrouping the whole CN family.
                raw_source = country_code
            else:
                conditions.append(
                    GeoPlace.country_iso == canonical_country_iso(country_code)
                )
        if raw_source:
            conditions.append(GeoPlace.source_iso == raw_source)
        if admin1_id is not None:
            if not isinstance(admin1_id, str) or not admin1_id:
                raise ValueError("admin1_id must be a non-empty string")
            conditions.append(GeoPlace.admin1_id == admin1_id)
        if bbox is not None:
            min_lng, min_lat, max_lng, max_lat = _validate_bbox(bbox)
            conditions.extend(
                (
                    GeoPlace.lng >= min_lng,
                    GeoPlace.lng <= max_lng,
                    GeoPlace.lat >= min_lat,
                    GeoPlace.lat <= max_lat,
                )
            )

        needle = _normalize_text(q) if q is not None else ""
        if needle:
            matched = list(
                self._session.execute(
                    select(GeoPlace).where(*conditions).order_by(*_order_for(level))
                ).scalars()
            )
            matched = [place for place in matched if _matches(place, needle)]
            missing = sum(1 for place in matched if place.lat is None or place.lng is None)
            if coordinates_only:
                matched = [
                    place for place in matched if place.lat is not None and place.lng is not None
                ]
            rows = matched[offset : offset + limit]
            return {"rows": rows, "total": len(matched), "missing_coordinates": missing}

        missing = int(
            self._session.execute(
                select(func.count())
                .select_from(GeoPlace)
                .where(*conditions, GeoPlace.lat.is_(None))
            ).scalar_one()
        )
        visible = list(conditions)
        if coordinates_only:
            visible.append(GeoPlace.lat.is_not(None))
        total = int(
            self._session.execute(
                select(func.count()).select_from(GeoPlace).where(*visible)
            ).scalar_one()
        )
        rows = list(
            self._session.execute(
                select(GeoPlace)
                .where(*visible)
                .order_by(*_order_for(level))
                .limit(limit)
                .offset(offset)
            ).scalars()
        )
        return {"rows": rows, "total": total, "missing_coordinates": missing}

    # ----- existing catalog shapes (guided adapter surface) ---------------
    def admin1_rows(self, source_iso: str) -> tuple[Admin1, ...]:
        """Return the exact ``Admin1`` dataclass rows for one raw source ISO."""

        iso = normalize_iso(source_iso)
        if not iso:
            raise ValueError("source_iso must be a two-letter ISO code")
        places = self._session.execute(
            select(GeoPlace)
            .where(GeoPlace.level == "admin1", GeoPlace.source_iso == iso)
            .order_by(GeoPlace.place_id.asc())
        ).scalars()
        rows: list[Admin1] = []
        for place in places:
            if not place.code or place.geoname_id is None or not place.ascii_name:
                raise ValueError(f"admin1 row {place.place_id} is incomplete")
            rows.append(
                Admin1(
                    admin1_id=place.place_id,
                    iso=place.source_iso,
                    code=place.code,
                    name=place.name,
                    ascii_name=place.ascii_name,
                    geoname_id=int(place.geoname_id),
                    name_zh=place.name_zh,
                    aliases=place.aliases,
                )
            )
        return tuple(rows)

    def city_rows(self, source_iso: str, admin1_id: str | None = None) -> dict[int, CityRow]:
        """Return ``{city_id: CityRow}`` for one raw source ISO.

        ``admin1_id`` narrows to cities whose exact admin1 code matched the
        directory (``source_iso.code``; leading zeros stay significant).
        """

        iso = normalize_iso(source_iso)
        if not iso:
            raise ValueError("source_iso must be a two-letter ISO code")
        statement = select(GeoPlace).where(
            GeoPlace.level == "city", GeoPlace.source_iso == iso
        )
        if admin1_id is not None:
            if not isinstance(admin1_id, str) or not admin1_id:
                raise ValueError("admin1_id must be a non-empty string")
            statement = statement.where(GeoPlace.admin1_id == admin1_id)
        places = self._session.execute(
            statement.order_by(GeoPlace.geoname_id.asc())
        ).scalars()
        rows: dict[int, CityRow] = {}
        for place in places:
            if place.lat is None or place.lng is None or place.population is None:
                raise ValueError(f"city row {place.place_id} is missing coordinates")
            city_id = int(place.place_id)
            rows[city_id] = CityRow(
                city_id=city_id,
                iso=place.source_iso,
                name=place.name,
                region=place.code or "",
                lat=float(place.lat),
                lng=float(place.lng),
                population=int(place.population),
            )
        return rows

    def countries(self) -> list[dict[str, Any]]:
        """Return the original countries.json payloads (sorted by ISO)."""

        places = self._session.execute(
            select(GeoPlace)
            .where(GeoPlace.level == "country")
            .order_by(GeoPlace.source_iso.asc())
        ).scalars()
        return [place.payload for place in places]


__all__ = [
    "BATCH_PLACE_IDS_MAX",
    "GeoCatalogStore",
    "QUERY_DEFAULT_LIMIT",
    "QUERY_MAX_LIMIT",
]
