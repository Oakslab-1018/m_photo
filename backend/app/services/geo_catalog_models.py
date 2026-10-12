"""SQLAlchemy models for the imported geographic catalog.

The JSON files under ``data/`` stay the pinned source and revision authority;
these tables are a read-optimized projection that
:mod:`backend.app.services.geo_catalog_import` rebuilds atomically after
Alembic migration 0004. Runtime lookups go through
:class:`backend.app.services.geo_catalog_store.GeoCatalogStore`.

Identity
--------
* ``country`` rows: ``place_id`` is the ISO alpha-2 code from countries.json.
* ``admin1`` rows: ``place_id`` is ``source_iso.code`` from the GeoNames
  directory (codes keep leading zeros, e.g. ``CA.01``).
* ``city`` rows: ``place_id`` is the decimal GeoNames id as a string.

``source_iso`` is always the raw source scope (``CN``/``HK``/``MO``/``TW``
stay distinct) and ``country_iso`` is the canonical product country produced
by the existing CN grouping policy (``TW``/``HK``/``MO`` -> ``CN``).

Coordinates are never fabricated: countries and cities must carry finite
catalog coordinates, while admin1 rows may keep ``lat``/``lng`` NULL and are
then marked ``coordinate_source='unavailable'``. ``(0, 0)`` is never accepted
as a placeholder and the database enforces the coordinate pair/range,
coordinate-source and coordinate-provenance invariants, so a partially
imported or hand-edited catalog cannot silently produce fake points.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base

LEVELS: tuple[str, ...] = ("country", "admin1", "city")

#: Admin1 coordinates derived from the highest-population matching city.
ADMIN1_COORDINATE_SOURCE = "admin1-largest-city-v1"
#: Coordinates taken verbatim from the source catalog (country/city).
CATALOG_COORDINATE_SOURCE = "catalog"
#: Directory row kept without coordinates (no matching city).
UNAVAILABLE_COORDINATE_SOURCE = "unavailable"


class GeoCatalogSource(Base):
    """One imported JSON source revision.

    ``fingerprint`` is the raw-file SHA-256, ``record_count`` the number of
    validated rows, ``import_version`` the import/derivation policy version
    and ``imported_at`` when the replacement transaction committed. Startup
    compares path/fingerprint/count/version before deciding to skip a rebuild.
    """

    __tablename__ = "geo_catalog_source"

    # countries | admin1 | cities
    source: Mapped[str] = mapped_column(String(20), primary_key=True)
    path: Mapped[str] = mapped_column(String(500), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    record_count: Mapped[int] = mapped_column(Integer, nullable=False)
    import_version: Mapped[str] = mapped_column(String(50), nullable=False)
    imported_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


class GeoPlace(Base):
    """One unified catalog row (country, admin1 or city)."""

    __tablename__ = "geo_place"
    __table_args__ = (
        UniqueConstraint("level", "place_id", name="uq_geo_place_identity"),
        CheckConstraint(
            "level IN ('country', 'admin1', 'city')", name="ck_geo_place_level"
        ),
        CheckConstraint(
            "(lat IS NULL AND lng IS NULL) OR (lat IS NOT NULL AND lng IS NOT NULL)",
            name="ck_geo_place_coordinate_pair",
        ),
        CheckConstraint(
            "lat IS NULL OR (lat >= -90.0 AND lat <= 90.0)",
            name="ck_geo_place_lat_range",
        ),
        CheckConstraint(
            "lng IS NULL OR (lng >= -180.0 AND lng <= 180.0)",
            name="ck_geo_place_lng_range",
        ),
        CheckConstraint(
            "level = 'admin1' OR (lat IS NOT NULL AND lng IS NOT NULL)",
            name="ck_geo_place_required_coordinates",
        ),
        CheckConstraint(
            "population IS NULL OR population >= 0", name="ck_geo_place_population"
        ),
        CheckConstraint(
            "coordinate_source IN ('admin1-largest-city-v1', 'catalog', 'unavailable')",
            name="ck_geo_place_coordinate_source",
        ),
        CheckConstraint(
            "(coordinate_source = 'unavailable' AND lat IS NULL AND lng IS NULL) "
            "OR (coordinate_source IN ('admin1-largest-city-v1', 'catalog') "
            "AND lat IS NOT NULL AND lng IS NOT NULL)",
            name="ck_geo_place_coordinate_source_consistency",
        ),
        CheckConstraint(
            "(coordinate_source = 'admin1-largest-city-v1' AND coordinate_city_id IS NOT NULL) "
            "OR (coordinate_source <> 'admin1-largest-city-v1' AND coordinate_city_id IS NULL)",
            name="ck_geo_place_coordinate_provenance",
        ),
        CheckConstraint(
            "lat IS NULL OR lng IS NULL OR lat <> 0.0 OR lng <> 0.0",
            name="ck_geo_place_no_zero_zero",
        ),
        Index("ix_geo_place_level_country", "level", "country_iso"),
        Index("ix_geo_place_level_source", "level", "source_iso"),
        Index("ix_geo_place_level_parent", "level", "parent_id"),
        Index("ix_geo_place_level_admin1", "level", "admin1_id"),
        Index("ix_geo_place_level_population", "level", "population"),
        Index("ix_geo_place_level_name", "level", "name"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    level: Mapped[str] = mapped_column(String(10), nullable=False)
    place_id: Mapped[str] = mapped_column(String(64), nullable=False)
    #: Raw source scope (never regrouped): CN / HK / MO / TW stay distinct.
    source_iso: Mapped[str] = mapped_column(String(2), nullable=False)
    #: Canonical product country (existing china-subordinate-regions policy).
    country_iso: Mapped[str] = mapped_column(String(2), nullable=False)
    #: Parent place id: country ISO for admin1; admin1 id (or country ISO when
    #: the city region is not in the admin1 directory) for cities.
    parent_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Own admin1 id on admin1 rows; matched ``source_iso.code`` on city rows.
    admin1_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Source directory code: admin1 code, city region code, ISO3 for countries.
    code: Mapped[str | None] = mapped_column(String(16), nullable=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    ascii_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    name_zh: Mapped[str | None] = mapped_column(String(255), nullable=True)
    aliases_json: Mapped[str | None] = mapped_column("aliases", Text, nullable=True)
    payload_json: Mapped[str] = mapped_column("payload", Text, nullable=False)
    population: Mapped[int | None] = mapped_column(Integer, nullable=True)
    geoname_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    lat: Mapped[float | None] = mapped_column(Float, nullable=True)
    lng: Mapped[float | None] = mapped_column(Float, nullable=True)
    coordinate_source: Mapped[str] = mapped_column(String(40), nullable=False)
    coordinate_city_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    @property
    def label(self) -> str:
        """Best display label, preferring a verified Chinese name."""

        return self.name_zh or self.name

    @property
    def aliases(self) -> tuple[str, ...]:
        """Decoded alias list (empty when the source has none)."""

        if not self.aliases_json:
            return ()
        try:
            value = json.loads(self.aliases_json)
        except (TypeError, ValueError):
            return ()
        if not isinstance(value, list):
            return ()
        return tuple(item for item in value if isinstance(item, str) and item)

    @property
    def payload(self) -> dict[str, Any]:
        """Original source object for this place (never re-serialized)."""

        try:
            value = json.loads(self.payload_json or "{}")
        except (TypeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}

    @property
    def coordinate_quality(self) -> str:
        """Documented quality label for the stored coordinate source.

        ``derived`` for ``admin1-largest-city-v1`` representative points,
        ``catalog`` for coordinates taken from the source file, and
        ``unavailable`` for directory rows kept without coordinates (``lat``/
        ``lng`` are NULL there; never render them as a point).
        """

        if self.coordinate_source == ADMIN1_COORDINATE_SOURCE:
            return "derived"
        if self.coordinate_source == CATALOG_COORDINATE_SOURCE:
            return "catalog"
        return "unavailable"


__all__ = [
    "ADMIN1_COORDINATE_SOURCE",
    "CATALOG_COORDINATE_SOURCE",
    "LEVELS",
    "UNAVAILABLE_COORDINATE_SOURCE",
    "GeoCatalogSource",
    "GeoPlace",
]
