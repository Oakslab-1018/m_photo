"""Import the pinned geographic catalog into indexed SQLite tables.

Revision ID: 0004_geo_catalog
Revises: 0003_guided_dialogue
Create Date: 2026-10-12

The JSON catalogs under ``data/`` stay the source/revision authority; these
tables are a read-optimized projection rebuilt by
``backend.app.services.geo_catalog_import`` after this migration runs.
``geo_catalog_source`` records one fingerprint/count/import-version row per
source file; ``geo_place`` stores the unified country/admin1/city rows with a
raw ``source_iso`` and the canonical ``country_iso`` produced by the existing
CN grouping policy. Check constraints mirror the ORM model exactly: coordinate
pairs are all-or-nothing, ranges are enforced, countries/cities always carry
coordinates and admin1 rows may keep a NULL pair (``unavailable``).
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004_geo_catalog"
down_revision: Union[str, None] = "0003_guided_dialogue"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "geo_catalog_source",
        sa.Column("source", sa.String(length=20), primary_key=True),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("fingerprint", sa.String(length=64), nullable=False),
        sa.Column("record_count", sa.Integer(), nullable=False),
        sa.Column("import_version", sa.String(length=50), nullable=False),
        sa.Column("imported_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "geo_place",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("level", sa.String(length=10), nullable=False),
        sa.Column("place_id", sa.String(length=64), nullable=False),
        sa.Column("source_iso", sa.String(length=2), nullable=False),
        sa.Column("country_iso", sa.String(length=2), nullable=False),
        sa.Column("parent_id", sa.String(length=64), nullable=True),
        sa.Column("admin1_id", sa.String(length=64), nullable=True),
        sa.Column("code", sa.String(length=16), nullable=True),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("ascii_name", sa.String(length=255), nullable=True),
        sa.Column("name_zh", sa.String(length=255), nullable=True),
        sa.Column("aliases", sa.Text(), nullable=True),
        sa.Column("payload", sa.Text(), nullable=False),
        sa.Column("population", sa.Integer(), nullable=True),
        sa.Column("geoname_id", sa.Integer(), nullable=True),
        sa.Column("lat", sa.Float(), nullable=True),
        sa.Column("lng", sa.Float(), nullable=True),
        sa.Column("coordinate_source", sa.String(length=40), nullable=False),
        sa.Column("coordinate_city_id", sa.Integer(), nullable=True),
        sa.UniqueConstraint("level", "place_id", name="uq_geo_place_identity"),
        sa.CheckConstraint(
            "level IN ('country', 'admin1', 'city')", name="ck_geo_place_level"
        ),
        sa.CheckConstraint(
            "(lat IS NULL AND lng IS NULL) OR (lat IS NOT NULL AND lng IS NOT NULL)",
            name="ck_geo_place_coordinate_pair",
        ),
        sa.CheckConstraint(
            "lat IS NULL OR (lat >= -90.0 AND lat <= 90.0)",
            name="ck_geo_place_lat_range",
        ),
        sa.CheckConstraint(
            "lng IS NULL OR (lng >= -180.0 AND lng <= 180.0)",
            name="ck_geo_place_lng_range",
        ),
        sa.CheckConstraint(
            "level = 'admin1' OR (lat IS NOT NULL AND lng IS NOT NULL)",
            name="ck_geo_place_required_coordinates",
        ),
        sa.CheckConstraint(
            "population IS NULL OR population >= 0", name="ck_geo_place_population"
        ),
        sa.CheckConstraint(
            "coordinate_source IN ('admin1-largest-city-v1', 'catalog', 'unavailable')",
            name="ck_geo_place_coordinate_source",
        ),
        sa.CheckConstraint(
            "(coordinate_source = 'unavailable' AND lat IS NULL AND lng IS NULL) "
            "OR (coordinate_source IN ('admin1-largest-city-v1', 'catalog') "
            "AND lat IS NOT NULL AND lng IS NOT NULL)",
            name="ck_geo_place_coordinate_source_consistency",
        ),
        sa.CheckConstraint(
            "(coordinate_source = 'admin1-largest-city-v1' AND coordinate_city_id IS NOT NULL) "
            "OR (coordinate_source <> 'admin1-largest-city-v1' AND coordinate_city_id IS NULL)",
            name="ck_geo_place_coordinate_provenance",
        ),
        sa.CheckConstraint(
            "lat IS NULL OR lng IS NULL OR lat <> 0.0 OR lng <> 0.0",
            name="ck_geo_place_no_zero_zero",
        ),
    )
    op.create_index("ix_geo_place_level_country", "geo_place", ["level", "country_iso"])
    op.create_index("ix_geo_place_level_source", "geo_place", ["level", "source_iso"])
    op.create_index("ix_geo_place_level_parent", "geo_place", ["level", "parent_id"])
    op.create_index("ix_geo_place_level_admin1", "geo_place", ["level", "admin1_id"])
    op.create_index("ix_geo_place_level_population", "geo_place", ["level", "population"])
    op.create_index("ix_geo_place_level_name", "geo_place", ["level", "name"])


def downgrade() -> None:
    op.drop_index("ix_geo_place_level_name", table_name="geo_place")
    op.drop_index("ix_geo_place_level_population", table_name="geo_place")
    op.drop_index("ix_geo_place_level_admin1", table_name="geo_place")
    op.drop_index("ix_geo_place_level_parent", table_name="geo_place")
    op.drop_index("ix_geo_place_level_source", table_name="geo_place")
    op.drop_index("ix_geo_place_level_country", table_name="geo_place")
    op.drop_table("geo_place")
    op.drop_table("geo_catalog_source")
