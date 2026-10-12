"""Settings-aware access to the SQLite geographic catalog.

This is the one entry point leaf readers should use:

* ``catalog_for_settings(settings)`` -> the runtime catalog database derived
  from ``settings.database_url`` (never the JSON-era path properties and never
  the shipped seed);
* ``catalog_for_settings(None)`` -> the immutable shipped seed, for offline
  domain/inference use where no runtime database exists.

Both forms return an already validated :class:`catalog_db.CatalogDatabase`.
There is no silent fallback: a settings object whose URL is not file-backed
SQLite raises instead of reading the seed in place of an uninitialized runtime
database.
"""
from __future__ import annotations

from catalog_db import CatalogDatabase, default_catalog_path

from ..config import Settings


def catalog_for_settings(settings: Settings | None = None) -> CatalogDatabase:
    """Return the validated catalog reader selected by ``settings``.

    ``None`` selects the immutable shipped seed (standard-library offline
    mode); a settings object selects its runtime database path, which is
    derived from ``settings.database_url``.
    """

    if settings is None:
        return CatalogDatabase(default_catalog_path())
    return CatalogDatabase(settings.catalog_db_path)


__all__ = ["catalog_for_settings"]
