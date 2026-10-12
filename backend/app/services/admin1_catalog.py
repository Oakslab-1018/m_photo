"""Read-only access to the authoritative GeoNames admin1 catalog (SQLite only).

The catalog now lives in the portable SQLite seed
(``data/geo_catalog.sqlite3``) and is read through the standard-library
:class:`catalog_db.CatalogDatabase`; ``data/admin1.json`` is never read. The
catalog remains the authoritative source for country -> admin1 -> city guided
navigation, and GeoNames admin1 codes stay strings so leading zeros remain
significant. The logical revision is the fingerprint stored for the ``admin1``
source at seed-build time, never a hash of the SQLite file.

Each row is exposed as a strict :class:`Admin1` dataclass; a malformed,
duplicated or internally inconsistent row raises :class:`Admin1CatalogError`
instead of allowing a caller to show fabricated geography.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from catalog_db import DEFAULT_CATALOG_PATH, CatalogDatabase, CatalogDatabaseError

REPO_ROOT = Path(__file__).resolve().parents[3]
#: Deprecated name retained for callers: the shipped SQLite seed, never JSON.
DEFAULT_ADMIN1_CATALOG_PATH = DEFAULT_CATALOG_PATH
DEFAULT_ADMIN1_PATH = DEFAULT_ADMIN1_CATALOG_PATH

_ADMIN1_ID_RE = re.compile(r"^(?P<iso>[A-Z]{2})\.(?P<code>[A-Z0-9]+)$")
_ISO_RE = re.compile(r"^[A-Z]{2}$")
_CODE_RE = re.compile(r"^[A-Z0-9]+$")


class Admin1CatalogError(ValueError):
    """Raised when the admin1 catalog is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Admin1:
    """One validated GeoNames admin1 row."""

    admin1_id: str
    iso: str
    code: str
    name: str
    ascii_name: str
    geoname_id: int
    name_zh: str | None = None
    aliases: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Return the best display label, preferring a verified Chinese name."""

        return self.name_zh or self.name

    def matches(self, query: str) -> bool:
        """Return whether ``query`` identifies this row by any stable label.

        Matching is case-insensitive and Unicode-normalized.  Codes and
        ``admin1_id`` remain exact because callers may need to distinguish
        e.g. ``SG.01`` from a city region code that was not present in the
        authoritative admin1 directory.
        """

        if not isinstance(query, str):
            return False
        needle = _normalize_for_match(query)
        if not needle:
            return False
        candidates = (
            self.admin1_id,
            self.code,
            self.name,
            self.ascii_name,
            self.name_zh or "",
            *self.aliases,
        )
        return any(needle == _normalize_for_match(candidate) for candidate in candidates)


@dataclass(frozen=True)
class Admin1Index:
    """Validated admin1 rows indexed by ID and country ISO code."""

    by_id: dict[str, Admin1]
    by_iso: dict[str, tuple[Admin1, ...]]
    revision: str


def _normalize_for_match(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold().strip()


def _row_error(index: int, message: str) -> Admin1CatalogError:
    return Admin1CatalogError(f"admin1 catalog item {index}: {message}")


def _coerce_name(value: Any, field: str, index: int, *, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise _row_error(index, f"{field} must be a string")
    text = " ".join(value.split())
    if not text:
        raise _row_error(index, f"{field} must not be blank")
    return text


def _coerce_geoname_id(value: Any, index: int) -> int:
    if isinstance(value, bool):
        raise _row_error(index, "geoname_id must be a positive integer")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise _row_error(index, "geoname_id must be a positive integer")
    if parsed <= 0:
        raise _row_error(index, "geoname_id must be a positive integer")
    return parsed


def _coerce_aliases(value: Any, index: int) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise _row_error(index, "aliases must be an array of strings")
    aliases: list[str] = []
    for alias_position, alias in enumerate(value):
        text = _coerce_name(alias, f"aliases[{alias_position}]", index)
        assert text is not None
        aliases.append(text)
    return tuple(aliases)


def admin1_from_mapping(item: Any, index: int = 0) -> Admin1:
    """Validate one catalog object into an :class:`Admin1` row."""

    if not isinstance(item, dict):
        raise _row_error(index, "item is not an object")

    admin1_id = item.get("admin1_id")
    if not isinstance(admin1_id, str):
        raise _row_error(index, "admin1_id must be a string")
    match = _ADMIN1_ID_RE.fullmatch(admin1_id)
    if match is None:
        raise _row_error(index, "admin1_id must have the form ISO.CODE")

    iso = item.get("iso")
    code = item.get("code")
    if not isinstance(iso, str) or _ISO_RE.fullmatch(iso) is None:
        raise _row_error(index, "iso must be two uppercase letters")
    if not isinstance(code, str) or _CODE_RE.fullmatch(code) is None:
        raise _row_error(index, "code must be a non-empty uppercase alphanumeric string")
    if iso != match.group("iso") or code != match.group("code"):
        raise _row_error(index, "admin1_id is inconsistent with iso/code")

    name = _coerce_name(item.get("name"), "name", index)
    ascii_name = _coerce_name(item.get("ascii_name"), "ascii_name", index)
    assert name is not None and ascii_name is not None
    geoname_id = _coerce_geoname_id(item.get("geoname_id"), index)
    name_zh = _coerce_name(item.get("name_zh"), "name_zh", index, optional=True)
    aliases = _coerce_aliases(item.get("aliases", ()), index)

    return Admin1(
        admin1_id=admin1_id,
        iso=iso,
        code=code,
        name=name,
        ascii_name=ascii_name,
        geoname_id=geoname_id,
        name_zh=name_zh,
        aliases=aliases,
    )


def load_admin1_index(path: Path | str | None = None) -> Admin1Index:
    """Load and validate the admin1 catalog from a SQLite catalog database.

    ``path`` is a SQLite catalog database path; ``None`` selects the immutable
    shipped seed. The function is fully offline: it only reads the supplied
    SQLite database and never contacts GeoNames. The stored logical
    ``admin1`` revision is returned unchanged.
    """

    candidate = None if path is None else Path(path)
    try:
        reader = CatalogDatabase(candidate)
        raw_rows = reader.admin1()
        revision = reader.revision("admin1")
    except CatalogDatabaseError as exc:
        raise Admin1CatalogError(f"admin1 catalog is unreadable: {exc}") from exc

    if not raw_rows:
        raise Admin1CatalogError("admin1 catalog must contain at least one row")

    validated: list[Admin1] = []
    seen_ids: set[str] = set()
    for position, item in enumerate(raw_rows):
        row = admin1_from_mapping(item, position)
        if row.admin1_id in seen_ids:
            raise Admin1CatalogError(
                f"duplicate admin1_id in admin1 catalog: {row.admin1_id}"
            )
        seen_ids.add(row.admin1_id)
        validated.append(row)

    validated.sort(key=lambda row: row.admin1_id)
    by_id = {row.admin1_id: row for row in validated}

    grouped: dict[str, list[Admin1]] = {}
    for row in validated:
        grouped.setdefault(row.iso, []).append(row)
    by_iso = {iso: tuple(group_rows) for iso, group_rows in grouped.items()}

    return Admin1Index(by_id=by_id, by_iso=by_iso, revision=revision)


__all__ = [
    "Admin1",
    "Admin1CatalogError",
    "Admin1Index",
    "DEFAULT_ADMIN1_CATALOG_PATH",
    "DEFAULT_ADMIN1_PATH",
    "admin1_from_mapping",
    "load_admin1_index",
]
