"""Country catalog loading and prompt templating.

The shipped read-only SQLite seed is the source of truth for the country
directory. The catalog is intentionally not a fixed allowlist: any non-empty
set of at least two unique two-letter ISO-style codes is accepted. This lets a
later catalog edit add or remove candidates without changing inference code.

A loaded catalog is normalized to uppercase ISO codes and sorted by ISO. The
``catalog_revision`` is a SHA-256 over the canonical catalog payload (``iso`` +
English ``name``, sorted by ISO) plus the prompt template version and the
three prompt templates. It is stable across restarts, independent of input
record order and non-identity metadata, and changes when any candidate name,
the model prompt set, or the template version changes.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import PROMPT_TEMPLATE_VERSION, PROMPT_TEMPLATES

try:
    from catalog_db import CatalogDatabase, CatalogDatabaseError
except ModuleNotFoundError as exc:  # support standalone script execution
    if exc.name != "catalog_db":
        raise
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from catalog_db import CatalogDatabase, CatalogDatabaseError

_ISO2_RE = re.compile(r"^[A-Z]{2}$")


class CatalogError(ValueError):
    """Raised when the country catalog is missing, malformed or incomplete."""


@dataclass(frozen=True)
class Country:
    iso: str
    name: str


def _read_catalog(path: Path | str) -> list[Country]:
    """Load and strictly validate country payloads from the SQLite seed."""

    try:
        payload = CatalogDatabase(path).countries()
    except CatalogDatabaseError as exc:
        raise CatalogError(f"country catalog database is unreadable: {path}") from exc
    if not isinstance(payload, list):
        raise CatalogError("country catalog database must return a list")

    countries: list[Country] = []
    seen: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise CatalogError(f"country catalog item {index} is not an object")
        iso = item.get("iso")
        name = item.get("name")
        if not isinstance(iso, str) or not iso.strip():
            raise CatalogError(f"country catalog item {index} has no ISO code")
        if not isinstance(name, str) or not name.strip():
            raise CatalogError(f"country catalog item {iso} has no English name")
        iso = iso.strip().upper()
        if not _ISO2_RE.fullmatch(iso):
            raise CatalogError(
                f"country catalog item {index} has an invalid ISO code: {iso!r}"
            )
        if iso in seen:
            raise CatalogError(f"duplicate ISO code in country catalog: {iso}")
        seen.add(iso)
        countries.append(Country(iso=iso, name=name.strip()))

    if len(countries) < 2:
        raise CatalogError("country catalog must contain at least two countries")
    return sorted(countries, key=lambda country: country.iso)


def render_prompts(country: Country) -> list[str]:
    """Return the three English prompt templates for one country."""
    return [template.format(country=country.name) for template in PROMPT_TEMPLATES]


def compute_catalog_revision(countries: Iterable[Country]) -> str:
    """Return the deterministic SHA-256 revision of the candidate catalog."""
    canonical = {
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_templates": list(PROMPT_TEMPLATES),
        "countries": [
            {"iso": country.iso, "name": country.name}
            for country in sorted(countries, key=lambda item: item.iso)
        ],
    }
    encoded = json.dumps(canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "CatalogError",
    "Country",
    "compute_catalog_revision",
    "render_prompts",
]
