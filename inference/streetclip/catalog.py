"""Country catalog loading and prompt templating.

The catalog is a fixed allowlist: ``data/countries.json`` must contain exactly
the 20 expected ISO-3166 alpha-2 codes, in any order, with no duplicates. Extra
or missing entries are startup errors rather than silent drops, so a later
catalog edit cannot silently remove a candidate.

``catalog_revision`` is a SHA-256 over the canonical catalog payload
(``iso`` + English ``name``, sorted by ISO) plus the prompt template version
and the three prompt templates. It is stable across restarts and changes when
any candidate name, the model prompt set or the template version changes.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .config import PROMPT_TEMPLATE_VERSION, PROMPT_TEMPLATES

EXPECTED_ISOS = (
    "AU",
    "BR",
    "CA",
    "CN",
    "DE",
    "EG",
    "ES",
    "FR",
    "GB",
    "GR",
    "IN",
    "IT",
    "JP",
    "KR",
    "MX",
    "NL",
    "RU",
    "SE",
    "TH",
    "US",
)


class CatalogError(ValueError):
    """Raised when the country catalog is missing, malformed or incomplete."""


@dataclass(frozen=True)
class Country:
    iso: str
    name: str


def _read_catalog(path: Path) -> list[Country]:
    if not path.is_file():
        raise CatalogError(f"country catalog not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise CatalogError(f"country catalog is unreadable: {path}") from exc
    except json.JSONDecodeError as exc:
        raise CatalogError(f"country catalog is not valid JSON: {path}") from exc
    if not isinstance(payload, list):
        raise CatalogError("country catalog must be a JSON array")

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
        name = name.strip()
        if iso in seen:
            raise CatalogError(f"duplicate ISO code in country catalog: {iso}")
        seen.add(iso)
        countries.append(Country(iso=iso, name=name))

    found = {country.iso for country in countries}
    expected = set(EXPECTED_ISOS)
    missing = sorted(expected - found)
    unexpected = sorted(found - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise CatalogError("country catalog is not the fixed 20-country set (" + "; ".join(details) + ")")
    if len(countries) != len(EXPECTED_ISOS):
        raise CatalogError("country catalog must contain exactly 20 unique countries")
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
    "EXPECTED_ISOS",
    "compute_catalog_revision",
    "render_prompts",
]
