"""Bounded, injectable place search providers.

This module intentionally owns its own environment-backed configuration so the
guided-dialogue feature can evolve without coupling to the OCR/model settings.
Only user-authored place terms and an optional confirmed country are sent to a
provider.  Image paths, OCR text, chat history and coordinates are never
forwarded.  Provider coordinates are also ignored: catalog city identity and
coordinates remain the sole source of map targets.
"""
from __future__ import annotations

import json
import logging
import math
import os
import sys
from pathlib import Path
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Mapping, Protocol, Sequence
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

PHOTON_DEFAULT_URL = "https://photon.komoot.io/api"
NOMINATIM_DEFAULT_URL = "https://nominatim.openstreetmap.org/search"
OSM_ATTRIBUTION = "© OpenStreetMap contributors"
OSM_LICENSE = "ODbL"

_MAX_RESPONSE_BYTES = 512 * 1024
_MAX_TERMS = 4
_MAX_TERM_LENGTH = 60


class GeocodingError(RuntimeError):
    """Base class for a provider failure that must degrade safely."""


class GeocodingUnavailable(GeocodingError):
    """The configured provider could not be used for this query."""


@dataclass(frozen=True)
class GeocodingConfig:
    """Standalone geocoder policy read directly from the process environment.

    The default is the unkeyed Photon endpoint after policy verification.
    Tests and batch evaluations are disabled automatically unless explicitly
    enabled with M_PHOTO_GEOCODER_FORCE_ENABLED=1.
    """

    enabled: bool = True
    provider: str = "photon"
    base_url: str = PHOTON_DEFAULT_URL
    language: str = "en"
    timeout_seconds: float = 4.0
    connect_timeout_seconds: float = 1.5
    min_interval_seconds: float = 1.0
    cache_max_entries: int = 512
    cache_ttl_seconds: float = 24 * 60 * 60
    negative_cache_ttl_seconds: float = 10 * 60
    result_limit: int = 5
    user_agent: str = "MPhoto/0.1 (bounded place search)"
    api_key: str = field(default="", repr=False)

    def __post_init__(self) -> None:
        provider = str(self.provider or "none").strip().lower()
        object.__setattr__(self, "provider", provider)
        if provider not in {"photon", "nominatim", "none"}:
            raise ValueError("geocoder provider must be photon, nominatim, or none")
        if provider == "nominatim" and self.base_url == PHOTON_DEFAULT_URL:
            object.__setattr__(self, "base_url", NOMINATIM_DEFAULT_URL)
        if self.timeout_seconds <= 0 or not math.isfinite(self.timeout_seconds):
            raise ValueError("geocoder timeout must be positive and finite")
        if self.connect_timeout_seconds <= 0 or not math.isfinite(self.connect_timeout_seconds):
            raise ValueError("geocoder connect timeout must be positive and finite")
        if self.min_interval_seconds < 1.0 or not math.isfinite(self.min_interval_seconds):
            raise ValueError("public geocoder interval must be at least one second")
        if not (1 <= int(self.cache_max_entries) <= 4096):
            raise ValueError("geocoder cache size must be between 1 and 4096")
        if self.cache_ttl_seconds <= 0 or self.negative_cache_ttl_seconds <= 0:
            raise ValueError("geocoder cache TTLs must be positive")
        if not (1 <= int(self.result_limit) <= 10):
            raise ValueError("geocoder result limit must be between 1 and 10")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "GeocodingConfig":
        source = dict(os.environ) if env is None else dict(env)

        def get(*names: str) -> str | None:
            for name in names:
                value = source.get(name)
                if value is not None and str(value).strip():
                    return str(value).strip()
            return None

        def truthy(value: str | None) -> bool:
            return str(value or "").strip().lower() in {"1", "true", "yes", "on"}

        def number(value: str | None, default: float, low: float, high: float) -> float:
            try:
                parsed = float(value) if value is not None else default
            except (TypeError, ValueError):
                return default
            if not math.isfinite(parsed):
                return default
            return min(high, max(low, parsed))

        provider = (get("M_PHOTO_GEOCODER_PROVIDER", "GEOCODER_PROVIDER") or "photon").lower()
        explicit_enabled = get("M_PHOTO_GEOCODER_ENABLED", "GEOCODER_ENABLED")
        enabled = True if explicit_enabled is None else truthy(explicit_enabled)
        disabled = truthy(get("M_PHOTO_GEOCODER_DISABLED", "GEOCODER_DISABLED"))
        argv_name = Path(sys.argv[0]).name.lower() if sys.argv else ""
        eval_mode = (
            any(
                truthy(source.get(name))
                for name in ("M_PHOTO_EVAL_MODE", "GEO_EVAL_MODE", "EVAL_MODE")
            )
            or bool(source.get("PYTEST_CURRENT_TEST"))
            or argv_name.startswith("eval_")
            or "evaluation" in argv_name
        )
        force_enabled = truthy(get("M_PHOTO_GEOCODER_FORCE_ENABLED"))
        if disabled or ((eval_mode or provider == "none") and not force_enabled):
            enabled = False

        configured_url = get("M_PHOTO_GEOCODER_BASE_URL", "GEOCODER_BASE_URL")
        if provider == "nominatim" and not configured_url:
            # Only the default Photon public endpoint is pre-approved. Nominatim
            # is enabled here as a self-host/explicit-URL adapter.
            enabled = False
        default_url = NOMINATIM_DEFAULT_URL if provider == "nominatim" else PHOTON_DEFAULT_URL
        base_url = configured_url or default_url
        timeout = number(
            get("M_PHOTO_GEOCODER_TIMEOUT_SECONDS", "GEOCODER_TIMEOUT_SECONDS"),
            4.0,
            0.5,
            10.0,
        )
        connect_timeout = min(
            timeout,
            number(
                get(
                    "M_PHOTO_GEOCODER_CONNECT_TIMEOUT_SECONDS",
                    "GEOCODER_CONNECT_TIMEOUT_SECONDS",
                ),
                1.5,
                0.25,
                timeout,
            ),
        )
        min_interval = number(
            get("M_PHOTO_GEOCODER_MIN_INTERVAL_SECONDS", "GEOCODER_MIN_INTERVAL_SECONDS"),
            1.0,
            1.0,
            10.0,
        )
        cache_size = int(number(
            get("M_PHOTO_GEOCODER_CACHE_MAX_ENTRIES", "GEOCODER_CACHE_MAX_ENTRIES"),
            512,
            1,
            4096,
        ))
        return cls(
            enabled=enabled,
            provider=provider,
            base_url=base_url,
            language=get("M_PHOTO_GEOCODER_LANGUAGE", "GEOCODER_LANGUAGE") or "en",
            timeout_seconds=timeout,
            connect_timeout_seconds=connect_timeout,
            min_interval_seconds=min_interval,
            cache_max_entries=cache_size,
            cache_ttl_seconds=number(
                get("M_PHOTO_GEOCODER_CACHE_TTL_SECONDS", "GEOCODER_CACHE_TTL_SECONDS"),
                24 * 60 * 60,
                60,
                30 * 24 * 60 * 60,
            ),
            negative_cache_ttl_seconds=number(
                get(
                    "M_PHOTO_GEOCODER_NEGATIVE_CACHE_TTL_SECONDS",
                    "GEOCODER_NEGATIVE_CACHE_TTL_SECONDS",
                ),
                10 * 60,
                30,
                24 * 60 * 60,
            ),
            result_limit=int(number(
                get("M_PHOTO_GEOCODER_MAX_RESULTS", "GEOCODER_MAX_RESULTS"),
                5,
                1,
                10,
            )),
            user_agent=get("M_PHOTO_GEOCODER_USER_AGENT", "GEOCODER_USER_AGENT")
            or "MPhoto/0.1 (bounded place search)",
            api_key=get("M_PHOTO_GEOCODER_API_KEY", "GEOCODER_API_KEY") or "",
        )


@dataclass(frozen=True)
class GeocoderRequest:
    terms: tuple[str, ...]
    quote: str = ""
    country_iso: str | None = None
    source_isos: tuple[str, ...] = ()
    language: str = "en"
    limit: int = 5

    def normalized_terms(self) -> tuple[str, ...]:
        result: list[str] = []
        for term in self.terms:
            text = " ".join(str(term or "").split())
            if not text or len(text) > _MAX_TERM_LENGTH:
                continue
            if text not in result:
                result.append(text)
            if len(result) >= _MAX_TERMS:
                break
        return tuple(result)

    def query(self) -> str:
        return " ".join(self.normalized_terms())


@dataclass(frozen=True)
class GeocoderHit:
    provider: str
    provider_place_id: str | None
    name: str
    display_name: str
    locality_names: tuple[str, ...]
    admin1_name: str | None
    country_code: str | None
    attribution: str | None
    license: str | None
    query: str


class PlaceGeocoder(Protocol):
    name: str

    def search(self, request: GeocoderRequest) -> tuple[GeocoderHit, ...]:
        ...


class _TTLCache:
    def __init__(self, max_entries: int) -> None:
        self._max_entries = max_entries
        self._values: dict[str, tuple[float, tuple[GeocoderHit, ...]]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> tuple[GeocoderHit, ...] | None:
        now = time.monotonic()
        with self._lock:
            item = self._values.get(key)
            if item is None:
                return None
            expires, hits = item
            if expires <= now:
                self._values.pop(key, None)
                return None
            return hits

    def put(
        self,
        key: str,
        hits: Sequence[GeocoderHit],
        *,
        ttl_seconds: float,
    ) -> None:
        if self._max_entries <= 0:
            return
        with self._lock:
            if len(self._values) >= self._max_entries:
                oldest = min(self._values, key=lambda item: self._values[item][0])
                self._values.pop(oldest, None)
            self._values[key] = (time.monotonic() + ttl_seconds, tuple(hits))


class _RateLimiter:
    def __init__(self, min_interval_seconds: float) -> None:
        self._interval = max(0.0, float(min_interval_seconds))
        self._last = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._interval - (now - self._last)
            if delay > 0:
                time.sleep(delay)
            self._last = time.monotonic()


def _validate_base_url(url: str) -> str:
    parts = urlsplit(str(url or "").strip())
    if parts.username or parts.password or parts.query or parts.fragment:
        raise GeocodingUnavailable("geocoder URL contains unsupported components")
    if parts.scheme == "https" and parts.hostname:
        return str(url).rstrip("/")
    loopback = parts.hostname in {"127.0.0.1", "localhost", "::1"}
    if parts.scheme == "http" and loopback:
        return str(url).rstrip("/")
    raise GeocodingUnavailable("geocoder URL must use HTTPS (HTTP is loopback-only)")


def _bounded_text(value: object, limit: int = 200) -> str:
    return str(value or "").strip()[:limit]


def _normalize_country(value: object) -> str | None:
    text = _bounded_text(value, 8).upper()
    return text if len(text) == 2 and text.isalpha() else None


class _HttpGeocoder:
    name = ""

    def __init__(
        self,
        config: GeocodingConfig,
        *,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.config = config
        self.transport = transport
        self._cache = _TTLCache(config.cache_max_entries)
        self._limiter = _RateLimiter(config.min_interval_seconds)
        self._clock = clock or time.monotonic

    def search(self, request: GeocoderRequest) -> tuple[GeocoderHit, ...]:
        terms = request.normalized_terms()
        if not terms:
            return ()
        normalized = GeocoderRequest(
            terms=terms,
            quote=request.quote,
            country_iso=_normalize_country(request.country_iso),
            source_isos=tuple(
                item for item in (_normalize_country(value) for value in request.source_isos)
                if item
            ),
            language=_bounded_text(request.language or self.config.language, 20),
            limit=min(int(self.config.result_limit), max(1, int(request.limit or 5))),
        )
        key = "|".join((
            self.name,
            normalized.query().casefold(),
            normalized.country_iso or "",
            ",".join(sorted(set(normalized.source_isos))),
            normalized.language.casefold(),
            str(normalized.limit),
        ))
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        hits = tuple(self._search(normalized)[: normalized.limit])
        ttl = self.config.cache_ttl_seconds if hits else self.config.negative_cache_ttl_seconds
        self._cache.put(key, hits, ttl_seconds=ttl)
        return hits

    def _search(self, request: GeocoderRequest) -> list[GeocoderHit]:
        raise NotImplementedError

    def _get_json(self, url: str, params: Mapping[str, object]) -> object:
        base = _validate_base_url(url)
        self._limiter.wait()
        headers = {
            "Accept": "application/json",
            "User-Agent": self.config.user_agent,
        }
        timeout = httpx.Timeout(
            self.config.timeout_seconds,
            connect=self.config.connect_timeout_seconds,
        )
        try:
            with httpx.Client(
                transport=self.transport,
                timeout=timeout,
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = client.get(base, params=params, headers=headers)
                response.raise_for_status()
                if len(response.content) > _MAX_RESPONSE_BYTES:
                    raise GeocodingUnavailable("geocoder response exceeded size limit")
                return response.json()
        except GeocodingUnavailable:
            raise
        except Exception as exc:
            raise GeocodingUnavailable(type(exc).__name__) from exc


class PhotonGeocoder(_HttpGeocoder):
    name = "photon"

    def _search(self, request: GeocoderRequest) -> list[GeocoderHit]:
        payload = self._get_json(
            self.config.base_url,
            {
                "q": request.query(),
                "limit": request.limit,
                "lang": request.language,
            },
        )
        features = payload.get("features") if isinstance(payload, dict) else None
        if not isinstance(features, list):
            raise GeocodingUnavailable("unexpected Photon response")
        hits: list[GeocoderHit] = []
        for feature in features[: request.limit]:
            if not isinstance(feature, dict):
                continue
            properties = feature.get("properties")
            if not isinstance(properties, dict):
                continue
            name = _bounded_text(properties.get("name"))
            display = " ".join(
                part for part in (
                    name,
                    _bounded_text(properties.get("street")),
                    _bounded_text(properties.get("city")),
                    _bounded_text(properties.get("state")),
                    _bounded_text(properties.get("country")),
                ) if part
            )
            locality_names: list[str] = []
            for key in ("city", "municipality", "town", "village", "county", "district", "locality", "name"):
                value = _bounded_text(properties.get(key))
                if value and value not in locality_names:
                    locality_names.append(value)
            provider_id = None
            osm_id = properties.get("osm_id")
            osm_type = _bounded_text(properties.get("osm_type"), 20)
            if osm_id is not None:
                provider_id = f"{osm_type}:{osm_id}" if osm_type else str(osm_id)
            hits.append(GeocoderHit(
                provider=self.name,
                provider_place_id=provider_id,
                name=name or (locality_names[0] if locality_names else request.query()),
                display_name=display or name or request.query(),
                locality_names=tuple(locality_names[:6]),
                admin1_name=_bounded_text(properties.get("state")) or None,
                country_code=_normalize_country(properties.get("countrycode")),
                attribution=OSM_ATTRIBUTION,
                license=OSM_LICENSE,
                query=request.query(),
            ))
        return hits


class NominatimGeocoder(_HttpGeocoder):
    name = "nominatim"

    def _search(self, request: GeocoderRequest) -> list[GeocoderHit]:
        params: dict[str, object] = {
            "q": request.query(),
            "format": "jsonv2",
            "addressdetails": 1,
            "limit": request.limit,
            "accept-language": request.language,
        }
        payload = self._get_json(self.config.base_url, params)
        if not isinstance(payload, list):
            raise GeocodingUnavailable("unexpected Nominatim response")
        hits: list[GeocoderHit] = []
        for item in payload[: request.limit]:
            if not isinstance(item, dict):
                continue
            address = item.get("address") if isinstance(item.get("address"), dict) else {}
            name = _bounded_text(item.get("name")) or _bounded_text(item.get("display_name"))
            locality_names: list[str] = []
            for key in (
                "city", "municipality", "town", "village", "county", "city_district",
                "state_district", "state",
            ):
                value = _bounded_text(address.get(key))
                if value and value not in locality_names:
                    locality_names.append(value)
            if name and name not in locality_names:
                locality_names.insert(0, name)
            provider_id = None
            osm_id = item.get("osm_id")
            osm_type = _bounded_text(item.get("osm_type"), 20)
            if osm_id is not None:
                provider_id = f"{osm_type}:{osm_id}" if osm_type else str(osm_id)
            hits.append(GeocoderHit(
                provider=self.name,
                provider_place_id=provider_id,
                name=name or request.query(),
                display_name=_bounded_text(item.get("display_name"), 400) or name or request.query(),
                locality_names=tuple(locality_names[:7]),
                admin1_name=_bounded_text(address.get("state")) or None,
                country_code=_normalize_country(address.get("country_code")),
                attribution=OSM_ATTRIBUTION,
                license=OSM_LICENSE,
                query=request.query(),
            ))
        return hits


class DisabledGeocoder:
    name = "disabled"

    def search(self, request: GeocoderRequest) -> tuple[GeocoderHit, ...]:
        return ()


@lru_cache(maxsize=8)
def _shared_geocoder(config: GeocodingConfig) -> PlaceGeocoder:
    # GuidedDialogueService is request-scoped; the network cache and limiter
    # must survive requests to prevent duplicate lookups and rate bursts.
    if config.provider == "photon":
        return PhotonGeocoder(config)
    if config.provider == "nominatim":
        return NominatimGeocoder(config)
    return DisabledGeocoder()


_SHARED_GEOCODERS_LOCK = threading.Lock()


def get_shared_geocoder(config: GeocodingConfig) -> PlaceGeocoder:
    """Share the bounded provider cache and serialize first construction."""

    with _SHARED_GEOCODERS_LOCK:
        return build_geocoder(config)


def build_geocoder(
    config: GeocodingConfig,
    *,
    transport: httpx.BaseTransport | None = None,
) -> PlaceGeocoder:
    """Build the configured provider; disabled config yields a safe no-op."""

    if not config.enabled or config.provider == "none":
        return DisabledGeocoder()
    if transport is None:
        return _shared_geocoder(config)
    if config.provider == "photon":
        return PhotonGeocoder(config, transport=transport)
    if config.provider == "nominatim":
        return NominatimGeocoder(config, transport=transport)
    return DisabledGeocoder()


__all__ = [
    "DisabledGeocoder",
    "GeocoderHit",
    "GeocoderRequest",
    "GeocodingConfig",
    "GeocodingError",
    "GeocodingUnavailable",
    "NOMINATIM_DEFAULT_URL",
    "NominatimGeocoder",
    "OSM_ATTRIBUTION",
    "OSM_LICENSE",
    "PHOTON_DEFAULT_URL",
    "PhotonGeocoder",
    "PlaceGeocoder",
    "build_geocoder",
    "get_shared_geocoder",
]
