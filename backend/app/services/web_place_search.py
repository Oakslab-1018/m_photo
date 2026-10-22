"""Bounded, optional web-search fallback for user place and OCR terms.

Only normalized user terms and an optional ISO country code are sent to the
configured provider. Search-result pages are never fetched, redirects are not
followed, and returned excerpts are treated as untrusted text.
"""
from __future__ import annotations

import html
import ipaddress
import json
import logging
import math
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

TAVILY_DEFAULT_URL = "https://api.tavily.com/search"
DEFAULT_TIMEOUT_SECONDS = 4.0
DEFAULT_CONNECT_TIMEOUT_SECONDS = 1.5
DEFAULT_MAX_RESULTS = 5
DEFAULT_CACHE_MAX_ENTRIES = 128
DEFAULT_SUCCESS_CACHE_TTL_SECONDS = 60 * 60
DEFAULT_EMPTY_CACHE_TTL_SECONDS = 5 * 60
DEFAULT_FAILURE_COOLDOWN_SECONDS = 30.0

_MAX_RESPONSE_BYTES = 512 * 1024
_MAX_QUERY_TERMS = 2
_MAX_TERM_LENGTH = 60
_MAX_TITLE_LENGTH = 240
_MAX_EXCERPT_LENGTH = 800
_MAX_URL_LENGTH = 2048
_PROVIDERS = frozenset({"none", "tavily", "searxng"})
_SKIPPED_HTML_TAGS = frozenset({"script", "style", "noscript", "template"})
_PRIVATE_HOST_SUFFIXES = (".localhost", ".local", ".internal", ".home", ".lan")
_TAG_FALLBACK_RE = re.compile(r"<[^>]{0,4096}>")

_NOTICE_DISABLED = "网页搜索未启用。"
_NOTICE_INVALID_CONFIG = "网页搜索配置无效，已停用。"
_NOTICE_UNAVAILABLE = "网页搜索暂时不可用，请稍后重试。"
_NOTICE_NO_KEY = "网页搜索未配置访问密钥，暂不可用。"
_NOTICE_NO_BASE_URL = "网页搜索地址未配置，暂不可用。"
_NOTICE_EMPTY = "未找到相关网页结果。"
_NOTICE_UNTRUSTED = "网页摘录来自第三方来源，请自行核实。"


@dataclass(frozen=True)
class WebPlaceSearchHit:
    title: str
    url: str
    excerpt: str
    provider: str
    query: str
    fetched_at: str


@dataclass(frozen=True)
class WebPlaceSearchResult:
    status: str
    hits: tuple[WebPlaceSearchHit, ...] = ()
    notice: str | None = None


@dataclass(frozen=True)
class WebSearchConfig:
    """Standalone web-search policy read directly from the environment.

    ``transport`` accepts an ``httpx.MockTransport`` for manual checks. Tests
    and evaluations are disabled by default unless
    ``M_PHOTO_WEB_SEARCH_FORCE_ENABLED=1`` is set, matching the geocoder policy.
    """

    provider: str = "none"
    api_key: str = field(default="", repr=False)
    base_url: str = ""
    enabled: bool = True
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    connect_timeout_seconds: float = DEFAULT_CONNECT_TIMEOUT_SECONDS
    max_results: int = DEFAULT_MAX_RESULTS
    cache_max_entries: int = DEFAULT_CACHE_MAX_ENTRIES
    success_cache_ttl_seconds: float = DEFAULT_SUCCESS_CACHE_TTL_SECONDS
    empty_cache_ttl_seconds: float = DEFAULT_EMPTY_CACHE_TTL_SECONDS
    failure_cooldown_seconds: float = DEFAULT_FAILURE_COOLDOWN_SECONDS
    transport: httpx.BaseTransport | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        provider = str(self.provider or "none").strip().lower()
        if provider not in _PROVIDERS:
            raise ValueError("web search provider must be none, tavily, or searxng")

        api_key = str(self.api_key or "").strip()
        if "\r" in api_key or "\n" in api_key:
            raise ValueError("web search API key must not contain line breaks")

        base_url = str(self.base_url or "").strip()
        if provider == "tavily" and not base_url:
            base_url = TAVILY_DEFAULT_URL

        timeout = float(self.timeout_seconds)
        connect_timeout = float(self.connect_timeout_seconds)
        success_ttl = float(self.success_cache_ttl_seconds)
        empty_ttl = float(self.empty_cache_ttl_seconds)
        cooldown = float(self.failure_cooldown_seconds)
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("web search timeout must be positive and finite")
        if connect_timeout <= 0 or not math.isfinite(connect_timeout):
            raise ValueError("web search connect timeout must be positive and finite")
        if success_ttl <= 0 or not math.isfinite(success_ttl):
            raise ValueError("web search success cache TTL must be positive and finite")
        if empty_ttl <= 0 or not math.isfinite(empty_ttl):
            raise ValueError("web search empty cache TTL must be positive and finite")
        if cooldown < 0 or not math.isfinite(cooldown):
            raise ValueError("web search failure cooldown must be non-negative and finite")

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "timeout_seconds", timeout)
        object.__setattr__(self, "connect_timeout_seconds", min(timeout, connect_timeout))
        object.__setattr__(
            self, "max_results", min(DEFAULT_MAX_RESULTS, max(1, int(self.max_results)))
        )
        object.__setattr__(
            self,
            "cache_max_entries",
            min(DEFAULT_CACHE_MAX_ENTRIES, max(0, int(self.cache_max_entries))),
        )
        object.__setattr__(self, "success_cache_ttl_seconds", success_ttl)
        object.__setattr__(self, "empty_cache_ttl_seconds", empty_ttl)
        object.__setattr__(self, "failure_cooldown_seconds", cooldown)

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "WebSearchConfig":
        source = dict(os.environ) if env is None else dict(env)

        def get(name: str) -> str | None:
            value = source.get(name)
            if value is None or not str(value).strip():
                return None
            return str(value).strip()

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

        provider = (get("M_PHOTO_WEB_SEARCH_PROVIDER") or "none").lower()
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
        force_enabled = truthy(get("M_PHOTO_WEB_SEARCH_FORCE_ENABLED"))
        enabled = provider != "none" and (force_enabled or not eval_mode)
        timeout = number(
            get("M_PHOTO_WEB_SEARCH_TIMEOUT_SECONDS"),
            DEFAULT_TIMEOUT_SECONDS,
            0.25,
            15.0,
        )
        connect_timeout = min(
            timeout,
            number(
                get("M_PHOTO_WEB_SEARCH_CONNECT_TIMEOUT_SECONDS"),
                DEFAULT_CONNECT_TIMEOUT_SECONDS,
                0.25,
                timeout,
            ),
        )
        return cls(
            provider=provider,
            api_key=get("M_PHOTO_WEB_SEARCH_API_KEY") or "",
            base_url=get("M_PHOTO_WEB_SEARCH_BASE_URL") or "",
            enabled=enabled,
            timeout_seconds=timeout,
            connect_timeout_seconds=connect_timeout,
        )


WebPlaceSearchConfig = WebSearchConfig


class _SearchFailure(RuntimeError):
    def __init__(self, reason: str, *, status_code: int | None = None) -> None:
        super().__init__(reason)
        self.reason = reason
        self.status_code = status_code


class _ResultCache:
    def __init__(self, max_entries: int) -> None:
        self._max_entries = max(0, int(max_entries))
        self._values: dict[str, tuple[float, WebPlaceSearchResult]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> WebPlaceSearchResult | None:
        now = time.monotonic()
        with self._lock:
            item = self._values.get(key)
            if item is None:
                return None
            expires_at, result = item
            if expires_at <= now:
                self._values.pop(key, None)
                return None
            return result

    def put(self, key: str, result: WebPlaceSearchResult, *, ttl_seconds: float) -> None:
        if self._max_entries <= 0:
            return
        with self._lock:
            if key not in self._values and len(self._values) >= self._max_entries:
                oldest = min(self._values, key=lambda item: self._values[item][0])
                self._values.pop(oldest, None)
            self._values[key] = (time.monotonic() + ttl_seconds, result)


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        normalized = tag.casefold()
        if normalized in _SKIPPED_HTML_TAGS:
            self._skip_depth += 1
        elif self._skip_depth == 0:
            self._parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del attrs
        if self._skip_depth == 0 and tag.casefold() not in _SKIPPED_HTML_TAGS:
            self._parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        normalized = tag.casefold()
        if normalized in _SKIPPED_HTML_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif self._skip_depth == 0:
            self._parts.append(" ")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def text(self) -> str:
        return "".join(self._parts)


def _clean_controls(value: object) -> str:
    return "".join(
        character
        for character in str(value or "")
        if character in "\t\n\r" or character.isprintable()
    )


def _clean_html_text(value: object, limit: int) -> str:
    raw = _clean_controls(value).strip()
    if not raw:
        return ""
    raw = raw[: max(4096, limit * 12)]
    parser = _HTMLTextExtractor()
    try:
        parser.feed(raw)
        parser.close()
        cleaned = parser.text()
    except Exception:
        cleaned = html.unescape(_TAG_FALLBACK_RE.sub(" ", raw))
    return " ".join(cleaned.split())[:limit].strip()


def _normalize_terms(terms: Sequence[str]) -> tuple[str, ...]:
    if isinstance(terms, str):
        iterator = iter((terms,))
    else:
        try:
            iterator = iter(terms)
        except TypeError:
            return ()

    normalized: list[str] = []
    seen: set[str] = set()
    try:
        for value in iterator:
            text = " ".join(_clean_controls(value).split())
            if not text or len(text) > _MAX_TERM_LENGTH:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(text)
            if len(normalized) >= _MAX_QUERY_TERMS:
                break
    except RuntimeError:
        pass
    return tuple(normalized)


def _normalize_country(value: object) -> str | None:
    text = " ".join(_clean_controls(value).split()).upper()
    if len(text) == 2 and text.isascii() and text.isalpha():
        return text
    return None


def _safe_result_url(value: object) -> str | None:
    text = str(value or "").strip()
    if not text or len(text) > _MAX_URL_LENGTH:
        return None
    if any(character.isspace() or ord(character) < 32 or character == "\\" for character in text):
        return None
    try:
        parts = urlsplit(text)
        hostname = parts.hostname
        _ = parts.port
    except ValueError:
        return None
    if parts.scheme.lower() not in {"http", "https"} or not hostname:
        return None
    if parts.username is not None or parts.password is not None:
        return None

    host = hostname.rstrip(".").casefold()
    if not host or host == "localhost" or host.endswith(_PRIVATE_HOST_SUFFIXES):
        return None
    try:
        address = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        if "." not in host and ":" not in host:
            return None
    else:
        if not address.is_global:
            return None
    return text


def _fetched_at() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _validate_base_url(value: str) -> str:
    # Provider endpoints are administrator-trusted and may be private or
    # self-hosted (notably SearXNG). Search-result URLs are validated
    # separately by _safe_result_url and must remain public HTTP(S).
    text = str(value or "").strip()
    if not text or len(text) > _MAX_URL_LENGTH:
        raise _SearchFailure("invalid_config")
    if any(character.isspace() or ord(character) < 32 for character in text):
        raise _SearchFailure("invalid_config")
    try:
        parts = urlsplit(text)
        _ = parts.port
    except ValueError:
        raise _SearchFailure("invalid_config") from None
    if (
        parts.scheme.lower() not in {"http", "https"}
        or not parts.hostname
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise _SearchFailure("invalid_config")
    return text.rstrip("/")


def _searxng_endpoint(base_url: str) -> str:
    base = _validate_base_url(base_url)
    path = urlsplit(base).path.rstrip("/")
    if path.endswith("/search"):
        return base
    return f"{base}/search"


class WebPlaceSearch:
    def __init__(
        self,
        config: WebSearchConfig,
        *,
        disabled_notice: str = _NOTICE_DISABLED,
    ) -> None:
        self.config = config
        self._disabled_notice = disabled_notice
        self._cache = _ResultCache(config.cache_max_entries)
        self._cooldown_lock = threading.Lock()
        self._cooldown_until = 0.0
        self._client = httpx.Client(
            transport=config.transport,
            timeout=httpx.Timeout(
                config.timeout_seconds,
                connect=config.connect_timeout_seconds,
            ),
            follow_redirects=False,
            trust_env=False,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        )

    def search(
        self,
        terms: Sequence[str],
        *,
        country_iso: str | None = None,
    ) -> WebPlaceSearchResult:
        if not self.config.enabled or self.config.provider == "none":
            return WebPlaceSearchResult(status="disabled", notice=self._disabled_notice)

        normalized_terms = _normalize_terms(terms)
        if not normalized_terms:
            return WebPlaceSearchResult(status="empty", notice=_NOTICE_EMPTY)

        country = _normalize_country(country_iso)
        query = " ".join((*normalized_terms, country) if country else normalized_terms)

        config_notice = self._configuration_notice()
        if config_notice is not None:
            return WebPlaceSearchResult(status="unavailable", notice=config_notice)

        cache_key = f"{self.config.provider}\0{query.casefold()}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        if self._is_cooling_down():
            return WebPlaceSearchResult(status="unavailable", notice=_NOTICE_UNAVAILABLE)

        try:
            if self.config.provider == "tavily":
                payload = self._search_tavily(query)
            else:
                payload = self._search_searxng(query)
            hits = self._parse_hits(payload, query=query)
        except _SearchFailure as exc:
            self._mark_failure()
            logger.warning(
                "Web place search failed provider=%s reason=%s status=%s",
                self.config.provider,
                exc.reason,
                exc.status_code,
            )
            return WebPlaceSearchResult(status="unavailable", notice=_NOTICE_UNAVAILABLE)
        except Exception:
            self._mark_failure()
            logger.warning(
                "Web place search failed provider=%s reason=unexpected_error",
                self.config.provider,
            )
            return WebPlaceSearchResult(status="unavailable", notice=_NOTICE_UNAVAILABLE)

        self._clear_failure()
        if hits:
            result = WebPlaceSearchResult(status="ok", hits=hits, notice=_NOTICE_UNTRUSTED)
            ttl = self.config.success_cache_ttl_seconds
        else:
            result = WebPlaceSearchResult(status="empty", notice=_NOTICE_EMPTY)
            ttl = self.config.empty_cache_ttl_seconds
        self._cache.put(cache_key, result, ttl_seconds=ttl)
        return result

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "WebPlaceSearch":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        del exc_type, exc, traceback
        self.close()

    def _configuration_notice(self) -> str | None:
        if self.config.provider == "tavily" and not self.config.api_key:
            return _NOTICE_NO_KEY
        if self.config.provider == "searxng" and not self.config.base_url:
            return _NOTICE_NO_BASE_URL
        try:
            _validate_base_url(self.config.base_url)
        except _SearchFailure:
            return _NOTICE_UNAVAILABLE
        return None

    def _search_tavily(self, query: str) -> object:
        return self._request_json(
            "POST",
            _validate_base_url(self.config.base_url),
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self.config.api_key}",
                "User-Agent": "MPhoto/0.1 (bounded web place search)",
            },
            json_payload={
                "query": query,
                "search_depth": "basic",
                "max_results": self.config.max_results,
                "include_answer": False,
                "include_raw_content": False,
            },
        )

    def _search_searxng(self, query: str) -> object:
        return self._request_json(
            "GET",
            _searxng_endpoint(self.config.base_url),
            headers={
                "Accept": "application/json",
                "User-Agent": "MPhoto/0.1 (bounded web place search)",
            },
            params={"q": query, "format": "json", "categories": "general"},
        )

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        params: Mapping[str, object] | None = None,
        json_payload: object | None = None,
    ) -> object:
        deadline = time.monotonic() + self.config.timeout_seconds
        remaining = max(0.001, deadline - time.monotonic())
        request_timeout = httpx.Timeout(
            remaining,
            connect=min(self.config.connect_timeout_seconds, remaining),
            read=remaining,
            write=remaining,
            pool=remaining,
        )
        try:
            with self._client.stream(
                method,
                url,
                params=params,
                json=json_payload,
                headers=headers,
                timeout=request_timeout,
            ) as response:
                status_code = response.status_code
                if 300 <= status_code < 400:
                    raise _SearchFailure("unexpected_redirect", status_code=status_code)
                if not 200 <= status_code < 300:
                    raise _SearchFailure("http_status", status_code=status_code)

                declared_length = response.headers.get("content-length")
                if declared_length:
                    try:
                        if int(declared_length) > _MAX_RESPONSE_BYTES:
                            raise _SearchFailure("response_too_large")
                    except ValueError:
                        pass

                chunks: list[bytes] = []
                total = 0
                for chunk in response.iter_bytes():
                    total += len(chunk)
                    if total > _MAX_RESPONSE_BYTES:
                        raise _SearchFailure("response_too_large")
                    if time.monotonic() > deadline:
                        raise _SearchFailure("timeout")
                    chunks.append(chunk)

            if time.monotonic() > deadline:
                raise _SearchFailure("timeout")
            return json.loads(b"".join(chunks))
        except _SearchFailure:
            raise
        except httpx.ConnectTimeout:
            raise _SearchFailure("connect_timeout") from None
        except httpx.ReadTimeout:
            raise _SearchFailure("read_timeout") from None
        except httpx.TimeoutException:
            raise _SearchFailure("timeout") from None
        except httpx.ConnectError:
            raise _SearchFailure("connect_error") from None
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise _SearchFailure("invalid_json") from None
        except httpx.HTTPError:
            raise _SearchFailure("transport_error") from None
        except Exception:
            raise _SearchFailure("unexpected_error") from None

    def _parse_hits(
        self,
        payload: object,
        *,
        query: str,
    ) -> tuple[WebPlaceSearchHit, ...]:
        if not isinstance(payload, dict):
            raise _SearchFailure("malformed_response")
        raw_results = payload.get("results")
        if not isinstance(raw_results, list):
            raise _SearchFailure("malformed_response")

        fetched_at = _fetched_at()
        hits: list[WebPlaceSearchHit] = []
        seen_urls: set[str] = set()
        for item in raw_results:
            if len(hits) >= self.config.max_results:
                break
            if not isinstance(item, dict):
                continue
            url = _safe_result_url(item.get("url"))
            if url is None:
                continue
            url_key = url.casefold()
            if url_key in seen_urls:
                continue
            seen_urls.add(url_key)

            hostname = urlsplit(url).hostname or ""
            title = _clean_html_text(item.get("title"), _MAX_TITLE_LENGTH)
            if not title:
                title = _clean_html_text(hostname, _MAX_TITLE_LENGTH)
            excerpt = _clean_html_text(item.get("content"), _MAX_EXCERPT_LENGTH)
            hits.append(
                WebPlaceSearchHit(
                    title=title,
                    url=url,
                    excerpt=excerpt,
                    provider=self.config.provider,
                    query=query,
                    fetched_at=fetched_at,
                )
            )
        return tuple(hits)

    def _is_cooling_down(self) -> bool:
        with self._cooldown_lock:
            return self._cooldown_until > time.monotonic()

    def _mark_failure(self) -> None:
        with self._cooldown_lock:
            self._cooldown_until = max(
                self._cooldown_until,
                time.monotonic() + self.config.failure_cooldown_seconds,
            )

    def _clear_failure(self) -> None:
        with self._cooldown_lock:
            self._cooldown_until = 0.0


_SINGLETON_LOCK = threading.Lock()
_WEB_PLACE_SEARCH: WebPlaceSearch | None = None


def get_web_place_search() -> WebPlaceSearch:
    """Return the process-wide search service built from current environment."""

    global _WEB_PLACE_SEARCH
    with _SINGLETON_LOCK:
        if _WEB_PLACE_SEARCH is None:
            try:
                config = WebSearchConfig.from_env()
                _WEB_PLACE_SEARCH = WebPlaceSearch(config)
            except Exception:
                # Invalid environment configuration must not break feature
                # construction. Do not include provider values or secrets.
                logger.warning("Web place search disabled reason=invalid_config")
                _WEB_PLACE_SEARCH = WebPlaceSearch(
                    WebSearchConfig(provider="none", enabled=False),
                    disabled_notice=_NOTICE_INVALID_CONFIG,
                )
        return _WEB_PLACE_SEARCH


__all__ = [
    "TAVILY_DEFAULT_URL",
    "WebPlaceSearch",
    "WebPlaceSearchConfig",
    "WebPlaceSearchHit",
    "WebPlaceSearchResult",
    "WebSearchConfig",
    "get_web_place_search",
]
