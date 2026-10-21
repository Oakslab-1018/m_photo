"""Bounded process-local cache for validated city model rankings.

Only raw, gateway-validated city payloads are stored. Session-specific clue
fusion, exclusions and candidate projection must be applied after every lookup.
The cache lock is held only for in-memory bookkeeping, never while hashing or
calling the model service.
"""
from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

_SUCCESS_TTL_SECONDS = 60.0 * 60.0
_FAILURE_TTL_SECONDS = 30.0
_MAX_RANKING_ENTRIES = 64
_MAX_PATH_FINGERPRINTS = 128


@dataclass(frozen=True)
class ImageFingerprint:
    """Identity for one immutable image file snapshot."""

    sha256: str
    size: int
    mtime_ns: int


@dataclass(frozen=True)
class CityRankingKey:
    """All inputs that can change one validated city ranking payload."""

    image_sha256: str
    image_size: int
    image_mtime_ns: int
    model_revision: str
    prompt_revision: str
    catalog_revision: str
    source_iso: str
    endpoint: str
    city_ids: tuple[int, ...] = ()


class CityRankingCache:
    """Thread-safe, bounded LRU/TTL cache with short negative entries."""

    def __init__(
        self,
        *,
        success_ttl_seconds: float = _SUCCESS_TTL_SECONDS,
        failure_ttl_seconds: float = _FAILURE_TTL_SECONDS,
        max_entries: int = _MAX_RANKING_ENTRIES,
        max_path_fingerprints: int = _MAX_PATH_FINGERPRINTS,
    ) -> None:
        self._success_ttl = max(0.0, float(success_ttl_seconds))
        self._failure_ttl = max(0.0, float(failure_ttl_seconds))
        self._max_entries = max(1, int(max_entries))
        self._max_path_fingerprints = max(1, int(max_path_fingerprints))
        self._lock = threading.Lock()
        # (expires_at, is_success, deep-copied payload)
        self._entries: OrderedDict[
            CityRankingKey, tuple[float, bool, dict[str, Any] | None]
        ] = OrderedDict()
        # normalized path -> ((size, mtime_ns), sha256)
        self._path_fingerprints: OrderedDict[
            str, tuple[tuple[int, int], str]
        ] = OrderedDict()

    def image_fingerprint(self, path: str | Path) -> ImageFingerprint | None:
        """Return a stable hash of the original image file.

        A digest is reused only for the exact same (size, mtime_ns) path
        snapshot. On a cache miss or stat change the original file is hashed
        outside the cache lock; model-service JPEG hashes are deliberately not
        treated as original-file identities.
        """

        image_path = Path(path)
        try:
            before = image_path.stat()
        except OSError:
            return None
        stat_key = (int(before.st_size), int(before.st_mtime_ns))
        normalized_path = os.path.realpath(str(image_path))

        with self._lock:
            cached = self._path_fingerprints.get(normalized_path)
            if cached is not None and cached[0] == stat_key:
                self._path_fingerprints.move_to_end(normalized_path)
                return ImageFingerprint(cached[1], stat_key[0], stat_key[1])

        digest = _hash_file(image_path)
        if digest is None:
            return None

        try:
            after = image_path.stat()
        except OSError:
            return None
        if (int(after.st_size), int(after.st_mtime_ns)) != stat_key:
            # The path changed while hashing; do not associate a stale digest.
            return None

        with self._lock:
            self._path_fingerprints[normalized_path] = (stat_key, digest)
            self._path_fingerprints.move_to_end(normalized_path)
            while len(self._path_fingerprints) > self._max_path_fingerprints:
                self._path_fingerprints.popitem(last=False)
        return ImageFingerprint(digest, stat_key[0], stat_key[1])

    def get(self, key: CityRankingKey) -> tuple[bool, dict[str, Any] | None]:
        """Return (hit, payload); a negative hit has payload=None."""

        now = time.monotonic()
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                return False, None
            expires_at, success, payload = item
            if expires_at <= now:
                self._entries.pop(key, None)
                return False, None
            self._entries.move_to_end(key)
            return True, deepcopy(payload) if success else None

    def put_success(self, key: CityRankingKey, payload: Mapping[str, Any]) -> None:
        self._put(key, True, dict(payload), self._success_ttl)

    def put_failure(self, key: CityRankingKey) -> None:
        # Short-lived only: a transient outage must not suppress retries forever.
        self._put(key, False, None, self._failure_ttl)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._path_fingerprints.clear()

    def _put(
        self,
        key: CityRankingKey,
        success: bool,
        payload: dict[str, Any] | None,
        ttl_seconds: float,
    ) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._entries[key] = (
                time.monotonic() + ttl_seconds,
                success,
                deepcopy(payload) if payload is not None else None,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)


def _hash_file(path: Path) -> str | None:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


CITY_RANKING_CACHE = CityRankingCache()


__all__ = [
    "CITY_RANKING_CACHE",
    "CityRankingCache",
    "CityRankingKey",
    "ImageFingerprint",
]
