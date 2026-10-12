"""HTTP-only adapter; no Torch/CUDA dependencies belong in the business backend.

The persisted ``model_evidence`` JSON keeps the immutable full-image ranking
under ``rankings`` and may carry a bounded, optional list of unique local-crop
rankings under ``crops``. No database migration is required: the contract is
validated whenever it is read back, and older payloads without ``crops`` stay
valid.

Two further optional contracts live here:

* ``reference`` — the labeled-reference sidecar attached by an inference
  service that has an index loaded. It is retrieval-only diagnostics and never
  reorders countries, so it is preserved but never fused into the ranking.
* City ranking (``/predict-cities``) — a stateless, on-demand lookup. Every
  returned city is checked against the backend's own copy of the GeoNames
  catalog before it is shown, so the displayed identity and coordinates come
  from the data catalog rather than from the model service.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import logging
import math
import re
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from PIL import Image, ImageOps

from ..config import Settings
from .catalog_access import catalog_for_settings
from .city_catalog import CityCatalogError, load_city_index

MODEL_ID = "geolocal/StreetCLIP"
MODEL_REVISION = "e3561ba2ad9bf14c9efd6b0092607b8497efbfea"
PROMPT_VERSION = "streetclip-country-prompts-v1"
PROMPTS = ["a photo from {country}", "a street-level photo taken in {country}",
           "a photograph of a street scene in {country}"]

MAX_CROPS = 8
MAX_CROP_REGION_SIDE = 100_000
# The city contract mirrors the inference service: at most CITY_TOP_N rows are
# returned, and the reference sidecar is bounded at 10 matches / 20 region rows.
MAX_CITY_ROWS = 10
MAX_REFERENCE_MATCHES = 10
MAX_REFERENCE_REGIONS = 20
CITY_SCORE_TYPES = {"city_text_cosine", "city_reference_hybrid"}
CITY_STATUSES = {"ok", "no_cities"}
REFERENCE_AGGREGATIONS = {"max", "top2mean"}
# Coordinates are catalog-sourced; a tolerance this tight only absorbs JSON
# float round-tripping, not a genuinely different catalog.
COORDINATE_TOLERANCE = 1e-4
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REFERENCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
logger = logging.getLogger(__name__)


def _finite_cosine(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {label}")
    number = float(value)
    if not math.isfinite(number) or not -1.0 <= number <= 1.0:
        raise ValueError(f"invalid {label}")
    return number


def _finite_coordinate(value: object, label: str, bound: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"invalid {label}")
    number = float(value)
    if not math.isfinite(number) or not -bound <= number <= bound:
        raise ValueError(f"invalid {label}")
    return number


def _required_text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"invalid {label}")
    text = " ".join(value.split())
    if not text and not allow_empty:
        raise ValueError(f"invalid {label}")
    return text


def _sha256_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ValueError(f"invalid {label}")
    return value


def catalog_revision(countries: list[dict]) -> str:
    payload = {"prompt_template_version": PROMPT_VERSION, "prompt_templates": PROMPTS,
               "countries": [{"iso": c["iso"], "name": c["name"]}
                             for c in sorted(countries, key=lambda c: c["iso"])]}
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def clean_image(path: str | Path) -> bytes:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.info.clear()
        rgb.thumbnail((2048, 2048))
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=90)
        return output.getvalue()


def _region_key(region: dict) -> tuple[float, float, int]:
    return (round(float(region["x"]), 6), round(float(region["y"]), 6), int(region["size"]))


class StreetClipGateway:
    def __init__(self, settings: Settings, *, transport: httpx.BaseTransport | None = None):
        self.settings = settings
        self.transport = transport
        self._catalog_database = None

    def _catalog(self):
        """Return the request's read-only runtime catalog handle."""

        if self._catalog_database is None:
            self._catalog_database = catalog_for_settings(self.settings)
        return self._catalog_database

    def _endpoint(self) -> tuple[str, float]:
        """Validated ``(base_url, timeout)`` or ``ValueError``."""
        url = self.settings.streetclip_url.rstrip("/")
        parts = urlsplit(url)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.query or parts.fragment:
            raise ValueError("invalid StreetCLIP URL")
        timeout = self.settings.streetclip_timeout_seconds
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid timeout")
        return url, timeout

    def _headers(self) -> dict[str, str]:
        return ({"Authorization": f"Bearer {self.settings.streetclip_api_key}"}
                if self.settings.streetclip_api_key else {})

    def country_isos(self) -> set[str]:
        """ISO codes of the loaded country catalog (the only valid city scope)."""
        return {c["iso"] for c in self._catalog().countries()}

    def rank_image(self, path: str | Path) -> dict | None:
        if not self.settings.streetclip_enabled:
            return None
        try:
            url, timeout = self._endpoint()
            payload = clean_image(path)
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(timeout, connect=min(timeout, 3)),
                              follow_redirects=False, trust_env=False) as client:
                response = client.post(url + "/predict", headers=self._headers(),
                                       files={"file": ("image.jpg", payload, "image/jpeg")})
                response.raise_for_status()
                evidence = self.validate(response.json())
            return {**evidence, "schema_version": 1, "provider_status": "ok",
                    "prompt_version": PROMPT_VERSION, "image_sha256": hashlib.sha256(payload).hexdigest()}
        except Exception as exc:
            # Never log response bodies, headers, token, path or raw exception text.
            logger.warning("StreetCLIP unavailable; using rules (%s)", type(exc).__name__)
            return None

    def rank_cities(self, path: str | Path, iso: str) -> dict | None:
        """Rank ``iso``'s catalog cities for the image at ``path``.

        Returns the validated payload, or ``None`` whenever the model service
        is disabled, unreachable or answered outside the contract: city
        candidates are an optional enrichment, so every failure degrades to
        "no city candidates" and never to an error the user cannot act on.
        """
        if not self.settings.streetclip_enabled:
            return None
        requested = str(iso or "").strip().upper()
        try:
            url, timeout = self._endpoint()
            payload = clean_image(path)
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(timeout, connect=min(timeout, 3)),
                              follow_redirects=False, trust_env=False) as client:
                response = client.post(url + "/predict-cities", headers=self._headers(),
                                       files={"file": ("image.jpg", payload, "image/jpeg")},
                                       data={"country_iso": requested})
                response.raise_for_status()
                result = self.validate_city_ranking(response.json(), requested)
            return {**result, "schema_version": 1, "provider_status": "ok",
                    "image_sha256": hashlib.sha256(payload).hexdigest()}
        except Exception as exc:
            # Same discipline as rank_image: type only, never payload or path.
            logger.warning("StreetCLIP city ranking unavailable; none (%s)", type(exc).__name__)
            return None

    # ---- contract validation -------------------------------------------------
    def _validate_rankings(self, data: object) -> dict:
        countries = self._catalog().countries()
        allowed = {c["iso"] for c in countries}
        if not isinstance(data, dict) or data.get("model") != MODEL_ID or data.get("model_revision") != MODEL_REVISION:
            raise ValueError("unexpected model")
        if data.get("catalog_revision") != catalog_revision(countries):
            raise ValueError("catalog mismatch")
        rows = data.get("rankings")
        if not isinstance(rows, list) or not rows:
            raise ValueError("incomplete ranking")
        seen, rankings = set(), []
        for row in rows:
            if not isinstance(row, dict) or row.get("iso") not in allowed or row["iso"] in seen:
                raise ValueError("unknown or duplicate country")
            value = row.get("raw_cosine")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not -1 <= value <= 1:
                raise ValueError("invalid cosine")
            rankings.append({"iso": row["iso"], "raw_cosine": float(value)})
            seen.add(row["iso"])
        # The provider must return exactly the configured directory once, and
        # the directory size is dynamic (new countries may be added later).
        if seen != allowed:
            raise ValueError("incomplete ranking")
        ordered = sorted(rankings, key=lambda row: (-row["raw_cosine"], row["iso"]))
        if rankings != ordered:
            raise ValueError("unsorted ranking")
        if len(ordered) < 2:
            raise ValueError("incomplete ranking")
        margin = ordered[0]["raw_cosine"] - ordered[1]["raw_cosine"]
        reported = data.get("margin")
        if isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isfinite(reported) or abs(reported - margin) > 1e-6:
            raise ValueError("invalid margin")
        if type(data.get("uncertain")) is not bool or data["uncertain"] != (margin < 0.015):
            raise ValueError("invalid uncertainty flag")
        return {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                "catalog_revision": data["catalog_revision"], "rankings": rankings,
                "margin": margin, "uncertain": data["uncertain"]}

    @staticmethod
    def _validate_region(region: object) -> dict:
        if not isinstance(region, dict):
            raise ValueError("invalid crop region")
        x, y, size = region.get("x"), region.get("y"), region.get("size")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (x, y)):
            raise ValueError("invalid crop region")
        if not 0.0 <= float(x) <= 1.0 or not 0.0 <= float(y) <= 1.0:
            raise ValueError("invalid crop region")
        if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= MAX_CROP_REGION_SIDE:
            raise ValueError("invalid crop region")
        return {"x": float(x), "y": float(y), "size": int(size)}

    @staticmethod
    def _validate_sha256(value: object) -> str:
        if not isinstance(value, str) or not _SHA256_RE.match(value):
            raise ValueError("invalid image hash")
        return value

    def validate(self, data: object) -> dict:
        """Validate and normalise a persisted/provided evidence payload.

        Backwards compatible: ``crops`` and ``reference`` are optional. A
        corrupted or oversized ``crops`` list invalidates the payload (callers
        fall back to rules rather than trusting partial model output). A
        corrupted ``reference`` sidecar only loses the sidecar: it is
        retrieval-only diagnostics, and the country ranking it rides along with
        is validated independently above, so a malformed diagnostic must not
        cost the user the whole model prior. An absent ``reference`` key stays
        absent so consumers can tell "no labeled library configured" from
        "configured, but nothing matched".
        """
        base = self._validate_rankings(data)
        raw_crops = data.get("crops") if isinstance(data, dict) else None
        if raw_crops is not None:
            if not isinstance(raw_crops, list) or len(raw_crops) > MAX_CROPS:
                raise ValueError("invalid crop evidence")
            crops, regions, digests = [], set(), set()
            for raw in raw_crops:
                if not isinstance(raw, dict):
                    raise ValueError("invalid crop entry")
                crop = self._validate_rankings(raw)
                crop["region"] = self._validate_region(raw.get("region"))
                digest = raw.get("image_sha256")
                if digest is not None:
                    crop["image_sha256"] = self._validate_sha256(digest)
                key = _region_key(crop["region"])
                if key in regions or (crop.get("image_sha256") and crop["image_sha256"] in digests):
                    raise ValueError("duplicate crop evidence")
                regions.add(key)
                if crop.get("image_sha256"):
                    digests.add(crop["image_sha256"])
                crops.append(crop)
            base["crops"] = crops
        try:
            reference = self._validate_reference(
                data.get("reference") if isinstance(data, dict) else None
            )
        except ValueError as exc:
            # Every ``_validate_reference`` message is a fixed string, so the
            # reason is safe to log. Unlike crops this is not fatal: see the
            # docstring for why the ranking survives a broken diagnostic.
            logger.warning("StreetCLIP reference diagnostics dropped (%s)", exc)
            reference = None
        if reference is not None:
            base["reference"] = reference
        return base

    def _validate_reference(self, data: object) -> dict | None:
        """Validate the optional labeled-reference sidecar (``None`` = absent).

        Only metadata and cosines are kept: the sidecar describes labeled
        reference *images*, never the uploaded photo's location. Cross-revision
        checks pin it to this deployment's model and country catalog so a
        sidecar built elsewhere cannot be replayed here.
        """
        if data is None:
            return None
        if not isinstance(data, dict):
            raise ValueError("invalid reference payload")
        if data.get("model_revision") != MODEL_REVISION:
            raise ValueError("reference model mismatch")
        countries = self._catalog().countries()
        allowed = {c["iso"] for c in countries}
        if data.get("catalog_revision") != catalog_revision(countries):
            raise ValueError("reference catalog mismatch")
        if data.get("aggregation") not in REFERENCE_AGGREGATIONS:
            raise ValueError("invalid reference aggregation")

        raw_matches = data.get("matches")
        raw_regions = data.get("regions")
        raw_coverage = data.get("coverage")
        if not isinstance(raw_matches, list) or len(raw_matches) > MAX_REFERENCE_MATCHES:
            raise ValueError("invalid reference matches")
        if not isinstance(raw_regions, list) or len(raw_regions) > MAX_REFERENCE_REGIONS:
            raise ValueError("invalid reference regions")
        if not isinstance(raw_coverage, list):
            raise ValueError("invalid reference coverage")

        coverage = []
        for entry in raw_coverage:
            code = _required_text(entry, "reference coverage iso").upper()
            if len(code) != 2 or code not in allowed:
                raise ValueError("invalid reference coverage iso")
            coverage.append(code)
        if coverage != sorted(set(coverage)):
            raise ValueError("unsorted reference coverage")

        matches, seen_ids = [], set()
        for row in raw_matches:
            if not isinstance(row, dict):
                raise ValueError("invalid reference match")
            reference_id = _required_text(row.get("reference_id"), "reference id")
            if not _REFERENCE_ID_RE.match(reference_id) or reference_id in seen_ids:
                raise ValueError("invalid reference id")
            seen_ids.add(reference_id)
            iso = _required_text(row.get("iso"), "reference iso").upper()
            if iso not in allowed or iso not in coverage:
                raise ValueError("unknown reference country")
            city_id = row.get("city_id")
            if city_id is not None:
                city_id = _required_text(city_id, "reference city_id")
            matches.append({
                "reference_id": reference_id,
                "iso": iso,
                "region": _required_text(row.get("region"), "reference region"),
                "city": _required_text(row.get("city"), "reference city", allow_empty=True),
                "city_id": city_id,
                "place": _required_text(row.get("place"), "reference place"),
                "viewpoint": _required_text(row.get("viewpoint"), "reference viewpoint", allow_empty=True),
                "lat": _finite_coordinate(row.get("lat"), "reference latitude", 90.0),
                "lng": _finite_coordinate(row.get("lng"), "reference longitude", 180.0),
                "raw_cosine": _finite_cosine(row.get("raw_cosine"), "reference cosine"),
            })

        regions = []
        for row in raw_regions:
            if not isinstance(row, dict):
                raise ValueError("invalid reference region")
            iso = _required_text(row.get("iso"), "reference region iso").upper()
            if iso not in allowed or iso not in coverage:
                raise ValueError("unknown reference region country")
            prototype_count = row.get("prototype_count")
            support = row.get("support")
            if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
                   for value in (prototype_count, support)):
                raise ValueError("invalid reference region counters")
            regions.append({
                "iso": iso,
                "region": _required_text(row.get("region"), "reference region"),
                "city": _required_text(row.get("city"), "reference city", allow_empty=True),
                "raw_cosine": _finite_cosine(row.get("raw_cosine"), "reference region cosine"),
                "prototype_count": int(prototype_count),
                "support": int(support),
            })
        if matches != sorted(matches, key=lambda row: (-row["raw_cosine"], row["reference_id"])):
            raise ValueError("unsorted reference matches")
        if regions != sorted(regions, key=lambda row: (-row["raw_cosine"], row["iso"], row["region"], row["city"])):
            raise ValueError("unsorted reference regions")
        return {
            "index_revision": _sha256_text(data.get("index_revision"), "reference index revision"),
            "model_revision": MODEL_REVISION,
            "catalog_revision": data["catalog_revision"],
            "aggregation": data["aggregation"],
            "matches": matches,
            "regions": regions,
            "coverage": coverage,
        }

    def validate_city_ranking(self, data: object, iso: str) -> dict:
        """Validate one ``/predict-cities`` payload for ``iso``.

        City identity, names and coordinates are resolved against the backend's
        own GeoNames catalog: a row the catalog does not contain under this ISO
        (or whose coordinates disagree) invalidates the whole payload, because
        city markers are shown on a map and must never be fabricated. The
        published row shape carries ``city_id`` as a decimal string.
        """
        requested = str(iso or "").strip().upper()
        if len(requested) != 2 or not requested.isalpha():
            raise ValueError("invalid country iso")
        if not isinstance(data, dict):
            raise ValueError("invalid city payload")
        if str(data.get("country_iso") or "").strip().upper() != requested:
            raise ValueError("city country mismatch")
        status = data.get("status")
        if status not in CITY_STATUSES:
            raise ValueError("invalid city status")

        scope = data.get("scope")
        if not isinstance(scope, dict):
            raise ValueError("invalid city scope")
        limit = scope.get("city_limit_per_country")
        considered = scope.get("cities_considered")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError("invalid city limit")
        if isinstance(considered, bool) or not isinstance(considered, int) or considered < 0:
            raise ValueError("invalid city count")
        revision = scope.get("city_catalog_revision")
        if revision is None:
            # Only the "catalog not loaded" degradation reports no revision, and
            # it can only ever produce an empty result.
            if status != "no_cities":
                raise ValueError("missing city catalog revision")
        else:
            revision = _sha256_text(revision, "city catalog revision")
            try:
                expected_revision = self._catalog().revision("cities")
            except (CityCatalogError, ValueError) as exc:
                raise ValueError("city catalog unavailable") from exc
            if revision != expected_revision:
                raise ValueError("city catalog mismatch")

        rows = data.get("city_scores")
        if not isinstance(rows, list) or len(rows) > MAX_CITY_ROWS:
            raise ValueError("invalid city rows")
        if status == "ok" and not rows:
            raise ValueError("empty city ranking")
        if status == "no_cities" and rows:
            raise ValueError("unexpected city rows")
        if considered < len(rows):
            raise ValueError("invalid city count")

        try:
            allowed = load_city_index(self._catalog().path).get(requested, {})
        except CityCatalogError as exc:
            raise ValueError("city catalog unavailable") from exc

        cities, seen = [], set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("invalid city entry")
            city_id = row.get("city_id")
            if isinstance(city_id, bool) or not isinstance(city_id, int) or city_id <= 0:
                raise ValueError("invalid city id")
            if city_id in seen:
                raise ValueError("duplicate city")
            seen.add(city_id)
            if str(row.get("iso") or "").strip().upper() != requested:
                raise ValueError("city outside the requested country")
            entry = allowed.get(city_id)
            if entry is None:
                raise ValueError("unknown city")
            if _required_text(row.get("name"), "city name") != entry.name:
                raise ValueError("city name mismatch")
            lat = _finite_coordinate(row.get("lat"), "city latitude", 90.0)
            lng = _finite_coordinate(row.get("lng"), "city longitude", 180.0)
            if abs(lat - entry.lat) > COORDINATE_TOLERANCE or abs(lng - entry.lng) > COORDINATE_TOLERANCE:
                raise ValueError("city coordinate mismatch")
            score_type = row.get("score_type")
            if score_type not in CITY_SCORE_TYPES:
                raise ValueError("invalid city score type")
            reference_cosine = row.get("reference_cosine")
            if reference_cosine is not None:
                reference_cosine = _finite_cosine(reference_cosine, "city reference cosine")
            # A hybrid row must carry the reference cosine it was fused from,
            # and a text-only row must not claim one.
            if (score_type == "city_reference_hybrid") != (reference_cosine is not None):
                raise ValueError("inconsistent city score type")
            score = row.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError("invalid city score")
            score = float(score)
            if not 0.0 <= score <= 1.0:
                raise ValueError("invalid city score")
            cities.append({
                "city_id": str(city_id),
                "iso": requested,
                "name": entry.name,
                "region": entry.region,
                "lat": entry.lat,
                "lng": entry.lng,
                "raw_cosine": _finite_cosine(row.get("raw_cosine"), "city cosine"),
                "score": score,
                "score_type": score_type,
                "reference_cosine": reference_cosine,
            })
        if cities != sorted(cities, key=lambda row: (-row["score"], int(row["city_id"]))):
            raise ValueError("unsorted city ranking")
        return {
            "country_iso": requested,
            "scope": {"city_limit_per_country": int(limit),
                      "city_catalog_revision": revision,
                      "cities_considered": int(considered)},
            "status": status,
            "city_scores": cities,
        }

    def validate_crop(self, data: object, region: dict) -> dict:
        """Build a validated crop record (ranking + click region) for storage."""
        validated = self.validate(data)
        if validated.get("crops"):
            raise ValueError("crop evidence must not nest crops")
        # Crop rankings are additional country evidence; a reference sidecar
        # describes the full image, so it is never duplicated per crop.
        validated.pop("reference", None)
        record = {**validated, "region": self._validate_region(region)}
        digest = data.get("image_sha256") if isinstance(data, dict) else None
        if digest is not None:
            record["image_sha256"] = self._validate_sha256(digest)
        return record

    def with_crop(self, evidence: dict, crop: dict) -> dict | None:
        """Return a new evidence payload with ``crop`` appended.

        Duplicate clicks (same region) and repeated imagery (same cleaned-image
        hash) are ignored, and the crop list is bounded by :data:`MAX_CROPS`.
        Returns ``None`` when the crop adds no new evidence. The input payload
        (including the immutable full-image ``rankings``) is never mutated.
        """
        if not isinstance(evidence, dict) or not isinstance(crop, dict):
            return None
        crops = list(evidence.get("crops") or [])
        if len(crops) >= MAX_CROPS:
            return None
        if not isinstance(crop.get("region"), dict):
            return None
        key = _region_key(crop["region"])
        digest = crop.get("image_sha256")
        for existing in crops:
            if not isinstance(existing, dict) or "region" not in existing:
                return None
            if _region_key(existing["region"]) == key:
                return None
            if digest and digest == existing.get("image_sha256"):
                return None
        updated = copy.deepcopy(evidence)
        updated["crops"] = crops + [copy.deepcopy(crop)]
        return updated
