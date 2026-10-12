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
MAX_SCENE_GROUPS = 6
MAX_SCENE_CANDIDATES = 2
MAX_SCENE_LABEL_LENGTH = 120
SCENE_GROUP_VALUES = {
    "road": frozenset({
        "arterial_road", "local_street", "residential_lane", "highway",
        "unpaved_road", "pedestrian_way",
    }),
    "building": frozenset({
        "high_rise", "mid_rise", "low_rise", "historic_masonry",
        "commercial_storefront", "industrial_warehouse",
    }),
    "vegetation": frozenset({
        "street_trees", "dense_green", "sparse_green", "grass_lawn",
        "flowering_plants", "dry_vegetation",
    }),
    "terrain": frozenset({
        "flat_ground", "sloped_ground", "hilly", "mountainous", "valley",
        "coastal_lowland",
    }),
    "water": frozenset({
        "no_visible_water", "river", "lake", "sea_ocean", "canal", "pond_fountain",
    }),
    "vehicle": frozenset({
        "cars", "motorcycles_scooters", "bicycles", "buses", "trucks",
        "no_visible_vehicles",
    }),
}
SCENE_DIMENSIONS = frozenset(SCENE_GROUP_VALUES)
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
_REGION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,199}$")
_SCENE_VALUE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
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


def build_inference_notice(
    settings: Settings,
    evidence: object | None,
    *,
    validation_failed: bool = False,
    has_trusted_evidence: bool = False,
) -> str | None:
    """Return a user-safe notice when this session has no usable model prior.

    The wording is session-scoped on purpose: an old fallback session may be
    opened after the local model has recovered, so it must not claim the
    current service is still disconnected.
    """
    if not settings.streetclip_enabled:
        return None
    if evidence is not None and not validation_failed:
        return None
    if validation_failed:
        notice = (
            "本会话保存的本地模型结果无效或与当前地点目录不一致，已停止使用；"
            "本次仅依据可确认的文字、地名等可信线索。"
        )
    else:
        notice = (
            "本会话未取得可用的本地模型结果，本次仅依据可确认的文字、地名等可信线索。"
            "如需模型排序，请重新分析或重新上传图片。"
        )
    if not has_trusted_evidence:
        notice += (
            " 暂无可确认的文字线索，暂时无法给出候选；"
            "建议重新分析图片或补充招牌文字、地名、通行方向等信息。"
        )
    return notice


def clean_image(path: str | Path) -> bytes:
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        rgb.info.clear()
        rgb.thumbnail((2048, 2048))
        output = io.BytesIO()
        rgb.save(output, format="JPEG", quality=90)
        return output.getvalue()


def _region_key(region: dict):
    """Stable crop-region identity with legacy click-region compatibility."""
    region_id = region.get("region_id")
    if isinstance(region_id, str) and region_id:
        return ("region_id", region_id)
    bbox = region.get("bbox_norm")
    if isinstance(bbox, dict):
        return (
            "bbox_norm",
            round(float(bbox["x"]), 6),
            round(float(bbox["y"]), 6),
            round(float(bbox["w"]), 6),
            round(float(bbox["h"]), 6),
            str(region.get("purpose") or ""),
        )
    return (
        "legacy_click",
        round(float(region["x"]), 6),
        round(float(region["y"]), 6),
        int(region["size"]),
    )


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

    def describe_region(self, png: bytes) -> dict:
        """Call the optional StreetCLIP scene-region endpoint.

        Scene hypotheses are display-only diagnostics. Missing endpoint,
        provider outage or malformed output all degrade to an explicit
        ``unavailable`` result so local OCR/quality analysis still succeeds.
        """
        base = {
            "provider": "streetclip",
            "model_revision": MODEL_REVISION,
            "enabled": bool(self.settings.streetclip_enabled),
            "available": False,
            "scene_groups": [],
        }
        if not self.settings.streetclip_enabled:
            return {**base, "state": "disabled"}
        if not isinstance(png, (bytes, bytearray)) or not png:
            return {**base, "state": "unavailable"}
        payload = bytes(png)
        if len(payload) > 16 * 1024 * 1024:
            return {**base, "state": "unavailable"}
        try:
            url, timeout = self._endpoint()
            with httpx.Client(
                transport=self.transport,
                timeout=httpx.Timeout(timeout, connect=min(timeout, 3)),
                follow_redirects=False,
                trust_env=False,
            ) as client:
                response = client.post(
                    url + "/analyze-region",
                    headers=self._headers(),
                    files={"file": ("region.png", payload, "image/png")},
                )
                response.raise_for_status()
                validated = self.validate_scene_region(
                    response.json(), hashlib.sha256(payload).hexdigest()
                )
            return {**base, **validated, "available": True, "state": "ok"}
        except Exception as exc:
            logger.warning("StreetCLIP scene region unavailable (%s)", type(exc).__name__)
            return {**base, "state": "unavailable"}

    def validate_scene_region(self, data: object, image_sha256: str) -> dict:
        """Validate the bounded scene-region response contract."""
        expected_hash = self._validate_sha256(image_sha256)
        if not isinstance(data, dict) or data.get("status") != "ok":
            raise ValueError("invalid scene response")
        if data.get("model_revision") != MODEL_REVISION:
            raise ValueError("scene model mismatch")
        if data.get("image_sha256") != expected_hash:
            raise ValueError("scene image mismatch")
        raw_groups = data.get("scene_groups")
        if not isinstance(raw_groups, list) or len(raw_groups) != len(SCENE_DIMENSIONS):
            raise ValueError("invalid scene groups")

        groups = []
        seen_dimensions = set()
        for raw in raw_groups:
            if not isinstance(raw, dict):
                raise ValueError("invalid scene group")
            dimension = _required_text(raw.get("dimension"), "scene dimension").lower()
            if dimension not in SCENE_DIMENSIONS or dimension in seen_dimensions:
                raise ValueError("invalid scene dimension")
            seen_dimensions.add(dimension)
            raw_candidates = raw.get("candidates")
            if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= MAX_SCENE_CANDIDATES:
                raise ValueError("invalid scene candidates")
            candidates = []
            seen_values = set()
            for candidate in raw_candidates:
                if not isinstance(candidate, dict):
                    raise ValueError("invalid scene candidate")
                value = _required_text(candidate.get("value"), "scene value")
                if (
                    not _SCENE_VALUE_RE.match(value)
                    or value not in SCENE_GROUP_VALUES.get(dimension, frozenset())
                    or value in seen_values
                ):
                    raise ValueError("invalid scene value")
                seen_values.add(value)
                label = _required_text(candidate.get("label"), "scene label")
                if len(label) > MAX_SCENE_LABEL_LENGTH:
                    raise ValueError("invalid scene label")
                candidates.append({
                    "value": value,
                    "label": label,
                    "raw_cosine": _finite_cosine(candidate.get("raw_cosine"), "scene cosine"),
                })
            candidates.sort(key=lambda row: (-row["raw_cosine"], row["value"]))
            margin = raw.get("margin")
            if isinstance(margin, bool) or not isinstance(margin, (int, float)):
                raise ValueError("invalid scene margin")
            margin = float(margin)
            if not math.isfinite(margin) or margin < 0.0 or margin > 2.0:
                raise ValueError("invalid scene margin")
            uncertain = raw.get("uncertain")
            if type(uncertain) is not bool:
                raise ValueError("invalid scene uncertainty")
            groups.append({
                "dimension": dimension,
                "candidates": candidates,
                "margin": margin,
                "uncertain": uncertain,
            })
        return {
            "model_revision": MODEL_REVISION,
            "image_sha256": expected_hash,
            "scene_groups": groups,
        }

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
        validated = {"x": float(x), "y": float(y), "size": int(size)}

        region_id = region.get("region_id")
        if region_id is not None:
            region_id = _required_text(region_id, "crop region id")
            if not _REGION_ID_RE.match(region_id):
                raise ValueError("invalid crop region id")
            validated["region_id"] = region_id

        purpose = region.get("purpose")
        if purpose is not None:
            if purpose not in {"auto", "ocr", "scene"}:
                raise ValueError("invalid crop purpose")
            validated["purpose"] = purpose

        bbox = region.get("bbox_norm")
        if bbox is not None:
            if not isinstance(bbox, dict):
                raise ValueError("invalid crop bbox")
            bx, by, bw, bh = (bbox.get("x"), bbox.get("y"), bbox.get("w"), bbox.get("h"))
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in (bx, by, bw, bh)
            ):
                raise ValueError("invalid crop bbox")
            bx, by, bw, bh = float(bx), float(by), float(bw), float(bh)
            if bx < 0.0 or by < 0.0 or bw <= 0.0 or bh <= 0.0:
                raise ValueError("invalid crop bbox")
            if bx + bw > 1.0 + 1e-9 or by + bh > 1.0 + 1e-9:
                raise ValueError("invalid crop bbox")
            rounded_x, rounded_y = round(bx, 6), round(by, 6)
            rounded_w = min(max(1e-6, round(bw, 6)), max(1e-6, round(1.0 - rounded_x, 6)))
            rounded_h = min(max(1e-6, round(bh, 6)), max(1e-6, round(1.0 - rounded_y, 6)))
            validated["bbox_norm"] = {
                "x": rounded_x,
                "y": rounded_y,
                "w": rounded_w,
                "h": rounded_h,
            }

        ranking_applied = region.get("ranking_applied")
        if ranking_applied is not None:
            if type(ranking_applied) is not bool:
                raise ValueError("invalid crop ranking flag")
            validated["ranking_applied"] = ranking_applied
        return validated

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
                region_identity = bool(
                    crop["region"].get("region_id") or crop["region"].get("bbox_norm")
                )
                duplicate_digest = bool(
                    not region_identity
                    and crop.get("image_sha256")
                    and crop["image_sha256"] in digests
                )
                if key in regions or duplicate_digest:
                    raise ValueError("duplicate crop evidence")
                regions.add(key)
                if crop.get("image_sha256") and not region_identity:
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

        Legacy click crops deduplicate repeated regions and imagery. Region-id
        crops keep explicit ROI provenance and replace an older crop for the
        same ROI when a fresh provider result is available. The crop list is
        bounded by :data:`MAX_CROPS`. Returns ``None`` when the crop adds no
        new evidence. The input payload (including the immutable full-image
        ``rankings``) is never mutated.
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
        region_identity = bool(
            crop["region"].get("region_id") or crop["region"].get("bbox_norm")
        )
        replace_index = None
        for index, existing in enumerate(crops):
            if not isinstance(existing, dict) or "region" not in existing:
                return None
            if _region_key(existing["region"]) == key:
                existing_region_identity = bool(
                    existing["region"].get("region_id") or existing["region"].get("bbox_norm")
                )
                if region_identity and existing_region_identity:
                    replace_index = index
                    break
                return None
            if not region_identity and digest and digest == existing.get("image_sha256"):
                return None
        updated = copy.deepcopy(evidence)
        if replace_index is None:
            updated["crops"] = crops + [copy.deepcopy(crop)]
        else:
            updated["crops"] = (
                crops[:replace_index] + [copy.deepcopy(crop)] + crops[replace_index + 1:]
            )
        return updated
