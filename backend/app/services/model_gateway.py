"""HTTP-only adapter; no Torch/CUDA dependencies belong in the business backend.

The persisted ``model_evidence`` JSON keeps the immutable full-image ranking
under ``rankings`` and may carry a bounded, optional list of unique local-crop
rankings under ``crops``. No database migration is required: the contract is
validated whenever it is read back, and older payloads without ``crops`` stay
valid.
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

MODEL_ID = "geolocal/StreetCLIP"
MODEL_REVISION = "e3561ba2ad9bf14c9efd6b0092607b8497efbfea"
PROMPT_VERSION = "streetclip-country-prompts-v1"
PROMPTS = ["a photo from {country}", "a street-level photo taken in {country}",
           "a photograph of a street scene in {country}"]

MAX_CROPS = 8
MAX_CROP_REGION_SIDE = 100_000
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
logger = logging.getLogger(__name__)


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

    def rank_image(self, path: str | Path) -> dict | None:
        if not self.settings.streetclip_enabled:
            return None
        try:
            url = self.settings.streetclip_url.rstrip("/")
            parts = urlsplit(url)
            if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.query or parts.fragment:
                raise ValueError("invalid StreetCLIP URL")
            timeout = self.settings.streetclip_timeout_seconds
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError("invalid timeout")
            payload = clean_image(path)
            headers = ({"Authorization": f"Bearer {self.settings.streetclip_api_key}"}
                       if self.settings.streetclip_api_key else {})
            with httpx.Client(transport=self.transport, timeout=httpx.Timeout(timeout, connect=min(timeout, 3)),
                              follow_redirects=False, trust_env=False) as client:
                response = client.post(url + "/predict", headers=headers,
                                       files={"file": ("image.jpg", payload, "image/jpeg")})
                response.raise_for_status()
                evidence = self.validate(response.json())
            return {**evidence, "schema_version": 1, "provider_status": "ok",
                    "prompt_version": PROMPT_VERSION, "image_sha256": hashlib.sha256(payload).hexdigest()}
        except Exception as exc:
            # Never log response bodies, headers, token, path or raw exception text.
            logger.warning("StreetCLIP unavailable; using rules (%s)", type(exc).__name__)
            return None

    # ---- contract validation -------------------------------------------------
    def _validate_rankings(self, data: object) -> dict:
        countries = json.loads(self.settings.countries_path.read_text(encoding="utf-8"))
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

        Backwards compatible: ``crops`` is optional. When present it must be a
        bounded list of unique, fully valid crop rankings; corrupted or
        oversized crop evidence invalidates the payload (callers fall back to
        rules rather than trusting partial model output).
        """
        base = self._validate_rankings(data)
        raw_crops = data.get("crops") if isinstance(data, dict) else None
        if raw_crops is None:
            return base
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
        return base

    def validate_crop(self, data: object, region: dict) -> dict:
        """Build a validated crop record (ranking + click region) for storage."""
        validated = self.validate(data)
        if validated.get("crops"):
            raise ValueError("crop evidence must not nest crops")
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
