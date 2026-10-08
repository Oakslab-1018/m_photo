"""HTTP-only adapter; no Torch/CUDA dependencies belong in the business backend."""
from __future__ import annotations

import hashlib
import io
import json
import logging
import math
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

    def validate(self, data: object) -> dict:
        countries = json.loads(self.settings.countries_path.read_text(encoding="utf-8"))
        allowed = {c["iso"] for c in countries}
        if not isinstance(data, dict) or data.get("model") != MODEL_ID or data.get("model_revision") != MODEL_REVISION:
            raise ValueError("unexpected model")
        if data.get("catalog_revision") != catalog_revision(countries):
            raise ValueError("catalog mismatch")
        rows = data.get("rankings")
        if not isinstance(rows, list) or len(rows) != len(allowed):
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
        ordered = sorted(rankings, key=lambda row: (-row["raw_cosine"], row["iso"]))
        if rankings != ordered:
            raise ValueError("unsorted ranking")
        margin = ordered[0]["raw_cosine"] - ordered[1]["raw_cosine"]
        reported = data.get("margin")
        if isinstance(reported, bool) or not isinstance(reported, (int, float)) or not math.isfinite(reported) or abs(reported - margin) > 1e-6:
            raise ValueError("invalid margin")
        if type(data.get("uncertain")) is not bool or data["uncertain"] != (margin < 0.015):
            raise ValueError("invalid uncertainty flag")
        return {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                "catalog_revision": data["catalog_revision"], "rankings": rankings,
                "margin": margin, "uncertain": data["uncertain"]}
