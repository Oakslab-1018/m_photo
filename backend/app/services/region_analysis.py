"""Focused local-region analysis for the point/ROI endpoint.

This module deliberately does not run the legacy color-only analyzers. It
records measurable pixel quality, real OCR text (when a real provider is
available), OCR-derived semantic clues and optional StreetCLIP scene
hypotheses. Scene hypotheses are display-only and never affect country rank.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageOps, ImageStat

from ..config import Settings
from ..domain.geolocation.clue import DIM_LABELS, VALUE_LABELS
from .model_gateway import MODEL_REVISION, StreetClipGateway
from .pipeline import PipelineService

logger = logging.getLogger("m_photo.region_analysis")

CONTEXT_SHORT_FRACTION = 0.6
CONTEXT_MIN_SHORT_FRACTION = 0.15
CONTEXT_MIN_AREA_FRACTION = 0.05
CONTEXT_MIN_SIDE_PX = 64
OCR_UPSCALE_TARGET_SHORT_SIDE = 320
OCR_UPSCALE_MAX = 3.0
WEAK_MARGIN_THRESHOLD = 0.03

_PURPOSES = {"auto", "ocr", "scene"}
_SCENE_DIMENSION_LABELS = {
    "road": "道路场景",
    "building": "建筑场景",
    "vegetation": "植被场景",
    "terrain": "地形场景",
    "water": "水体场景",
    "vehicle": "车辆场景",
}
_SCENE_VALUE_LABELS = {
    "arterial_road": "主干道",
    "local_street": "城市街道",
    "residential_lane": "住宅道路",
    "highway": "高速公路",
    "unpaved_road": "土路或碎石路",
    "pedestrian_way": "步行道",
    "high_rise": "高层建筑",
    "mid_rise": "中层建筑",
    "low_rise": "低层建筑",
    "historic_masonry": "历史砖石建筑",
    "commercial_storefront": "商业店面",
    "industrial_warehouse": "工业或仓储建筑",
    "street_trees": "行道树",
    "dense_green": "茂密绿地",
    "sparse_green": "稀疏植被",
    "grass_lawn": "草地或草坪",
    "flowering_plants": "花卉植物",
    "dry_vegetation": "干燥植被",
    "flat_ground": "平坦地面",
    "sloped_ground": "坡地",
    "hilly": "丘陵",
    "mountainous": "山地",
    "valley": "谷地",
    "coastal_lowland": "沿海低地",
    "no_visible_water": "未见水体",
    "lake": "湖泊",
    "river": "河流",
    "sea_ocean": "海洋",
    "canal": "运河",
    "pond_fountain": "水池或喷泉",
    "cars": "汽车",
    "motorcycles_scooters": "摩托车或踏板车",
    "bicycles": "自行车",
    "buses": "公交车",
    "trucks": "卡车",
    "no_visible_vehicles": "未见车辆",
}


@dataclass(frozen=True)
class RegionPlan:
    image_width: int
    image_height: int
    orientation_applied: bool
    image_sha256: str
    bbox_px: tuple[int, int, int, int]
    bbox_norm: dict[str, float]
    context_bbox_px: tuple[int, int, int, int] | None
    context_bbox_norm: dict[str, float] | None
    click_x: float
    click_y: float
    scale: float
    purpose: str
    region_id: str


def provider_signature(settings: Settings, pipeline: PipelineService) -> str:
    """Stable non-secret signature of the providers that produced a report."""
    try:
        status = pipeline.ocr_status()
    except Exception:
        status = {"provider": pipeline.ocr_provider, "state": "unavailable"}
    payload = {
        "contract": "region-analysis-v1",
        "ocr": {
            "provider": status.get("provider") or pipeline.ocr_provider,
            "model_type": status.get("model_type") or pipeline.ocr_model_type,
            "models": status.get("models") or {},
            "min_text_score": settings.ocr_min_text_score,
            "language_min_confidence": settings.ocr_language_min_confidence,
        },
        "model": {
            "enabled": bool(settings.streetclip_enabled),
            "url": settings.streetclip_url,
            "timeout_seconds": settings.streetclip_timeout_seconds,
            "model_revision": MODEL_REVISION,
            "scene_contract": "streetclip-scene-v1",
            "api_key_configured": bool(settings.streetclip_api_key),
        },
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class RegionAnalysisService:
    """Build stable ROI plans and the detailed RegionReport payload."""

    def __init__(
        self,
        settings: Settings,
        pipeline: PipelineService,
        gateway: StreetClipGateway,
    ) -> None:
        self.settings = settings
        self.pipeline = pipeline
        self.gateway = gateway

    def plan(
        self,
        image_path: str | Path,
        *,
        x: float,
        y: float,
        scale: float,
        image_region: dict[str, float] | None,
        purpose: str,
    ) -> RegionPlan:
        with Image.open(image_path) as opened:
            orientation = opened.getexif().get(274)
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.info.clear()
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError("empty image")
        bbox = _detail_bbox(
            width=width,
            height=height,
            x=x,
            y=y,
            scale=scale,
            image_region=image_region,
        )
        context = _context_bbox(
            width=width,
            height=height,
            detail=bbox,
            purpose=purpose,
        )
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        bbox_norm = _norm_bbox(bbox, width, height)
        context_norm = _norm_bbox(context, width, height) if context else None
        identity_payload = {
            "image_sha256": digest,
            "orientation_applied": orientation not in {None, 1},
            "bbox_px": list(bbox),
            "context_bbox_px": list(context) if context else None,
            "purpose": purpose,
        }
        region_id = "ra_" + hashlib.sha256(
            json.dumps(identity_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        return RegionPlan(
            image_width=width,
            image_height=height,
            orientation_applied=orientation not in {None, 1},
            image_sha256=digest,
            bbox_px=bbox,
            bbox_norm=bbox_norm,
            context_bbox_px=context,
            context_bbox_norm=context_norm,
            click_x=float(x),
            click_y=float(y),
            scale=float(scale),
            purpose=purpose,
            region_id=region_id,
        )

    def analyze(
        self,
        image_path: str | Path,
        plan: RegionPlan,
        *,
        allow_crop_ranking: bool,
        full_margin: float | None,
        full_uncertain: bool | None,
    ) -> dict[str, Any]:
        with Image.open(image_path) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGB")
        image.info.clear()
        if image.size != (plan.image_width, plan.image_height):
            raise ValueError("image dimensions changed during region analysis")

        detail = image.crop(plan.bbox_px)
        detail.info.clear()
        detail_quality = _quality_metrics(detail, image.size)
        self.settings.upload_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="region-analysis-", dir=str(self.settings.upload_dir)) as temp_dir:
            temp = Path(temp_dir)
            detail_path = temp / "detail.png"
            detail.save(detail_path, format="PNG")

            ocr_status = _ocr_report(self.pipeline.ocr_status())
            clues: list[dict[str, Any]] = []
            ocr_factor = 1.0
            if ocr_status["enabled"]:
                ocr_factor = _ocr_upscale_factor(detail.size)
                ocr_path = detail_path
                if ocr_factor > 1.0:
                    upscaled = detail.resize(
                        (
                            max(1, int(round(detail.width * ocr_factor))),
                            max(1, int(round(detail.height * ocr_factor))),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                    upscaled.info.clear()
                    ocr_path = temp / "detail-ocr.png"
                    upscaled.save(ocr_path, format="PNG")
                result = self.pipeline.analyze_region(
                    str(ocr_path),
                    image_size=image.size,
                    crop_px=plan.bbox_px,
                    ocr_scale=ocr_factor,
                    region_id=plan.region_id,
                    region_bbox_norm=plan.bbox_norm,
                )
                clues = list(result.get("clues") or [])
                ocr_status = _ocr_report(result.get("ocr_status") or ocr_status)

            clues = _finalize_clues(clues, plan=plan, ocr_status=ocr_status)
            observations: list[dict[str, Any]] = []
            observations.append(_pixel_observation(plan, detail_quality))
            observations.extend(_clue_observations(clues, plan, detail_quality))

            if plan.purpose in {"auto", "scene"}:
                scene = self.gateway.describe_region(detail_path.read_bytes())
            else:
                scene = {
                    "provider": "streetclip",
                    "enabled": self.settings.streetclip_enabled,
                    "available": False,
                    "state": "not_requested",
                    "scene_groups": [],
                }
            scene_observations = _scene_observations(scene, plan)
            observations.extend(scene_observations)

            crop_evidence = None
            rank_attempted = False
            rank_available = False
            rank_margin = None
            rank_uncertain = None
            weak_prior = full_uncertain is True or full_margin is None or full_margin < WEAK_MARGIN_THRESHOLD
            rank_requested = self._should_try_context_rank(
                image=image,
                plan=plan,
                allow_crop_ranking=allow_crop_ranking,
                full_margin=full_margin,
                full_uncertain=full_uncertain,
            )
            context_eligible = False
            if rank_requested:
                context = image.crop(plan.context_bbox_px)
                context_quality = _quality_metrics(context, image.size)
                context_eligible = _context_rank_eligible(context.size, image.size, context_quality)
                if context_eligible:
                    rank_attempted = True
                    context.info.clear()
                    context_path = temp / "context.png"
                    context.save(context_path, format="PNG")
                    raw_crop = self.gateway.rank_image(context_path)
                    if raw_crop is not None:
                        rank_available = True
                        rank_margin = _optional_float(raw_crop.get("margin"))
                        rank_uncertain = raw_crop.get("uncertain") if isinstance(raw_crop.get("uncertain"), bool) else None
                        if raw_crop.get("uncertain") is False:
                            region_meta = {
                                "x": plan.click_x,
                                "y": plan.click_y,
                                "size": plan.bbox_px[2] - plan.bbox_px[0],
                                "region_id": plan.region_id,
                                "purpose": plan.purpose,
                                "bbox_norm": plan.context_bbox_norm or plan.bbox_norm,
                                "ranking_applied": True,
                            }
                            crop_evidence = self.gateway.validate_crop(
                                {
                                    **raw_crop,
                                    "image_sha256": hashlib.sha256(context_path.read_bytes()).hexdigest(),
                                },
                                region_meta,
                            )

            ranking_applied = crop_evidence is not None
            ranking_note = _ranking_note(
                enabled=self.settings.streetclip_enabled,
                purpose=plan.purpose,
                allow_crop_ranking=allow_crop_ranking,
                weak_prior=weak_prior,
                context_available=plan.context_bbox_px is not None,
                context_eligible=context_eligible,
                rank_attempted=rank_attempted,
                rank_available=rank_available,
                rank_uncertain=rank_uncertain,
                ranking_applied=ranking_applied,
            )
            model_status = _model_report(
                enabled=self.settings.streetclip_enabled,
                scene=scene,
                purpose=plan.purpose,
                rank_attempted=rank_attempted,
                rank_available=rank_available,
                ranking_applied=ranking_applied,
                rank_margin=rank_margin,
                rank_uncertain=rank_uncertain,
                ranking_note=ranking_note,
            )
            if ranking_applied:
                observations.append(_crop_ranking_observation(plan, rank_margin))

            questions = _question_suggestions(
                plan=plan,
                quality=detail_quality,
                ocr_status=ocr_status,
                model_status=model_status,
                observations=observations,
            )
            applied = bool(clues or ranking_applied)
            summary = _summary(
                plan=plan,
                quality=detail_quality,
                ocr_status=ocr_status,
                model_status=model_status,
                observations=observations,
                clues=clues,
                applied=applied,
            )
            report = {
                "id": plan.region_id,
                "purpose": plan.purpose,
                "scale": round(plan.scale, 6),
                "bbox_px": list(plan.bbox_px),
                "bbox_norm": plan.bbox_norm,
                "context_bbox_norm": plan.context_bbox_norm,
                "image": {
                    "width": plan.image_width,
                    "height": plan.image_height,
                    "orientation_applied": plan.orientation_applied,
                    "detail_width": detail.width,
                    "detail_height": detail.height,
                    "ocr_scale": round(ocr_factor, 6),
                },
                "quality": {
                    "issues": list(detail_quality["issues"]),
                    "metrics": dict(detail_quality["metrics"]),
                },
                "ocr": ocr_status,
                "model": model_status,
                "observations": observations,
                "question_suggestions": questions,
                "summary": summary,
                "dedupe": "new" if applied else "no_new_evidence",
            }
            cacheable = _cacheable(
                purpose=plan.purpose,
                ocr=ocr_status,
                scene_attempted=plan.purpose in {"auto", "scene"} and self.settings.streetclip_enabled,
                scene_available=bool(scene.get("available")),
                rank_attempted=rank_attempted,
                rank_available=rank_available,
            )
            return {
                "plan": plan,
                "report": report,
                "clues": clues,
                "crop_evidence": crop_evidence,
                "cacheable": cacheable,
                "scene_available": bool(scene.get("available")),
                "scene_state": str(scene.get("state") or "unavailable"),
                "rank_attempted": rank_attempted,
                "rank_available": rank_available,
                "ocr_available": bool(ocr_status["available"]),
                "ocr_enabled": bool(ocr_status["enabled"]),
            }

    def _should_try_context_rank(
        self,
        *,
        image: Image.Image,
        plan: RegionPlan,
        allow_crop_ranking: bool,
        full_margin: float | None,
        full_uncertain: bool | None,
    ) -> bool:
        if not (allow_crop_ranking and self.settings.streetclip_enabled):
            return False
        if plan.purpose not in {"auto", "scene"} or plan.context_bbox_px is None:
            return False
        if plan.context_bbox_px == plan.bbox_px:
            return False
        weak_prior = full_uncertain is True or full_margin is None or full_margin < WEAK_MARGIN_THRESHOLD
        return weak_prior

def _detail_bbox(
    *,
    width: int,
    height: int,
    x: float,
    y: float,
    scale: float,
    image_region: dict[str, float] | None,
) -> tuple[int, int, int, int]:
    if isinstance(image_region, dict):
        rx, ry, rw, rh = (
            float(image_region["x"]),
            float(image_region["y"]),
            float(image_region["w"]),
            float(image_region["h"]),
        )
        left = max(0, min(width - 1, int(math.floor(rx * width))))
        top = max(0, min(height - 1, int(math.floor(ry * height))))
        right = max(left + 1, min(width, int(math.ceil((rx + rw) * width))))
        bottom = max(top + 1, min(height, int(math.ceil((ry + rh) * height))))
        return left, top, right, bottom
    short = min(width, height)
    side = max(1, min(max(width, height), int(round(short * float(scale)))))
    cx = float(x) * width
    cy = float(y) * height
    left = max(0, min(width - 1, int(round(cx - side / 2.0))))
    top = max(0, min(height - 1, int(round(cy - side / 2.0))))
    right = min(width, left + side)
    bottom = min(height, top + side)
    return max(0, right - side), max(0, bottom - side), right, bottom


def _context_bbox(
    *,
    width: int,
    height: int,
    detail: tuple[int, int, int, int],
    purpose: str,
) -> tuple[int, int, int, int] | None:
    if purpose not in {"auto", "scene"}:
        return None
    left, top, right, bottom = detail
    detail_w, detail_h = right - left, bottom - top
    target = max(1, int(round(min(width, height) * CONTEXT_SHORT_FRACTION)))
    context_w = min(width, max(detail_w, target))
    context_h = min(height, max(detail_h, target))
    cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
    c_left = max(0, min(width - context_w, int(round(cx - context_w / 2.0))))
    c_top = max(0, min(height - context_h, int(round(cy - context_h / 2.0))))
    context = (c_left, c_top, c_left + context_w, c_top + context_h)
    if context == detail:
        return None
    return context


def _norm_bbox(bbox: tuple[int, int, int, int], width: int, height: int) -> dict[str, float]:
    left, top, right, bottom = bbox
    return {
        "x": round(left / width, 6),
        "y": round(top / height, 6),
        "w": max(1e-6, round((right - left) / width, 6)),
        "h": max(1e-6, round((bottom - top) / height, 6)),
    }


def _quality_metrics(image: Image.Image, full_size: tuple[int, int]) -> dict[str, Any]:
    gray = image.convert("L")
    stat = ImageStat.Stat(gray)
    brightness_mean = float(stat.mean[0]) / 255.0
    brightness_std = float(stat.stddev[0]) / 255.0
    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_energy = float(ImageStat.Stat(edges).mean[0]) / 255.0
    hsv = image.convert("HSV")
    saturation_mean = float(ImageStat.Stat(hsv).mean[1]) / 255.0
    rgb_mean = [round(float(value) / 255.0, 4) for value in ImageStat.Stat(image).mean]
    area_ratio = (image.width * image.height) / float(full_size[0] * full_size[1])
    issues: list[str] = []
    if min(image.size) < 64:
        issues.append("区域过小")
    if brightness_mean < 0.18:
        issues.append("图像偏暗")
    elif brightness_mean > 0.90:
        issues.append("图像过亮")
    if brightness_std < 0.08:
        issues.append("对比度偏低")
    if edge_energy < 0.015:
        issues.append("清晰度可能不足")
    return {
        "issues": issues,
        "metrics": {
            "width": image.width,
            "height": image.height,
            "area_ratio": round(area_ratio, 6),
            "brightness_mean": round(brightness_mean, 6),
            "brightness_std": round(brightness_std, 6),
            "edge_energy": round(edge_energy, 6),
            "saturation_mean": round(saturation_mean, 6),
            "rgb_mean": rgb_mean,
            "limitations": [
                "像素统计只反映亮度、对比度、清晰度和色彩分布，不能单独判断建筑风格、气候、植被类型或具体地点。"
            ],
        },
    }


def _context_rank_eligible(
    context_size: tuple[int, int],
    full_size: tuple[int, int],
    quality: dict[str, Any],
) -> bool:
    metrics = quality["metrics"]
    short = min(context_size)
    if short < max(CONTEXT_MIN_SIDE_PX, int(min(full_size) * CONTEXT_MIN_SHORT_FRACTION)):
        return False
    if metrics["area_ratio"] < CONTEXT_MIN_AREA_FRACTION:
        return False
    if not (0.12 <= metrics["brightness_mean"] <= 0.92):
        return False
    if metrics["brightness_std"] < 0.06 or metrics["edge_energy"] < 0.008:
        return False
    return True


def _ocr_upscale_factor(size: tuple[int, int]) -> float:
    short = min(size)
    if short <= 0 or short >= OCR_UPSCALE_TARGET_SHORT_SIDE:
        return 1.0
    factor = round(min(OCR_UPSCALE_MAX, OCR_UPSCALE_TARGET_SHORT_SIDE / short), 2)
    return factor if factor > 1.01 else 1.0


def _ocr_report(status: dict[str, Any]) -> dict[str, Any]:
    provider = str(status.get("provider") or "mock")
    enabled = bool(status.get("enabled")) if "enabled" in status else provider != "mock"
    state = str(status.get("state") or "idle")
    available = bool(status.get("available")) if "available" in status else enabled and state == "ok"
    if not enabled:
        state = "disabled"
        available = False
    line_count = status.get("line_count")
    if isinstance(line_count, bool) or not isinstance(line_count, int) or line_count < 0:
        line_count = 0
    return {
        "provider": provider,
        "enabled": enabled,
        "available": available,
        "state": state,
        "line_count": line_count,
    }


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None

def _finalize_clues(
    clues: list[dict[str, Any]],
    *,
    plan: RegionPlan,
    ocr_status: dict[str, Any],
) -> list[dict[str, Any]]:
    finalized: list[dict[str, Any]] = []
    for clue in clues:
        if not isinstance(clue, dict):
            continue
        extra = dict(clue.get("extra") or {})
        origin = str(extra.get("origin_source") or clue.get("source") or "ocr")
        provider = str(extra.get("provider") or ocr_status.get("provider") or "unknown")
        bbox_norm = _valid_bbox(extra.get("bbox_norm"))
        quad_norm = _valid_bbox(extra.get("quad_norm"))
        if bbox_norm is None and quad_norm is not None:
            bbox_norm = quad_norm
        trust = _clue_trust(clue, bbox_norm=bbox_norm, origin=origin)
        extra.update({
            "origin_source": origin,
            "provider": provider,
            "origin_provider": provider,
            "region_id": plan.region_id,
            "region_bbox_norm": dict(plan.bbox_norm),
            "trust": trust,
        })
        if bbox_norm is not None:
            extra["bbox_norm"] = bbox_norm
        if quad_norm is not None:
            extra["quad_norm"] = quad_norm
        clue["extra"] = extra
        finalized.append(clue)
    return finalized


def _clue_trust(clue: dict[str, Any], *, bbox_norm: dict[str, float] | None, origin: str) -> dict[str, Any]:
    extra = clue.get("extra") if isinstance(clue.get("extra"), dict) else {}
    confidence = _bounded_float(clue.get("confidence"), 0.0, 1.0)
    if clue.get("dimension") == "signage":
        reliable = bool(extra.get("reliable")) and confidence >= 0.5
        return {
            "level": "high" if reliable and bbox_norm is not None else "medium",
            "reliable": reliable,
            "validated": bbox_norm is not None,
            "affects_ranking": False,
            "confidence_kind": "recognition_score",
            "confidence": confidence,
        }
    reliable = bool(extra.get("reliable"))
    validated = bool(extra.get("validated"))
    affects_ranking = bool(extra.get("affects_ranking")) and not bool(extra.get("display_only"))
    return {
        "level": "high" if reliable and validated and affects_ranking else "medium" if reliable else "low",
        "reliable": reliable,
        "validated": validated,
        "affects_ranking": affects_ranking,
        "confidence_kind": "derived_text_confidence",
        "confidence": confidence,
    }


def _pixel_observation(plan: RegionPlan, quality: dict[str, Any]) -> dict[str, Any]:
    metrics = quality["metrics"]
    evidence = (
        f"亮度均值 {metrics['brightness_mean']:.3f}，亮度标准差 {metrics['brightness_std']:.3f}，"
        f"边缘能量 {metrics['edge_energy']:.3f}，平均饱和度 {metrics['saturation_mean']:.3f}。"
        "这些是区域实际像素统计，只用于判断可读性和图像质量，不是地点语义判断。"
    )
    if quality["issues"]:
        evidence += " 当前质量问题：" + "、".join(quality["issues"]) + "。"
    return {
        "id": _observation_id("pixel", plan.region_id, 0),
        "kind": "pixel_signal",
        "dimension": "quality",
        "value": "region_pixels",
        "text": evidence,
        "label": "区域像素质量",
        "evidence": evidence,
        "source": "local_image",
        "provider": "pillow",
        "bbox_norm": dict(plan.bbox_norm),
        "quad_norm": None,
        "trust": {
            "level": "high",
            "reliable": True,
            "validated": True,
            "affects_ranking": False,
            "confidence_kind": "measured",
        },
    }


def _clue_observations(
    clues: list[dict[str, Any]],
    plan: RegionPlan,
    quality: dict[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    for index, clue in enumerate(clues):
        extra = clue.get("extra") if isinstance(clue.get("extra"), dict) else {}
        bbox_norm = _valid_bbox(extra.get("bbox_norm"))
        quad_norm = _valid_bbox(extra.get("quad_norm")) if extra.get("quad_norm") else None
        confidence = _bounded_float(clue.get("confidence"), 0.0, 1.0)
        provider = str(extra.get("provider") or "unknown")
        if clue.get("dimension") == "signage":
            text = str(clue.get("value") or "")
            evidence = f"真实 OCR 识别到文字「{text[:120]}」，识别分数 {confidence:.3f}。"
            observations.append({
                "id": _observation_id("ocr", plan.region_id, index, text),
                "kind": "ocr_text",
                "dimension": "signage",
                "value": text[:200],
                "text": text[:200],
                "label": "OCR 文字",
                "evidence": evidence,
                "source": "ocr",
                "provider": provider,
                "bbox_norm": bbox_norm,
                "quad_norm": quad_norm,
                "trust": {
                    "level": "high" if confidence >= 0.5 and bbox_norm is not None else "medium",
                    "reliable": confidence >= 0.5,
                    "validated": bbox_norm is not None,
                    "affects_ranking": False,
                    "confidence_kind": "recognition_score",
                    "confidence": confidence,
                },
            })
            continue
        dimension = str(clue.get("dimension") or "")
        value = str(clue.get("value") or "")
        label = _clue_label(dimension, value)
        text = str(extra.get("matched_text") or value)
        observations.append({
            "id": _observation_id("semantic", plan.region_id, index, f"{dimension}:{value}"),
            "kind": "semantic",
            "dimension": dimension,
            "value": value[:200],
            "text": text[:200] or None,
            "label": label,
            "evidence": str(clue.get("evidence") or f"基于 OCR 文字「{text[:80]}」派生，不是颜色推断。"),
            "source": str(clue.get("source") or "semantic"),
            "provider": provider,
            "bbox_norm": bbox_norm,
            "quad_norm": quad_norm,
            "trust": {
                "level": "high" if extra.get("reliable") and extra.get("validated") else "medium" if extra.get("reliable") else "low",
                "reliable": bool(extra.get("reliable")),
                "validated": bool(extra.get("validated")),
                "affects_ranking": bool(extra.get("affects_ranking")) and not bool(extra.get("display_only")),
                "confidence_kind": "derived_text_confidence",
                "confidence": confidence,
            },
        })
    return observations

def _scene_observations(scene: dict[str, Any], plan: RegionPlan) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []
    if not scene.get("available"):
        return observations
    for group_index, group in enumerate(scene.get("scene_groups") or []):
        dimension = str(group.get("dimension") or "")
        uncertain = bool(group.get("uncertain"))
        for candidate_index, candidate in enumerate(group.get("candidates") or []):
            value = str(candidate.get("value") or "")
            raw_cosine = _bounded_float(candidate.get("raw_cosine"), -1.0, 1.0)
            label = _SCENE_VALUE_LABELS.get(value, _SCENE_DIMENSION_LABELS.get(dimension, "场景假设"))
            evidence = (
                f"StreetCLIP 场景模型原始余弦 {raw_cosine:.4f}（不是地点概率，也不参与国家排序；"
                "仅供参考复核）。"
            )
            observations.append({
                "id": _observation_id("scene", plan.region_id, group_index * 10 + candidate_index, f"{dimension}:{value}"),
                "kind": "scene_hypothesis",
                "dimension": dimension,
                "value": value,
                "text": str(candidate.get("label") or value),
                "label": label,
                "evidence": evidence,
                "source": "streetclip_scene",
                "provider": "streetclip",
                "bbox_norm": dict(plan.bbox_norm),
                "quad_norm": None,
                "trust": {
                    "level": "low",
                    "reliable": False,
                    "validated": False,
                    "affects_ranking": False,
                    "confidence_kind": "raw_cosine",
                    "raw_cosine": raw_cosine,
                    "uncertain": uncertain,
                },
            })
    return observations


def _crop_ranking_observation(plan: RegionPlan, margin: float | None) -> dict[str, Any]:
    margin_text = "未知" if margin is None else f"{margin:.4f}"
    evidence = (
        f"上下文裁剪仅作为弱排序证据应用，模型原始排序间隔为 {margin_text}；"
        "该值不是概率，也不改变全图原始余弦。"
    )
    return {
        "id": _observation_id("crop_ranking", plan.region_id, 0),
        "kind": "crop_ranking",
        "dimension": "country_prior",
        "value": "context_crop",
        "text": evidence,
        "label": "上下文裁剪排序",
        "evidence": evidence,
        "source": "streetclip_crop",
        "provider": "streetclip",
        "bbox_norm": dict(plan.context_bbox_norm or plan.bbox_norm),
        "quad_norm": None,
        "trust": {
            "level": "medium",
            "reliable": True,
            "validated": True,
            "affects_ranking": True,
            "confidence_kind": "ranking_margin",
            "uncertain": False,
        },
    }


def _ranking_note(
    *,
    enabled: bool,
    purpose: str,
    allow_crop_ranking: bool,
    weak_prior: bool,
    context_available: bool,
    context_eligible: bool,
    rank_attempted: bool,
    rank_available: bool,
    rank_uncertain: bool | None,
    ranking_applied: bool,
) -> str:
    if ranking_applied:
        return "已应用质量合格的上下文裁剪作为弱排序证据。"
    if not enabled:
        return "国家模型未启用，未应用裁剪排序。"
    if purpose == "ocr":
        return "本次 purpose=ocr，未请求场景或裁剪排序。"
    if not allow_crop_ranking:
        return "缺少全图模型先验，未应用裁剪排序。"
    if not weak_prior:
        return "全图排序间隔充足，本轮未请求上下文裁剪排序。"
    if not context_available:
        return "没有可用的更大上下文裁剪，未应用裁剪排序。"
    if not context_eligible:
        return "上下文裁剪过小或质量不足，未应用裁剪排序。"
    if not rank_attempted:
        return "未发起上下文裁剪排序。"
    if not rank_available:
        return "裁剪排序模型当前不可用，未应用排序。"
    if rank_uncertain is True:
        return "裁剪排序结果不确定，未应用排序。"
    return "裁剪排序没有提供可用证据，未应用排序。"


def _model_report(
    *,
    enabled: bool,
    scene: dict[str, Any],
    purpose: str,
    rank_attempted: bool,
    rank_available: bool,
    ranking_applied: bool,
    rank_margin: float | None,
    rank_uncertain: bool | None,
    ranking_note: str,
) -> dict[str, Any]:
    scene_attempted = purpose in {"auto", "scene"} and enabled
    scene_available = bool(scene.get("available"))
    scene_state = str(scene.get("state") or "unavailable")
    available = scene_available or rank_available
    if not enabled:
        state = "disabled"
    elif purpose == "ocr" and not rank_attempted:
        state = "not_requested"
    elif available:
        failed_attempt = (scene_attempted and not scene_available) or (rank_attempted and not rank_available)
        state = "partial" if failed_attempt else "ok"
    else:
        state = "unavailable"
    return {
        "provider": "streetclip",
        "enabled": enabled,
        "available": available,
        "state": state,
        "ranking_applied": ranking_applied,
        "margin": rank_margin,
        "uncertain": rank_uncertain,
        "scene_available": scene_available if scene_attempted else None,
        "scene_state": scene_state if scene_attempted else None,
        "ranking_note": ranking_note,
    }


def _cacheable(
    *,
    purpose: str,
    ocr: dict[str, Any],
    scene_attempted: bool,
    scene_available: bool,
    rank_attempted: bool,
    rank_available: bool,
) -> bool:
    failures = []
    if ocr["enabled"] and not ocr["available"]:
        failures.append("ocr")
    if scene_attempted and not scene_available:
        failures.append("scene")
    if rank_attempted and not rank_available:
        failures.append("rank")
    any_success = bool(ocr["available"] or scene_available or rank_available)
    return any_success and not failures

def _question_suggestions(
    *,
    plan: RegionPlan,
    quality: dict[str, Any],
    ocr_status: dict[str, Any],
    model_status: dict[str, Any],
    observations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ocr_obs = [item for item in observations if item.get("kind") == "ocr_text"]
    semantic_obs = [item for item in observations if item.get("kind") == "semantic"]
    scene_obs = [item for item in observations if item.get("kind") == "scene_hypothesis"]
    pixel_obs = [item for item in observations if item.get("kind") == "pixel_signal"]
    questions: list[dict[str, Any]] = []
    if ocr_obs:
        text = "、".join(str(item.get("text") or item.get("value") or "")[:40] for item in ocr_obs[:3])
        questions.append({
            "id": _question_id(plan.region_id, "ocr_text"),
            "text": f"识别到文字「{text}」，它是否与画面中的招牌或路牌一致？",
            "reason": "该问题来自本次真实 OCR 行，不代表地名已经确认。",
            "basis_observation_ids": [str(item["id"]) for item in ocr_obs[:3]],
        })
    elif ocr_status["enabled"] and ocr_status["available"]:
        basis = [str(item["id"]) for item in pixel_obs]
        if quality["issues"]:
            text = "当前区域未识别到可靠文字，是否尝试框选更清晰或更大的文字区域？"
            reason = "OCR 已运行，但画面质量限制了文字识别。"
        else:
            text = "当前区域未识别到可靠文字，是否可框选包含招牌、路牌或门牌的区域？"
            reason = "OCR 已运行但没有达到阈值的文字行。"
        questions.append({
            "id": _question_id(plan.region_id, "ocr_empty"),
            "text": text,
            "reason": reason,
            "basis_observation_ids": basis,
        })
    elif ocr_status["enabled"]:
        questions.append({
            "id": _question_id(plan.region_id, "ocr_unavailable"),
            "text": "OCR 当前不可用；你能否直接描述画面中可辨认的文字？",
            "reason": f"OCR 状态为 {ocr_status['state']}，没有产生文字识别结果。",
            "basis_observation_ids": [str(item["id"]) for item in pixel_obs],
        })
    else:
        questions.append({
            "id": _question_id(plan.region_id, "ocr_disabled"),
            "text": "当前未启用真实 OCR；画面中是否包含可人工辨认的文字或路牌？",
            "reason": "报告没有把模拟 OCR 或颜色统计写成真实文字证据。",
            "basis_observation_ids": [str(item["id"]) for item in pixel_obs],
        })
    if semantic_obs:
        questions.append({
            "id": _question_id(plan.region_id, "semantic_check"),
            "text": "识别文字中出现的语言或地名线索，是否与实际画面相符？",
            "reason": "这些线索来自 OCR 文本目录匹配，需要用户复核。",
            "basis_observation_ids": [str(item["id"]) for item in semantic_obs[:3]],
        })
    if scene_obs:
        labels = "、".join(str(item.get("label") or "") for item in scene_obs[:3])
        questions.append({
            "id": _question_id(plan.region_id, "scene_check"),
            "text": f"场景模型给出参考标签「{labels}」，画面是否符合其中任一描述？",
            "reason": "场景标签仅为低可信假设，原始余弦不是地点概率。",
            "basis_observation_ids": [str(item["id"]) for item in scene_obs[:3]],
        })
    if model_status.get("uncertain") is True and model_status.get("available"):
        questions.append({
            "id": _question_id(plan.region_id, "rank_uncertain"),
            "text": "局部模型排序仍有不确定性；你能补充画面中最明显的道路、建筑或语言线索吗？",
            "reason": "模型返回不确定标记，不能用局部排序确认地点。",
            "basis_observation_ids": [str(item["id"]) for item in (semantic_obs + ocr_obs)[:3]],
        })
    return questions[:4]


def _summary(
    *,
    plan: RegionPlan,
    quality: dict[str, Any],
    ocr_status: dict[str, Any],
    model_status: dict[str, Any],
    observations: list[dict[str, Any]],
    clues: list[dict[str, Any]],
    applied: bool,
) -> str:
    parts: list[str] = []
    if ocr_status["enabled"] and ocr_status["available"]:
        count = sum(1 for item in observations if item.get("kind") == "ocr_text")
        if count:
            parts.append(f"真实 OCR 识别到 {count} 行文字")
        else:
            parts.append("真实 OCR 已运行，但未识别到达到阈值的文字")
    elif ocr_status["enabled"]:
        parts.append(f"OCR 当前不可用（{ocr_status['state']}），未产生文字识别结果")
    else:
        parts.append("OCR 未启用，未产生真实文字识别结果")
    if clues:
        parts.append(f"本次新增 {len(clues)} 条文字派生线索，仅用于候选排序")
    if plan.purpose in {"auto", "scene"}:
        scene_count = sum(1 for item in observations if item.get("kind") == "scene_hypothesis")
        if scene_count:
            parts.append(f"场景模型给出 {scene_count} 个低可信参考假设，不参与国家排序")
        elif not model_status.get("enabled"):
            parts.append("场景与国家模型未启用，未生成场景假设")
        elif model_status.get("scene_state") == "disabled":
            parts.append("场景模型未启用，未生成场景假设")
        elif model_status.get("scene_state") in {"unavailable", "failed"}:
            parts.append("场景模型端点暂不可用，未生成场景假设")
    if model_status.get("ranking_applied"):
        parts.append("已把质量合格的上下文裁剪作为弱国家排序证据，全图原始余弦保持不变")
    elif model_status.get("enabled") and model_status.get("uncertain") is True:
        parts.append("局部国家排序存在不确定性，因此未应用裁剪排序")
    if quality["issues"]:
        parts.append("图像质量提示：" + "、".join(quality["issues"]))
    parts.append("像素统计不能单独判断建筑风格、植被类型、气候或具体地点")
    if applied:
        parts.append("这些证据不会自动确认拍摄地")
    else:
        parts.append("本轮未新增可用于排序的证据，仅记录区域质量与模型可用性")
    return "；".join(parts) + "。"


def _valid_bbox(value: Any) -> dict[str, float] | None:
    if not isinstance(value, dict):
        return None
    try:
        x, y, w, h = float(value["x"]), float(value["y"]), float(value["w"]), float(value["h"])
    except (KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(item) for item in (x, y, w, h)):
        return None
    if x < 0.0 or y < 0.0 or w <= 0.0 or h <= 0.0:
        return None
    if x + w > 1.0 + 1e-9 or y + h > 1.0 + 1e-9:
        return None
    return {"x": round(x, 6), "y": round(y, 6), "w": round(w, 6), "h": round(h, 6)}


def _bounded_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return minimum
    if not math.isfinite(number):
        return minimum
    return max(minimum, min(maximum, number))


def _clue_label(dimension: str, value: str) -> str:
    dimension_label = DIM_LABELS.get(dimension, dimension or "文字线索")
    value_label = VALUE_LABELS.get(value)
    return f"{dimension_label}：{value_label or value}" if value else dimension_label


def _observation_id(kind: str, region_id: str, index: int, text: str = "") -> str:
    digest = hashlib.sha256(f"{kind}|{region_id}|{index}|{text}".encode()).hexdigest()[:10]
    return f"obs_{kind}_{digest}"


def _question_id(region_id: str, suffix: str) -> str:
    digest = hashlib.sha256(f"{region_id}|{suffix}".encode()).hexdigest()[:10]
    return f"q_{suffix}_{digest}"


__all__ = ["RegionAnalysisService", "RegionPlan", "provider_signature"]
