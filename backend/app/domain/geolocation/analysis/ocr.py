"""OCR analyzer backed by RapidOCR + ONNX Runtime (CPU only).

Modes
-----
``mock`` (default)
    No OCR engine is imported or downloaded; :meth:`OCRAnalyzer.analyze`
    returns an empty clue list so the pipeline stays offline and deterministic.

``rapidocr``
    Lazily loads RapidOCR with the selected PP-OCRv5 det/cls/rec profile on the
    ``onnxruntime`` engine (CPU): ``server`` by default, ``mobile`` for lower
    latency. No torch, no easyocr and no TLS monkeypatching is involved; the
    optional ``OCR_DET/REC/CLS_MODEL_PATH`` settings point at locally cached
    ``.onnx`` files instead of model IDs.

If RapidOCR (or its model files) is unavailable while ``rapidocr`` mode is
requested, the analyzer degrades to ``mock`` and records a *diagnostic status*
(``unavailable`` / ``failed``) in :attr:`OCRAnalyzer.last_status` plus a log
line. It never fabricates real-looking text clues.

Thread safety
-------------
Engine construction and inference are serialized behind one instance lock, so a
single RapidOCR session is never shared across concurrent analyzer threads.
"""
from __future__ import annotations

import logging
import math
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .base import BaseAnalyzer
from ..clue import Clue
from ....config import DEFAULT_OCR_MODEL_TYPE, normalize_ocr_model_type

logger = logging.getLogger("m_photo.ocr")

MOCK_PROVIDER = "mock"
RAPIDOCR_PROVIDER = "rapidocr"
VALID_PROVIDERS = (MOCK_PROVIDER, RAPIDOCR_PROVIDER)

# Model identity is pinned explicitly: PP-OCRv5 det/cls/rec on CPU.  RapidOCR
# 3.10.0 is pinned in pyproject.toml; its ``default_models.yaml`` records the
# exact URL + SHA256 for every profile below.  Server is the accuracy-oriented
# default; mobile remains available for lower-latency deployments.
PPOCR_VERSION = "PP-OCRv5"
OCR_ENGINE = "onnxruntime"
DEFAULT_MODEL_TYPE = DEFAULT_OCR_MODEL_TYPE


@dataclass(frozen=True)
class OCRModelProfile:
    """One complete PP-OCRv5 det/cls/rec model family."""

    name: str
    det_model: str
    cls_model: str
    rec_model: str


MOBILE_MODEL_PROFILE = OCRModelProfile(
    name="mobile",
    det_model="ch_PP-OCRv5_det_mobile",
    cls_model="ch_PP-LCNet_x0_25_textline_ori_cls_mobile",
    rec_model="ch_PP-OCRv5_rec_mobile",
)
SERVER_MODEL_PROFILE = OCRModelProfile(
    name="server",
    det_model="ch_PP-OCRv5_det_server",
    cls_model="ch_PP-LCNet_x1_0_textline_ori_cls_server",
    rec_model="ch_PP-OCRv5_rec_server",
)
MODEL_PROFILES: Dict[str, OCRModelProfile] = {
    profile.name: profile for profile in (MOBILE_MODEL_PROFILE, SERVER_MODEL_PROFILE)
}

#: Backwards-compatible imports for callers that still use the old mobile keys.
DET_MODEL_KEY = MOBILE_MODEL_PROFILE.det_model
CLS_MODEL_KEY = MOBILE_MODEL_PROFILE.cls_model
REC_MODEL_KEY = MOBILE_MODEL_PROFILE.rec_model
DEFAULT_MIN_TEXT_SCORE = 0.5
# Detection resize policy: cap the long side instead of upscaling the short
# side. Verified against PP-OCRv5 det on small/wide signage crops: ``min``-side
# upscaling fails, while ``max`` @1280 recognizes them reliably.
DET_LIMIT_SIDE_LEN = 1280
DET_LIMIT_TYPE = "max"

#: Provider/mode argument accepted from the environment (``real`` is a legacy
#: alias kept for backwards compatibility with ``GEO_AI_MODE``).
_LEGACY_ALIASES = {"real": RAPIDOCR_PROVIDER, "true": RAPIDOCR_PROVIDER, "1": RAPIDOCR_PROVIDER}

_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\\[^\s'\"<>|]+")
_UNC_PATH_RE = re.compile(r"\\\\[^\s'\"<>|]+")
_POSIX_PATH_RE = re.compile(r"(?<![\w.])/(?:[\w.\-]+/)+[\w.\-]*")


def _model_label(path: str | Path | None, fallback: str) -> str:
    """Return a provenance-safe model label without exposing a local path."""
    if not path:
        return fallback
    name = str(path).replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    return name or fallback


def sanitize_diagnostic(message: str) -> str:
    """Strip paths/URLs from engine errors before logging or exposing them."""
    if not message:
        return ""
    scrubbed = _URL_RE.sub("<url>", str(message))
    scrubbed = _WIN_PATH_RE.sub("<path>", scrubbed)
    scrubbed = _UNC_PATH_RE.sub("<path>", scrubbed)
    scrubbed = _POSIX_PATH_RE.sub("<path>", scrubbed)
    scrubbed = scrubbed.replace("onnxruntime\\", "onnxruntime")
    return scrubbed[:300]


@dataclass
class OCRLine:
    """One recognized text line with provenance kept intact."""

    text: str
    confidence: float
    bbox: List[List[float]]
    provider: str = RAPIDOCR_PROVIDER
    det_model: str = SERVER_MODEL_PROFILE.det_model
    rec_model: str = SERVER_MODEL_PROFILE.rec_model
    cls_model: str = SERVER_MODEL_PROFILE.cls_model
    model_type: str = SERVER_MODEL_PROFILE.name
    model_version: str = PPOCR_VERSION

    def to_extra(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {
            # ``origin_source`` is the clue source ('ocr'); ``provider`` names
            # the concrete engine so point_click fusion keeps provenance.
            "origin_source": "ocr",
            "provider": self.provider,
            # Lines are already filtered by the provider's text-score floor, so
            # they may be consumed as reliable OCR evidence by fusion.
            "reliable": True,
            "line_confidence": self.confidence,
            "det_model": self.det_model,
            "rec_model": self.rec_model,
            "cls_model": self.cls_model,
            "model_type": self.model_type,
            "model_version": self.model_version,
        }
        # Invalid/zero-area provider boxes are omitted rather than represented
        # by a fabricated all-zero quad.
        if _normalize_bbox(self.bbox):
            extra["bbox"] = [list(point) for point in self.bbox]
            extra["quad"] = [list(point) for point in self.bbox]
        return extra


@dataclass
class OCRStatus:
    """Safe, non-secret diagnostic snapshot of the OCR provider state."""

    provider: str
    state: str = "idle"  # idle | ok | unavailable | failed
    engine: str | None = None
    model_version: str | None = None
    model_type: str | None = None
    models: Dict[str, str | None] = field(default_factory=dict)
    detail: str | None = None
    line_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "state": self.state,
            "enabled": self.provider != MOCK_PROVIDER,
            "available": self.provider != MOCK_PROVIDER and self.state == "ok",
            "engine": self.engine,
            "model_version": self.model_version,
            "model_type": self.model_type,
            "models": dict(self.models),
            "detail": self.detail,
            "line_count": self.line_count,
        }


def normalize_provider(value: str | None, default: str = MOCK_PROVIDER) -> str:
    """Resolve ``OCR_PROVIDER`` input; unknown values fall back to ``default``."""
    if value is None:
        return default
    candidate = str(value).strip().lower()
    if not candidate:
        return default
    return _LEGACY_ALIASES.get(candidate, candidate)


class _EngineUnavailable(RuntimeError):
    """Raised internally when RapidOCR cannot be loaded."""


class _InferenceFailed(RuntimeError):
    """Raised internally when a loaded engine fails during inference."""


class RapidOCREngine:
    """Minimal lazy adapter around :class:`rapidocr.RapidOCR` (PP-OCRv5/CPU).

    ``model_type`` selects the complete server or mobile det/cls/rec family.

    The heavy dependency is imported only inside :meth:`load`, which lets the
    backend run with mock mode even when RapidOCR is not installed.
    """

    def __init__(
        self,
        *,
        model_type: str = DEFAULT_MODEL_TYPE,
        model_dir: str | Path | None = None,
        det_model_path: str | Path | None = None,
        rec_model_path: str | Path | None = None,
        cls_model_path: str | Path | None = None,
        min_text_score: float = DEFAULT_MIN_TEXT_SCORE,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        self.model_type = normalize_ocr_model_type(model_type)
        self.model_dir = str(model_dir) if model_dir else None
        self.det_model_path = str(det_model_path) if det_model_path else None
        self.rec_model_path = str(rec_model_path) if rec_model_path else None
        self.cls_model_path = str(cls_model_path) if cls_model_path else None
        self.min_text_score = float(min_text_score)
        self._loader = loader
        self._engine: Any | None = None
        self._lock = threading.Lock()

    @property
    def profile(self) -> OCRModelProfile:
        return MODEL_PROFILES[self.model_type]

    def model_identity(self) -> Dict[str, str]:
        """Return safe per-task labels for selected or explicit models."""
        profile = self.profile
        return {
            "det_model": _model_label(self.det_model_path, profile.det_model),
            "rec_model": _model_label(self.rec_model_path, profile.rec_model),
            "cls_model": _model_label(self.cls_model_path, profile.cls_model),
            "model_type": self.model_type,
            "model_version": PPOCR_VERSION,
        }

    # -- construction -------------------------------------------------
    def _build_params(self) -> Dict[str, Any]:
        # RapidOCR requires Enum values for these four keys.
        try:
            from rapidocr import EngineType, ModelType, OCRVersion
        except Exception:  # pragma: no cover - only hit when injectable
            EngineType = ModelType = OCRVersion = None  # type: ignore[assignment]

        if EngineType is not None:
            engine_type: Any = EngineType.ONNXRUNTIME
            version: Any = OCRVersion.PPOCRV5
            model_type: Any = (
                ModelType.SERVER if self.model_type == "server" else ModelType.MOBILE
            )
        else:  # pragma: no cover - fallback for exotic injected runtimes
            engine_type, version, model_type = (
                "onnxruntime",
                "PP-OCRv5",
                self.model_type,
            )

        params: Dict[str, Any] = {
            "Global.text_score": self.min_text_score,
            "Global.log_level": "warning",
            "Det.engine_type": engine_type,
            "Det.ocr_version": version,
            "Det.lang_type": "ch",
            "Det.model_type": model_type,
            "Det.limit_side_len": DET_LIMIT_SIDE_LEN,
            "Det.limit_type": DET_LIMIT_TYPE,
            "Cls.engine_type": engine_type,
            "Cls.ocr_version": version,
            "Cls.lang_type": "ch",
            "Cls.model_type": model_type,
            "Rec.engine_type": engine_type,
            "Rec.ocr_version": version,
            "Rec.lang_type": "ch",
            "Rec.model_type": model_type,
        }
        if self.model_dir:
            params["Global.model_root_dir"] = self.model_dir
        if self.det_model_path:
            params["Det.model_path"] = self.det_model_path
        if self.rec_model_path:
            params["Rec.model_path"] = self.rec_model_path
        if self.cls_model_path:
            params["Cls.model_path"] = self.cls_model_path
        return params

    def load(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            self._engine = self._create_engine()
            return self._engine

    def _create_engine(self) -> Any:
        loader = self._loader
        if loader is None:
            try:  # lazy import keeps mock startup free of RapidOCR
                from rapidocr import RapidOCR as runtime

                return runtime(params=self._build_params())
            except Exception as exc:  # pragma: no cover - depends on install
                raise _EngineUnavailable(f"rapidocr import failed: {exc}") from exc

        # Injected runtimes may be a zero-arg callable or accept ``params=``.
        try:
            return loader()
        except TypeError:
            pass
        except Exception as exc:
            raise _EngineUnavailable(f"rapidocr init failed: {exc}") from exc
        try:
            return loader(self._build_params())
        except TypeError:
            try:
                return loader(params=self._build_params())
            except Exception as exc:
                raise _EngineUnavailable(f"rapidocr init failed: {exc}") from exc
        except Exception as exc:
            raise _EngineUnavailable(f"rapidocr init failed: {exc}") from exc

    # -- inference ----------------------------------------------------
    def recognize(self, image_path: str | Path) -> List[OCRLine]:
        engine = self.load()
        with self._lock:
            try:
                output = engine(str(image_path))
            except Exception as exc:
                raise _InferenceFailed(f"rapidocr inference failed: {exc}") from exc
        return self._parse_output(output)

    def _parse_output(self, output: Any) -> List[OCRLine]:
        """Convert RapidOCR output into :class:`OCRLine` objects.

        Accepts the ``RapidOCROutput`` dataclass (``.txts``/``.scores``/
        ``.boxes``), a ``to_json()``-style list of dicts, or a JSON string.
        """
        if output is None:
            return []
        records = output
        if hasattr(output, "to_json"):
            try:
                records = output.to_json()
            except Exception:
                records = None
        elif hasattr(output, "txts"):
            records = [
                {"txt": txt, "score": score, "box": box}
                for txt, score, box in zip(output.txts or (), output.scores or (), output.boxes or ())
            ]
        if records is None:
            return []
        if isinstance(records, str):
            import json

            records = json.loads(records)
        identity = self.model_identity()
        lines: List[OCRLine] = []
        for record in records or ():
            if not isinstance(record, dict):
                continue
            text = str(record.get("txt") or "").strip()
            if not text:
                continue
            try:
                confidence = float(record.get("score", 0.0))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence) or confidence < self.min_text_score:
                # Enforce the configured floor after provider parsing too: a
                # custom/older provider must not bypass Global.text_score.
                continue
            confidence = max(0.0, min(1.0, confidence))
            bbox = _normalize_bbox(record.get("box"))
            lines.append(
                OCRLine(text=text, confidence=confidence, bbox=bbox, **identity)
            )
        return lines


def _normalize_bbox(box: Any) -> List[List[float]]:
    """Coerce a RapidOCR quad box into four finite, non-zero-area points."""
    points: List[List[float]] = []
    if box is not None:
        for point in box:
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, IndexError, ValueError):
                return []
            if not math.isfinite(x) or not math.isfinite(y):
                return []
            points.append([x, y])
    if len(points) != 4:
        return []
    area = abs(sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )) / 2.0
    return points if area > 1e-6 else []


class OCRAnalyzer(BaseAnalyzer):
    """Pipeline analyzer exposing per-line signage clues from OCR."""

    name = "ocr"

    def __init__(
        self,
        provider: str = MOCK_PROVIDER,
        *,
        model_type: str = DEFAULT_MODEL_TYPE,
        model_dir: str | Path | None = None,
        det_model_path: str | Path | None = None,
        rec_model_path: str | Path | None = None,
        cls_model_path: str | Path | None = None,
        min_text_score: float = DEFAULT_MIN_TEXT_SCORE,
        engine: Any | None = None,
    ) -> None:
        self.provider = normalize_provider(provider)
        self.model_type = normalize_ocr_model_type(model_type)
        if engine is not None and hasattr(engine, "model_type"):
            self.model_type = normalize_ocr_model_type(engine.model_type)
        self.model_dir = model_dir
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.cls_model_path = cls_model_path
        self.min_text_score = min_text_score
        self.last_status = OCRStatus(
            provider=self.provider, model_type=self.model_type
        )
        if engine is None:
            self._engine = RapidOCREngine(
                model_type=self.model_type,
                model_dir=model_dir,
                det_model_path=det_model_path,
                rec_model_path=rec_model_path,
                cls_model_path=cls_model_path,
                min_text_score=min_text_score,
            )
        elif hasattr(engine, "recognize"):
            self._engine = engine
        else:
            # Accept a bare RapidOCR-compatible runtime and serialize it.
            self._engine = RapidOCREngine(
                model_type=self.model_type, loader=lambda _params: engine
            )
        self._refresh_status_identity()

    def _identity(self) -> Dict[str, str]:
        identity = getattr(self._engine, "model_identity", None)
        if callable(identity):
            try:
                return dict(identity())
            except Exception:
                pass
        profile = MODEL_PROFILES[self.model_type]
        return {
            "det_model": _model_label(self.det_model_path, profile.det_model),
            "rec_model": _model_label(self.rec_model_path, profile.rec_model),
            "cls_model": _model_label(self.cls_model_path, profile.cls_model),
            "model_type": self.model_type,
            "model_version": PPOCR_VERSION,
        }

    def _refresh_status_identity(self) -> None:
        identity = self._identity()
        self.model_type = normalize_ocr_model_type(identity["model_type"])
        self.last_status.model_type = self.model_type
        self.last_status.models = {
            "det": identity["det_model"],
            "rec": identity["rec_model"],
            "cls": identity["cls_model"],
        }

    # ``mode`` kept as an alias so legacy call sites keep working.
    @property
    def mode(self) -> str:
        return self.provider

    def status(self) -> Dict[str, Any]:
        """Safe diagnostic snapshot with basename-only model provenance."""
        return self.last_status.to_dict()

    def _fail(self, state: str, detail: str) -> List[Clue]:
        detail = sanitize_diagnostic(detail)
        self.last_status.state = state
        self.last_status.detail = detail
        self.last_status.line_count = 0
        logger.warning("[OCRAnalyzer] OCR provider '%s' %s: %s", self.provider, state, detail)
        return []

    def analyze(self, image_path: str) -> List[Clue]:
        if self.provider == MOCK_PROVIDER:
            self._refresh_status_identity()
            self.last_status.state = "ok"
            self.last_status.detail = "mock provider: no OCR performed"
            self.last_status.line_count = 0
            return []
        if self.provider not in VALID_PROVIDERS:
            return self._fail("unavailable", f"unknown OCR provider: {self.provider!r}")

        try:
            lines = self._filter_lines(self._engine.recognize(image_path))
        except _EngineUnavailable as exc:
            return self._fail("unavailable", str(exc))
        except _InferenceFailed as exc:
            return self._fail("failed", str(exc))
        except Exception as exc:  # defensive: never take the pipeline down
            return self._fail("failed", type(exc).__name__)

        self._refresh_status_identity()
        self.last_status.state = "ok"
        self.last_status.engine = OCR_ENGINE
        self.last_status.model_version = PPOCR_VERSION
        self.last_status.detail = None
        self.last_status.line_count = len(lines)

        clues: List[Clue] = []
        for line in lines:
            clues.append(
                Clue(
                    dimension="signage",
                    value=line.text[:200],
                    confidence=line.confidence,
                    evidence=f"OCR 识别到文字: {line.text[:80]}",
                    source=self.name,
                    extra=line.to_extra(),
                )
            )
        return clues

    def _filter_lines(self, lines: Sequence[OCRLine]) -> List[OCRLine]:
        """Apply the score floor to provider output, including injected engines."""
        accepted: List[OCRLine] = []
        for line in lines or ():
            if not isinstance(line, OCRLine):
                continue
            text = str(line.text or "").strip()
            try:
                confidence = float(line.confidence)
            except (TypeError, ValueError):
                continue
            if not text or not math.isfinite(confidence) or confidence < self.min_text_score:
                continue
            line.text = text
            line.confidence = max(0.0, min(1.0, confidence))
            line.bbox = _normalize_bbox(line.bbox)
            accepted.append(line)
        return accepted

    def analyze_lines(self, image_path: str) -> Tuple[List[Clue], List[OCRLine]]:
        """Return both clues and raw :class:`OCRLine` objects (used by pipeline)."""
        clues = self.analyze(image_path)
        if self.provider != MOCK_PROVIDER and self.last_status.state != "ok":
            return clues, []
        return clues, _lines_from_clues(clues)


def _lines_from_clues(clues: Sequence[Clue]) -> List[OCRLine]:
    lines: List[OCRLine] = []
    for clue in clues:
        extra = clue.extra or {}
        lines.append(
            OCRLine(
                text=clue.value,
                confidence=float(clue.confidence),
                bbox=extra.get("bbox") or extra.get("quad") or [],
                provider=str(extra.get("provider") or RAPIDOCR_PROVIDER),
                det_model=str(extra.get("det_model") or SERVER_MODEL_PROFILE.det_model),
                rec_model=str(extra.get("rec_model") or SERVER_MODEL_PROFILE.rec_model),
                cls_model=str(extra.get("cls_model") or SERVER_MODEL_PROFILE.cls_model),
                model_type=str(extra.get("model_type") or SERVER_MODEL_PROFILE.name),
                model_version=str(extra.get("model_version") or PPOCR_VERSION),
            )
        )
    return lines
