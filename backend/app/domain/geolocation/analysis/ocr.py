"""OCR analyzer backed by RapidOCR + ONNX Runtime.

Modes
-----
"mock"
    No OCR engine is imported or downloaded; analyze returns an empty clue list so the pipeline stays offline and deterministic.

"rapidocr"
    Lazily loads RapidOCR with the selected PP-OCRv5 det/cls/rec profile.
    device='cpu' uses the in-process CPU runtime. device='cuda' uses a
    loopback GPU OCR service so the backend process never imports the CPU and
    GPU ONNX Runtime distributions together.

CUDA is fail-closed: the requested device must match the actual execution
providers of every det/cls/rec session. RapidOCR itself includes a CPU fallback
provider, so configuration flags alone are never treated as proof that CUDA was
used. A provider mismatch returns no OCR clues rather than mislabeling them.

Thread safety
-------------
Engine construction and inference are serialized behind one instance lock, so a
single RapidOCR session is never shared across concurrent analyzer threads.
"""
from __future__ import annotations

import json
import logging
import math
import platform
import re
import threading
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

from .base import BaseAnalyzer
from ..clue import Clue
from ....config import (
    DEFAULT_OCR_CPU_INTER_OP_THREADS,
    DEFAULT_OCR_CPU_INTRA_OP_THREADS,
    DEFAULT_OCR_DEVICE,
    DEFAULT_OCR_GPU_TIMEOUT_SECONDS,
    DEFAULT_OCR_GPU_URL,
    DEFAULT_OCR_MODEL_TYPE,
    normalize_ocr_device,
    normalize_ocr_model_type,
)

logger = logging.getLogger("m_photo.ocr")

MOCK_PROVIDER = "mock"
RAPIDOCR_PROVIDER = "rapidocr"
VALID_PROVIDERS = (MOCK_PROVIDER, RAPIDOCR_PROVIDER)

PPOCR_VERSION = "PP-OCRv5"
OCR_ENGINE = "onnxruntime"
DEFAULT_MODEL_TYPE = DEFAULT_OCR_MODEL_TYPE
CUDA_EXECUTION_PROVIDER = "CUDAExecutionProvider"
CPU_EXECUTION_PROVIDER = "CPUExecutionProvider"


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
DET_LIMIT_SIDE_LEN = 1280
DET_LIMIT_TYPE = "max"

_LEGACY_ALIASES = {"real": RAPIDOCR_PROVIDER, "true": RAPIDOCR_PROVIDER, "1": RAPIDOCR_PROVIDER}

_URL_RE = re.compile(r"https?://[^\s'\"<>]+", re.IGNORECASE)
_WIN_PATH_RE = re.compile(r"[A-Za-z]:\[^\s'\"<>|]+")
_UNC_PATH_RE = re.compile(r"\\[^\s'\"<>|]+")
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


def _distribution_version(*names: str) -> str | None:
    for name in names:
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
        except Exception:
            continue
    return None


def _runtime_metadata(requested_device: str) -> Dict[str, Any]:
    ort_distribution = (
        "onnxruntime-gpu" if _distribution_version("onnxruntime-gpu") else "onnxruntime"
    )
    return {
        "python": platform.python_version(),
        "rapidocr": _distribution_version("rapidocr"),
        "onnxruntime": _distribution_version("onnxruntime-gpu", "onnxruntime"),
        "onnxruntime_distribution": ort_distribution,
        "requested_device": requested_device,
    }


def _copy_providers(value: Any) -> Dict[str, List[str]]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, List[str]] = {}
    for task in ("det", "cls", "rec"):
        providers = value.get(task)
        if isinstance(providers, (list, tuple)):
            result[task] = [str(item) for item in providers if str(item)]
    return result


def _copy_runtime(value: Any, fallback: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    return dict(fallback)


@dataclass
class OCRLine:
    """One recognized text line with runtime and model provenance intact."""

    text: str
    confidence: float
    bbox: List[List[float]]
    provider: str = RAPIDOCR_PROVIDER
    det_model: str = SERVER_MODEL_PROFILE.det_model
    rec_model: str = SERVER_MODEL_PROFILE.rec_model
    cls_model: str = SERVER_MODEL_PROFILE.cls_model
    model_type: str = SERVER_MODEL_PROFILE.name
    model_version: str = PPOCR_VERSION
    requested_device: str = DEFAULT_OCR_DEVICE
    effective_device: str | None = DEFAULT_OCR_DEVICE
    transport: str = "inprocess"
    execution_provider: str | None = CPU_EXECUTION_PROVIDER
    execution_providers: Dict[str, List[str]] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)

    def to_extra(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {
            "origin_source": "ocr",
            "provider": self.provider,
            "reliable": True,
            "line_confidence": self.confidence,
            "det_model": self.det_model,
            "rec_model": self.rec_model,
            "cls_model": self.cls_model,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "requested_device": self.requested_device,
            "effective_device": self.effective_device,
            "transport": self.transport,
            "execution_provider": self.execution_provider,
            "execution_providers": {
                task: list(providers)
                for task, providers in self.execution_providers.items()
            },
            "runtime": dict(self.runtime),
        }
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
    requested_device: str = DEFAULT_OCR_DEVICE
    effective_device: str | None = None
    transport: str = "inprocess"
    execution_provider: str | None = None
    execution_providers: Dict[str, List[str]] = field(default_factory=dict)
    runtime: Dict[str, Any] = field(default_factory=dict)

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
            "requested_device": self.requested_device,
            "effective_device": self.effective_device,
            "transport": self.transport,
            "execution_provider": self.execution_provider,
            "execution_providers": {
                task: list(providers)
                for task, providers in self.execution_providers.items()
            },
            "runtime": dict(self.runtime),
        }


def normalize_provider(value: str | None, default: str = MOCK_PROVIDER) -> str:
    """Resolve OCR_PROVIDER input; unknown values fall through safely."""
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


def _session_providers(runtime: Any, attribute: str) -> List[str]:
    model = getattr(runtime, attribute, None)
    session_wrapper = getattr(model, "session", None)
    session = getattr(session_wrapper, "session", session_wrapper)
    getter = getattr(session, "get_providers", None)
    if not callable(getter):
        return []
    try:
        return [str(item) for item in getter() if str(item)]
    except Exception:
        return []


def _extract_providers(runtime: Any) -> Dict[str, List[str]]:
    return {
        "det": _session_providers(runtime, "text_det"),
        "cls": _session_providers(runtime, "text_cls"),
        "rec": _session_providers(runtime, "text_rec"),
    }


def _providers_are_cuda_first(providers: Dict[str, List[str]]) -> bool:
    return all(
        providers.get(task)
        and str(providers[task][0]) == CUDA_EXECUTION_PROVIDER
        for task in ("det", "cls", "rec")
    )


def _force_real_session_load(runtime: Any) -> None:
    for method_name in ("_load_det_model", "_load_cls_model", "_load_rec_model"):
        loader = getattr(runtime, method_name, None)
        if callable(loader):
            loader()


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


def _parse_ocr_records(
    records: Any,
    *,
    min_text_score: float,
    identity: Dict[str, Any],
    runtime: Dict[str, Any],
) -> List[OCRLine]:
    if records is None:
        return []
    if isinstance(records, str):
        try:
            records = json.loads(records)
        except (TypeError, ValueError):
            return []
    if not isinstance(records, (list, tuple)):
        return []

    execution_providers = _copy_providers(runtime.get("execution_providers"))
    line_runtime = _copy_runtime(runtime.get("runtime"), {})
    requested_device = str(runtime.get("requested_device") or DEFAULT_OCR_DEVICE)
    effective_device = runtime.get("effective_device")
    transport = str(runtime.get("transport") or "inprocess")
    execution_provider = runtime.get("execution_provider")

    lines: List[OCRLine] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        text = str(record.get("txt") or record.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(record.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(confidence) or confidence < min_text_score:
            continue
        confidence = max(0.0, min(1.0, confidence))
        bbox = _normalize_bbox(record.get("box") or record.get("bbox"))
        lines.append(
            OCRLine(
                text=text,
                confidence=confidence,
                bbox=bbox,
                provider=str(identity.get("provider") or RAPIDOCR_PROVIDER),
                det_model=str(identity.get("det_model") or SERVER_MODEL_PROFILE.det_model),
                rec_model=str(identity.get("rec_model") or SERVER_MODEL_PROFILE.rec_model),
                cls_model=str(identity.get("cls_model") or SERVER_MODEL_PROFILE.cls_model),
                model_type=str(identity.get("model_type") or SERVER_MODEL_PROFILE.name),
                model_version=str(identity.get("model_version") or PPOCR_VERSION),
                requested_device=requested_device,
                effective_device=str(effective_device) if effective_device else None,
                transport=transport,
                execution_provider=(
                    str(execution_provider) if execution_provider else None
                ),
                execution_providers={
                    task: list(providers)
                    for task, providers in execution_providers.items()
                },
                runtime=dict(line_runtime),
            )
        )
    return lines


class RapidOCREngine:
    """Lazy adapter around rapidocr.RapidOCR for one explicit device."""

    def __init__(
        self,
        *,
        model_type: str = DEFAULT_MODEL_TYPE,
        model_dir: str | Path | None = None,
        det_model_path: str | Path | None = None,
        rec_model_path: str | Path | None = None,
        cls_model_path: str | Path | None = None,
        min_text_score: float = DEFAULT_MIN_TEXT_SCORE,
        device: str = DEFAULT_OCR_DEVICE,
        cpu_intra_op_threads: int = DEFAULT_OCR_CPU_INTRA_OP_THREADS,
        cpu_inter_op_threads: int = DEFAULT_OCR_CPU_INTER_OP_THREADS,
        loader: Callable[..., Any] | None = None,
    ) -> None:
        self.model_type = normalize_ocr_model_type(model_type)
        self.model_dir = str(model_dir) if model_dir else None
        self.det_model_path = str(det_model_path) if det_model_path else None
        self.rec_model_path = str(rec_model_path) if rec_model_path else None
        self.cls_model_path = str(cls_model_path) if cls_model_path else None
        self.min_text_score = float(min_text_score)
        self.device = normalize_ocr_device(device)
        self.cpu_intra_op_threads = max(1, int(cpu_intra_op_threads))
        self.cpu_inter_op_threads = max(1, int(cpu_inter_op_threads))
        self._loader = loader
        self._engine: Any | None = None
        self._lock = threading.Lock()
        self._execution_providers: Dict[str, List[str]] = {}
        self._effective_device: str | None = None
        self._runtime: Dict[str, Any] = _runtime_metadata(self.device)

    @property
    def profile(self) -> OCRModelProfile:
        return MODEL_PROFILES[self.model_type]

    def model_identity(self) -> Dict[str, str]:
        profile = self.profile
        return {
            "provider": RAPIDOCR_PROVIDER,
            "det_model": _model_label(self.det_model_path, profile.det_model),
            "rec_model": _model_label(self.rec_model_path, profile.rec_model),
            "cls_model": _model_label(self.cls_model_path, profile.cls_model),
            "model_type": self.model_type,
            "model_version": PPOCR_VERSION,
        }

    def runtime_identity(self) -> Dict[str, Any]:
        return dict(self._runtime)

    def provider_status(self) -> Dict[str, Any]:
        providers = {
            task: list(values) for task, values in self._execution_providers.items()
        }
        first = providers.get("det", [None])[0]
        return {
            "requested_device": self.device,
            "effective_device": self._effective_device,
            "transport": "inprocess",
            "execution_provider": first,
            "execution_providers": providers,
            "runtime": dict(self._runtime),
        }

    def _build_params(self) -> Dict[str, Any]:
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
            "EngineConfig.onnxruntime.use_cuda": self.device == "cuda",
            "EngineConfig.onnxruntime.intra_op_num_threads": self.cpu_intra_op_threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": self.cpu_inter_op_threads,
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
        if self.device == "cuda":
            params.update(
                {
                    "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": 0,
                    "EngineConfig.onnxruntime.cuda_ep_cfg.arena_extend_strategy": "kSameAsRequested",
                    "EngineConfig.onnxruntime.cuda_ep_cfg.cudnn_conv_algo_search": "HEURISTIC",
                    "EngineConfig.onnxruntime.cuda_ep_cfg.do_copy_in_default_stream": True,
                }
            )
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

    def _preflight_cuda(self) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise _EngineUnavailable(f"onnxruntime import failed: {exc}") from exc
        providers = [str(item) for item in ort.get_available_providers()]
        device = str(ort.get_device())
        if CUDA_EXECUTION_PROVIDER not in providers or device != "GPU":
            raise _EngineUnavailable(
                "cuda requested but CUDAExecutionProvider is unavailable"
            )

    def _create_engine(self) -> Any:
        loader = self._loader
        if loader is None:
            try:
                if self.device == "cuda":
                    self._preflight_cuda()
                from rapidocr import RapidOCR as runtime

                engine = runtime(params=self._build_params())
                try:
                    _force_real_session_load(engine)
                except Exception as exc:
                    raise _EngineUnavailable(f"rapidocr model load failed: {exc}") from exc
                self._execution_providers = _extract_providers(engine)
                if self.device == "cuda" and not _providers_are_cuda_first(
                    self._execution_providers
                ):
                    raise _EngineUnavailable(
                        "cuda requested but det/cls/rec did not all use CUDAExecutionProvider"
                    )
                if not _providers_are_cuda_first(self._execution_providers) and self.device == "cpu":
                    first = self._execution_providers.get("det") or []
                    if first and str(first[0]) != CPU_EXECUTION_PROVIDER:
                        raise _EngineUnavailable(
                            "cpu requested but sessions did not use CPUExecutionProvider"
                        )
                self._effective_device = self.device
                self._runtime = _runtime_metadata(self.device)
                import onnxruntime as ort

                self._runtime["available_providers"] = [
                    str(item) for item in ort.get_available_providers()
                ]
                return engine
            except _EngineUnavailable:
                raise
            except Exception as exc:
                raise _EngineUnavailable(f"rapidocr import failed: {exc}") from exc

        try:
            return loader()
        except TypeError:
            pass
        except Exception as exc:
            raise _EngineUnavailable(f"rapidocr init failed: {exc}") from exc
        try:
            engine = loader(self._build_params())
        except TypeError:
            try:
                engine = loader(params=self._build_params())
            except Exception as exc:
                raise _EngineUnavailable(f"rapidocr init failed: {exc}") from exc
        except Exception as exc:
            raise _EngineUnavailable(f"rapidocr init failed: {exc}") from exc
        self._effective_device = self.device
        self._runtime = _runtime_metadata(self.device)
        self._runtime["engine_source"] = "injected"
        return engine

    def _parse_output(self, output: Any) -> List[OCRLine]:
        records = output
        if hasattr(output, "to_json"):
            try:
                records = output.to_json()
            except Exception:
                records = None
        elif hasattr(output, "txts"):
            records = [
                {"txt": txt, "score": score, "box": box}
                for txt, score, box in zip(
                    output.txts or (), output.scores or (), output.boxes or ()
                )
            ]
        return _parse_ocr_records(
            records,
            min_text_score=self.min_text_score,
            identity=self.model_identity(),
            runtime=self.provider_status(),
        )

    def recognize(self, image_path: str | Path) -> List[OCRLine]:
        engine = self.load()
        with self._lock:
            try:
                output = engine(str(image_path))
            except Exception as exc:
                raise _InferenceFailed(f"rapidocr inference failed: {exc}") from exc
            return self._parse_output(output)


class RemoteOCREngine:
    """HTTP adapter for the isolated CUDA RapidOCR service."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_OCR_GPU_URL,
        api_key: str | None = None,
        timeout_seconds: float = DEFAULT_OCR_GPU_TIMEOUT_SECONDS,
        model_type: str = DEFAULT_MODEL_TYPE,
        min_text_score: float = DEFAULT_MIN_TEXT_SCORE,
        client: Any | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.api_key = str(api_key or "")
        self.timeout_seconds = float(timeout_seconds)
        self.model_type = normalize_ocr_model_type(model_type)
        self.min_text_score = float(min_text_score)
        self._client = client
        self._owns_client = client is None
        self._engine: Any | None = None
        self._lock = threading.Lock()
        self._status: Dict[str, Any] = {}
        self._runtime = _runtime_metadata("cuda")

    @property
    def device(self) -> str:
        return "cuda"

    def _client_instance(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(
                base_url=self.base_url,
                timeout=self.timeout_seconds,
                trust_env=False,
            )
        return self._client

    def _headers(self) -> Dict[str, str]:
        if not self.api_key:
            return {}
        return {"Authorization": f"Bearer {self.api_key}"}

    def _validate_status(self, status: Any, *, require_ok: bool = True) -> Dict[str, Any]:
        if not isinstance(status, dict):
            raise _EngineUnavailable("GPU OCR service returned no status")
        state = str(status.get("state") or "unavailable")
        if require_ok and state != "ok":
            detail = sanitize_diagnostic(str(status.get("detail") or state))
            raise _EngineUnavailable(f"GPU OCR service is {state}: {detail}")
        if str(status.get("provider") or "") != RAPIDOCR_PROVIDER:
            raise _EngineUnavailable("GPU OCR service provider mismatch")
        if str(status.get("model_type") or "") != self.model_type:
            raise _EngineUnavailable("GPU OCR service model profile mismatch")
        if str(status.get("model_version") or "") != PPOCR_VERSION:
            raise _EngineUnavailable("GPU OCR service model version mismatch")
        models = status.get("models")
        if not isinstance(models, dict) or not all(
            models.get(key) for key in ("det", "cls", "rec")
        ):
            raise _EngineUnavailable("GPU OCR service model provenance is incomplete")
        if str(status.get("requested_device") or "").lower() != "cuda":
            raise _EngineUnavailable("GPU OCR service did not request CUDA")
        if str(status.get("effective_device") or "").lower() != "cuda":
            raise _EngineUnavailable("GPU OCR service did not run on CUDA")
        providers = _copy_providers(status.get("execution_providers"))
        if not _providers_are_cuda_first(providers):
            raise _EngineUnavailable(
                "GPU OCR service det/cls/rec sessions are not CUDA-first"
            )
        return status

    def _record_status(self, status: Dict[str, Any]) -> None:
        self._status = dict(status)
        self._runtime = _copy_runtime(status.get("runtime"), _runtime_metadata("cuda"))
        self._runtime.setdefault("requested_device", "cuda")

    def load(self) -> Any:
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                response = self._client_instance().get("/health", headers=self._headers())
                if response.status_code >= 400:
                    raise _EngineUnavailable(
                        f"GPU OCR health check failed with HTTP {response.status_code}"
                    )
                status = self._validate_status(response.json())
                self._record_status(status)
                self._engine = self
                return self._engine
            except _EngineUnavailable:
                raise
            except Exception as exc:
                raise _EngineUnavailable(f"GPU OCR service unavailable: {exc}") from exc

    def model_identity(self) -> Dict[str, str]:
        models = self._status.get("models") if isinstance(self._status, dict) else {}
        if not isinstance(models, dict):
            models = {}
        profile = MODEL_PROFILES[self.model_type]
        return {
            "provider": RAPIDOCR_PROVIDER,
            "det_model": str(models.get("det") or profile.det_model),
            "rec_model": str(models.get("rec") or profile.rec_model),
            "cls_model": str(models.get("cls") or profile.cls_model),
            "model_type": str(self._status.get("model_type") or self.model_type),
            "model_version": str(self._status.get("model_version") or PPOCR_VERSION),
        }

    def runtime_identity(self) -> Dict[str, Any]:
        return dict(self._runtime)

    def provider_status(self) -> Dict[str, Any]:
        providers = _copy_providers(self._status.get("execution_providers"))
        first = providers.get("det", [None])[0] if providers.get("det") else None
        return {
            "requested_device": str(
                self._status.get("requested_device") or "cuda"
            ),
            "effective_device": self._status.get("effective_device"),
            "transport": "http",
            "execution_provider": first,
            "execution_providers": providers,
            "runtime": dict(self._runtime),
        }

    def recognize(self, image_path: str | Path) -> List[OCRLine]:
        self.load()
        try:
            payload = Path(image_path).read_bytes()
        except Exception as exc:
            raise _InferenceFailed(f"image read failed: {exc}") from exc
        with self._lock:
            try:
                response = self._client_instance().post(
                    "/recognize",
                    content=payload,
                    headers={
                        **self._headers(),
                        "Content-Type": "application/octet-stream",
                    },
                )
            except Exception as exc:
                raise _EngineUnavailable(f"GPU OCR request failed: {exc}") from exc
            if response.status_code == 503:
                detail = ""
                try:
                    body = response.json()
                    detail = str(body.get("detail") or "")
                except Exception:
                    pass
                raise _EngineUnavailable(
                    f"GPU OCR unavailable: {sanitize_diagnostic(detail)}"
                )
            if response.status_code >= 400:
                detail = ""
                try:
                    body = response.json()
                    detail = str(body.get("detail") or "")
                except Exception:
                    pass
                raise _InferenceFailed(
                    f"GPU OCR inference failed with HTTP {response.status_code}: "
                    f"{sanitize_diagnostic(detail)}"
                )
            try:
                body = response.json()
            except Exception as exc:
                raise _InferenceFailed("GPU OCR returned invalid JSON") from exc
            status = self._validate_status(body.get("status"))
            self._record_status(status)
            return _parse_ocr_records(
                body.get("lines"),
                min_text_score=self.min_text_score,
                identity=self.model_identity(),
                runtime=self.provider_status(),
            )

    def close(self) -> None:
        if self._owns_client and self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._engine = None


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
        device: str = DEFAULT_OCR_DEVICE,
        gpu_url: str | None = None,
        gpu_api_key: str | None = None,
        gpu_timeout_seconds: float = DEFAULT_OCR_GPU_TIMEOUT_SECONDS,
        cpu_intra_op_threads: int = DEFAULT_OCR_CPU_INTRA_OP_THREADS,
        cpu_inter_op_threads: int = DEFAULT_OCR_CPU_INTER_OP_THREADS,
        engine: Any | None = None,
    ) -> None:
        self.provider = normalize_provider(provider)
        self.model_type = normalize_ocr_model_type(model_type)
        self.device = normalize_ocr_device(device)
        if engine is not None and hasattr(engine, "device"):
            try:
                self.device = normalize_ocr_device(str(engine.device))
            except (TypeError, ValueError):
                pass
        if engine is not None and hasattr(engine, "model_type"):
            self.model_type = normalize_ocr_model_type(engine.model_type)
        self.model_dir = model_dir
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.cls_model_path = cls_model_path
        self.min_text_score = min_text_score
        self.last_status = OCRStatus(
            provider=self.provider,
            model_type=self.model_type,
            requested_device=self.device,
            transport="http" if self.device == "cuda" else "inprocess",
            runtime=_runtime_metadata(self.device),
        )
        if engine is None:
            if self.provider == RAPIDOCR_PROVIDER and self.device == "cuda":
                self._engine = RemoteOCREngine(
                    base_url=gpu_url or DEFAULT_OCR_GPU_URL,
                    api_key=gpu_api_key,
                    timeout_seconds=gpu_timeout_seconds,
                    model_type=self.model_type,
                    min_text_score=min_text_score,
                )
            else:
                self._engine = RapidOCREngine(
                    model_type=self.model_type,
                    model_dir=model_dir,
                    det_model_path=det_model_path,
                    rec_model_path=rec_model_path,
                    cls_model_path=cls_model_path,
                    min_text_score=min_text_score,
                    device=self.device,
                    cpu_intra_op_threads=cpu_intra_op_threads,
                    cpu_inter_op_threads=cpu_inter_op_threads,
                )
        elif hasattr(engine, "recognize"):
            self._engine = engine
        else:
            self._engine = RapidOCREngine(
                model_type=self.model_type,
                min_text_score=min_text_score,
                device=self.device,
                cpu_intra_op_threads=cpu_intra_op_threads,
                cpu_inter_op_threads=cpu_inter_op_threads,
                loader=lambda _params: engine,
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
            "provider": RAPIDOCR_PROVIDER,
            "det_model": _model_label(self.det_model_path, profile.det_model),
            "rec_model": _model_label(self.rec_model_path, profile.rec_model),
            "cls_model": _model_label(self.cls_model_path, profile.cls_model),
            "model_type": self.model_type,
            "model_version": PPOCR_VERSION,
        }

    def _engine_provider_status(self) -> Dict[str, Any]:
        method = getattr(self._engine, "provider_status", None)
        if not callable(method):
            return {}
        try:
            value = method()
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}

    def _refresh_status_identity(self) -> None:
        identity = self._identity()
        self.model_type = normalize_ocr_model_type(identity["model_type"])
        self.last_status.model_type = self.model_type
        self.last_status.models = {
            "det": identity["det_model"],
            "rec": identity["rec_model"],
            "cls": identity["cls_model"],
        }
        self.last_status.model_version = str(identity.get("model_version") or PPOCR_VERSION)
        provider_status = self._engine_provider_status()
        if provider_status:
            requested = provider_status.get("requested_device")
            if requested:
                try:
                    self.device = normalize_ocr_device(str(requested))
                except ValueError:
                    pass
            self.last_status.requested_device = self.device
            effective = provider_status.get("effective_device")
            self.last_status.effective_device = (
                str(effective) if effective else None
            )
            self.last_status.transport = str(
                provider_status.get("transport")
                or ("http" if self.device == "cuda" else "inprocess")
            )
            execution_provider = provider_status.get("execution_provider")
            self.last_status.execution_provider = (
                str(execution_provider) if execution_provider else None
            )
            self.last_status.execution_providers = _copy_providers(
                provider_status.get("execution_providers")
            )
            self.last_status.runtime = _copy_runtime(
                provider_status.get("runtime"),
                self.last_status.runtime,
            )

    @property
    def mode(self) -> str:
        return self.provider

    def status(self) -> Dict[str, Any]:
        """Safe raw diagnostic snapshot with model/runtime provenance."""
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
            self.last_status.effective_device = None
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
        if not self.last_status.model_version:
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
        """Return both clues and raw OCRLine objects used by the pipeline."""
        clues = self.analyze(image_path)
        if self.provider != MOCK_PROVIDER and self.last_status.state != "ok":
            return clues, []
        return clues, _lines_from_clues(clues)


def _lines_from_clues(clues: Sequence[Clue]) -> List[OCRLine]:
    lines: List[OCRLine] = []
    for clue in clues:
        extra = clue.extra or {}
        providers = _copy_providers(extra.get("execution_providers"))
        runtime = _copy_runtime(extra.get("runtime"), {})
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
                requested_device=str(extra.get("requested_device") or DEFAULT_OCR_DEVICE),
                effective_device=(
                    str(extra["effective_device"])
                    if extra.get("effective_device")
                    else None
                ),
                transport=str(extra.get("transport") or "inprocess"),
                execution_provider=(
                    str(extra["execution_provider"])
                    if extra.get("execution_provider")
                    else None
                ),
                execution_providers=providers,
                runtime=runtime,
            )
        )
    return lines
