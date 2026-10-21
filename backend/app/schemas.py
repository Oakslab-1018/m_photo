"""Pydantic request/response models for the public API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_serializer, model_validator

FeedbackType = Literal["confirm", "reject", "answer", "free_text"]
MessageRole = Literal["user", "ai", "system"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    ai_mode: str
    data_dir: str
    version: str
    streetclip_enabled: bool = False
    deepseek_configured: bool = False
    deepseek_model: str = "deepseek-flash"


class CountryModel(BaseModel):
    """Country knowledge-base entry (schema kept compatible with data/countries.json)."""

    iso: str
    name: str
    name_zh: str
    lat: float = 0.0
    lng: float = 0.0
    continent: str | None = None
    driving_side: str | None = None
    languages_spoken: list[str] = Field(default_factory=list)
    features: dict[str, list[str]] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class CountriesResponse(BaseModel):
    countries: list[CountryModel]


class ClueModel(BaseModel):
    dimension: str
    value: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: str = ""
    source: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class MatchedClueModel(BaseModel):
    dimension: str
    value: str
    gain: float = 0.0
    evidence: str = ""

    model_config = {"extra": "allow"}


class CountryScoreModel(BaseModel):
    raw_cosine: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    score_type: str = "rule_weight"
    iso: str
    name: str
    name_zh: str
    score: float
    matched_clues: list[MatchedClueModel] = Field(default_factory=list)
    lat: float = 0.0
    lng: float = 0.0
    # Optional non-probabilistic fusion breakdown (full cosine + crop mean + clue boost).
    fusion: dict[str, Any] | None = None
    # Grouped-country provenance. These fields stay absent for ordinary
    # single-source countries so existing response shapes are preserved.
    parent_iso: str | None = None
    source_regions: list[dict[str, Any]] | None = None
    is_grouped: bool = False
    # ``reference_matches`` / ``reference_regions`` / ``reference_cosine`` are
    # deliberately *not* declared: they must stay absent from the JSON unless a
    # labeled reference index is configured, and ``extra="allow"`` passes them
    # through only when the engine attached them. Declaring them would emit
    # ``null``/``[]`` for every deployment and make "no library configured"
    # indistinguishable from "configured, nothing matched".

    model_config = {"extra": "allow"}

    @model_serializer(mode="wrap")
    def _drop_absent_group_metadata(self, handler):
        data = handler(self)
        if data.get("parent_iso") is None:
            data.pop("parent_iso", None)
        if data.get("source_regions") is None:
            data.pop("source_regions", None)
        if data.get("is_grouped") is not True:
            data.pop("is_grouped", None)
        return data


class ActionModel(BaseModel):
    kind: str
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


class ConfidenceRouteModel(BaseModel):
    """Deterministic product routing; neither margin is a probability."""

    tier: Literal["high", "medium", "low"]
    action: Literal["direct_top1", "confirm_candidates", "ask_clue"]
    model_only_margin: float | None = Field(default=None, allow_inf_nan=False)
    fused_margin: float | None = Field(default=None, allow_inf_nan=False)
    effective_margin: float | None = Field(default=None, allow_inf_nan=False)
    prior_top_iso: str | None = None
    top1_iso: str | None = None
    top3: list[CountryScoreModel] = Field(default_factory=list)
    leader_changed: bool = False
    reason: str = ""
    policy_version: str

    model_config = {"extra": "forbid"}


class ChatMessageModel(BaseModel):
    id: int
    role: MessageRole
    content: str
    meta: dict[str, Any] = Field(default_factory=dict)
    created_at: str | None = None

    model_config = {"extra": "allow"}


class UploadResponse(BaseModel):
    session_id: int
    image_url: str
    image_filename: str
    clues: list[ClueModel] = Field(default_factory=list)
    scores: list[CountryScoreModel] = Field(default_factory=list)
    action: ActionModel | None = None
    confidence: ConfidenceRouteModel | None = None
    inference_notice: str | None = None
    messages: list[ChatMessageModel] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    """Feedback payload; required fields depend on ``type`` (mirrors the UI)."""

    type: FeedbackType
    iso: str | None = Field(default=None, max_length=10)
    dimension: str | None = None
    value: str | None = None
    text: str | None = None

    model_config = {"extra": "allow"}

    @field_validator("iso")
    @classmethod
    def _normalise_iso(cls, value: str | None) -> str | None:
        return value.strip().upper() if value else value

    @model_validator(mode="after")
    def _require_fields_for_type(self) -> "FeedbackRequest":
        missing: list[str] = []
        if self.type in {"confirm", "reject"}:
            if not self.iso:
                missing.append("iso")
        elif self.type == "answer":
            if not self.dimension:
                missing.append("dimension")
            if not self.value:
                missing.append("value")
        elif self.type == "free_text":
            if not (self.text and self.text.strip()):
                missing.append("text")
        if missing:
            raise ValueError(f"missing required field(s) for type={self.type}: {', '.join(missing)}")
        return self


class FeedbackResponse(BaseModel):
    scores: list[CountryScoreModel] = Field(default_factory=list)
    action: ActionModel | None = None
    confidence: ConfidenceRouteModel | None = None
    messages: list[ChatMessageModel] = Field(default_factory=list)
    clues: list[ClueModel] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    done: bool | None = None
    iso: str | None = None


GuidedLocationLevel = Literal["country", "admin1", "city"]
GuidedStage = Literal["country", "city", "resolved"]
ReviewedClueStatus = Literal["pending", "confirmed", "rejected", "uncertain"]
GuidedEventType = Literal[
    "confirm_clue",
    "reject_clue",
    "confirm_location",
    "reject_location",
    "answer",
    "unknown",
    "retract",
    "backtrack",
    "free_text",
    "adopt_observation",
]


class GuidedLocationTarget(BaseModel):
    """Catalog-validated location identity used by the guided flow."""

    level: GuidedLocationLevel
    id: str = Field(min_length=1, max_length=120)
    label: str = Field(min_length=1, max_length=200)
    country_iso: str = Field(min_length=2, max_length=2)
    source_iso: str | None = Field(default=None, min_length=2, max_length=2)
    admin1_id: str | None = Field(default=None, max_length=120)
    city_id: str | None = Field(default=None, max_length=32)
    lat: float | None = Field(default=None, ge=-90.0, le=90.0, allow_inf_nan=False)
    lng: float | None = Field(default=None, ge=-180.0, le=180.0, allow_inf_nan=False)
    coordinate_source: str | None = Field(default=None, max_length=40)
    coordinate_quality: str | None = Field(default=None, max_length=40)
    coordinate_city_id: int | None = Field(default=None, ge=1)

    model_config = {"extra": "forbid"}


class PlaceSourceModel(BaseModel):
    """One bounded source supporting a catalog city match."""

    provider: str = Field(min_length=1, max_length=40)
    provider_place_id: str | None = Field(default=None, max_length=200)
    label: str = Field(min_length=1, max_length=200)
    query: str = Field(default="", max_length=200)
    attribution: str | None = Field(default=None, max_length=200)
    license: str | None = Field(default=None, max_length=80)
    fetched_at: str = Field(default="", max_length=80)

    model_config = {"extra": "forbid"}


class PlaceMatchModel(BaseModel):
    """User-text place evidence that resolved to authoritative catalog rows."""

    query_terms: list[str] = Field(default_factory=list, max_length=4)
    clue_ids: list[str] = Field(default_factory=list, max_length=20)
    match_kind: Literal["catalog_exact", "geocoder_locality"]
    ambiguous: bool = False
    sources: list[PlaceSourceModel] = Field(default_factory=list, max_length=4)

    model_config = {"extra": "forbid"}


class GuidedCandidateModel(BaseModel):
    """One current option; catalog-only rows intentionally omit score fields."""

    target: GuidedLocationTarget
    score: float | None = Field(default=None, allow_inf_nan=False)
    score_type: str | None = None
    reason: str | None = None
    selection_basis: Literal["user_text", "catalog_search"] | None = None
    place_match: PlaceMatchModel | None = None

    model_config = {"extra": "forbid"}

    @model_serializer(mode="wrap")
    def _drop_absent_optional_fields(self, handler):
        data = handler(self)
        for key in ("score", "score_type", "reason", "selection_basis", "place_match"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class ReviewedClueModel(BaseModel):
    """User-review state layered over immutable machine evidence."""

    id: str = Field(min_length=1, max_length=120)
    dimension: str = Field(min_length=1, max_length=64)
    value: str = Field(max_length=256)
    description: str = ""
    source: str = ""
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    status: ReviewedClueStatus

    model_config = {"extra": "forbid"}


class GuidedQuestionOptionModel(BaseModel):
    value: str = Field(min_length=1, max_length=200)
    label: str = Field(min_length=1, max_length=200)

    model_config = {"extra": "forbid"}


class GuidedQuestionModel(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    kind: Literal["review_clue", "ask", "confirm_location", "request_clue"]
    message: str = Field(min_length=1, max_length=1000)
    dimension: str | None = Field(default=None, max_length=64)
    clue_id: str | None = Field(default=None, max_length=120)
    target: GuidedLocationTarget | None = None
    options: list[GuidedQuestionOptionModel] | None = None

    model_config = {"extra": "forbid"}


class ConfirmedPathModel(BaseModel):
    country: GuidedLocationTarget | None = None
    admin1: GuidedLocationTarget | None = None
    city: GuidedLocationTarget | None = None

    model_config = {"extra": "forbid"}


class GuidedDialogueModel(BaseModel):
    version: Literal[1] = 1
    revision: int = Field(ge=0)
    stage: GuidedStage
    confirmed: ConfirmedPathModel = Field(default_factory=ConfirmedPathModel)
    exclusions: list[GuidedLocationTarget] = Field(default_factory=list)
    clues: list[ReviewedClueModel] = Field(default_factory=list)
    pending_question: GuidedQuestionModel | None = None

    model_config = {"extra": "forbid"}


class GuidedEventTarget(BaseModel):
    level: GuidedLocationLevel
    id: str = Field(min_length=1, max_length=120)

    model_config = {"extra": "forbid"}


class GuidedEventRequest(BaseModel):
    type: GuidedEventType
    event_id: str = Field(min_length=1, max_length=200)
    expected_revision: int = Field(ge=0)
    question_id: str | None = Field(default=None, max_length=200)
    target: GuidedEventTarget | None = None
    clue_id: str | None = Field(default=None, max_length=120)
    value: str | None = Field(default=None, max_length=500)
    text: str | None = Field(default=None, max_length=2000)
    level: GuidedLocationLevel | None = None
    # Adopt-observation events carry identifiers only. The service reads the
    # authoritative text/provenance from the current session's persisted AI
    # report; client text/value fields are never used for this event type.
    report_id: str | None = Field(default=None, min_length=1, max_length=120)
    observation_id: str | None = Field(default=None, min_length=1, max_length=120)

    model_config = {"extra": "forbid"}

    @field_validator("report_id", "observation_id")
    @classmethod
    def _normalise_observation_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None

    @model_validator(mode="after")
    def _require_observation_ids(self) -> "GuidedEventRequest":
        if self.type == "adopt_observation":
            missing = [
                name for name in ("report_id", "observation_id")
                if not getattr(self, name)
            ]
            if missing:
                raise ValueError(
                    "missing required field(s) for type=adopt_observation: "
                    + ", ".join(missing)
                )
        return self


class GuidedReceiptModel(BaseModel):
    event_id: str
    status: Literal["applied", "unchanged", "unrecognized"]
    summary: str

    model_config = {"extra": "forbid"}


class GuidedDialogueResponse(BaseModel):
    dialogue: GuidedDialogueModel
    candidates: list[GuidedCandidateModel] = Field(default_factory=list)
    action: ActionModel | None = None
    confidence: ConfidenceRouteModel | None = None
    messages: list[ChatMessageModel] = Field(default_factory=list)
    scores: list[CountryScoreModel] = Field(default_factory=list)
    clues: list[ClueModel] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    receipt: GuidedReceiptModel | None = None
    inference_notice: str | None = None


class MessagesResponse(BaseModel):
    messages: list[ChatMessageModel] = Field(default_factory=list)


class CitiesRequest(BaseModel):
    """Canonical country plus optional raw source-region scope for cities."""

    iso: str = Field(min_length=2, max_length=2)
    source_iso: str | None = Field(default=None, min_length=2, max_length=2)

    @field_validator("iso")
    @classmethod
    def _normalise_iso(cls, value: str) -> str:
        normalised = value.strip().upper()
        if len(normalised) != 2 or not normalised.isalpha():
            raise ValueError("iso must be a two-letter country code")
        return normalised

    @field_validator("source_iso")
    @classmethod
    def _normalise_source_iso(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalised = value.strip().upper()
        if len(normalised) != 2 or not normalised.isalpha():
            raise ValueError("source_iso must be a two-letter source code")
        return normalised


class CityScoreModel(BaseModel):
    """One catalog city candidate; representative point, never photo GPS."""

    # Serialised as a decimal string: the UI keys map markers and list rows on it.
    city_id: str
    # Public country identity is canonical (for example CN for a raw TW city).
    # ``source_iso``/``source_region`` retain the validated catalog provenance.
    iso: str
    country_iso: str | None = None
    source_iso: str | None = None
    source_region: dict[str, Any] | None = None
    name: str
    name_zh: str | None = None
    region: str = ""
    lat: float
    lng: float
    raw_cosine: float = Field(ge=-1.0, le=1.0, allow_inf_nan=False)
    score: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    score_type: str = "city_text_cosine"
    reference_cosine: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    # Backend-only clue rerank provenance. Both stay ``None`` and are dropped
    # from the JSON unless this session's own evidence matched this row, so an
    # unfused city payload keeps its exact previous shape. ``city_clue_fusion``
    # is produced only here, *after* gateway validation; the provider contract
    # still rejects that value at the gateway.
    matched_clues: list[MatchedClueModel] | None = None
    fusion: dict[str, Any] | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_clue_evidence(self, handler):
        data = handler(self)
        for key in ("country_iso", "source_iso", "source_region", "matched_clues", "fusion"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class CityScopeModel(BaseModel):
    """Coverage counters for one city query; absent when the model is unreachable."""

    city_limit_per_country: int | None = Field(default=None, gt=0)
    city_catalog_revision: str | None = None
    cities_considered: int | None = Field(default=None, ge=0)


class CitiesResponse(BaseModel):
    """City candidates for one country, or an explicit degraded status.

    ``unavailable`` means the model service is off or unreachable; ``no_cities``
    means the catalog holds no city for that country. Neither is an error.
    """

    city_scores: list[CityScoreModel] = Field(default_factory=list)
    country_iso: str
    # Raw source scope when the response covers one region; absent for merged
    # canonical queries (for example CN with CN/TW/HK/MO scopes).
    source_iso: str | None = None
    source_region: dict[str, Any] | None = None
    source_isos: list[str] = Field(default_factory=list)
    scope: CityScopeModel | None = None
    status: Literal["ok", "no_cities", "unavailable"]
    # Honest partial-coverage note, e.g. when one scoped provider call failed
    # while another returned usable rows.
    note: str | None = None
    # Present only when session clues actually changed a returned row, or when
    # exact evidence matched a catalog city outside the returned rows (then
    # ``clue_note`` explains it was *not* scored). Both keys are omitted when
    # absent so existing responses keep their exact shape.
    clue_dimensions: list[str] | None = None
    clue_note: str | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_city_clue_evidence(self, handler):
        data = handler(self)
        for key in ("source_iso", "source_region", "note", "clue_dimensions", "clue_note"):
            if data.get(key) is None:
                data.pop(key, None)
        if not data.get("source_isos"):
            data.pop("source_isos", None)
        return data


class ImageRegionModel(BaseModel):
    """Normalized top-left crop rectangle in the EXIF-oriented image."""

    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    w: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    h: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _inside_image(self) -> "ImageRegionModel":
        if self.x + self.w > 1.0 + 1e-9 or self.y + self.h > 1.0 + 1e-9:
            raise ValueError("image_region must fit inside the normalized image")
        return self


class AnalyzePointRequest(BaseModel):
    # allow_inf_nan=False keeps NaN/inf out: they would otherwise slip through
    # the ge/le comparisons on some platforms.
    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    scale: float = Field(default=0.25, ge=0.08, le=0.8, allow_inf_nan=False)
    image_region: ImageRegionModel | None = None
    purpose: Literal["auto", "ocr", "scene"] = "auto"
    client_request_id: str | None = Field(default=None, max_length=128)

    @field_validator("client_request_id")
    @classmethod
    def _normalise_request_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


class RegionBBoxModel(BaseModel):
    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    w: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)
    h: float = Field(gt=0.0, le=1.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def _inside_image(self) -> "RegionBBoxModel":
        if self.x + self.w > 1.0 + 1e-9 or self.y + self.h > 1.0 + 1e-9:
            raise ValueError("bbox must fit inside the normalized image")
        return self


class CropInfoModel(BaseModel):
    x: float
    y: float
    size: int
    region_id: str | None = None
    bbox_px: list[int] | None = None
    bbox_norm: RegionBBoxModel | None = None
    scale: float | None = None
    purpose: Literal["auto", "ocr", "scene"] | None = None

    @model_serializer(mode="wrap")
    def _drop_absent_region_fields(self, handler):
        data = handler(self)
        for key in ("region_id", "bbox_px", "bbox_norm", "scale", "purpose"):
            if data.get(key) is None:
                data.pop(key, None)
        return data


class RegionTrustModel(BaseModel):
    level: Literal["high", "medium", "low"]
    reliable: bool
    validated: bool
    affects_ranking: bool
    confidence_kind: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0, allow_inf_nan=False)
    raw_cosine: float | None = Field(default=None, ge=-1.0, le=1.0, allow_inf_nan=False)
    uncertain: bool | None = None


class RegionObservationModel(BaseModel):
    id: str
    kind: Literal["ocr_text", "semantic", "pixel_signal", "scene_hypothesis", "crop_ranking"]
    dimension: str
    value: str
    text: str | None = None
    label: str
    evidence: str
    source: str
    provider: str | None = None
    bbox_norm: RegionBBoxModel | None = None
    quad_norm: RegionBBoxModel | None = None
    trust: RegionTrustModel


class RegionQuestionModel(BaseModel):
    id: str
    text: str
    reason: str
    basis_observation_ids: list[str] = Field(default_factory=list)


class RegionQualityModel(BaseModel):
    issues: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class RegionImageModel(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    orientation_applied: bool
    detail_width: int = Field(gt=0)
    detail_height: int = Field(gt=0)
    ocr_scale: float = Field(gt=0.0, le=3.0, allow_inf_nan=False)


class RegionOcrModel(BaseModel):
    provider: str
    enabled: bool
    available: bool
    state: str
    line_count: int = Field(ge=0)


class RegionModelModel(BaseModel):
    provider: str
    enabled: bool
    available: bool
    state: str
    ranking_applied: bool
    margin: float | None = Field(default=None, allow_inf_nan=False)
    uncertain: bool | None = None
    scene_available: bool | None = None
    scene_state: str | None = None
    ranking_note: str | None = None


class RegionReportModel(BaseModel):
    id: str
    purpose: Literal["auto", "ocr", "scene"]
    scale: float = Field(ge=0.08, le=0.8, allow_inf_nan=False)
    bbox_px: list[int]
    bbox_norm: RegionBBoxModel
    context_bbox_norm: RegionBBoxModel | None = None
    image: RegionImageModel
    quality: RegionQualityModel
    ocr: RegionOcrModel
    model: RegionModelModel
    observations: list[RegionObservationModel] = Field(default_factory=list)
    question_suggestions: list[RegionQuestionModel] = Field(default_factory=list)
    summary: str
    dedupe: Literal["new", "duplicate_region", "duplicate_request", "no_new_evidence"]

    @model_validator(mode="after")
    def _validate_bbox_length(self) -> "RegionReportModel":
        if len(self.bbox_px) != 4:
            raise ValueError("bbox_px must contain left, top, right, bottom")
        return self


class AnalyzePointResponse(BaseModel):
    new_clues: list[ClueModel] = Field(default_factory=list)
    scores: list[CountryScoreModel] = Field(default_factory=list)
    action: ActionModel | None = None
    confidence: ConfidenceRouteModel | None = None
    messages: list[ChatMessageModel] = Field(default_factory=list)
    clues: list[ClueModel] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    crop_info: CropInfoModel
    focus_region: RegionReportModel | None = None
    model_status: RegionModelModel | None = None
    ocr_status: RegionOcrModel | None = None
    applied: bool = True


class ErrorResponse(BaseModel):
    error: str
    detail: Any | None = None


__all__ = [
    "ActionModel",
    "AnalyzePointRequest",
    "AnalyzePointResponse",
    "ChatMessageModel",
    "CitiesRequest",
    "CitiesResponse",
    "CityScopeModel",
    "CityScoreModel",
    "ClueModel",
    "ConfidenceRouteModel",
    "CountriesResponse",
    "CountryModel",
    "CountryScoreModel",
    "CropInfoModel",
    "ErrorResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "ConfirmedPathModel",
    "GuidedCandidateModel",
    "GuidedDialogueModel",
    "GuidedDialogueResponse",
    "GuidedEventRequest",
    "GuidedEventTarget",
    "GuidedLocationTarget",
    "GuidedQuestionModel",
    "GuidedQuestionOptionModel",
    "PlaceMatchModel",
    "PlaceSourceModel",
    "GuidedReceiptModel",
    "HealthResponse",
    "MatchedClueModel",
    "MessagesResponse",
    "ImageRegionModel",
    "RegionBBoxModel",
    "RegionImageModel",
    "RegionModelModel",
    "RegionObservationModel",
    "RegionOcrModel",
    "RegionQualityModel",
    "RegionQuestionModel",
    "RegionReportModel",
    "RegionTrustModel",
    "ReviewedClueModel",
    "UploadResponse",
]
