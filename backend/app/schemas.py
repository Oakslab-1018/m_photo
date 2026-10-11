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
    messages: list[ChatMessageModel] = Field(default_factory=list)
    clues: list[ClueModel] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    done: bool | None = None
    iso: str | None = None


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


class AnalyzePointRequest(BaseModel):
    # allow_inf_nan=False keeps NaN/inf out: they would otherwise slip through
    # the ge/le comparisons on some platforms.
    x: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    y: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)


class CropInfoModel(BaseModel):
    x: float
    y: float
    size: int


class AnalyzePointResponse(BaseModel):
    new_clues: list[ClueModel] = Field(default_factory=list)
    scores: list[CountryScoreModel] = Field(default_factory=list)
    action: ActionModel | None = None
    messages: list[ChatMessageModel] = Field(default_factory=list)
    clues: list[ClueModel] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
    crop_info: CropInfoModel


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
    "CountriesResponse",
    "CountryModel",
    "CountryScoreModel",
    "CropInfoModel",
    "ErrorResponse",
    "FeedbackRequest",
    "FeedbackResponse",
    "HealthResponse",
    "MatchedClueModel",
    "MessagesResponse",
    "UploadResponse",
]
