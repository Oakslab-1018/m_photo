"""Pydantic request/response models for the public API."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

FeedbackType = Literal["confirm", "reject", "answer", "free_text"]
MessageRole = Literal["user", "ai", "system"]


class HealthResponse(BaseModel):
    status: Literal["ok"]
    database: Literal["ok"]
    ai_mode: str
    data_dir: str
    version: str


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
    iso: str
    name: str
    name_zh: str
    score: float
    matched_clues: list[MatchedClueModel] = Field(default_factory=list)
    lat: float = 0.0
    lng: float = 0.0

    model_config = {"extra": "allow"}


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
