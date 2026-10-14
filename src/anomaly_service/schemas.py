"""Request/response contracts shared by the API, the worker and the inference layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

Severity = Literal["low", "medium", "high"]
CauseKind = Literal["magnitude", "correlation"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]

_EXAMPLE_REQUEST: dict[str, Any] = {
    "timestamps": ["2024-01-01T00:00:00Z", "2024-01-01T00:01:00Z", "2024-01-01T00:02:00Z"],
    "series": {"f00": [41.2, 40.8, 41.0], "f01": [12.5, 12.9, 12.7]},
    "options": {"top_k": 3, "language": "pt"},
}


class DetectionOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    top_k: int = Field(3, ge=1, le=50, description="Maximum number of root causes per anomaly.")
    z_thresh: float = Field(
        3.0, gt=0, description="Robust z-score a variable must exceed to be a statistical root cause."
    )
    max_gap: int = Field(
        5, ge=0, description="Anomalous windows separated by up to this many normal windows are merged."
    )
    include_scores: bool = Field(False, description="Return the score of every window.")
    include_warmup: bool = Field(
        False, description="Also report anomalies in the warm-up (first max_window-1 points)."
    )
    language: Literal["pt", "en"] = Field("pt", description="Language of the textual explanation.")


class DetectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [_EXAMPLE_REQUEST]})

    timestamps: list[datetime] | None = Field(
        None, description="One timestamp per point. Optional: without it, anomalies are reported by index."
    )
    series: dict[str, list[float | None]] = Field(
        ..., min_length=1, description="Column-oriented series: feature name -> values (null = missing)."
    )
    options: DetectionOptions = Field(default_factory=DetectionOptions)

    @model_validator(mode="after")
    def _same_length(self) -> DetectionRequest:
        lengths = {len(values) for values in self.series.values()}
        if len(lengths) != 1:
            raise ValueError("all series must have the same length")
        n = lengths.pop()
        if n == 0:
            raise ValueError("series must not be empty")
        if self.timestamps is not None and len(self.timestamps) != n:
            raise ValueError(f"timestamps has {len(self.timestamps)} items but the series have {n}")
        return self

    @property
    def n_points(self) -> int:
        return len(next(iter(self.series.values())))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.series, dtype=float)  # null -> NaN


class JobRequest(DetectionRequest):
    callback_url: HttpUrl | None = Field(
        None, description="If set, the result is POSTed here when the job finishes."
    )


class ModelRef(BaseModel):
    name: str
    version: str | None = None
    alias: str | None = None
    run_id: str | None = None


class RootCause(BaseModel):
    variable: str
    contribution_pct: float = Field(description="Share of the anomaly attributed to this variable (%).")
    peak_z: float = Field(description="Peak robust z-score of its reconstruction error vs. its own normal.")
    kind: CauseKind = Field(
        description="magnitude: the value itself is off. correlation: its relation to the others broke."
    )
    observed: float | None = Field(description="Value received at the variable's peak deviation.")
    expected: float | None = Field(description="Value the model reconstructed (what normal looks like).")


class Anomaly(BaseModel):
    start_index: int
    end_index: int
    peak_index: int
    start: datetime | None = None
    end: datetime | None = None
    peak: datetime | None = None
    max_score: float
    threshold_ratio: float = Field(description="max_score / threshold.")
    severity: Severity
    root_causes: list[RootCause]
    explanation: str


class ScorePoint(BaseModel):
    index: int
    timestamp: datetime | None = None
    score: float
    is_anomaly: bool


class DetectionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelRef
    threshold: float
    n_points: int
    warmup_points: int = Field(description="Leading points without full history (less reliable).")
    ignored_features: list[str] = Field(
        default_factory=list, description="Extra columns not used by the model."
    )
    anomalies: list[Anomaly]
    scores: list[ScorePoint] | None = None


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model: ModelRef
    source: Literal["registry", "local"]
    loaded_at: datetime
    threshold: float
    features: list[str]
    window_sizes: list[int]
    stride: int
    seq_len: int
    warmup_points: int


class JobAccepted(BaseModel):
    job_id: str
    status: JobStatus
    status_url: str


class JobState(BaseModel):
    job_id: str
    status: JobStatus
    submitted_at: datetime | None = None
    result: DetectionResponse | None = None
    error: str | None = None
