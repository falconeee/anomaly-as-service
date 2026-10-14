"""Inference layer shared by the API (sync) and the worker (async).

- InferenceService: one loaded model -> detect(request) -> DetectionResponse.
- ModelManager: owns the CURRENT InferenceService, loads it from the MLflow registry
  (models:/<name>@<alias>) or a local bundle, and hot-swaps it when the alias moves.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from anomaly_service.config import Settings
from anomaly_service.inference.events import group_events, severity
from anomaly_service.inference.explain import explain, root_causes
from anomaly_service.logger import logger
from anomaly_service.model import MSCVAE, load_model
from anomaly_service.observability import DETECTIONS, record_detection, set_model_info
from anomaly_service.schemas import (
    Anomaly,
    DetectionRequest,
    DetectionResponse,
    ModelInfo,
    ModelRef,
    ScorePoint,
)


class ModelUnavailableError(RuntimeError):
    """No model could be loaded (registry unreachable, alias missing...). Usually transient."""


@dataclass(frozen=True)
class LoadedModel:
    model: MSCVAE
    ref: ModelRef
    source: str  # "registry" | "local"
    loaded_at: datetime


class InferenceService:
    def __init__(self, loaded: LoadedModel):
        self.loaded = loaded

    @property
    def model(self) -> MSCVAE:
        return self.loaded.model

    @property
    def ref(self) -> ModelRef:
        return self.loaded.ref

    def info(self) -> ModelInfo:
        model = self.model
        return ModelInfo(
            model=self.ref,
            source="registry" if self.loaded.source == "registry" else "local",
            loaded_at=self.loaded.loaded_at,
            threshold=float(model.threshold),
            features=[str(f) for f in model.feature_names],
            window_sizes=list(model.window_sizes),
            stride=model.stride,
            seq_len=model.seq_len,
            warmup_points=model.warmup_rows,
        )

    def detect(self, request: DetectionRequest, mode: str = "sync") -> DetectionResponse:
        started = time.perf_counter()
        try:
            response, frame, max_score = self._detect(request)
        except Exception:
            DETECTIONS.labels(mode, "error").inc()
            raise
        model = self.model
        record_detection(
            mode,
            response,
            time.perf_counter() - started,
            frame,
            model.generator.mean,
            model.generator.std,
            max_score,
        )
        return response

    def _detect(self, request: DetectionRequest):
        model = self.model
        options = request.options
        threshold = float(model.threshold)

        frame = request.to_frame()
        aligned = model.generator.align_columns(frame)  # FeatureSchemaError if features are missing
        ignored = [str(c) for c in frame.columns if c not in set(aligned.columns)]
        scores = model.score_windows(aligned)

        warmup_windows = int(np.searchsorted(scores.row_index, model.warmup_rows))
        if options.include_warmup:
            warmup_windows = 0
        events = group_events(scores.phi, threshold, max_gap=options.max_gap, skip=warmup_windows)
        raw = aligned.to_numpy(dtype=float)
        ts = request.timestamps

        anomalies = []
        for event in events:
            rows = (
                int(scores.row_index[event.start]),
                int(scores.row_index[event.end]),
                int(scores.row_index[event.peak]),
            )
            stamps = (ts[rows[0]], ts[rows[1]], ts[rows[2]]) if ts else (None, None, None)
            ratio = event.max_score / threshold
            sev = severity(ratio)
            causes = root_causes(model, scores, event, raw, top_k=options.top_k, z_thresh=options.z_thresh)
            anomalies.append(
                Anomaly(
                    start_index=rows[0],
                    end_index=rows[1],
                    peak_index=rows[2],
                    start=stamps[0],
                    end=stamps[1],
                    peak=stamps[2],
                    max_score=round(event.max_score, 4),
                    threshold_ratio=round(ratio, 3),
                    severity=sev,
                    root_causes=causes,
                    explanation=explain(sev, ratio, rows, stamps, causes, options.language),
                )
            )

        score_points = None
        if options.include_scores:
            score_points = [
                ScorePoint(
                    index=int(r),
                    timestamp=ts[int(r)] if ts else None,
                    score=float(s),
                    is_anomaly=bool(s > threshold),
                )
                for r, s in zip(scores.row_index, scores.phi, strict=True)
            ]

        response = DetectionResponse(
            model=self.ref,
            threshold=threshold,
            n_points=request.n_points,
            warmup_points=min(model.warmup_rows, request.n_points),
            ignored_features=ignored,
            anomalies=anomalies,
            scores=score_points,
        )
        return response, aligned, float(scores.phi.max())


def load_from_registry(name: str, alias: str, version: str | None = None) -> LoadedModel:
    import mlflow.pyfunc
    from mlflow import MlflowClient

    client = MlflowClient()
    if version:
        mv = client.get_model_version(name, version)
    else:
        mv = client.get_model_version_by_alias(name, alias)
    # Load the RESOLVED version, not the alias: the alias could move between the two calls.
    pyfunc_model = mlflow.pyfunc.load_model(f"models:/{name}/{mv.version}")
    model: MSCVAE = pyfunc_model.unwrap_python_model().model
    ref = ModelRef(name=name, version=str(mv.version), alias=alias, run_id=mv.run_id)
    return LoadedModel(model=model, ref=ref, source="registry", loaded_at=datetime.now(UTC))


def load_from_path(path: str, device: str = "cpu") -> LoadedModel:
    model = load_model(path, device=device)
    ref = ModelRef(name=Path(path).name, version=None, alias=None, run_id=None)
    return LoadedModel(model=model, ref=ref, source="local", loaded_at=datetime.now(UTC))


class ModelManager:
    """Thread-safe holder of the current InferenceService."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._service: InferenceService | None = None
        self._lock = threading.Lock()
        self._last_check = 0.0
        self.last_error: str | None = None

    @property
    def service(self) -> InferenceService | None:
        return self._service

    def _resolve_version(self) -> str:
        from mlflow import MlflowClient

        mv = MlflowClient().get_model_version_by_alias(self.settings.model_name, self.settings.model_alias)
        return str(mv.version)

    def load(self, version: str | None = None) -> InferenceService:
        """(Re)loads the model and swaps it in atomically; in-flight requests keep the old one."""
        s = self.settings
        try:
            if s.model_path:
                loaded = load_from_path(s.model_path, device=s.model_device)
            else:
                loaded = load_from_registry(s.model_name, s.model_alias, version)
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            raise ModelUnavailableError(self.last_error) from exc
        service = InferenceService(loaded)
        with self._lock:
            self._service = service
            self._last_check = time.monotonic()
        self.last_error = None
        set_model_info(loaded.ref)
        logger.info("Serving model %s v%s (%s)", loaded.ref.name, loaded.ref.version, loaded.source)
        return service

    def refresh(self) -> bool:
        """Reloads if @alias now points to another version. Returns True when it swapped."""
        self._last_check = time.monotonic()
        if self.settings.model_path:
            return False
        version = self._resolve_version()
        current = self._service.ref.version if self._service else None
        if version == current:
            return False
        logger.info("@%s moved %s -> %s, reloading", self.settings.model_alias, current, version)
        self.load(version)
        return True

    def get(self) -> InferenceService:
        """Current service, loading it on first use and refreshing it when stale (worker side)."""
        if self._service is None:
            return self.load()
        interval = self.settings.model_refresh_interval_s
        if interval > 0 and time.monotonic() - self._last_check > interval:
            try:
                self.refresh()
            except Exception as exc:  # keep serving the current model
                logger.warning("Model refresh failed, keeping v%s: %s", self._service.ref.version, exc)
        return self._service
