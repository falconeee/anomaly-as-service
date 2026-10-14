"""Prometheus metrics: service health AND model behavior.

HTTP latency/throughput comes from prometheus-fastapi-instrumentator. The metrics below are
the ML-specific part: what the model is seeing (input drift) and saying (score levels,
anomaly rate), plus which version is serving.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from anomaly_service.logger import logger
from anomaly_service.schemas import DetectionResponse, ModelRef

DETECTIONS = Counter(
    "anomaly_detections_total", "Detection requests processed", ["mode", "outcome"]
)  # mode: sync|async, outcome: ok|error
ANOMALY_EVENTS = Counter("anomaly_events_total", "Anomaly events detected", ["severity"])
INFERENCE_SECONDS = Histogram(
    "anomaly_inference_seconds",
    "Model inference + explanation time per request",
    ["mode"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
)
REQUEST_POINTS = Histogram(
    "anomaly_request_points",
    "Points per detection request",
    buckets=(60, 250, 1_000, 5_000, 20_000, 100_000, 500_000),
)
MAX_SCORE_RATIO = Histogram(
    "anomaly_max_score_ratio",
    "max(score) / threshold per request: a drifting distribution signals model or data drift",
    buckets=(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10, 50),
)
FEATURE_DRIFT = Gauge(
    "anomaly_feature_drift_z",
    "|mean(last request) - mean(train)| / std(train), per feature",
    ["feature"],
)
MODEL_INFO = Gauge("anomaly_model_info", "Model currently served (always 1)", ["name", "version", "alias"])


def set_model_info(ref: ModelRef) -> None:
    MODEL_INFO.clear()
    MODEL_INFO.labels(ref.name, ref.version or "local", ref.alias or "-").set(1)


def record_detection(
    mode: str,
    response: DetectionResponse,
    seconds: float,
    frame: pd.DataFrame,
    train_mean: pd.Series,
    train_std: pd.Series,
    max_score: float,
) -> None:
    DETECTIONS.labels(mode, "ok").inc()
    INFERENCE_SECONDS.labels(mode).observe(seconds)
    REQUEST_POINTS.observe(response.n_points)
    MAX_SCORE_RATIO.observe(max_score / response.threshold if response.threshold > 0 else 0.0)
    for anomaly in response.anomalies:
        ANOMALY_EVENTS.labels(anomaly.severity).inc()
    drift = (frame.mean() - train_mean).abs() / train_std
    for feature, value in drift.items():
        if np.isfinite(value):
            FEATURE_DRIFT.labels(str(feature)).set(float(value))


class QueueDepthCollector(Collector):
    """Reads the Celery queue length from Redis at scrape time (API side)."""

    def __init__(self, redis_client, queue: str):
        self._redis = redis_client
        self._queue = queue

    def collect(self) -> Iterator[GaugeMetricFamily]:
        metric = GaugeMetricFamily(
            "anomaly_queue_depth", "Detection jobs waiting in the queue", labels=["queue"]
        )
        try:
            metric.add_metric([self._queue], float(self._redis.llen(self._queue)))
        except Exception as exc:  # metrics must never break the scrape
            logger.debug("queue depth unavailable: %s", exc)
        yield metric
