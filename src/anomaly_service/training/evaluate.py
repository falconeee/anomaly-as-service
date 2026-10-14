"""Evaluation of time-series anomaly detection and root-cause analysis.

Detection
---------
- point-wise precision / recall / F1 at the model's own threshold (what production sees);
- the same after **point adjustment** (PA): if any point of a true anomaly segment is
  flagged, the whole segment counts as detected. PA is the protocol used by most papers on
  SMD/SMAP/MSL, but it inflates scores a lot (a random detector can reach a high PA-F1 on long
  segments), so both are reported;
- AUC-PR (average precision): threshold-free quality of the score itself;
- best F1: F1 at the best possible threshold (an "oracle", upper bound for threshold tuning).

Root cause
----------
HitRate@P% (Su et al., 2019): for each true anomaly with G ground-truth features, the share
of them that appear in the top ceil(P% x G) features ranked by the model's attribution.
"""

from __future__ import annotations

import math
import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from anomaly_service.model import MSCVAE, WindowScores


@dataclass(frozen=True)
class GroundTruthEvent:
    start: int  # first row (inclusive)
    end: int  # last row (exclusive)
    features: tuple[str, ...]


def contiguous_segments(labels: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every run of 1s in a 0/1 array."""
    padded = np.concatenate([[0], np.asarray(labels, dtype=np.int8), [0]])
    changes = np.flatnonzero(np.diff(padded))
    return list(zip(changes[0::2].tolist(), changes[1::2].tolist(), strict=True))


def point_adjust(pred: np.ndarray, labels: np.ndarray) -> np.ndarray:
    adjusted = np.asarray(pred, dtype=bool).copy()
    for start, end in contiguous_segments(labels):
        if adjusted[start:end].any():
            adjusted[start:end] = True
    return adjusted


def precision_recall_f1(pred: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(labels, dtype=bool)
    tp = float(np.sum(pred & truth))
    precision = tp / max(float(pred.sum()), 1.0)
    recall = tp / max(float(truth.sum()), 1.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def precision_recall_curve(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Precision and recall when thresholding at each distinct score, highest first."""
    order = np.argsort(-np.asarray(scores, dtype=float), kind="stable")
    sorted_scores = np.asarray(scores, dtype=float)[order]
    truth = np.asarray(labels, dtype=bool)[order]
    tp = np.cumsum(truth)
    # Keep only the last position of each tied score: a threshold cannot split ties.
    distinct = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(sorted_scores) - 1]
    tp = tp[distinct]
    flagged = distinct + 1
    precision = tp / flagged
    recall = tp / max(int(truth.sum()), 1)
    return precision, recall


def average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    precision, recall = precision_recall_curve(scores, labels)
    recall_steps = np.diff(np.r_[0.0, recall])
    return float(np.sum(recall_steps * precision))


def best_f1(scores: np.ndarray, labels: np.ndarray) -> float:
    precision, recall = precision_recall_curve(scores, labels)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-12), 0)
    return float(f1.max()) if len(f1) else 0.0


def scores_to_rows(row_index: np.ndarray, values: np.ndarray, n_rows: int) -> np.ndarray:
    """
    Expands per-window values to per-row values. Window k ends at row_index[k], and each row
    takes the value of the most recent window that ended at or before it (relevant when
    stride > 1; with stride 1 it is the identity).
    """
    positions = np.searchsorted(row_index, np.arange(n_rows), side="right") - 1
    return np.asarray(values)[np.clip(positions, 0, None)]


def detection_metrics(phi_rows: np.ndarray, threshold: float, labels: np.ndarray) -> dict[str, float]:
    pred = phi_rows > threshold
    precision, recall, f1 = precision_recall_f1(pred, labels)
    pa_precision, pa_recall, pa_f1 = precision_recall_f1(point_adjust(pred, labels), labels)
    normal = ~np.asarray(labels, dtype=bool)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pa_precision": pa_precision,
        "pa_recall": pa_recall,
        "pa_f1": pa_f1,
        "auc_pr": average_precision(phi_rows, labels),
        "best_f1": best_f1(phi_rows, labels),
        "false_alarm_rate": float(pred[normal].mean()) if normal.any() else 0.0,
    }


def rca_hit_rate(
    scores: WindowScores,
    attribution: np.ndarray,
    events: Sequence[GroundTruthEvent],
    percentages: Sequence[int] = (100, 150),
) -> dict[str, float]:
    """Mean HitRate@P% over `events` (features unknown to the model are ignored)."""
    names = list(scores.feature_names)
    hits: dict[int, list[float]] = {p: [] for p in percentages}
    for event in events:
        truth = [f for f in event.features if f in names]
        in_event = (scores.row_index >= event.start) & (scores.row_index < event.end)
        if not truth or not in_event.any():
            continue
        ranking = [names[i] for i in np.argsort(-attribution[in_event].sum(axis=0), kind="stable")]
        for p in percentages:
            top = set(ranking[: math.ceil(p / 100 * len(truth))])
            hits[p].append(len(top.intersection(truth)) / len(truth))
    return {f"hitrate_{p}": float(np.mean(v)) for p, v in hits.items() if v}


def measure_latency_ms(model: MSCVAE, df: pd.DataFrame, repeats: int = 3) -> float:
    """Median wall time of one full inference pass (score + RCA inputs) over `df`."""
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.score_windows(df)
        timings.append((time.perf_counter() - start) * 1000)
    return float(np.median(timings))


def evaluate_model(
    model: MSCVAE,
    test: pd.DataFrame,
    labels: np.ndarray,
    events: Sequence[GroundTruthEvent] = (),
    latency_rows: int = 1000,
) -> dict[str, float]:
    """All detection, RCA and latency metrics of `model` on a labeled test set."""
    scores = model.score_windows(test)
    phi_rows = scores_to_rows(scores.row_index, scores.phi, len(test))
    metrics = detection_metrics(phi_rows, float(model.threshold), labels)
    metrics.update(rca_hit_rate(scores, model.rca_window_scores(scores), events))
    metrics["latency_ms"] = measure_latency_ms(model, test.iloc[:latency_rows])
    return metrics
