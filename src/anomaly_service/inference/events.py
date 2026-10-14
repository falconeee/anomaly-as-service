"""Turning per-window scores into anomaly EVENTS (contiguous periods above the threshold)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anomaly_service.schemas import Severity


@dataclass(frozen=True)
class WindowEvent:
    start: int  # first window index
    end: int  # last window index (inclusive)
    peak: int  # window with the highest score
    max_score: float


def group_events(phi: np.ndarray, threshold: float, max_gap: int = 0, skip: int = 0) -> list[WindowEvent]:
    """
    Groups windows with phi > threshold into events. Two anomalous runs separated by at most
    `max_gap` normal windows become one event (one incident, not a flickering series of
    alerts). The first `skip` windows (warm-up) are ignored.
    """
    above = np.asarray(phi) > threshold
    above[:skip] = False
    idx = np.flatnonzero(above)
    if idx.size == 0:
        return []

    events = []
    start = prev = int(idx[0])
    for i in idx[1:]:
        if i - prev - 1 > max_gap:
            events.append(_event(phi, start, prev))
            start = int(i)
        prev = int(i)
    events.append(_event(phi, start, prev))
    return events


def _event(phi: np.ndarray, start: int, end: int) -> WindowEvent:
    peak = start + int(np.argmax(phi[start : end + 1]))
    return WindowEvent(start=start, end=end, peak=peak, max_score=float(phi[peak]))


def severity(threshold_ratio: float) -> Severity:
    if threshold_ratio >= 3.0:
        return "high"
    if threshold_ratio >= 1.5:
        return "medium"
    return "low"
