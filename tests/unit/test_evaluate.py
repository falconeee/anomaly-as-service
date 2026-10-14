import numpy as np
import pytest

from anomaly_service.model.mscvae import WindowScores
from anomaly_service.training.evaluate import (
    GroundTruthEvent,
    average_precision,
    best_f1,
    contiguous_segments,
    detection_metrics,
    point_adjust,
    precision_recall_f1,
    rca_hit_rate,
    scores_to_rows,
)


def test_contiguous_segments():
    assert contiguous_segments(np.array([0, 1, 1, 0, 0, 1, 0, 1])) == [(1, 3), (5, 6), (7, 8)]
    assert contiguous_segments(np.zeros(4)) == []


def test_point_adjust_fills_detected_segments_only():
    labels = np.array([0, 1, 1, 1, 0, 1, 1, 0])
    pred = np.array([0, 0, 1, 0, 0, 0, 0, 1])

    np.testing.assert_array_equal(point_adjust(pred, labels), [0, 1, 1, 1, 0, 0, 0, 1])


def test_point_adjust_inflates_f1():
    # One hit inside a long segment: point-wise F1 is poor, point-adjusted F1 is perfect.
    labels = np.r_[np.zeros(50), np.ones(50)]
    pred = np.zeros(100)
    pred[60] = 1

    _, _, f1 = precision_recall_f1(pred, labels)
    _, _, pa_f1 = precision_recall_f1(point_adjust(pred, labels), labels)
    assert f1 < 0.05
    assert pa_f1 == pytest.approx(1.0)


def test_average_precision_known_values():
    labels = np.array([1, 0, 1, 0])
    # Ranking 1, 0, 1, 0 -> precision at the positives: 1/1 and 2/3.
    assert average_precision(np.array([4.0, 3.0, 2.0, 1.0]), labels) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision(np.array([4.0, 1.0, 3.0, 0.0]), labels) == pytest.approx(1.0)


def test_best_f1_finds_the_oracle_threshold():
    scores = np.array([0.1, 0.2, 0.9, 0.8, 0.3])
    labels = np.array([0, 0, 1, 1, 0])
    assert best_f1(scores, labels) == pytest.approx(1.0)


def test_detection_metrics_at_threshold():
    phi = np.array([0.0, 5.0, 5.0, 0.0, 5.0])
    labels = np.array([0, 1, 1, 0, 0])

    metrics = detection_metrics(phi, threshold=1.0, labels=labels)

    assert metrics["precision"] == pytest.approx(2 / 3)
    assert metrics["recall"] == pytest.approx(1.0)
    assert metrics["false_alarm_rate"] == pytest.approx(1 / 3)


def test_scores_to_rows_forward_fills_strided_windows():
    expanded = scores_to_rows(np.array([0, 3, 6]), np.array([1.0, 2.0, 3.0]), n_rows=8)
    np.testing.assert_array_equal(expanded, [1, 1, 1, 2, 2, 2, 3, 3])


def test_rca_hit_rate():
    names = ["a", "b", "c", "d"]
    n = 10
    attribution = np.zeros((n, 4))
    attribution[2:5, 1] = 5.0  # event 1 blames "b" (correct)
    attribution[6:9, 3] = 5.0  # event 2 blames "d" (truth is "a" and "c")
    attribution[6:9, 0] = 1.0
    scores = WindowScores(
        feature_names=names,
        row_index=np.arange(n),
        phi=np.zeros(n),
        s_diag=attribution,
        s_off=attribution,
        s_val=attribution,
        recon_z=attribution,
    )
    events = [GroundTruthEvent(2, 5, ("b",)), GroundTruthEvent(6, 9, ("a", "c", "unknown"))]

    rates = rca_hit_rate(scores, attribution, events, percentages=(100, 150))

    # Event 1 ranks "b" first -> 1/1 at both levels. Event 2 (truth {a, c}; "unknown" ignored)
    # ranks d, a, then the zero-score ties in column order (b, c):
    # top-2 = {d, a} -> 1/2 and top-ceil(1.5 x 2) = {d, a, b} -> still 1/2.
    assert rates["hitrate_100"] == pytest.approx(0.75)
    assert rates["hitrate_150"] == pytest.approx(0.75)
