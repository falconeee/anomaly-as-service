from datetime import datetime

import numpy as np

from anomaly_service.inference.events import group_events, severity
from anomaly_service.inference.explain import explain, root_causes
from anomaly_service.schemas import RootCause


def test_group_events_merges_small_gaps_and_finds_peaks():
    phi = np.array([0, 5, 6, 0, 7, 0, 0, 0, 9, 0], dtype=float)

    events = group_events(phi, threshold=1.0, max_gap=1)

    assert [(e.start, e.end, e.peak, e.max_score) for e in events] == [(1, 4, 4, 7.0), (8, 8, 8, 9.0)]
    assert len(group_events(phi, threshold=1.0, max_gap=0)) == 3


def test_group_events_skips_warmup_and_handles_no_anomaly():
    phi = np.array([9, 9, 0, 5], dtype=float)

    assert [(e.start, e.end) for e in group_events(phi, 1.0, skip=2)] == [(3, 3)]
    assert group_events(np.zeros(5), 1.0) == []


def test_severity_bands():
    assert [severity(r) for r in (1.1, 1.5, 2.9, 3.0, 10)] == ["low", "medium", "medium", "high", "high"]


def test_root_causes_blame_the_spiked_feature(fitted_model, test_series):
    spike = next(a for a in test_series.anomalies if a.kind == "spike")
    scores = fitted_model.score_windows(test_series.data)
    (event,) = [
        e
        for e in group_events(scores.phi, fitted_model.threshold, max_gap=5)
        if e.start <= spike.end and e.end >= spike.start
    ]
    raw = test_series.data[fitted_model.feature_names].to_numpy()

    causes = root_causes(fitted_model, scores, event, raw, top_k=3)

    top = causes[0]
    assert top.variable == spike.features[0]
    assert top.kind == "magnitude"
    assert top.observed is not None and top.expected is not None and top.observed > top.expected
    assert 0 < sum(c.contribution_pct for c in causes) <= 100.01


def cause(variable, pct, z, kind, observed, expected) -> RootCause:
    return RootCause(
        variable=variable, contribution_pct=pct, peak_z=z, kind=kind, observed=observed, expected=expected
    )


def test_explanation_templates():
    causes = [
        cause("cpu", 70, 8.2, "magnitude", 97.0, 40.0),
        cause("mem", 30, 4.1, "correlation", 50.0, 48.0),
    ]
    stamps = (datetime(2024, 1, 1, 10, 5), datetime(2024, 1, 1, 10, 20), datetime(2024, 1, 1, 10, 12))

    pt = explain("high", 4.13, (5, 20, 12), stamps, causes, "pt")
    en = explain("high", 4.13, (5, 20, 12), (None, None, None), causes, "en")

    assert pt.startswith("Anomalia de severidade alta entre 2024-01-01T10:05:00")
    assert "score 4.1× o limiar" in pt
    assert "cpu (70% da contribuição): valor 97 acima do esperado (40), desvio de 8.2σ" in pt
    assert "mem (30% da contribuição): a relação com as demais variáveis se rompeu" in pt
    assert en.startswith("High-severity anomaly between points 5 and 20 (peak at point 12)")
    assert "mem (30% of the contribution): its relation to the other variables broke" in en


def test_huge_z_scores_are_capped_in_the_text_only():
    huge = cause("x", 100, 16265.2, "magnitude", 2.0, 1.0)
    text = explain("low", 1.2, (0, 1, 0), (None, None, None), [huge])
    assert "desvio de >100σ" in text
