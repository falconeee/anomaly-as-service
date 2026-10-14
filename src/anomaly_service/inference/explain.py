"""Deterministic root-cause analysis and textual explanation of one anomaly event.

The numbers come from the model's own RCA (MSCVAE.score_windows + rca_window_scores); the
text is a template over those numbers. Same input -> same explanation, no extra model.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np

from anomaly_service.inference.events import WindowEvent
from anomaly_service.model import MSCVAE, WindowScores
from anomaly_service.schemas import RootCause, Severity


def _finite(value: float) -> float | None:
    return round(float(value), 6) if math.isfinite(value) else None


def root_causes(
    model: MSCVAE,
    scores: WindowScores,
    event: WindowEvent,
    raw: np.ndarray,
    top_k: int = 3,
    z_thresh: float = 3.0,
) -> list[RootCause]:
    """
    Root causes of `event`, strongest first. `raw` holds the input values (rows x features,
    in the model's feature order) and supplies the observed value of each cause.
    """
    windows = slice(event.start, event.end + 1)
    attribution = model.rca_window_scores(scores)[windows]
    contrib = attribution.sum(axis=0)
    peak_z = attribution.max(axis=0)
    selected = model.select_contributors(contrib, peak_z, z_thresh=z_thresh)[:top_k]
    total = float(contrib[selected].sum())

    # Is the variable blamed for its own value (diagonal + raw value channels) or for its
    # relations with the others (off-diagonal channel)?
    magnitude = (model.rca_self_weight * scores.s_diag + scores.s_val)[windows].sum(axis=0)
    relational = (model.rca_rel_weight * scores.s_off)[windows].sum(axis=0)

    causes = []
    for i in selected:
        # Observed vs expected at the window where THIS variable deviates the most.
        window = event.start + int(np.argmax(attribution[:, i]))
        row = int(scores.row_index[window])
        expected = model.denormalize(scores.recon_z[window])[i]
        causes.append(
            RootCause(
                variable=str(scores.feature_names[i]),
                contribution_pct=round(100 * float(contrib[i]) / total, 2) if total > 0 else 0.0,
                peak_z=round(float(peak_z[i]), 3),
                kind="correlation" if relational[i] > magnitude[i] else "magnitude",
                observed=_finite(raw[row, i]),
                expected=_finite(expected),
            )
        )
    return causes


_TEXT: dict[str, dict[str, Any]] = {
    "pt": {
        "severity": {"low": "baixa", "medium": "média", "high": "alta"},
        "span_time": "entre {start} e {end} (pico em {peak})",
        "span_index": "entre os pontos {start} e {end} (pico no ponto {peak})",
        "header": "Anomalia de severidade {severity} {span}; score {ratio:.1f}× o limiar.",
        "causes": "Principais causas: {items}.",
        "above": "acima",
        "below": "abaixo",
        "magnitude": "{var} ({pct:.0f}% da contribuição): valor {obs} {direction} do esperado ({exp}), "
        "desvio de {z} em relação ao seu erro normal",
        "correlation": "{var} ({pct:.0f}% da contribuição): a relação com as demais variáveis se rompeu "
        "(valor {obs}, esperado {exp}), desvio de {z}",
        "none": "Nenhuma variável se destacou individualmente; a anomalia é difusa.",
    },
    "en": {
        "severity": {"low": "low", "medium": "medium", "high": "high"},
        "span_time": "between {start} and {end} (peak at {peak})",
        "span_index": "between points {start} and {end} (peak at point {peak})",
        "header": "{Severity}-severity anomaly {span}; score {ratio:.1f}x the threshold.",
        "causes": "Main causes: {items}.",
        "above": "above",
        "below": "below",
        "magnitude": "{var} ({pct:.0f}% of the contribution): value {obs} {direction} the expected ({exp}), "
        "{z} off its normal error",
        "correlation": "{var} ({pct:.0f}% of the contribution): its relation to the other variables broke "
        "(value {obs}, expected {exp}), {z} off",
        "none": "No single variable stands out; the anomaly is diffuse.",
    },
}


def _num(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.4g}"


def _sigma(z: float) -> str:
    # A variable whose normal error is ~0 yields astronomic z-scores: true, but unreadable.
    return f"{z:.1f}σ" if z < 100 else ">100σ"


def _when(index: int, timestamp: datetime | None) -> str:
    return timestamp.isoformat(timespec="seconds") if timestamp is not None else str(index)


def explain(
    severity: Severity,
    threshold_ratio: float,
    indices: tuple[int, int, int],
    timestamps: tuple[datetime | None, datetime | None, datetime | None],
    causes: list[RootCause],
    language: str = "pt",
) -> str:
    """One paragraph describing when, how strong and why. `indices` = (start, end, peak)."""
    text = _TEXT[language]
    has_time = all(t is not None for t in timestamps)
    start, end, peak = (_when(i, t if has_time else None) for i, t in zip(indices, timestamps, strict=True))
    span = text["span_time" if has_time else "span_index"].format(start=start, end=end, peak=peak)
    sev = text["severity"][severity]
    header = text["header"].format(severity=sev, Severity=sev.capitalize(), span=span, ratio=threshold_ratio)

    items = []
    for cause in causes:
        observed, expected = cause.observed, cause.expected
        direction = text["above"] if (observed or 0) >= (expected or 0) else text["below"]
        items.append(
            text[cause.kind].format(
                var=cause.variable,
                pct=cause.contribution_pct,
                obs=_num(observed),
                exp=_num(expected),
                direction=direction,
                z=_sigma(cause.peak_z),
            )
        )
    tail = text["causes"].format(items="; ".join(items)) if items else text["none"]
    return f"{header} {tail}"
