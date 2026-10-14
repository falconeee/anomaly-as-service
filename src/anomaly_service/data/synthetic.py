"""Synthetic multivariate time series with labeled, injected anomalies.

Normal behavior is built from a few shared latent factors (sinusoids with different periods)
mixed into every feature, plus AR(1) noise. Features are therefore CORRELATED, which is
exactly the structure MSCVAE learns from its correlation matrices.

Three anomaly kinds cover the two decoder routes of the model:

- ``spike``: a feature jumps by `magnitude` standard deviations (value route);
- ``drift``: a feature ramps away linearly up to `magnitude` std (value route, slowly);
- ``correlation_break``: a feature is mirrored around its local mean. Its range and
  marginal distribution stay plausible, but it now moves AGAINST its partners
  (matrix route; invisible to per-feature thresholds).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

AnomalyKind = Literal["spike", "drift", "correlation_break"]
ANOMALY_KINDS: tuple[AnomalyKind, ...] = ("spike", "correlation_break", "drift")


@dataclass(frozen=True)
class InjectedAnomaly:
    kind: AnomalyKind
    start: int  # first row (inclusive)
    end: int  # last row (exclusive)
    features: tuple[str, ...]
    magnitude: float = 6.0  # in standard deviations of the affected feature


@dataclass
class SyntheticSeries:
    data: pd.DataFrame  # DatetimeIndex named "timestamp", one column per feature
    labels: np.ndarray  # (n_rows,) int8, 1 inside injected anomalies
    anomalies: list[InjectedAnomaly]


def feature_names(n_features: int) -> list[str]:
    return [f"f{i:02d}" for i in range(n_features)]


def generate_normal(
    n_rows: int,
    n_features: int = 8,
    n_factors: int = 3,
    freq: str = "1min",
    start: str = "2024-01-01",
    seed: int = 0,
) -> pd.DataFrame:
    """Correlated, anomaly-free multivariate series."""
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)[:, None]

    # Structural parameters are drawn BEFORE the noise, so the same seed yields the same
    # system for any n_rows (and a longer series extends a shorter one).
    periods = rng.uniform(30, 300, n_factors)
    phases = rng.uniform(0, 2 * np.pi, n_factors)
    loadings = rng.normal(0, 1, (n_factors, n_features))  # how each feature mixes the factors
    offsets = rng.uniform(10, 100, n_features)
    scales = rng.uniform(1, 10, n_features)
    factors = np.sin(2 * np.pi * t / periods + phases)  # (T, K)

    # AR(1) noise: consecutive points are correlated, like real sensor noise.
    noise = np.zeros((n_rows, n_features))
    eps = rng.normal(0, 0.1, (n_rows, n_features))
    for i in range(1, n_rows):
        noise[i] = 0.5 * noise[i - 1] + eps[i]

    values = offsets + scales * (factors @ loadings + noise)

    index = pd.date_range(start, periods=n_rows, freq=freq, name="timestamp")
    return pd.DataFrame(values, index=index, columns=feature_names(n_features))


def inject_anomalies(df: pd.DataFrame, anomalies: Sequence[InjectedAnomaly]) -> SyntheticSeries:
    """Returns a copy of `df` with `anomalies` applied, plus point-wise labels."""
    out = df.copy()
    labels = np.zeros(len(df), dtype=np.int8)
    std = df.std()

    for anomaly in anomalies:
        rows = slice(anomaly.start, anomaly.end)
        length = anomaly.end - anomaly.start
        for feature in anomaly.features:
            col = out.columns.get_loc(feature)
            segment = out.iloc[rows, col].to_numpy()
            if anomaly.kind == "spike":
                segment = segment + anomaly.magnitude * std[feature]
            elif anomaly.kind == "drift":
                segment = segment + np.linspace(0, anomaly.magnitude * std[feature], length)
            elif anomaly.kind == "correlation_break":
                segment = 2 * segment.mean() - segment
            else:
                raise ValueError(f"Unknown anomaly kind: {anomaly.kind!r}")
            out.iloc[rows, col] = segment
        labels[rows] = 1

    return SyntheticSeries(data=out, labels=labels, anomalies=list(anomalies))


def make_dataset(
    n_train: int = 2000,
    n_test: int = 1000,
    n_features: int = 8,
    n_anomalies: int = 3,
    warmup: int = 100,
    seed: int = 0,
) -> tuple[pd.DataFrame, SyntheticSeries]:
    """
    One continuous normal series split into a clean training part and a test part with
    `n_anomalies` non-overlapping anomalies (kinds cycle spike -> correlation_break -> drift).
    The first `warmup` test rows are kept clean so the model has history before the first event.
    """
    rng = np.random.default_rng(seed)
    full = generate_normal(n_train + n_test, n_features=n_features, seed=seed)
    train, test = full.iloc[:n_train], full.iloc[n_train:]

    names = feature_names(n_features)
    slot = (n_test - warmup) // n_anomalies
    if slot < 60:
        raise ValueError("n_test - warmup must leave at least 60 rows per anomaly")

    anomalies = []
    for i in range(n_anomalies):
        kind = ANOMALY_KINDS[i % len(ANOMALY_KINDS)]
        length = int(rng.integers(5, 15)) if kind == "spike" else int(rng.integers(20, min(80, slot // 2)))
        start = warmup + i * slot + int(rng.integers(0, slot - length))
        feature = names[int(rng.integers(0, n_features))]
        anomalies.append(InjectedAnomaly(kind=kind, start=start, end=start + length, features=(feature,)))

    return train, inject_anomalies(test, anomalies)
