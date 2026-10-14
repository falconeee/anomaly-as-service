"""Phase 0 smoke tests: the package installs, the model imports and a tiny fit/predict runs.

These are intentionally coarse. Phase 1 adds the real model tests (determinism,
save/load round-trip, column order, NaNs, short series).
"""

import numpy as np
import pandas as pd

import anomaly_service
from anomaly_service.model import MSCVAE


def _correlated_series(n_rows: int = 400, n_features: int = 4, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n_rows)
    base = np.sin(2 * np.pi * t / 50)
    data = {f"x{i}": base * (i + 1) + rng.normal(0, 0.1, n_rows) for i in range(n_features)}
    return pd.DataFrame(data)


def test_package_exposes_version():
    assert anomaly_service.__version__


def test_tiny_fit_and_predict_on_cpu():
    df = _correlated_series()
    model = MSCVAE(window_sizes=(5, 10), stride=1, seq_len=3, device="cpu")

    model.fit(df, epochs=1, batch_size=64, verbose=False)
    out = model.predict(df)

    assert model.threshold is not None and np.isfinite(model.threshold)
    assert len(out["phi"]) == len(df)
    assert np.all(np.isfinite(out["phi"]))
