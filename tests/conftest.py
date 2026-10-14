"""Shared fixtures.

Training is the slow part, so a small model is fitted ONCE per test session on synthetic
data and shared (read-only) by every test that needs a fitted model.
"""

import os

# Must be set before anything imports the Celery app (it reads settings at import time):
# tasks run inline (eager) with in-memory broker/backend, and no registry polling.
os.environ.update(
    {
        "CELERY_TASK_ALWAYS_EAGER": "true",
        "CELERY_BROKER_URL": "memory://",
        "CELERY_RESULT_BACKEND": "cache+memory://",
        "MODEL_REFRESH_INTERVAL_S": "0",
        "MLFLOW_DISABLE_AGENT_HINT": "1",
    }
)

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from anomaly_service.data.synthetic import SyntheticSeries, make_dataset  # noqa: E402
from anomaly_service.model import MSCVAE  # noqa: E402

# Small windows/epochs keep the session fixture to a couple of seconds on CPU.
MODEL_KWARGS = {"window_sizes": (5, 10, 20), "stride": 1, "seq_len": 3, "device": "cpu", "seed": 7}
FIT_KWARGS = {"epochs": 5, "batch_size": 64, "verbose": False}


@pytest.fixture(scope="session")
def dataset() -> tuple[pd.DataFrame, SyntheticSeries]:
    return make_dataset(n_train=800, n_test=600, n_features=6, n_anomalies=3, warmup=60, seed=1)


@pytest.fixture(scope="session")
def train_df(dataset) -> pd.DataFrame:
    return dataset[0]


@pytest.fixture(scope="session")
def test_series(dataset) -> SyntheticSeries:
    return dataset[1]


@pytest.fixture(scope="session")
def fitted_model(train_df) -> MSCVAE:
    model = MSCVAE(**MODEL_KWARGS)
    model.fit(train_df, **FIT_KWARGS)
    return model


@pytest.fixture(scope="session")
def bundle(fitted_model, tmp_path_factory):
    return fitted_model.save(tmp_path_factory.mktemp("model") / "bundle")


def to_payload(df: pd.DataFrame, **options) -> dict:
    """DataFrame (DatetimeIndex) -> JSON body of POST /v1/detect."""
    return {
        "timestamps": [t.isoformat() for t in df.index],
        "series": {str(c): df[c].tolist() for c in df.columns},
        "options": options,
    }
