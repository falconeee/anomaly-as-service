"""Full MLflow lifecycle against a throwaway SQLite tracking store.

train -> register (@challenger) -> promote (@champion) -> retrain -> gate -> rollback.
"""

import mlflow
import pytest
from mlflow import MlflowClient

from anomaly_service.data.csv_source import write_sample_csv
from anomaly_service.data.synthetic import generate_normal
from anomaly_service.training.config import (
    CSVDataConfig,
    ModelConfig,
    PromotionPolicy,
    SyntheticDataConfig,
    TrainConfig,
    TrainingConfig,
)
from anomaly_service.training.promote import (
    CHALLENGER,
    CHAMPION,
    load_registered,
    rollback,
    status,
)
from anomaly_service.training.train import train

NAME = "mscvae-test"


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # artifacts land in ./mlruns under tmp_path
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow.set_tracking_uri(uri)
    yield MlflowClient(uri)
    mlflow.set_tracking_uri(None)


def tiny_config(**promotion) -> TrainConfig:
    return TrainConfig(
        experiment="test",
        registered_model=NAME,
        data=SyntheticDataConfig(n_train=800, n_test=600, n_features=5, n_anomalies=3),
        model=ModelConfig(window_sizes=[5, 10], seq_len=2),
        training=TrainingConfig(epochs=2, batch_size=64, device="cpu"),
        promotion=PromotionPolicy(latency_rows=200, **promotion),
    )


def test_train_register_promote_and_rollback(tracking):
    first = train(tiny_config(), register=True, promote=True)

    assert first.model_version == "1"
    assert first.decision is not None and first.decision.promoted
    assert str(tracking.get_model_version_by_alias(NAME, CHAMPION).version) == "1"
    run = tracking.get_run(first.run_id)
    assert {"f1", "pa_f1", "auc_pr", "hitrate_100", "latency_ms"} <= set(run.data.metrics)
    assert run.data.tags["data.digest"]

    # Same data + seed -> same quality: the gate (within tolerance) promotes v2.
    second = train(tiny_config(), register=True, promote=True)
    assert second.decision is not None and second.decision.promoted
    assert str(tracking.get_model_version_by_alias(NAME, CHAMPION).version) == "2"
    assert tracking.get_model_version(NAME, "1").tags["replaced_by"] == "2"
    assert tracking.get_model_version(NAME, "2").tags["promotion.compared_with"] == "1"

    rollback(NAME, "1")
    assert str(tracking.get_model_version_by_alias(NAME, CHAMPION).version) == "1"
    rows = status(NAME)
    assert [row["version"] for row in rows] == ["1", "2"]
    assert [row["aliases"] for row in rows] == ["champion", "challenger"]


def test_rejected_challenger_keeps_the_champion(tracking):
    train(tiny_config(), register=True, promote=True)

    # An impossible latency budget makes the gate reject the new version.
    result = train(tiny_config(max_latency_ms=0.0), register=True, promote=True)

    assert result.decision is not None and not result.decision.promoted
    assert str(tracking.get_model_version_by_alias(NAME, CHAMPION).version) == "1"
    assert str(tracking.get_model_version_by_alias(NAME, CHALLENGER).version) == "2"
    assert tracking.get_model_version(NAME, "2").tags["promotion.decision"] == "rejected"


def test_registered_model_loads_through_pyfunc(tracking):
    result = train(tiny_config(), register=True)
    uri = f"models:/{NAME}/{result.model_version}"
    batch = generate_normal(100, n_features=5, seed=9).reset_index(drop=True)

    # Generic MLflow consumers get per-window scores...
    out = mlflow.pyfunc.load_model(uri).predict(batch)
    assert list(out.columns) == ["row", "score", "is_anomaly"]
    assert len(out) == 100

    # ...and the service unwraps the full MSCVAE (RCA included).
    model = load_registered(NAME, result.model_version)
    assert model.feature_names == list(batch.columns)
    assert model.score_windows(batch).phi.shape == (100,)


def test_own_csv_data_trains_registers_and_serves(tracking, tmp_path):
    csv = write_sample_csv(tmp_path / "meu_sistema.csv", n_rows=1500, n_features=4)
    cfg = tiny_config()
    cfg.data = CSVDataConfig(path=str(csv))

    result = train(cfg, register=True, promote=True)

    assert result.decision is not None and result.decision.promoted
    model = load_registered(NAME, result.model_version)
    assert model.feature_names == ["sensor_0", "sensor_1", "sensor_2", "sensor_3"]
    assert result.metrics["auc_pr"] > 0  # the sample has labels in its test part
