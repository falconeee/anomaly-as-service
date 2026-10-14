"""Training configuration (YAML validated by Pydantic) and dataset loading."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field

from anomaly_service.logger import logger
from anomaly_service.training.evaluate import GroundTruthEvent


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in the YAML is an error, not a no-op


class SyntheticDataConfig(_Strict):
    source: Literal["synthetic"] = "synthetic"
    n_train: int = 4000
    n_test: int = 2000
    n_features: int = 8
    n_anomalies: int = 4
    seed: int = 0


class SMDDataConfig(_Strict):
    source: Literal["smd"] = "smd"
    machine: str = "machine-1-1"
    data_dir: str = "data/smd"
    drop_constant: bool = True


class CSVDataConfig(_Strict):
    """Your own data: one CSV, split chronologically into train/test (see data/csv_source.py)."""

    source: Literal["csv"] = "csv"
    path: str
    timestamp_column: str | None = "timestamp"
    label_column: str | None = "label"  # optional 0/1 column, only used for evaluation
    features: list[str] | None = None  # default: every numeric column
    test_fraction: float = 0.3
    drop_constant: bool = True


DataConfig = Annotated[SyntheticDataConfig | SMDDataConfig | CSVDataConfig, Field(discriminator="source")]


class ModelConfig(_Strict):
    window_sizes: list[int] = [10, 30, 60]
    stride: int = 1
    seq_len: int = 5
    latent_dim: int | None = None
    seed: int = 42


class TrainingConfig(_Strict):
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    gain: float = 1.0
    # Last share of the training data held out to calibrate the threshold/RCA baselines.
    calibration_fraction: float = 0.2
    device: str = "auto"  # auto -> cuda, then mps, then cpu


class PromotionPolicy(_Strict):
    metric: str = "f1"  # compared between challenger and champion on the SAME test set
    tolerance: float = 0.01  # challenger may be this much worse and still win (avoid flapping)
    max_latency_ms: float = 2000.0  # hard limit for one pass over `latency_rows` rows
    latency_rows: int = 1000


class TrainConfig(_Strict):
    experiment: str
    registered_model: str
    data: DataConfig
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    promotion: PromotionPolicy = PromotionPolicy()

    @classmethod
    def from_yaml(cls, path: str | Path) -> TrainConfig:
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))


@dataclass
class TrainData:
    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    labels: np.ndarray
    events: list[GroundTruthEvent]
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def digest(self) -> str:
        """Content hash of train + test, logged as lineage (same data -> same digest)."""
        h = hashlib.sha256()
        for df in (self.train, self.test):
            h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
        return h.hexdigest()[:16]


def load_data(cfg: SyntheticDataConfig | SMDDataConfig | CSVDataConfig) -> TrainData:
    if isinstance(cfg, CSVDataConfig):
        from anomaly_service.data.csv_source import load_csv

        dataset = load_csv(
            cfg.path,
            timestamp_column=cfg.timestamp_column,
            label_column=cfg.label_column,
            features=cfg.features,
            test_fraction=cfg.test_fraction,
            drop_constant=cfg.drop_constant,
        )
        if not dataset.has_labels:
            logger.warning(
                "%s has no label column: detection metrics (f1, auc_pr...) will be 0 and the "
                "promotion gate can only check latency",
                cfg.path,
            )
        return TrainData(
            name=Path(cfg.path).stem,
            train=dataset.train,
            test=dataset.test,
            labels=dataset.labels,
            events=[],
            info={"dropped_features": dataset.dropped_features, "has_labels": dataset.has_labels},
        )

    if isinstance(cfg, SMDDataConfig):
        from anomaly_service.data import smd

        machine = smd.load(cfg.machine, cfg.data_dir, drop_constant=cfg.drop_constant)
        return TrainData(
            name=f"smd-{cfg.machine}",
            train=machine.train,
            test=machine.test,
            labels=machine.labels,
            events=[GroundTruthEvent(i.start, i.end, i.features) for i in machine.interpretations],
            info={"dropped_features": machine.dropped_features},
        )

    from anomaly_service.data.synthetic import make_dataset

    train, test = make_dataset(
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        n_features=cfg.n_features,
        n_anomalies=cfg.n_anomalies,
        seed=cfg.seed,
    )
    return TrainData(
        name="synthetic",
        train=train,
        test=test.data,
        labels=test.labels,
        events=[GroundTruthEvent(a.start, a.end, a.features) for a in test.anomalies],
        info={"anomalies": [f"{a.kind}:{a.start}-{a.end}:{','.join(a.features)}" for a in test.anomalies]},
    )
