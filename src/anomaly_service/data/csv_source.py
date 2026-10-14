"""Loader for user-provided data: one CSV file, one row per time step, one column per variable.

Expected layout::

    timestamp,cpu,memory,latency_ms,label
    2024-01-01T00:00:00,41.2,63.0,120.5,0
    2024-01-01T00:01:00,40.8,63.4,118.9,0
    ...

- `timestamp` (optional): parsed and used as the index; rows are sorted by it.
- `label` (optional): 1 = known anomaly, 0 = normal. Only used to EVALUATE the model.
- every other numeric column is a feature (or only those listed in `features`).

The file is split chronologically: the first (1 - test_fraction) rows train the model, the
last test_fraction rows evaluate it. The training part should be (mostly) normal behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class CSVDataset:
    train: pd.DataFrame
    test: pd.DataFrame
    labels: np.ndarray  # (len(test),) 0/1; all zeros when the file has no label column
    has_labels: bool
    dropped_features: list[str]


def load_csv(
    path: str | Path,
    timestamp_column: str | None = "timestamp",
    label_column: str | None = "label",
    features: list[str] | None = None,
    test_fraction: float = 0.3,
    drop_constant: bool = True,
) -> CSVDataset:
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1")
    df = pd.read_csv(path)

    if timestamp_column and timestamp_column in df.columns:
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        df = df.sort_values(timestamp_column).set_index(timestamp_column)
        df.index.name = "timestamp"

    has_labels = bool(label_column and label_column in df.columns)
    labels_all = df.pop(label_column).fillna(0).astype(np.int8).to_numpy() if has_labels else None

    if features is None:
        features = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in {path}: {missing}")
    data = df[features].astype(float)
    data.columns = [str(c) for c in data.columns]

    split = int(len(data) * (1 - test_fraction))
    if split < 2 or len(data) - split < 1:
        raise ValueError(f"{path} has too few rows ({len(data)}) for test_fraction={test_fraction}")
    train, test = data.iloc[:split], data.iloc[split:]

    # A variable that never changes during training has std ~0: any movement in test would
    # explode its z-score. It carries no information for the model, so it is removed.
    dropped: list[str] = []
    if drop_constant:
        dropped = [c for c in train.columns if not train[c].std() > 1e-8]
        train, test = train.drop(columns=dropped), test.drop(columns=dropped)

    labels = labels_all[split:] if labels_all is not None else np.zeros(len(test), dtype=np.int8)
    return CSVDataset(train=train, test=test, labels=labels, has_labels=has_labels, dropped_features=dropped)


def write_sample_csv(path: str | Path, n_rows: int = 4500, n_features: int = 6, seed: int = 0) -> Path:
    """
    Writes a CSV in the expected layout (timestamp, features, label) from the synthetic
    generator, with anomalies injected only in the last 30% (the test part of the split).
    Handy for trying the "own data" workflow end to end.
    """
    from anomaly_service.data.synthetic import make_dataset

    n_test = int(n_rows * 0.3)
    train, test = make_dataset(
        n_train=n_rows - n_test, n_test=n_test, n_features=n_features, n_anomalies=3, seed=seed
    )
    names = [f"sensor_{i}" for i in range(n_features)]
    frame = pd.concat([train, test.data])
    frame.columns = names
    frame["label"] = np.concatenate([np.zeros(len(train), dtype=np.int8), test.labels])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.round(4).to_csv(path, index_label="timestamp")
    return path


if __name__ == "__main__":
    import sys

    target = sys.argv[1] if len(sys.argv) > 1 else "datasets/exemplo.csv"
    print(f"wrote {write_sample_csv(target)}")
