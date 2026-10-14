import numpy as np
import pandas as pd
import pytest

from anomaly_service.data.csv_source import load_csv, write_sample_csv
from anomaly_service.data.examples import build_request


def write(tmp_path, frame: pd.DataFrame):
    path = tmp_path / "data.csv"
    frame.to_csv(path, index=False)
    return path


def test_sample_csv_round_trips_with_labels_only_in_test(tmp_path):
    path = write_sample_csv(tmp_path / "sample.csv", n_rows=1000)

    data = load_csv(path)

    assert data.has_labels
    assert list(data.train.columns) == [f"sensor_{i}" for i in range(6)]
    assert len(data.train) == 700 and len(data.test) == 300
    assert data.labels.sum() > 0
    assert isinstance(data.train.index, pd.DatetimeIndex)
    assert data.test.index[0] > data.train.index[-1]


def test_rows_are_sorted_by_timestamp_and_constant_columns_dropped(tmp_path):
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="1min")[::-1],
            "a": np.arange(10.0),
            "flat": 1.0,
            "host": "srv-1",  # non-numeric: ignored
        }
    )

    data = load_csv(write(tmp_path, frame), test_fraction=0.2)

    assert data.dropped_features == ["flat"]
    assert list(data.train.columns) == ["a"]
    assert data.train["a"].tolist() == [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
    assert not data.has_labels and data.labels.tolist() == [0, 0]


def test_without_timestamps_the_index_is_positional(tmp_path):
    frame = pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) ** 2})

    data = load_csv(write(tmp_path, frame), timestamp_column=None, label_column=None)
    body = build_request(data.test)

    assert "timestamps" not in body
    assert set(body["series"]) == {"a", "b"}


def test_unknown_feature_is_an_error(tmp_path):
    frame = pd.DataFrame({"a": np.arange(10.0)})
    with pytest.raises(ValueError, match="not found"):
        load_csv(write(tmp_path, frame), features=["a", "b"])
