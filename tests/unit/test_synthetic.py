import numpy as np
import pandas as pd
import pytest

from anomaly_service.data.synthetic import InjectedAnomaly, generate_normal, inject_anomalies, make_dataset


def test_same_seed_same_data():
    train_a, test_a = make_dataset(n_train=300, n_test=400, seed=3)
    train_b, test_b = make_dataset(n_train=300, n_test=400, seed=3)

    pd.testing.assert_frame_equal(train_a, train_b)
    pd.testing.assert_frame_equal(test_a.data, test_b.data)
    assert test_a.anomalies == test_b.anomalies


def test_labels_match_injected_anomalies():
    _, test = make_dataset(n_train=300, n_test=700, n_anomalies=3, warmup=100, seed=0)

    expected = np.zeros(700, dtype=np.int8)
    for anomaly in test.anomalies:
        assert 100 <= anomaly.start < anomaly.end <= 700
        expected[anomaly.start : anomaly.end] = 1
    np.testing.assert_array_equal(test.labels, expected)
    assert [a.kind for a in test.anomalies] == ["spike", "correlation_break", "drift"]

    # Non-overlapping, in chronological order.
    for prev, nxt in zip(test.anomalies, test.anomalies[1:], strict=False):
        assert prev.end <= nxt.start


def test_features_are_correlated():
    df = generate_normal(2000, n_features=6, seed=0)

    off_diagonal = np.abs(df.corr().to_numpy()[~np.eye(6, dtype=bool)])
    assert off_diagonal.mean() > 0.3


def test_spike_shifts_by_magnitude_std():
    df = generate_normal(500, n_features=3, seed=0)
    spike = InjectedAnomaly(kind="spike", start=200, end=210, features=("f01",), magnitude=5.0)

    out = inject_anomalies(df, [spike]).data

    shift = out["f01"].iloc[200:210] - df["f01"].iloc[200:210]
    assert shift.to_numpy() == pytest.approx(5.0 * df["f01"].std())
    pd.testing.assert_frame_equal(out.drop(index=out.index[200:210]), df.drop(index=df.index[200:210]))


def test_correlation_break_keeps_level_but_inverts_movement():
    df = generate_normal(500, n_features=3, seed=0)
    brk = InjectedAnomaly(kind="correlation_break", start=100, end=160, features=("f00",))

    before = df["f00"].iloc[100:160]
    after = inject_anomalies(df, [brk]).data["f00"].iloc[100:160]

    assert after.mean() == pytest.approx(before.mean())
    assert np.corrcoef(before, after)[0, 1] == pytest.approx(-1.0)


def test_longer_series_extends_a_shorter_one():
    short = generate_normal(300, n_features=4, seed=5)
    long = generate_normal(900, n_features=4, seed=5)

    pd.testing.assert_frame_equal(long.iloc[:300], short)
