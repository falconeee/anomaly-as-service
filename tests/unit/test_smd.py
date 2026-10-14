"""SMD loader against tiny fake files (no network: cached files are never re-downloaded)."""

import numpy as np

from anomaly_service.data import smd


def write_fake_machine(data_dir, machine="machine-x-1", n=20):
    rng = np.random.default_rng(0)
    train = rng.uniform(size=(n, smd.N_METRICS))
    train[:, 4] = 0.0  # constant in train
    test = rng.uniform(size=(n, smd.N_METRICS))
    labels = np.zeros(n, dtype=int)
    labels[5:9] = 1
    files = {
        "train": "\n".join(",".join(f"{v:.6f}" for v in row) for row in train),
        "test": "\n".join(",".join(f"{v:.6f}" for v in row) for row in test),
        "test_label": "\n".join(str(v) for v in labels),
        "interpretation_label": "5-8:1,5,12\n",
    }
    for split, content in files.items():
        path = data_dir / split / f"{machine}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    return machine


def test_load_parses_splits_and_drops_constant_metrics(tmp_path):
    machine = write_fake_machine(tmp_path)

    data = smd.load(machine, tmp_path)

    assert data.dropped_features == ["m04"]
    assert data.train.shape == (20, 37) and data.test.shape == (20, 37)
    assert "m04" not in data.train.columns
    assert data.labels.sum() == 4
    assert data.test.index[0] > data.train.index[-1]  # test continues after train


def test_interpretation_labels_are_zero_indexed_and_end_exclusive(tmp_path):
    machine = write_fake_machine(tmp_path)

    (event,) = smd.load(machine, tmp_path).interpretations

    assert (event.start, event.end) == (5, 9)
    assert event.features == ("m00", "m04", "m11")
