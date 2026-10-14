"""Server Machine Dataset (SMD) loader.

SMD (Su et al., KDD 2019, "OmniAnomaly") has 28 server machines, each with 38 metrics
sampled every minute for ~5 weeks, split in half into train (normal) and test (labeled).
Values are already min-max normalized to [0, 1].

Two label files exist for the test half:
- ``test_label``: 0/1 per row (detection ground truth);
- ``interpretation_label``: ``start-end:d1,d2,...`` per anomaly, listing the metrics
  (1-indexed) that caused it (root-cause ground truth).

Files are downloaded once from the OmniAnomaly GitHub repository and cached in `data_dir`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from anomaly_service.logger import logger

BASE_URL = "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/master/ServerMachineDataset"
SPLITS = ("train", "test", "test_label", "interpretation_label")
N_METRICS = 38


@dataclass(frozen=True)
class Interpretation:
    start: int  # first test row (inclusive)
    end: int  # last test row (exclusive)
    features: tuple[str, ...]


@dataclass
class SMDMachine:
    machine: str
    train: pd.DataFrame
    test: pd.DataFrame
    labels: np.ndarray  # (len(test),) 0/1
    interpretations: list[Interpretation]
    dropped_features: list[str]


def metric_name(index: int) -> str:
    """0-indexed metric -> column name (kept stable even when constant metrics are dropped)."""
    return f"m{index:02d}"


def download(machine: str = "machine-1-1", data_dir: str | Path = "data/smd") -> dict[str, Path]:
    """Downloads the four files of `machine` (skipping those already cached)."""
    paths = {}
    for split in SPLITS:
        path = Path(data_dir) / split / f"{machine}.txt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"{BASE_URL}/{split}/{machine}.txt"
            logger.info("Downloading %s", url)
            response = httpx.get(url, timeout=60, follow_redirects=True)
            response.raise_for_status()
            path.write_bytes(response.content)
        paths[split] = path
    return paths


def _parse_interpretations(path: Path) -> list[Interpretation]:
    items = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        span, dims = line.split(":")
        start, end = (int(x) for x in span.split("-"))
        # Dimensions are 1-indexed in the file; `end` is inclusive there.
        features = tuple(metric_name(int(d) - 1) for d in dims.split(","))
        items.append(Interpretation(start=start, end=end + 1, features=features))
    return items


def load(
    machine: str = "machine-1-1",
    data_dir: str | Path = "data/smd",
    drop_constant: bool = True,
    freq: str = "1min",
) -> SMDMachine:
    """
    Loads one machine as DataFrames with a synthetic 1-minute DatetimeIndex (SMD ships no
    timestamps; test continues right after train).

    With `drop_constant`, metrics that never change in TRAIN are removed: their z-score
    std would be ~0, so any movement in test would explode the attribute matrices.
    """
    paths = download(machine, data_dir)
    train = np.loadtxt(paths["train"], delimiter=",")
    test = np.loadtxt(paths["test"], delimiter=",")
    labels = np.loadtxt(paths["test_label"], dtype=np.int8)

    columns = [metric_name(i) for i in range(N_METRICS)]
    index = pd.date_range("2024-01-01", periods=len(train) + len(test), freq=freq, name="timestamp")
    train_df = pd.DataFrame(train, columns=columns, index=index[: len(train)])
    test_df = pd.DataFrame(test, columns=columns, index=index[len(train) :])

    dropped: list[str] = []
    if drop_constant:
        dropped = [c for c in columns if train_df[c].std() < 1e-8]
        train_df = train_df.drop(columns=dropped)
        test_df = test_df.drop(columns=dropped)

    interpretations = _parse_interpretations(paths["interpretation_label"])
    return SMDMachine(
        machine=machine,
        train=train_df,
        test=test_df,
        labels=labels,
        interpretations=interpretations,
        dropped_features=dropped,
    )
