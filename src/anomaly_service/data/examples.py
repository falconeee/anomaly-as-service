"""Builds example request bodies for POST /v1/detect from a training config's TEST split.

    python -m anomaly_service.data.examples --config configs/train_synthetic.yaml \\
        --out examples/request_synthetic.json
    python -m anomaly_service.data.examples --config configs/train_smd.yaml \\
        --start 15500 --rows 1500 --out examples/request_smd.json

Using the training config guarantees the features match the model trained from it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from anomaly_service.training.config import TrainConfig, load_data


def build_request(df: pd.DataFrame, **options: Any) -> dict[str, Any]:
    body: dict[str, Any] = {}
    if isinstance(df.index, pd.DatetimeIndex):  # timestamps are optional in the API
        body["timestamps"] = [t.isoformat() for t in df.index]
    body["series"] = {str(c): [None if pd.isna(v) else round(float(v), 6) for v in df[c]] for c in df.columns}
    body["options"] = options
    return body


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="examples", description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--start", type=int, default=0, help="first test row")
    parser.add_argument("--rows", type=int, help="number of rows (default: all)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    data = load_data(TrainConfig.from_yaml(args.config).data)
    end = None if args.rows is None else args.start + args.rows
    window = data.test.iloc[args.start : end]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(build_request(window, top_k=3, language="pt")))
    labeled = int(data.labels[args.start : end].sum())
    print(f"{out}: {len(window)} points x {window.shape[1]} features ({labeled} labeled anomalous points)")


if __name__ == "__main__":
    main()
