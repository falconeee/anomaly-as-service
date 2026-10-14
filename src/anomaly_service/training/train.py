"""Training pipeline: data -> fit -> evaluate -> log to MLflow -> (register) -> (promote).

Usage::

    python -m anomaly_service.training.train --config configs/train_synthetic.yaml --register --promote

The tracking server comes from the MLFLOW_TRACKING_URI environment variable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import warnings
from dataclasses import dataclass
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.data.pandas_dataset import from_pandas
from mlflow.models import ModelSignature, infer_signature

from anomaly_service.logger import logger
from anomaly_service.model import MSCVAE
from anomaly_service.model import pyfunc as pyfunc_module
from anomaly_service.model.pyfunc import BUNDLE_ARTIFACT, predict_frame
from anomaly_service.training.config import TrainConfig, TrainData, load_data
from anomaly_service.training.evaluate import evaluate_model, scores_to_rows
from anomaly_service.training.promote import CHALLENGER, PromotionDecision, promote_challenger

RUNTIME_PACKAGES = ("torch", "numpy", "pandas", "scipy", "mlflow-skinny")


@dataclass
class TrainResult:
    run_id: str
    metrics: dict[str, float]
    model_version: str | None
    decision: PromotionDecision | None


def git_sha() -> str:
    """Commit the code was trained from; GIT_SHA wins (set at Docker build time)."""
    if sha := os.getenv("GIT_SHA"):
        return sha
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout
        return f"{sha}-dirty" if dirty.strip() else sha
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in d.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def pip_requirements() -> list[str]:
    # Local version labels (e.g. torch "+cpu") are platform specific: pin the public version.
    return [f"{p}=={pkg_version(p).split('+')[0]}" for p in RUNTIME_PACKAGES]


def log_scores_plot(model: MSCVAE, data: TrainData) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # the `train` dependency group is optional
        logger.warning("matplotlib not installed; skipping score plot")
        return

    scores = model.score_windows(data.test)
    phi = scores_to_rows(scores.row_index, scores.phi, len(data.test))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(np.arange(len(phi)), phi, lw=0.7, label="score (phi)")
    ax.axhline(model.threshold, color="tab:red", ls="--", lw=1, label="threshold (SPOT)")
    for event in data.events:
        ax.axvspan(event.start, event.end, color="tab:orange", alpha=0.25)
    ax.set_yscale("log")
    ax.set_xlabel("test row")
    ax.set_title(f"{data.name}: anomaly score vs threshold (shaded = labeled anomalies)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    mlflow.log_figure(fig, "plots/test_scores.png")
    plt.close(fig)


def signature(model_input: pd.DataFrame, model_output: pd.DataFrame) -> ModelSignature:
    with warnings.catch_warnings():
        # MLflow warns that integer columns cannot hold missing values; the output `row`
        # column is an index and never has any.
        warnings.filterwarnings("ignore", message=".*integer column.*")
        return infer_signature(model_input, model_output)


def model_card(cfg: TrainConfig, data: TrainData, metrics: dict[str, float], sha: str) -> str:
    rows = "\n".join(f"| {k} | {v:.4f} |" for k, v in sorted(metrics.items()))
    return f"""# {cfg.registered_model}

**MSCVAE** multivariate time-series anomaly detector with root-cause attribution.

- Data: `{data.name}`: {len(data.train)} training rows, {len(data.test)} test rows,
  {data.train.shape[1]} features, digest `{data.digest}`
- Code: git `{sha}`
- Windows {cfg.model.window_sizes}, stride {cfg.model.stride}, seq_len {cfg.model.seq_len},
  {cfg.training.epochs} epochs

## Intended use
Unsupervised detection of anomalous periods in the SAME system (same features, same
operating regime) the model was trained on, with per-variable explanations.

## Evaluation (labeled test set)
| metric | value |
|---|---|
{rows}

## Limitations
- One model per feature set: a request must carry exactly the training features.
- The first `max(window_sizes) - 1` points of a request lack history (warm-up) and are less reliable.
- Trained on presumed-normal data; anomalies inside the training period lower sensitivity.
"""


def train(cfg: TrainConfig, register: bool = False, promote: bool = False) -> TrainResult:
    mlflow.set_experiment(cfg.experiment)
    data = load_data(cfg.data)
    sha = git_sha()
    logger.info("Data %s: train=%s test=%s", data.name, data.train.shape, data.test.shape)

    with mlflow.start_run(run_name=f"{cfg.registered_model}-{sha}") as run:
        mlflow.set_tags({"git_sha": sha, "data.name": data.name, "data.digest": data.digest})
        mlflow.log_params(flatten(cfg.model_dump(exclude={"experiment", "registered_model"})))
        mlflow.log_params({f"data.info.{k}": str(v)[:500] for k, v in data.info.items()})
        mlflow.log_input(
            from_pandas(data.train.reset_index(), name=f"{data.name}-train", digest=data.digest),
            context="training",
        )
        mlflow.log_text(yaml.safe_dump(cfg.model_dump(), sort_keys=False), "config.yaml")

        model = MSCVAE(
            latent_dim=cfg.model.latent_dim,
            window_sizes=cfg.model.window_sizes,
            stride=cfg.model.stride,
            seq_len=cfg.model.seq_len,
            device=None if cfg.training.device == "auto" else cfg.training.device,
            seed=cfg.model.seed,
        )

        def on_epoch(epoch: int, epoch_metrics: dict[str, float]) -> None:
            mlflow.log_metrics({f"train.{k}": v for k, v in epoch_metrics.items()}, step=epoch)

        model.fit(
            data.train,
            epochs=cfg.training.epochs,
            batch_size=cfg.training.batch_size,
            lr=cfg.training.lr,
            gain=cfg.training.gain,
            callback=on_epoch,
            calibration_fraction=cfg.training.calibration_fraction,
        )
        logger.info("Evaluating on %d test rows", len(data.test))
        metrics = evaluate_model(
            model, data.test, data.labels, data.events, latency_rows=cfg.promotion.latency_rows
        )
        metrics["threshold"] = float(model.threshold)
        log_scores_plot(model, data)
        card = model_card(cfg, data, metrics, sha)
        mlflow.log_text(card, "model_card.md")

        with tempfile.TemporaryDirectory() as tmp:
            bundle = model.save(Path(tmp) / "bundle")
            example = data.test.iloc[:200].reset_index(drop=True)
            info = mlflow.pyfunc.log_model(
                name="model",
                python_model=str(Path(pyfunc_module.__file__).with_name("pyfunc_model.py")),
                artifacts={BUNDLE_ARTIFACT: str(bundle)},
                signature=signature(example, predict_frame(model, example)),
                input_example=example,
                pip_requirements=pip_requirements(),
                registered_model_name=cfg.registered_model if register else None,
                metadata={"features": model.feature_names, "threshold": float(model.threshold)},
            )
        # Run metrics (for comparing runs) and the same numbers linked to the logged model.
        mlflow.log_metrics(metrics)
        mlflow.log_metrics(metrics, model_id=info.model_id)
        logger.info("Metrics: %s", {k: round(v, 4) for k, v in metrics.items()})

    model_version = None
    if register:
        client = MlflowClient()
        model_version = str(info.registered_model_version)
        for key, value in {"git_sha": sha, "data.digest": data.digest, "data.name": data.name}.items():
            client.set_model_version_tag(cfg.registered_model, model_version, key, value)
        client.update_model_version(cfg.registered_model, model_version, description=card)
        client.set_registered_model_alias(cfg.registered_model, CHALLENGER, model_version)
        logger.info("Registered %s v%s as @%s", cfg.registered_model, model_version, CHALLENGER)

    decision = None
    if promote:
        if not register:
            raise ValueError("--promote requires --register")
        decision = promote_challenger(cfg.registered_model, data, cfg.promotion)

    return TrainResult(run.info.run_id, metrics, model_version, decision)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="train", description="Train, evaluate and log an MSCVAE model.")
    parser.add_argument("--config", required=True, help="training YAML (see configs/)")
    parser.add_argument("--register", action="store_true", help="register the model and tag it @challenger")
    parser.add_argument("--promote", action="store_true", help="run the champion/challenger quality gate")
    parser.add_argument("--epochs", type=int, help="override training.epochs (quick runs)")
    args = parser.parse_args(argv)

    cfg = TrainConfig.from_yaml(args.config)
    if args.epochs is not None:
        cfg.training.epochs = args.epochs
    result = train(cfg, register=args.register, promote=args.promote)
    print(f"run_id={result.run_id} version={result.model_version}")
    if result.decision is not None:
        print(f"promoted={result.decision.promoted} reason={result.decision.reason}")


if __name__ == "__main__":
    main()
