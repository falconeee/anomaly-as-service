"""Save and load a fitted MSCVAE as a directory bundle.

Bundle layout (format_version 1)::

    weights.pt       torch state_dict of the network (tensors only)
    metadata.json    constructor config, feature schema, scaler, threshold, versions
    baselines.npz    per-variable median/MAD reconstruction-error baselines (RCA)

Nothing is pickled. Weights are read with ``torch.load(weights_only=True)`` and the rest is
plain JSON/NPZ (``allow_pickle=False``), so opening a bundle can never execute code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from anomaly_service import __version__
from anomaly_service.model.mscvae import MSCVAE, MSCVAE_Hybrid

FORMAT_VERSION = 1
WEIGHTS_FILE = "weights.pt"
METADATA_FILE = "metadata.json"
BASELINES_FILE = "baselines.npz"

_BASELINE_ATTRS = (
    "mat_diag_med_",
    "mat_diag_mad_",
    "mat_off_med_",
    "mat_off_mad_",
    "val_err_med_",
    "val_err_mad_",
)


def _json_float(value: Any) -> float | None:
    """JSON has no NaN/inf: encode them as null."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def save_model(model: MSCVAE, path: str | Path) -> Path:
    """Writes a fitted model to the directory `path` (created if needed) and returns it."""
    if model.model is None or model.threshold is None or model.mat_diag_med_ is None:
        raise ValueError("Only a fitted model can be saved. Call .fit() first.")

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    generator = model.generator
    features = generator.feature_names

    metadata = {
        "format_version": FORMAT_VERSION,
        "versions": {"anomaly_service": __version__, "torch": torch.__version__},
        "config": {
            "n_features": model.n_features,
            "latent_dim": model.model.latent_dim,
            "window_sizes": list(model.window_sizes),
            "stride": model.stride,
            "seq_len": model.seq_len,
            "seed": model.seed,
        },
        "features": features,
        "scaler": {
            "mean": [_json_float(v) for v in generator.mean.to_numpy()],
            "std": [_json_float(v) for v in generator.std.to_numpy()],
            "min_physical": [_json_float(generator.min_physical_vals.get(f)) for f in features],
        },
        "threshold": float(model.threshold),
        "gain": float(model.gain),
        "rca": {"self_weight": model.rca_self_weight, "rel_weight": model.rca_rel_weight},
    }

    # Tensors are moved to CPU so a model trained on MPS/CUDA loads anywhere.
    state = {k: v.detach().cpu() for k, v in model.model.state_dict().items()}
    torch.save(state, path / WEIGHTS_FILE)
    (path / METADATA_FILE).write_text(json.dumps(metadata, indent=2))
    np.savez(path / BASELINES_FILE, **{attr: getattr(model, attr) for attr in _BASELINE_ATTRS})
    return path


def read_metadata(path: str | Path) -> dict[str, Any]:
    """Reads and validates a bundle's metadata without loading the weights."""
    metadata: dict[str, Any] = json.loads((Path(path) / METADATA_FILE).read_text())
    version = metadata.get("format_version")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported model bundle format_version={version!r} (expected {FORMAT_VERSION})")
    return metadata


def load_model(path: str | Path, device: str = "cpu") -> MSCVAE:
    """Rebuilds a model saved with save_model(), ready for inference on `device`."""
    path = Path(path)
    metadata = read_metadata(path)
    cfg = metadata["config"]
    features = metadata["features"]

    model = MSCVAE(
        n_features=cfg["n_features"],
        latent_dim=cfg["latent_dim"],
        window_sizes=cfg["window_sizes"],
        stride=cfg["stride"],
        seq_len=cfg["seq_len"],
        device=device,
        seed=cfg["seed"],
    )

    # Scaler state. null (non-finite at save time) comes back as NaN for mean/std; a missing
    # physical floor simply means "do not clip" for that variable.
    scaler = metadata["scaler"]
    generator = model.generator
    generator.mean = pd.Series(scaler["mean"], index=features, dtype=float)
    generator.std = pd.Series(scaler["std"], index=features, dtype=float)
    generator.min_physical_vals = {
        f: v for f, v in zip(features, scaler["min_physical"], strict=True) if v is not None
    }

    network = MSCVAE_Hybrid(
        n_features=cfg["n_features"], n_scales=len(model.window_sizes), latent_dim=cfg["latent_dim"]
    )
    state = torch.load(path / WEIGHTS_FILE, map_location="cpu", weights_only=True)
    network.load_state_dict(state)
    model.model = network.to(model.device).eval()

    model.threshold = metadata["threshold"]
    model.gain = metadata["gain"]
    model.rca_self_weight = metadata["rca"]["self_weight"]
    model.rca_rel_weight = metadata["rca"]["rel_weight"]

    with np.load(path / BASELINES_FILE, allow_pickle=False) as baselines:
        for attr in _BASELINE_ATTRS:
            setattr(model, attr, baselines[attr])

    return model
