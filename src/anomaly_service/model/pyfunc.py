"""MLflow pyfunc wrapper around the MSCVAE bundle.

The model is logged with MLflow's "models from code" mode: instead of cloudpickling a
PythonModel instance, MLflow stores the small script `pyfunc_model.py` and runs it at load
time. Together with the pickle-free bundle (ADR 0002), loading a registered model never
unpickles anything.

Any MLflow consumer (``mlflow.pyfunc.load_model``, ``mlflow models serve``, batch scoring)
gets per-window scores. The service unwraps the model (``unwrap_python_model().model``) to
reach the full MSCVAE API, root-cause analysis included.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
from mlflow.pyfunc.model import PythonModel

from anomaly_service.model.mscvae import MSCVAE
from anomaly_service.model.persistence import load_model

BUNDLE_ARTIFACT = "bundle"


class MSCVAEPyfunc(PythonModel):
    model: MSCVAE | None = None

    def load_context(self, context: Any) -> None:
        self.model = load_model(context.artifacts[BUNDLE_ARTIFACT], device=os.getenv("MODEL_DEVICE", "cpu"))

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict[str, Any] | None = None):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return predict_frame(self.model, model_input)


def predict_frame(model: MSCVAE, df: pd.DataFrame) -> pd.DataFrame:
    """One row per generated window: input row it ends at, score and anomaly flag."""
    scores = model.score_windows(df)
    return pd.DataFrame(
        {
            "row": scores.row_index.astype("int64"),
            "score": scores.phi.astype("float64"),
            "is_anomaly": scores.phi > model.threshold,
        }
    )
