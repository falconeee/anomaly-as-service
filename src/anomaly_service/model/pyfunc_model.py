"""Entry point for MLflow "models from code" (see model/pyfunc.py).

MLflow copies this file into the logged model and executes it when the model is loaded.
"""

from mlflow.models import set_model

from anomaly_service.model.pyfunc import MSCVAEPyfunc

set_model(MSCVAEPyfunc())
