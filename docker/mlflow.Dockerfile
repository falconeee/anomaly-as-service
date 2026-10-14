# MLflow tracking server + model registry.
#   metadata  -> PostgreSQL   (--backend-store-uri)
#   artifacts -> /mlartifacts (--artifacts-destination; swap for s3://... in the cloud)
# With --serve-artifacts the server proxies artifact uploads/downloads, so clients only need
# MLFLOW_TRACKING_URI: no storage credentials on the API, worker or trainer.
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /bin/

# Same MLflow version as uv.lock (clients and server should match).
ARG MLFLOW_VERSION=3.17.1
# psycopg (v3): SQLAlchemy 2.1 maps postgresql:// to it by default (psycopg2 is legacy).
RUN uv pip install --system --no-cache "mlflow==${MLFLOW_VERSION}" "psycopg[binary]" boto3

RUN useradd --uid 10001 --create-home mlflow \
    && mkdir -p /mlartifacts \
    && chown mlflow /mlartifacts
USER mlflow

ENV MLFLOW_DISABLE_AGENT_HINT=1 \
    MLFLOW_ARTIFACTS_DESTINATION=/mlartifacts \
    MLFLOW_ALLOWED_HOSTS=mlflow:5000,localhost:5000,127.0.0.1:5000 \
    MLFLOW_WORKERS=2
EXPOSE 5000

# --allowed-hosts: MLflow rejects unknown Host headers (DNS-rebinding protection); the
# compose service name must be listed for other containers to reach it.
CMD ["sh", "-c", "exec mlflow server --host 0.0.0.0 --port 5000 --workers \"$MLFLOW_WORKERS\" --backend-store-uri \"$MLFLOW_BACKEND_STORE_URI\" --artifacts-destination \"$MLFLOW_ARTIFACTS_DESTINATION\" --serve-artifacts --allowed-hosts \"$MLFLOW_ALLOWED_HOSTS\""]
