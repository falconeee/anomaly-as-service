"""Service settings, read from environment variables (and an optional .env file).

Every field maps to an upper-case env var: `model_name` <- MODEL_NAME, `api_key` <- API_KEY...
MLflow itself reads MLFLOW_TRACKING_URI, so it is not duplicated here.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Model ------------------------------------------------------------------------
    model_name: str = "mscvae-synthetic"
    model_alias: str = "champion"
    # Serve a local bundle directory instead of the registry (development / tests).
    model_path: str | None = None
    # How often to check whether @alias moved to a new version (0 disables hot reload).
    model_refresh_interval_s: float = 60.0
    model_device: str = "cpu"

    # --- API --------------------------------------------------------------------------
    # When unset, authentication is disabled (local development only).
    api_key: SecretStr | None = None
    sync_max_points: int = 20_000
    async_max_points: int = 500_000

    # --- Queue (Celery + Redis) -------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None  # defaults to redis_url
    celery_result_backend: str | None = None  # defaults to redis_url
    # Run tasks inline in the caller instead of a worker (tests). Job status still needs
    # a result backend and Redis for the job registry.
    celery_task_always_eager: bool = False
    detection_queue: str = "detections"
    job_result_ttl_s: int = 86_400
    job_time_limit_s: int = 600
    worker_metrics_port: int = 9100

    # --- Observability ----------------------------------------------------------------
    log_level: str = "INFO"
    log_json: bool = False

    @property
    def broker_url(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def result_backend(self) -> str:
        return self.celery_result_backend or self.redis_url


@lru_cache
def get_settings() -> Settings:
    return Settings()
