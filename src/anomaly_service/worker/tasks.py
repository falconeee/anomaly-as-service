"""Celery tasks: asynchronous detection and webhook delivery.

The worker runs with the `solo` pool (one task at a time, in the main process) and is scaled
by running more containers. That avoids forking a process that already initialized PyTorch
(a classic source of deadlocks) and keeps one model copy and one metrics endpoint per container.
"""

from __future__ import annotations

from typing import Any

import httpx
from celery.signals import task_prerun, worker_ready
from prometheus_client import start_http_server

from anomaly_service.config import get_settings
from anomaly_service.inference.service import ModelManager, ModelUnavailableError
from anomaly_service.logger import configure_logging, job_id_var, logger
from anomaly_service.model import FeatureSchemaError
from anomaly_service.schemas import JobRequest
from anomaly_service.worker.celery_app import celery_app

settings = get_settings()
# One model per worker process, loaded once and hot-swapped when @alias moves.
manager = ModelManager(settings)


@worker_ready.connect
def _on_worker_ready(**_: Any) -> None:
    configure_logging(settings.log_level, settings.log_json)
    try:
        start_http_server(settings.worker_metrics_port)
    except OSError as exc:
        logger.warning("Worker metrics endpoint not started: %s", exc)
    try:
        manager.load()  # warm start: the first job does not pay the model load
    except ModelUnavailableError as exc:
        logger.warning("No model available yet; jobs will retry until one is: %s", exc)


@task_prerun.connect
def _bind_job_id(task_id: str | None = None, **_: Any) -> None:
    job_id_var.set(task_id)


@celery_app.task(
    bind=True,
    name="anomaly.detect",
    autoretry_for=(ModelUnavailableError,),  # transient: registry/model not reachable yet
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=5,
)
def detect_task(self, payload: dict[str, Any]) -> dict[str, Any]:
    request = JobRequest.model_validate(payload)
    service = manager.get()
    try:
        response = service.detect(request, mode="async")
    except FeatureSchemaError as exc:
        # Permanent error: do not retry. Re-raised as a plain ValueError so the result
        # backend can serialize it (the custom exception has a non-standard constructor).
        raise ValueError(f"{exc}; extra columns: {exc.extra}") from None

    result = response.model_dump(mode="json")
    logger.info("Job done: %d points, %d anomalies", response.n_points, len(response.anomalies))
    if request.callback_url is not None:
        deliver_webhook.delay(
            str(request.callback_url), {"job_id": self.request.id, "status": "succeeded", "result": result}
        )
    return result


@celery_app.task(
    name="anomaly.webhook",
    autoretry_for=(httpx.HTTPError,),
    retry_backoff=True,
    retry_backoff_max=300,
    max_retries=6,
)
def deliver_webhook(url: str, payload: dict[str, Any]) -> int:
    """Separate task so a flaky receiver is retried without re-running the detection."""
    response = httpx.post(url, json=payload, timeout=10)
    response.raise_for_status()
    return response.status_code
