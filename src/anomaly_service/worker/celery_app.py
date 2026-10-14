"""Celery application (Redis as broker and result backend).

Delivery semantics are the main design choice here:

- task_acks_late + task_reject_on_worker_lost: a message is acknowledged only AFTER the
  task finishes, so a worker crash puts the job back in the queue (at-least-once delivery).
  Detection is a pure function of its input, so running it twice is harmless (idempotent).
- worker_prefetch_multiplier=1: a worker reserves one job at a time. Jobs are long and
  CPU-bound; prefetching would park jobs behind a busy worker while others sit idle.
- visibility_timeout > task_time_limit: with Redis, an unacknowledged message is redelivered
  after the visibility timeout. It must exceed the longest job, or long jobs run twice.
"""

from __future__ import annotations

from celery import Celery

from anomaly_service.config import Settings, get_settings


def create_celery(settings: Settings) -> Celery:
    app = Celery(
        "anomaly_service",
        broker=settings.broker_url,
        backend=settings.result_backend,
        include=["anomaly_service.worker.tasks"],
    )
    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_default_queue=settings.detection_queue,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
        task_track_started=True,
        task_time_limit=settings.job_time_limit_s,
        task_soft_time_limit=max(1, settings.job_time_limit_s - 30),
        result_expires=settings.job_result_ttl_s,
        broker_transport_options={"visibility_timeout": settings.job_time_limit_s * 2},
        broker_connection_retry_on_startup=True,
        worker_hijack_root_logger=False,
        # Eager mode runs tasks inline in the caller instead of a worker (tests).
        task_always_eager=settings.celery_task_always_eager,
        task_store_eager_result=True,
    )
    return app


celery_app = create_celery(get_settings())
