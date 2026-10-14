"""FastAPI application factory.

Run with:  uvicorn anomaly_service.api.main:create_app --factory --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from prometheus_client import REGISTRY
from prometheus_fastapi_instrumentator import Instrumentator
from redis import Redis

from anomaly_service import __version__
from anomaly_service.api import routes
from anomaly_service.config import Settings, get_settings
from anomaly_service.inference.service import ModelManager, ModelUnavailableError
from anomaly_service.jobs import JobStore
from anomaly_service.logger import configure_logging, logger, request_id_var
from anomaly_service.model import FeatureSchemaError
from anomaly_service.observability import QueueDepthCollector

DESCRIPTION = """
Detects anomalies in **multivariate time series** and explains them: when, how severe and
which variables caused it (magnitude vs. broken correlation), with observed vs. expected values.

- `POST /v1/detect`: synchronous, for series up to `SYNC_MAX_POINTS` points.
- `POST /v1/jobs` + `GET /v1/jobs/{id}`: asynchronous via a queue, for large series.

The model (MSCVAE) is served from the MLflow Model Registry alias `@champion` and is
hot-reloaded when the alias moves.
"""

_queue_collector: QueueDepthCollector | None = None


async def _refresh_forever(manager: ModelManager, interval_s: float) -> None:
    """Polls the registry and hot-swaps the model when @alias points to a new version."""
    while True:
        await asyncio.sleep(interval_s)
        try:
            if manager.service is None:
                await run_in_threadpool(manager.load)
            else:
                await run_in_threadpool(manager.refresh)
        except Exception as exc:
            logger.warning("Model refresh failed: %s", exc)


def create_app(settings: Settings | None = None, redis_client: Redis | None = None) -> FastAPI:
    settings = settings or get_settings()
    redis = redis_client or Redis.from_url(settings.redis_url, decode_responses=True)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        manager = ModelManager(settings)
        app.state.manager = manager
        try:
            await run_in_threadpool(manager.load)
        except ModelUnavailableError as exc:
            # Start anyway (readiness reports 503) so the platform does not crash-loop while
            # no model has been promoted yet; the refresh loop picks it up later.
            logger.warning("Starting without a model: %s", exc)
        refresher = None
        if settings.model_refresh_interval_s > 0 and not settings.model_path:
            refresher = asyncio.create_task(_refresh_forever(manager, settings.model_refresh_interval_s))
        yield
        if refresher is not None:
            refresher.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await refresher

    app = FastAPI(
        title="Anomaly-as-a-Service",
        version=__version__,
        description=DESCRIPTION,
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.redis = redis
    app.state.jobs = JobStore(redis, settings.job_result_ttl_s)

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        # Propagate the caller's X-Request-ID (or create one) into every log line.
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_var.set(rid)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(FeatureSchemaError)
    async def schema_error(_: Request, exc: FeatureSchemaError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={"detail": {"message": str(exc), "missing": exc.missing, "extra": exc.extra}},
        )

    for router in (routes.detection, routes.jobs, routes.models, routes.admin, routes.health):
        app.include_router(router)

    Instrumentator(excluded_handlers=["/metrics", "/health/.*"]).instrument(app).expose(
        app, endpoint="/metrics", include_in_schema=False
    )
    global _queue_collector
    if _queue_collector is None:
        _queue_collector = QueueDepthCollector(redis, settings.detection_queue)
        REGISTRY.register(_queue_collector)
    return app
