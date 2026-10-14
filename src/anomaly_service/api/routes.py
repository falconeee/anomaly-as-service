"""HTTP routes: detection (sync), jobs (async), model info, admin and health."""

from __future__ import annotations

from typing import Annotated
from uuid import uuid4

from celery.result import AsyncResult
from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response, status
from fastapi.concurrency import run_in_threadpool

from anomaly_service.api.deps import (
    JobStoreDep,
    ManagerDep,
    ServiceDep,
    SettingsDep,
    require_api_key,
)
from anomaly_service.inference.service import ModelUnavailableError
from anomaly_service.schemas import (
    DetectionRequest,
    DetectionResponse,
    JobAccepted,
    JobRequest,
    JobState,
    JobStatus,
    ModelInfo,
    ModelRef,
)
from anomaly_service.worker.celery_app import celery_app
from anomaly_service.worker.tasks import detect_task

detection = APIRouter(prefix="/v1", tags=["detection"], dependencies=[Depends(require_api_key)])
jobs = APIRouter(prefix="/v1/jobs", tags=["jobs"], dependencies=[Depends(require_api_key)])
models = APIRouter(prefix="/v1/models", tags=["models"], dependencies=[Depends(require_api_key)])
admin = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_api_key)])
health = APIRouter(prefix="/health", tags=["health"])

# Celery state -> public job status
_STATUS: dict[str, JobStatus] = {
    "PENDING": "queued",
    "RECEIVED": "queued",
    "STARTED": "running",
    "RETRY": "running",
    "SUCCESS": "succeeded",
    "FAILURE": "failed",
    "REVOKED": "failed",
}


@detection.post(
    "/detect",
    response_model=DetectionResponse,
    response_model_exclude_none=True,
    summary="Detect and explain anomalies (synchronous)",
    responses={413: {"description": "Too many points: use POST /v1/jobs"}, 422: {"description": "Bad input"}},
)
async def detect(request: DetectionRequest, service: ServiceDep, settings: SettingsDep) -> DetectionResponse:
    if request.n_points > settings.sync_max_points:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"{request.n_points} points exceed the synchronous limit of {settings.sync_max_points}; "
            "submit it as a job with POST /v1/jobs",
        )
    # Inference is CPU-bound: run it in the thread pool so the event loop keeps serving.
    return await run_in_threadpool(service.detect, request, "sync")


@jobs.post(
    "",
    response_model=JobAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit a detection job (asynchronous, via the queue)",
    responses={200: {"description": "Idempotency-Key already used: the existing job is returned"}},
)
def submit_job(
    job: JobRequest,
    request: Request,
    response: Response,
    store: JobStoreDep,
    settings: SettingsDep,
    manager: ManagerDep,
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key", max_length=200)] = None,
) -> JobAccepted:
    if job.n_points > settings.async_max_points:
        raise HTTPException(
            status.HTTP_413_CONTENT_TOO_LARGE,
            f"{job.n_points} points exceed the job limit of {settings.async_max_points}",
        )
    # Fail fast on a schema mismatch when the API knows the model (saves a queue round trip).
    if manager.service is not None:
        manager.service.model.generator.align_columns(job.to_frame().iloc[:1])

    job_id = uuid4().hex
    if idempotency_key:
        existing = store.claim(idempotency_key, job_id)
        if existing is not None:
            response.status_code = status.HTTP_200_OK
            return _accepted(existing, request, response, store)

    store.create(job_id, job.n_points)
    detect_task.apply_async(args=[job.model_dump(mode="json")], task_id=job_id)
    return _accepted(job_id, request, response, store)


def _accepted(job_id: str, request: Request, response: Response, store) -> JobAccepted:
    url = str(request.url_for("get_job", job_id=job_id).path)
    response.headers["Location"] = url
    state = AsyncResult(job_id, app=celery_app).state
    return JobAccepted(job_id=job_id, status=_STATUS.get(state, "queued"), status_url=url)


@jobs.get("/{job_id}", response_model=JobState, response_model_exclude_none=True, summary="Job status/result")
def get_job(job_id: str, store: JobStoreDep) -> JobState:
    meta = store.get(job_id)
    if meta is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Job {job_id} not found (unknown or expired)")
    result = AsyncResult(job_id, app=celery_app)
    job_status = _STATUS.get(result.state, "queued")
    state = JobState(job_id=job_id, status=job_status, submitted_at=meta.get("submitted_at"))
    if job_status == "succeeded":
        state.result = DetectionResponse.model_validate(result.result)
    elif job_status == "failed":
        state.error = f"{type(result.result).__name__}: {result.result}"
    return state


@models.get("/current", response_model=ModelInfo, summary="Model being served")
def current_model(service: ServiceDep) -> ModelInfo:
    return service.info()


@admin.post("/reload", summary="Reload the model now (instead of waiting for the next refresh)")
async def reload_model(manager: ManagerDep) -> dict[str, object]:
    try:
        service = await run_in_threadpool(manager.load)
    except ModelUnavailableError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    ref: ModelRef = service.ref
    return {"reloaded": True, "model": ref.model_dump()}


@health.get("/live", summary="Liveness: the process is up")
def live() -> dict[str, str]:
    return {"status": "ok"}


@health.get("/ready", summary="Readiness: a model is loaded (queue status is informative)")
def ready(request: Request, response: Response, manager: ManagerDep) -> dict[str, object]:
    try:
        queue_ok = bool(request.app.state.redis.ping())
    except Exception:
        queue_ok = False
    model_ok = manager.service is not None
    if not model_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    version = manager.service.ref.version if manager.service else None
    return {
        "status": "ok" if model_ok else "unavailable",
        "model": model_ok,
        "model_version": version,
        "queue": queue_ok,
    }
