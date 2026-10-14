"""FastAPI dependencies: settings, model, job store and API-key authentication."""

from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader

from anomaly_service.config import Settings
from anomaly_service.inference.service import InferenceService, ModelManager
from anomaly_service.jobs import JobStore

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False, description="Service API key")


def get_settings_dep(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


def get_manager(request: Request) -> ModelManager:
    manager: ModelManager = request.app.state.manager
    return manager


def get_job_store(request: Request) -> JobStore:
    store: JobStore = request.app.state.jobs
    return store


def require_api_key(
    settings: Annotated[Settings, Depends(get_settings_dep)],
    api_key: Annotated[str | None, Security(_api_key_header)] = None,
) -> None:
    expected = settings.api_key
    if expected is None:  # auth disabled (local development)
        return
    # compare_digest: constant-time comparison, so the key cannot be guessed by timing.
    if api_key is None or not secrets.compare_digest(api_key, expected.get_secret_value()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key", headers={"WWW-Authenticate": "ApiKey"}
        )


def get_service(manager: Annotated[ModelManager, Depends(get_manager)]) -> InferenceService:
    service = manager.service
    if service is None:
        detail = "No model loaded. Train and promote a model to @champion, then call POST /admin/reload."
        if manager.last_error:
            detail += f" Last error: {manager.last_error}"
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail)
    return service


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ManagerDep = Annotated[ModelManager, Depends(get_manager)]
ServiceDep = Annotated[InferenceService, Depends(get_service)]
JobStoreDep = Annotated[JobStore, Depends(get_job_store)]
