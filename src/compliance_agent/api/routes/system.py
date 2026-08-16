"""Liveness, readiness, and API metadata endpoints."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from compliance_agent.api.routes.alerts import router as alerts_router
from compliance_agent.api.routes.cases import router as cases_router
from compliance_agent.config import Settings, get_settings


class HealthResponse(BaseModel):
    status: Literal["ok"]
    service: str
    version: str


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    checks: dict[str, Literal["ok"]]


system_router = APIRouter(tags=["system"])
api_router = APIRouter(prefix="/api/v1", tags=["api"])
api_router.include_router(alerts_router)
api_router.include_router(cases_router)
SettingsDependency = Annotated[Settings, Depends(get_settings)]


@system_router.get("/health", response_model=HealthResponse)
async def health(settings: SettingsDependency) -> HealthResponse:
    """Return process liveness without checking external dependencies."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )


@system_router.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Not ready"}},
)
async def ready(settings: SettingsDependency) -> ReadinessResponse:
    """Return whether required service configuration is available."""

    if not settings.has_openai_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Required service configuration is unavailable.",
        )

    return ReadinessResponse(status="ready", checks={"openai_api_key": "ok"})


@api_router.get("", response_model=HealthResponse)
async def api_info(settings: SettingsDependency) -> HealthResponse:
    """Describe the versioned API entrypoint."""

    return HealthResponse(
        status="ok",
        service=settings.app_name,
        version=settings.app_version,
    )
