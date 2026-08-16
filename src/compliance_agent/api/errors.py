"""Stable public error contracts and FastAPI exception handlers."""

import logging
from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException

from compliance_agent.services.recommendations import (
    RecommendationAuthenticationError,
    RecommendationConfigurationError,
    RecommendationError,
    RecommendationInvalidOutputError,
    RecommendationQuotaError,
    RecommendationRateLimitError,
    RecommendationRefusalError,
    RecommendationTimeoutError,
)

logger = logging.getLogger(__name__)


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    details: list[dict[str, Any]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else "Request failed"
    body = ErrorResponse(
        error=ErrorDetail(
            code=f"http_{exc.status_code}",
            message=message,
            request_id=_request_id(request),
        )
    )
    return JSONResponse(status_code=exc.status_code, content=body.model_dump())


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "location": ".".join(str(part) for part in error["loc"]),
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    body = ErrorResponse(
        error=ErrorDetail(
            code="validation_error",
            message="The request did not match the expected format.",
            request_id=_request_id(request),
            details=details,
        )
    )
    return JSONResponse(status_code=422, content=body.model_dump())


async def unexpected_exception_handler(request: Request, _exc: Exception) -> JSONResponse:
    body = ErrorResponse(
        error=ErrorDetail(
            code="internal_error",
            message="The service encountered an unexpected error.",
            request_id=_request_id(request),
        )
    )
    return JSONResponse(status_code=500, content=body.model_dump())


async def recommendation_exception_handler(
    request: Request,
    exc: RecommendationError,
) -> JSONResponse:
    """Map expected provider failures without exposing SDK or credential details."""

    if isinstance(exc, RecommendationInvalidOutputError):
        logger.warning(
            "AI recommendation output rejected: reason=%s request_id=%s",
            exc.reason,
            _request_id(request),
        )
    status_code, code, message = _recommendation_error_details(exc)
    body = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=_request_id(request),
        )
    )
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _recommendation_error_details(exc: RecommendationError) -> tuple[int, str, str]:
    if isinstance(exc, RecommendationQuotaError):
        return 429, "ai_quota_exhausted", "The AI project has no available API quota."
    if isinstance(exc, RecommendationRateLimitError):
        return 429, "ai_rate_limited", "The AI service is temporarily rate limited."
    if isinstance(exc, RecommendationTimeoutError):
        return 504, "ai_timeout", "The AI service did not respond before the timeout."
    if isinstance(exc, RecommendationRefusalError):
        return 422, "ai_refusal", "The AI service declined to analyze this case."
    if isinstance(exc, RecommendationInvalidOutputError):
        return 502, "ai_invalid_output", "The AI service returned an unusable response."
    if isinstance(exc, RecommendationAuthenticationError):
        return 503, "ai_authentication_failed", "The AI service is not configured correctly."
    if isinstance(exc, RecommendationConfigurationError):
        return 503, "ai_not_configured", "The AI service is not configured."
    return 503, "ai_unavailable", "The AI service is temporarily unavailable."
