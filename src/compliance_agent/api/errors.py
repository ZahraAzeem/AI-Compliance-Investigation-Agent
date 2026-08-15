"""Stable public error contracts and FastAPI exception handlers."""

from typing import Any

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException


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
