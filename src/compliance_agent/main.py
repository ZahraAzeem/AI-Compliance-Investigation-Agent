"""FastAPI application factory."""

import re
from collections.abc import Awaitable, Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException

from compliance_agent.api.errors import (
    http_exception_handler,
    unexpected_exception_handler,
    validation_exception_handler,
)
from compliance_agent.api.routes.system import api_router, system_router
from compliance_agent.config import Settings, get_settings

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application instance with explicit, testable configuration."""

    resolved_settings = settings or get_settings()
    app = FastAPI(
        title=resolved_settings.app_name,
        version=resolved_settings.app_version,
    )
    app.dependency_overrides[get_settings] = lambda: resolved_settings

    @app.middleware("http")
    async def add_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        supplied_request_id = request.headers.get(REQUEST_ID_HEADER, "")
        request_id = (
            supplied_request_id if _SAFE_REQUEST_ID.fullmatch(supplied_request_id) else str(uuid4())
        )
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001 - HTTP boundary converts unknown failures safely.
            response = await unexpected_exception_handler(request, exc)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, unexpected_exception_handler)

    app.include_router(system_router)
    app.include_router(api_router)
    return app


app = create_app()
