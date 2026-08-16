"""Tests for the FastAPI foundation."""

import httpx
import pytest

from compliance_agent.config import Settings
from compliance_agent.main import REQUEST_ID_HEADER, create_app


def _client(settings: Settings, *, raise_app_exceptions: bool = True) -> httpx.AsyncClient:
    transport = httpx.ASGITransport(
        app=create_app(settings), raise_app_exceptions=raise_app_exceptions
    )
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_health_is_live_without_openai_configuration() -> None:
    settings = Settings(_env_file=None, environment="test", groq_api_key=None)

    async with _client(settings) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "AI Compliance Investigation Agent",
        "version": "0.1.0",
    }
    assert response.headers[REQUEST_ID_HEADER]


@pytest.mark.asyncio
async def test_readiness_fails_safely_without_required_configuration() -> None:
    settings = Settings(_env_file=None, environment="test", groq_api_key=None)

    async with _client(settings) as client:
        response = await client.get("/ready", headers={REQUEST_ID_HEADER: "test-request-123"})

    assert response.status_code == 503
    assert response.headers[REQUEST_ID_HEADER] == "test-request-123"
    assert response.json() == {
        "error": {
            "code": "http_503",
            "message": "Required service configuration is unavailable.",
            "request_id": "test-request-123",
            "details": None,
        }
    }


@pytest.mark.asyncio
async def test_readiness_reports_presence_without_exposing_secret() -> None:
    secret = "test-secret-that-must-not-be-returned"
    settings = Settings(_env_file=None, environment="test", groq_api_key=secret)

    async with _client(settings) as client:
        response = await client.get("/ready")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"ai_provider": "ok", "ai_api_key": "ok"},
    }
    assert secret not in response.text


@pytest.mark.asyncio
async def test_unsafe_request_id_is_replaced() -> None:
    settings = Settings(_env_file=None, environment="test")

    async with _client(settings) as client:
        response = await client.get("/health", headers={REQUEST_ID_HEADER: "contains spaces"})

    assert response.status_code == 200
    assert response.headers[REQUEST_ID_HEADER] != "contains spaces"


@pytest.mark.asyncio
async def test_versioned_api_entrypoint_exists() -> None:
    settings = Settings(_env_file=None, environment="test")

    async with _client(settings) as client:
        response = await client.get("/api/v1")

    assert response.status_code == 200
    assert response.json()["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_not_found_uses_stable_error_shape() -> None:
    settings = Settings(_env_file=None, environment="test")

    async with _client(settings) as client:
        response = await client.get("/does-not-exist")

    assert response.status_code == 404
    assert response.json()["error"]["code"] == "http_404"
    assert response.json()["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]


@pytest.mark.asyncio
async def test_unexpected_error_is_safe_and_traceable() -> None:
    settings = Settings(_env_file=None, environment="test")
    app = create_app(settings)

    @app.get("/test-only-error")
    async def test_only_error() -> None:
        raise RuntimeError("sensitive internal detail")

    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/test-only-error")

    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.json()["error"]["request_id"] == response.headers[REQUEST_ID_HEADER]
    assert "sensitive internal detail" not in response.text
