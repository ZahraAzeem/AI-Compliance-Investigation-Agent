"""Tests for safe Server-Sent Events investigation progress."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic import TypeAdapter

from compliance_agent.api.routes.investigations import (
    get_investigation_agent_service,
    investigation_event_generator,
)
from compliance_agent.config import Settings
from compliance_agent.domain import (
    ComplianceRecommendation,
    InvestigationCase,
    TransactionMonitoringCase,
)
from compliance_agent.main import create_app
from compliance_agent.services.agent import (
    InvestigationAgentService,
    InvestigationProgress,
    ToolPlanningResult,
)
from compliance_agent.services.recommendations import (
    ProviderRecommendation,
    RecommendationService,
    RecommendationTimeoutError,
    TokenUsage,
)
from compliance_agent.services.tools import SUMMARIZE_TRANSACTIONS, ToolCallRequest

SAMPLE_CASE_PATH = Path(__file__).parents[1] / "examples" / "sample_case.json"
CASE_ADAPTER = TypeAdapter(InvestigationCase)


@pytest.fixture
def case() -> TransactionMonitoringCase:
    payload = json.loads(SAMPLE_CASE_PATH.read_text(encoding="utf-8"))
    validated = CASE_ADAPTER.validate_python(payload)
    assert isinstance(validated, TransactionMonitoringCase)
    return validated


class FakePlanner:
    async def plan(self, _case: InvestigationCase) -> ToolPlanningResult:
        return ToolPlanningResult(
            calls=[
                ToolCallRequest(
                    call_id="call-summary",
                    name=SUMMARIZE_TRANSACTIONS,
                    arguments={"transaction_ids": ["TX-9001", "TX-9002"]},
                )
            ],
            response_id="plan_stream_test",
            model="test-planner",
            latency_ms=10,
            usage=TokenUsage(input_tokens=100, output_tokens=20, total_tokens=120),
        )


class FakeRecommendationProvider:
    async def recommend(
        self,
        _case: InvestigationCase,
        *,
        tool_outcomes: list | None = None,
    ) -> ProviderRecommendation:
        assert tool_outcomes and tool_outcomes[0].status == "completed"
        recommendation = ComplianceRecommendation.model_validate(
            {
                "case_id": "CASE-2026-001",
                "risk_level": "high",
                "recommended_action": "request_information",
                "confidence": 0.72,
                "rationale": "Exact transaction totals require human review.",
                "evidence": [
                    {
                        "source": "tool_result",
                        "source_id": "call-summary",
                        "summary": "The deterministic transaction total is USD 40,600.00.",
                        "supports_recommendation": True,
                    }
                ],
                "policy_citations": [],
                "uncertainties": ["The stated purpose is unverified."],
                "missing_information": ["Source-of-funds evidence."],
                "human_review_required": True,
            }
        )
        return ProviderRecommendation(
            recommendation=recommendation,
            response_id="recommend_stream_test",
            model="test-recommender",
            latency_ms=20,
            retry_count=0,
            usage=TokenUsage(input_tokens=200, output_tokens=80, total_tokens=280),
        )


class FailingService:
    async def run(self, *_args: object, **_kwargs: object) -> None:
        raise RecommendationTimeoutError


class CancellationAwareService:
    def __init__(self) -> None:
        self.cancelled = asyncio.Event()

    async def run(
        self,
        _case: InvestigationCase,
        *,
        on_progress,
    ) -> None:
        await on_progress(InvestigationProgress(stage="planning_started"))
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            raise


class ConnectedRequest:
    def __init__(self) -> None:
        self.state = SimpleNamespace(request_id="disconnect-test")

    async def is_disconnected(self) -> bool:
        return False


def _service() -> InvestigationAgentService:
    return InvestigationAgentService(
        FakePlanner(),
        RecommendationService(FakeRecommendationProvider(), max_output_retries=0),
        max_tool_calls=4,
    )


def _parse_sse(body: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    normalized = body.replace("\r\n", "\n").strip()
    for block in normalized.split("\n\n"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            if line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            fields[name] = value.lstrip()
        if "event" in fields:
            events.append(
                {
                    "id": fields["id"],
                    "event": fields["event"],
                    "data": json.loads(fields["data"]),
                }
            )
    return events


@pytest.mark.asyncio
async def test_stream_orders_safe_events_and_reuses_final_contract(
    case: TransactionMonitoringCase,
) -> None:
    service = _service()
    settings = Settings(_env_file=None, environment="test", groq_api_key="test-key")
    app = create_app(settings)
    app.dependency_overrides[get_investigation_agent_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        normal_response = await client.post(
            "/api/v1/investigations/run",
            json=case.model_dump(mode="json"),
        )
        stream_response = await client.post(
            "/api/v1/investigations/stream",
            headers={"X-Request-ID": "stream-order-test"},
            json=case.model_dump(mode="json"),
        )

    assert stream_response.status_code == 200
    assert stream_response.headers["content-type"].startswith("text/event-stream")
    assert stream_response.headers["cache-control"] == "no-store"
    assert stream_response.headers["x-accel-buffering"] == "no"

    events = _parse_sse(stream_response.text)
    assert [event["event"] for event in events] == [
        "started",
        "progress",
        "progress",
        "tool_call",
        "progress",
        "completed",
    ]
    assert [event["id"] for event in events] == ["1", "2", "3", "4", "5", "6"]
    assert [events[index]["data"]["stage"] for index in (1, 2, 4)] == [
        "planning_started",
        "planning_completed",
        "recommendation_started",
    ]
    tool_event = events[3]["data"]
    assert tool_event["tool_name"] == SUMMARIZE_TRANSACTIONS
    assert tool_event["status"] == "completed"
    assert "output" not in tool_event
    assert "arguments" not in tool_event
    assert events[-1]["data"]["result"] == normal_response.json()


@pytest.mark.asyncio
async def test_stream_converts_provider_failure_to_safe_terminal_event(
    case: TransactionMonitoringCase,
) -> None:
    settings = Settings(_env_file=None, environment="test", groq_api_key="test-key")
    app = create_app(settings)
    app.dependency_overrides[get_investigation_agent_service] = lambda: FailingService()
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/investigations/stream",
            json=case.model_dump(mode="json"),
        )

    events = _parse_sse(response.text)
    assert [event["event"] for event in events] == ["started", "failed"]
    assert events[-1]["data"]["code"] == "ai_timeout"
    assert events[-1]["data"]["retryable"] is True
    assert "credential" not in response.text.lower()


@pytest.mark.asyncio
async def test_closing_stream_cancels_background_investigation(
    case: TransactionMonitoringCase,
) -> None:
    service = CancellationAwareService()
    generator = investigation_event_generator(ConnectedRequest(), case, service)

    first_event = await anext(generator)
    assert first_event.event == "started"
    await generator.aclose()

    await asyncio.wait_for(service.cancelled.wait(), timeout=1)
