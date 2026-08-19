"""Tests for the bounded model → tools → recommendation workflow."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import TypeAdapter

from compliance_agent.api.routes.investigations import get_investigation_agent_service
from compliance_agent.config import Settings
from compliance_agent.domain import (
    ComplianceRecommendation,
    InvestigationCase,
    TransactionMonitoringCase,
)
from compliance_agent.main import create_app
from compliance_agent.providers.tool_planner import CompatibleToolPlanner
from compliance_agent.services.agent import (
    InvestigationAgentService,
    ToolPlanningResult,
)
from compliance_agent.services.recommendations import (
    ProviderRecommendation,
    RecommendationInvalidOutputError,
    RecommendationService,
    RecommendationTimeoutError,
    TokenUsage,
)
from compliance_agent.services.tools import (
    LOOKUP_CUSTOMER_RISK,
    SUMMARIZE_TRANSACTIONS,
    ToolCallOutcome,
    ToolCallRequest,
)

SAMPLE_CASE_PATH = Path(__file__).parents[1] / "examples" / "sample_case.json"
CASE_ADAPTER = TypeAdapter(InvestigationCase)


@pytest.fixture
def case() -> TransactionMonitoringCase:
    payload = json.loads(SAMPLE_CASE_PATH.read_text(encoding="utf-8"))
    validated = CASE_ADAPTER.validate_python(payload)
    assert isinstance(validated, TransactionMonitoringCase)
    return validated


def _recommendation(tool_source_id: str = "call-summary") -> ComplianceRecommendation:
    return ComplianceRecommendation.model_validate(
        {
            "case_id": "CASE-2026-001",
            "risk_level": "high",
            "recommended_action": "request_information",
            "confidence": 0.72,
            "rationale": "Deterministic totals and missing documents require human review.",
            "evidence": [
                {
                    "source": "tool_result",
                    "source_id": tool_source_id,
                    "summary": "The deterministic transaction total is USD 40,600.00.",
                    "supports_recommendation": True,
                }
            ],
            "policy_citations": [],
            "uncertainties": ["The stated business purpose remains unverified."],
            "missing_information": ["Invoices and source-of-funds evidence."],
            "human_review_required": True,
        }
    )


def _planning(calls: list[ToolCallRequest]) -> ToolPlanningResult:
    return ToolPlanningResult(
        calls=calls,
        response_id="plan_test_123",
        model="test-planner",
        latency_ms=20,
        usage=TokenUsage(input_tokens=200, output_tokens=50, total_tokens=250),
    )


class FakePlanner:
    def __init__(self, planning: ToolPlanningResult) -> None:
        self.planning = planning

    async def plan(self, _case: InvestigationCase) -> ToolPlanningResult:
        return self.planning


class SlowPlanner:
    async def plan(self, _case: InvestigationCase) -> ToolPlanningResult:
        await asyncio.sleep(0.05)
        return _planning([])


class ToolAwareRecommendationProvider:
    def __init__(self, recommendation: ComplianceRecommendation) -> None:
        self.recommendation = recommendation
        self.received_outcomes: list[ToolCallOutcome] | None = None

    async def recommend(
        self,
        _case: InvestigationCase,
        *,
        tool_outcomes: list[ToolCallOutcome] | None = None,
    ) -> ProviderRecommendation:
        self.received_outcomes = tool_outcomes
        return ProviderRecommendation(
            recommendation=self.recommendation,
            response_id="recommend_test_123",
            model="test-recommender",
            latency_ms=30,
            retry_count=0,
            usage=TokenUsage(input_tokens=300, output_tokens=100, total_tokens=400),
        )


def _agent(
    calls: list[ToolCallRequest],
    recommendation: ComplianceRecommendation,
    *,
    max_tool_calls: int = 4,
    max_elapsed_seconds: float = 60.0,
) -> tuple[InvestigationAgentService, ToolAwareRecommendationProvider]:
    provider = ToolAwareRecommendationProvider(recommendation)
    service = InvestigationAgentService(
        FakePlanner(_planning(calls)),
        RecommendationService(provider, max_output_retries=0),
        max_tool_calls=max_tool_calls,
        max_elapsed_seconds=max_elapsed_seconds,
    )
    return service, provider


@pytest.mark.asyncio
async def test_agent_executes_tools_before_grounded_recommendation(
    case: TransactionMonitoringCase,
) -> None:
    calls = [
        ToolCallRequest(
            call_id="call-risk",
            name=LOOKUP_CUSTOMER_RISK,
            arguments={"customer_id": "CUST-1042"},
        ),
        ToolCallRequest(
            call_id="call-summary",
            name=SUMMARIZE_TRANSACTIONS,
            arguments={"transaction_ids": ["TX-9001", "TX-9002"]},
        ),
    ]
    service, provider = _agent(calls, _recommendation())

    result = await service.run(case)

    assert [outcome.status for outcome in result.tool_outcomes] == [
        "completed",
        "completed",
    ]
    assert result.tool_outcomes[1].output["totals_by_currency"] == {"USD": "40600.00"}
    assert provider.received_outcomes == result.tool_outcomes
    assert result.recommendation.recommendation.evidence[0].source == "tool_result"


@pytest.mark.asyncio
async def test_agent_rejects_recommendation_citing_unexecuted_tool(
    case: TransactionMonitoringCase,
) -> None:
    service, _provider = _agent([], _recommendation("call-never-executed"))

    with pytest.raises(RecommendationInvalidOutputError) as error:
        await service.run(case)

    assert error.value.reason == "unknown_tool_evidence_id"


@pytest.mark.asyncio
async def test_agent_rejects_model_tool_calls_above_limit(
    case: TransactionMonitoringCase,
) -> None:
    calls = [
        ToolCallRequest(
            call_id=f"call-{index}",
            name=LOOKUP_CUSTOMER_RISK,
            arguments={"customer_id": "CUST-1042"},
        )
        for index in range(3)
    ]
    service, _provider = _agent(calls, _recommendation(), max_tool_calls=2)

    with pytest.raises(RecommendationInvalidOutputError) as error:
        await service.run(case)

    assert error.value.reason == "tool_call_limit_exceeded"


@pytest.mark.asyncio
async def test_agent_marks_result_partial_when_a_tool_call_is_rejected(
    case: TransactionMonitoringCase,
) -> None:
    calls = [
        ToolCallRequest(
            call_id="call-summary",
            name=SUMMARIZE_TRANSACTIONS,
            arguments={"transaction_ids": []},
        ),
        ToolCallRequest(
            call_id="call-not-allowed",
            name="delete_customer",
            arguments={"customer_id": "CUST-1042"},
        ),
    ]
    service, _provider = _agent(calls, _recommendation())

    result = await service.run(case)

    assert result.status == "partial"
    assert result.tool_outcomes[0].status == "completed"
    assert result.tool_outcomes[1].status == "rejected"
    assert result.tool_outcomes[1].error_code == "unknown_tool"


@pytest.mark.asyncio
async def test_agent_enforces_total_elapsed_time_limit(
    case: TransactionMonitoringCase,
) -> None:
    provider = ToolAwareRecommendationProvider(_recommendation())
    service = InvestigationAgentService(
        SlowPlanner(),
        RecommendationService(provider, max_output_retries=0),
        max_tool_calls=4,
        max_elapsed_seconds=0.001,
    )

    with pytest.raises(RecommendationTimeoutError):
        await service.run(case)


@pytest.mark.asyncio
async def test_compatible_planner_parses_only_structured_tool_calls(
    case: TransactionMonitoringCase,
) -> None:
    tool_call = SimpleNamespace(
        id="call-summary",
        function=SimpleNamespace(
            name=SUMMARIZE_TRANSACTIONS,
            arguments='{"transaction_ids":["TX-9001","TX-9002"]}',
        ),
    )
    create = AsyncMock(
        return_value=SimpleNamespace(
            id="chatcmpl-plan",
            model="openai/gpt-oss-20b",
            choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[tool_call]))],
            usage=SimpleNamespace(prompt_tokens=200, completion_tokens=50, total_tokens=250),
        )
    )
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    settings = Settings(_env_file=None, environment="test", groq_api_key="test-key")
    planner = CompatibleToolPlanner(settings, client=client)

    result = await planner.plan(case)

    assert result.calls == [
        ToolCallRequest(
            call_id="call-summary",
            name=SUMMARIZE_TRANSACTIONS,
            arguments={"transaction_ids": ["TX-9001", "TX-9002"]},
        )
    ]
    kwargs = create.await_args.kwargs
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is False
    assert kwargs["store"] is False
    assert [tool["function"]["name"] for tool in kwargs["tools"]] == [
        LOOKUP_CUSTOMER_RISK,
        SUMMARIZE_TRANSACTIONS,
    ]


@pytest.mark.asyncio
async def test_investigation_endpoint_returns_auditable_workflow(
    case: TransactionMonitoringCase,
) -> None:
    calls = [
        ToolCallRequest(
            call_id="call-summary",
            name=SUMMARIZE_TRANSACTIONS,
            arguments={"transaction_ids": []},
        )
    ]
    service, _provider = _agent(calls, _recommendation())
    settings = Settings(_env_file=None, environment="test", groq_api_key="test-key")
    app = create_app(settings)
    app.dependency_overrides[get_investigation_agent_service] = lambda: service
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/api/v1/investigations/run",
            json=case.model_dump(mode="json"),
        )

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["planning"]["requested_tool_count"] == 1
    assert body["tool_outcomes"][0]["name"] == SUMMARIZE_TRANSACTIONS
    assert body["recommendation"]["recommendation"]["human_review_required"] is True
    assert body["human_review_required"] is True
