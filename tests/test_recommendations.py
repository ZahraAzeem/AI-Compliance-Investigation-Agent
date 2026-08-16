"""Tests for structured recommendation orchestration and its HTTP boundary."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from openai import APITimeoutError
from pydantic import TypeAdapter

from compliance_agent.api.routes.recommendations import get_recommendation_service
from compliance_agent.config import Settings
from compliance_agent.domain import (
    ComplianceRecommendation,
    InvestigationCase,
    TransactionMonitoringCase,
)
from compliance_agent.main import create_app
from compliance_agent.providers.openai_recommendations import (
    SYSTEM_INSTRUCTIONS,
    GroqRecommendationProvider,
    OpenAIRecommendationProvider,
)
from compliance_agent.services.recommendations import (
    ProviderRecommendation,
    RecommendationInvalidOutputError,
    RecommendationQuotaError,
    RecommendationRateLimitError,
    RecommendationRefusalError,
    RecommendationService,
    RecommendationTimeoutError,
    RecommendationUnavailableError,
    TokenUsage,
)

SAMPLE_CASE_PATH = Path(__file__).parents[1] / "examples" / "sample_case.json"
CASE_ADAPTER = TypeAdapter(InvestigationCase)


@pytest.fixture
def case() -> TransactionMonitoringCase:
    payload = json.loads(SAMPLE_CASE_PATH.read_text(encoding="utf-8"))
    validated = CASE_ADAPTER.validate_python(payload)
    assert isinstance(validated, TransactionMonitoringCase)
    return validated


@pytest.fixture
def recommendation() -> ComplianceRecommendation:
    return ComplianceRecommendation.model_validate(
        {
            "case_id": "CASE-2026-001",
            "risk_level": "high",
            "recommended_action": "request_information",
            "confidence": 0.78,
            "rationale": (
                "The supplied transactions exceed the stated baseline, while the business "
                "purpose remains unverified."
            ),
            "evidence": [
                {
                    "source": "case_input",
                    "source_id": "ALT-DEMO-001",
                    "summary": "The synthetic monitoring alert identifies unusual activity.",
                    "supports_recommendation": True,
                },
                {
                    "source": "case_input",
                    "source_id": "TX-9001",
                    "summary": "The first supplied outbound wire contributes to the pattern.",
                    "supports_recommendation": True,
                },
            ],
            "policy_citations": [],
            "uncertainties": ["The commercial relationship is not independently verified."],
            "missing_information": ["Supporting invoices and source-of-funds evidence."],
            "human_review_required": True,
        }
    )


def _provider_result(
    recommendation: ComplianceRecommendation,
) -> ProviderRecommendation:
    return ProviderRecommendation(
        recommendation=recommendation,
        response_id="resp_test_123",
        model="test-model",
        latency_ms=42,
        retry_count=0,
        usage=TokenUsage(input_tokens=300, output_tokens=120, total_tokens=420),
    )


class FakeProvider:
    def __init__(self, result: ProviderRecommendation) -> None:
        self.result = result

    async def recommend(self, _case: InvestigationCase) -> ProviderRecommendation:
        return self.result


class SequenceProvider:
    def __init__(self, results: list[ProviderRecommendation]) -> None:
        self.results = results
        self.call_count = 0

    async def recommend(self, _case: InvestigationCase) -> ProviderRecommendation:
        result = self.results[self.call_count]
        self.call_count += 1
        return result


class FailingService:
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def recommend(self, _case: InvestigationCase) -> ProviderRecommendation:
        raise self.error


def _client(
    service: RecommendationService | FailingService,
) -> httpx.AsyncClient:
    settings = Settings(_env_file=None, environment="test", groq_api_key="test-key")
    app = create_app(settings)
    app.dependency_overrides[get_recommendation_service] = lambda: service
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


@pytest.mark.asyncio
async def test_service_accepts_case_grounded_recommendation(
    case: TransactionMonitoringCase,
    recommendation: ComplianceRecommendation,
) -> None:
    expected = _provider_result(recommendation)
    service = RecommendationService(FakeProvider(expected))

    assert await service.recommend(case) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    [
        {"case_id": "A-DIFFERENT-CASE"},
        {"evidence.0.source_id": "TX-NOT-IN-CASE"},
        {"evidence.0.source": "tool_result"},
        {
            "policy_citations": [
                {
                    "policy_id": "POLICY-NOT-RETRIEVED",
                    "section_id": "SECTION-1",
                    "excerpt": "Unsupported policy claim.",
                }
            ]
        },
    ],
)
async def test_service_rejects_ungrounded_output(
    case: TransactionMonitoringCase,
    recommendation: ComplianceRecommendation,
    mutation: dict[str, object],
) -> None:
    payload = recommendation.model_dump(mode="json")
    for key, value in mutation.items():
        if key == "evidence.0.source_id":
            payload["evidence"][0]["source_id"] = value
        elif key == "evidence.0.source":
            payload["evidence"][0]["source"] = value
        else:
            payload[key] = value
    invalid = ComplianceRecommendation.model_validate(payload)
    service = RecommendationService(FakeProvider(_provider_result(invalid)))

    with pytest.raises(RecommendationInvalidOutputError) as error:
        await service.recommend(case)

    assert error.value.reason != "unspecified"


@pytest.mark.asyncio
async def test_openai_adapter_uses_structured_output_and_non_storage(
    case: TransactionMonitoringCase,
    recommendation: ComplianceRecommendation,
) -> None:
    parse = AsyncMock(
        return_value=SimpleNamespace(
            output_parsed=recommendation,
            id="resp_openai_test",
            model="gpt-test",
            usage=SimpleNamespace(input_tokens=250, output_tokens=100, total_tokens=350),
        )
    )
    fake_client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    settings = Settings(
        _env_file=None,
        environment="test",
        ai_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-test",
        ai_max_output_tokens=900,
    )
    provider = OpenAIRecommendationProvider(settings, client=fake_client)

    result = await provider.recommend(case)

    assert result.recommendation == recommendation
    kwargs = parse.await_args.kwargs
    assert kwargs["text_format"] is ComplianceRecommendation
    assert kwargs["store"] is False
    assert kwargs["max_output_tokens"] == 900
    assert kwargs["instructions"] == SYSTEM_INSTRUCTIONS
    assert "never claim an action was" in kwargs["instructions"]
    assert "avoid repeating names" in kwargs["instructions"]
    case_content = kwargs["input"][0]["content"]
    assert "<case_data>" in case_content
    assert case.investigator_notes in case_content
    assert '"proposed_control_action":"temporary_hold"' in case_content
    assert '"control_execution_status":"not_provided"' in case_content
    assert '"control_action":' not in case_content


@pytest.mark.asyncio
async def test_groq_adapter_uses_selected_groq_model(
    case: TransactionMonitoringCase,
    recommendation: ComplianceRecommendation,
) -> None:
    parse = AsyncMock(
        return_value=SimpleNamespace(
            output_parsed=recommendation,
            id="resp_groq_test",
            model="openai/gpt-oss-20b",
            usage=SimpleNamespace(input_tokens=200, output_tokens=80, total_tokens=280),
        )
    )
    fake_client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    settings = Settings(
        _env_file=None,
        environment="test",
        ai_provider="groq",
        groq_api_key="test-key",
        groq_model="openai/gpt-oss-20b",
    )
    provider = GroqRecommendationProvider(settings, client=fake_client)

    result = await provider.recommend(case)

    assert result.model == "openai/gpt-oss-20b"
    assert parse.await_args.kwargs["model"] == "openai/gpt-oss-20b"


@pytest.mark.asyncio
async def test_openai_adapter_retries_transient_failure_and_reports_count(
    case: TransactionMonitoringCase,
    recommendation: ComplianceRecommendation,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timeout = APITimeoutError(request=httpx.Request("POST", "https://api.openai.com/v1/responses"))
    parse = AsyncMock(
        side_effect=[
            timeout,
            SimpleNamespace(
                output_parsed=recommendation,
                id="resp_after_retry",
                model="gpt-test",
                usage=SimpleNamespace(input_tokens=250, output_tokens=100, total_tokens=350),
            ),
        ]
    )
    sleep = AsyncMock()
    monkeypatch.setattr("compliance_agent.providers.openai_recommendations.asyncio.sleep", sleep)
    fake_client = SimpleNamespace(responses=SimpleNamespace(parse=parse))
    settings = Settings(
        _env_file=None,
        environment="test",
        ai_provider="openai",
        openai_api_key="test-key",
        openai_model="gpt-test",
        ai_max_retries=1,
    )
    provider = OpenAIRecommendationProvider(settings, client=fake_client)

    result = await provider.recommend(case)

    assert result.retry_count == 1
    assert parse.await_count == 2
    sleep.assert_awaited_once()


@pytest.mark.asyncio
async def test_service_retries_one_ungrounded_output(
    case: TransactionMonitoringCase,
    recommendation: ComplianceRecommendation,
) -> None:
    invalid_payload = recommendation.model_dump(mode="json")
    invalid_payload["evidence"][0]["source_id"] = "TX-NOT-IN-CASE"
    invalid = ComplianceRecommendation.model_validate(invalid_payload)
    provider = SequenceProvider([_provider_result(invalid), _provider_result(recommendation)])
    service = RecommendationService(provider, max_output_retries=1)

    result = await service.recommend(case)

    assert result.recommendation == recommendation
    assert result.retry_count == 1
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_recommendation_endpoint_returns_structured_result(
    case: TransactionMonitoringCase,
    recommendation: ComplianceRecommendation,
) -> None:
    service = RecommendationService(FakeProvider(_provider_result(recommendation)))

    async with _client(service) as client:
        response = await client.post("/api/v1/recommendations", json=case.model_dump(mode="json"))

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["recommendation"]["human_review_required"] is True
    assert body["response_id"] == "resp_test_123"
    assert body["retry_count"] == 0
    assert body["usage"] == {
        "input_tokens": 300,
        "output_tokens": 120,
        "total_tokens": 420,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "status_code", "code"),
    [
        (RecommendationRateLimitError(), 429, "ai_rate_limited"),
        (RecommendationQuotaError(), 429, "ai_quota_exhausted"),
        (RecommendationRefusalError(), 422, "ai_refusal"),
        (RecommendationInvalidOutputError(), 502, "ai_invalid_output"),
        (RecommendationUnavailableError(), 503, "ai_unavailable"),
        (RecommendationTimeoutError(), 504, "ai_timeout"),
    ],
)
async def test_recommendation_endpoint_maps_expected_failures_safely(
    case: TransactionMonitoringCase,
    error: Exception,
    status_code: int,
    code: str,
) -> None:
    async with _client(FailingService(error)) as client:
        response = await client.post(
            "/api/v1/recommendations",
            headers={"X-Request-ID": "recommendation-failure-test"},
            json=case.model_dump(mode="json"),
        )

    assert response.status_code == status_code
    assert response.json()["error"]["code"] == code
    assert response.json()["error"]["request_id"] == "recommendation-failure-test"
    assert "test-key" not in response.text
