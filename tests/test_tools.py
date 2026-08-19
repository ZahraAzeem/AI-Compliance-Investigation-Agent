"""Tests for allowlisted, read-only compliance tool execution."""

import json
from pathlib import Path

import httpx
import pytest
from pydantic import TypeAdapter

from compliance_agent.config import Settings
from compliance_agent.domain import InvestigationCase, TransactionMonitoringCase
from compliance_agent.main import create_app
from compliance_agent.services.tools import (
    LOOKUP_CUSTOMER_RISK,
    SUMMARIZE_TRANSACTIONS,
    ToolCallLimitExceededError,
    ToolCallRequest,
    chat_completion_tool_definitions,
    execute_tool_calls,
)

SAMPLE_CASE_PATH = Path(__file__).parents[1] / "examples" / "sample_case.json"
CASE_ADAPTER = TypeAdapter(InvestigationCase)


@pytest.fixture
def case() -> TransactionMonitoringCase:
    payload = json.loads(SAMPLE_CASE_PATH.read_text(encoding="utf-8"))
    validated = CASE_ADAPTER.validate_python(payload)
    assert isinstance(validated, TransactionMonitoringCase)
    return validated


def _call(call_id: str, name: str, arguments: dict[str, object]) -> ToolCallRequest:
    return ToolCallRequest(call_id=call_id, name=name, arguments=arguments)


def _client(*, max_tool_calls: int = 4) -> httpx.AsyncClient:
    settings = Settings(
        _env_file=None,
        environment="test",
        groq_api_key="test-key",
        agent_max_tool_calls=max_tool_calls,
    )
    transport = httpx.ASGITransport(app=create_app(settings))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_model_sees_only_fixed_strict_tool_definitions() -> None:
    definitions = chat_completion_tool_definitions()

    assert [definition["function"]["name"] for definition in definitions] == [
        LOOKUP_CUSTOMER_RISK,
        SUMMARIZE_TRANSACTIONS,
    ]
    assert all(
        definition["function"]["parameters"]["additionalProperties"] is False
        for definition in definitions
    )


def test_transaction_summary_uses_exact_decimal_calculation(
    case: TransactionMonitoringCase,
) -> None:
    outcomes = execute_tool_calls(
        case,
        [_call("call-summary", SUMMARIZE_TRANSACTIONS, {"transaction_ids": []})],
        max_tool_calls=4,
    )

    assert outcomes[0].status == "completed"
    assert outcomes[0].arguments_summary == {"requested_transaction_count": 0}
    assert outcomes[0].output == {
        "transaction_count": 2,
        "outbound_count": 2,
        "inbound_count": 0,
        "totals_by_currency": {"USD": "40600.00"},
        "counterparty_countries": ["AE"],
        "transaction_ids": ["TX-9001", "TX-9002"],
        "source": "deterministic_case_calculation",
    }


def test_customer_risk_lookup_cannot_access_another_customer(
    case: TransactionMonitoringCase,
) -> None:
    outcomes = execute_tool_calls(
        case,
        [_call("call-risk", LOOKUP_CUSTOMER_RISK, {"customer_id": "CUST-OTHER"})],
        max_tool_calls=4,
    )

    assert outcomes[0].status == "rejected"
    assert outcomes[0].arguments_summary == {"customer_id": "CUST-OTHER"}
    assert outcomes[0].error_code == "customer_not_in_case"
    assert outcomes[0].output is None


def test_unknown_tool_is_rejected(case: TransactionMonitoringCase) -> None:
    outcomes = execute_tool_calls(
        case,
        [_call("call-unknown", "run_arbitrary_python", {"code": "print('unsafe')"})],
        max_tool_calls=4,
    )

    assert outcomes[0].status == "rejected"
    assert outcomes[0].arguments_summary == {"provided_argument_count": 1}
    assert outcomes[0].error_code == "unknown_tool"


def test_extra_tool_arguments_are_rejected(case: TransactionMonitoringCase) -> None:
    outcomes = execute_tool_calls(
        case,
        [
            _call(
                "call-extra",
                LOOKUP_CUSTOMER_RISK,
                {"customer_id": "CUST-1042", "include_secrets": True},
            )
        ],
        max_tool_calls=4,
    )

    assert outcomes[0].status == "rejected"
    assert outcomes[0].error_code == "invalid_tool_arguments"


def test_duplicate_call_is_not_executed_twice(case: TransactionMonitoringCase) -> None:
    calls = [
        _call("call-first", LOOKUP_CUSTOMER_RISK, {"customer_id": "CUST-1042"}),
        _call("call-second", LOOKUP_CUSTOMER_RISK, {"customer_id": "CUST-1042"}),
    ]

    outcomes = execute_tool_calls(case, calls, max_tool_calls=4)

    assert outcomes[0].status == "completed"
    assert outcomes[1].status == "rejected"
    assert outcomes[1].error_code == "duplicate_tool_call"


def test_tool_call_limit_is_enforced_before_execution(
    case: TransactionMonitoringCase,
) -> None:
    calls = [
        _call(f"call-{index}", LOOKUP_CUSTOMER_RISK, {"customer_id": "CUST-1042"})
        for index in range(3)
    ]

    with pytest.raises(ToolCallLimitExceededError):
        execute_tool_calls(case, calls, max_tool_calls=2)


@pytest.mark.asyncio
async def test_tool_execution_endpoint_returns_mock_results(
    case: TransactionMonitoringCase,
) -> None:
    payload = {
        "case": case.model_dump(mode="json"),
        "calls": [
            {
                "call_id": "postman-risk-lookup",
                "name": LOOKUP_CUSTOMER_RISK,
                "arguments": {"customer_id": "CUST-1042"},
            },
            {
                "call_id": "postman-transaction-summary",
                "name": SUMMARIZE_TRANSACTIONS,
                "arguments": {"transaction_ids": ["TX-9001", "TX-9002"]},
            },
        ],
    }

    async with _client() as client:
        response = await client.post("/api/v1/tools/execute", json=payload)

    assert response.status_code == 200
    assert response.json()["status"] == "completed"
    assert response.json()["is_mock"] is True
    assert [outcome["status"] for outcome in response.json()["outcomes"]] == [
        "completed",
        "completed",
    ]


@pytest.mark.asyncio
async def test_tool_execution_endpoint_returns_safe_limit_error(
    case: TransactionMonitoringCase,
) -> None:
    payload = {
        "case": case.model_dump(mode="json"),
        "calls": [
            {
                "call_id": f"call-{index}",
                "name": LOOKUP_CUSTOMER_RISK,
                "arguments": {"customer_id": "CUST-1042"},
            }
            for index in range(3)
        ],
    }

    async with _client(max_tool_calls=2) as client:
        response = await client.post("/api/v1/tools/execute", json=payload)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "http_422"
    assert "CUST-1042" not in response.text
