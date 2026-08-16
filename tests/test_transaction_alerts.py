"""Tests for deterministic, synthetic transaction-monitoring alerts."""

import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from compliance_agent.config import Settings
from compliance_agent.domain import TransactionRuleEvaluationRequest
from compliance_agent.main import create_app
from compliance_agent.services.rule_engine import (
    MONTHLY_AED_VOLUME_RULE_ID,
    NON_FAMILY_FREQUENCY_RULE_ID,
    RULESET_VERSION,
    evaluate_transaction_rules,
)

SAMPLE_ACTIVITY_PATH = Path(__file__).parents[1] / "examples" / "sample_transaction_activity.json"
EVALUATED_AT = datetime(2026, 8, 16, 12, 0, tzinfo=UTC)


@pytest.fixture
def sample_activity() -> dict[str, object]:
    return json.loads(SAMPLE_ACTIVITY_PATH.read_text(encoding="utf-8"))


def _client() -> httpx.AsyncClient:
    settings = Settings(_env_file=None, environment="test")
    transport = httpx.ASGITransport(app=create_app(settings))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_activity_triggers_frequency_and_volume_alerts(
    sample_activity: dict[str, object],
) -> None:
    request = TransactionRuleEvaluationRequest.model_validate(sample_activity)

    alerts = evaluate_transaction_rules(request, evaluated_at=EVALUATED_AT)

    assert [alert.rule_id for alert in alerts] == [
        NON_FAMILY_FREQUENCY_RULE_ID,
        MONTHLY_AED_VOLUME_RULE_ID,
    ]
    assert all(alert.control_action == "temporary_hold" for alert in alerts)
    assert all(alert.is_mock for alert in alerts)
    assert alerts[0].metrics.non_family_outbound_count == 5
    assert alerts[0].metrics.outbound_aed_total == 25_000


def test_rule_evaluation_is_idempotent(sample_activity: dict[str, object]) -> None:
    request = TransactionRuleEvaluationRequest.model_validate(sample_activity)

    first = evaluate_transaction_rules(request, evaluated_at=EVALUATED_AT)
    second = evaluate_transaction_rules(request, evaluated_at=EVALUATED_AT)

    assert [alert.alert_id for alert in first] == [alert.alert_id for alert in second]


def test_below_threshold_activity_returns_no_alerts(
    sample_activity: dict[str, object],
) -> None:
    below_threshold = deepcopy(sample_activity)
    transactions = below_threshold["transactions"]
    assert isinstance(transactions, list)
    below_threshold["transactions"] = transactions[:4]

    request = TransactionRuleEvaluationRequest.model_validate(below_threshold)

    assert evaluate_transaction_rules(request, evaluated_at=EVALUATED_AT) == []


def test_transactions_outside_evaluation_month_are_ignored(
    sample_activity: dict[str, object],
) -> None:
    outside_month = deepcopy(sample_activity)
    transactions = outside_month["transactions"]
    assert isinstance(transactions, list)
    for transaction in transactions:
        transaction["occurred_at"] = transaction["occurred_at"].replace("2026-08", "2026-07")

    request = TransactionRuleEvaluationRequest.model_validate(outside_month)

    assert evaluate_transaction_rules(request, evaluated_at=EVALUATED_AT) == []


@pytest.mark.asyncio
async def test_alert_evaluation_endpoint_returns_explainable_results(
    sample_activity: dict[str, object],
) -> None:
    async with _client() as client:
        response = await client.post("/api/v1/alerts/evaluate/transactions", json=sample_activity)

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "evaluated"
    assert body["ruleset"] == RULESET_VERSION
    assert body["alert_count"] == 2
    assert body["evaluated_rule_ids"] == [
        NON_FAMILY_FREQUENCY_RULE_ID,
        MONTHLY_AED_VOLUME_RULE_ID,
    ]
    assert body["alerts"][0]["metrics"] == {
        "evaluation_month": "2026-08-01",
        "outbound_aed_total": "25000.00",
        "non_family_outbound_count": 5,
    }
