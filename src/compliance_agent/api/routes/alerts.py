"""Deterministic alert-evaluation routes."""

from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel

from compliance_agent.domain import TransactionMonitoringAlert, TransactionRuleEvaluationRequest
from compliance_agent.services.rule_engine import (
    EVALUATED_RULE_IDS,
    RULESET_VERSION,
    evaluate_transaction_rules,
)

router = APIRouter(prefix="/alerts", tags=["alerts"])


class TransactionAlertEvaluationResponse(BaseModel):
    status: Literal["evaluated"]
    ruleset: str
    evaluated_rule_ids: list[str]
    alert_count: int
    alerts: list[TransactionMonitoringAlert]


@router.post("/evaluate/transactions", response_model=TransactionAlertEvaluationResponse)
async def evaluate_transaction_activity(
    request: TransactionRuleEvaluationRequest,
) -> TransactionAlertEvaluationResponse:
    """Evaluate synthetic transaction rules without AI or external side effects."""

    alerts = evaluate_transaction_rules(request)
    return TransactionAlertEvaluationResponse(
        status="evaluated",
        ruleset=RULESET_VERSION,
        evaluated_rule_ids=EVALUATED_RULE_IDS,
        alert_count=len(alerts),
        alerts=alerts,
    )
