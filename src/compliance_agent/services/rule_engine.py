"""Small deterministic portfolio rule engine with fictional thresholds."""

from datetime import UTC, datetime
from decimal import Decimal
from uuid import NAMESPACE_URL, uuid5

from compliance_agent.domain import (
    ControlAction,
    RecipientRelationship,
    TransactionDirection,
    TransactionMonitoringAlert,
    TransactionRuleEvaluationRequest,
    TransferPurpose,
)
from compliance_agent.domain.models import Transaction, TransactionAlertMetrics

RULESET_VERSION = "synthetic-portfolio-v1"
NON_FAMILY_FREQUENCY_RULE_ID = "synthetic-non-family-frequency"
MONTHLY_AED_VOLUME_RULE_ID = "synthetic-monthly-aed-volume"
NON_FAMILY_TRANSFER_COUNT_THRESHOLD = 5
MONTHLY_AED_VOLUME_THRESHOLD = Decimal("25000.00")
EVALUATED_RULE_IDS = [NON_FAMILY_FREQUENCY_RULE_ID, MONTHLY_AED_VOLUME_RULE_ID]


def _is_in_evaluation_month(
    transaction: Transaction, request: TransactionRuleEvaluationRequest
) -> bool:
    occurred_at = transaction.occurred_at.astimezone(UTC)
    return (
        occurred_at.year == request.evaluation_month.year
        and occurred_at.month == request.evaluation_month.month
    )


def _alert_id(request: TransactionRuleEvaluationRequest, rule_id: str) -> str:
    unique_key = (
        f"{RULESET_VERSION}:{request.customer.customer_id}:{request.evaluation_month}:{rule_id}"
    )
    return f"ALT-{uuid5(NAMESPACE_URL, unique_key)}"


def evaluate_transaction_rules(
    request: TransactionRuleEvaluationRequest,
    *,
    evaluated_at: datetime | None = None,
) -> list[TransactionMonitoringAlert]:
    """Evaluate exact synthetic rules without using AI or executing a control action."""

    created_at = evaluated_at or datetime.now(UTC)
    monthly_outbound = [
        transaction
        for transaction in request.transactions
        if transaction.direction is TransactionDirection.OUTBOUND
        and _is_in_evaluation_month(transaction, request)
    ]
    monthly_aed_outbound = [
        transaction for transaction in monthly_outbound if transaction.currency == "AED"
    ]
    non_family_outbound = [
        transaction
        for transaction in monthly_outbound
        if transaction.recipient_relationship
        not in {RecipientRelationship.SELF, RecipientRelationship.IMMEDIATE_FAMILY}
        and transaction.transfer_purpose is not TransferPurpose.FAMILY_SUPPORT
    ]

    monthly_aed_total = sum(
        (transaction.amount for transaction in monthly_aed_outbound),
        start=Decimal("0.00"),
    )
    metrics = TransactionAlertMetrics(
        evaluation_month=request.evaluation_month,
        outbound_aed_total=monthly_aed_total,
        non_family_outbound_count=len(non_family_outbound),
    )
    alerts: list[TransactionMonitoringAlert] = []

    if len(non_family_outbound) >= NON_FAMILY_TRANSFER_COUNT_THRESHOLD:
        alerts.append(
            TransactionMonitoringAlert(
                alert_id=_alert_id(request, NON_FAMILY_FREQUENCY_RULE_ID),
                customer_id=request.customer.customer_id,
                created_at=created_at,
                rule_id=NON_FAMILY_FREQUENCY_RULE_ID,
                rule_version=RULESET_VERSION,
                summary=(
                    "Synthetic frequency rule triggered for outbound transfers to non-family "
                    "recipients with a non-family transfer purpose."
                ),
                control_action=ControlAction.TEMPORARY_HOLD,
                related_transaction_ids=[
                    transaction.transaction_id for transaction in non_family_outbound
                ],
                metrics=metrics,
            )
        )

    if monthly_aed_total >= MONTHLY_AED_VOLUME_THRESHOLD:
        alerts.append(
            TransactionMonitoringAlert(
                alert_id=_alert_id(request, MONTHLY_AED_VOLUME_RULE_ID),
                customer_id=request.customer.customer_id,
                created_at=created_at,
                rule_id=MONTHLY_AED_VOLUME_RULE_ID,
                rule_version=RULESET_VERSION,
                summary="Synthetic monthly AED outbound-volume rule triggered.",
                control_action=ControlAction.TEMPORARY_HOLD,
                related_transaction_ids=[
                    transaction.transaction_id for transaction in monthly_aed_outbound
                ],
                metrics=metrics,
            )
        )

    return alerts
