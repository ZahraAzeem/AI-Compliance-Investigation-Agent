"""Strict, read-only compliance tools and bounded local execution."""

import json
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from compliance_agent.domain import InvestigationCase
from compliance_agent.domain.models import Identifier

LOOKUP_CUSTOMER_RISK = "lookup_customer_risk"
SUMMARIZE_TRANSACTIONS = "summarize_transactions"


class StrictToolModel(BaseModel):
    """Reject undeclared model-generated tool arguments."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class LookupCustomerRiskInput(StrictToolModel):
    customer_id: Identifier


class SummarizeTransactionsInput(StrictToolModel):
    transaction_ids: list[Identifier] = Field(default_factory=list, max_length=500)

    @field_validator("transaction_ids")
    @classmethod
    def transaction_ids_must_be_unique(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("transaction_ids must not contain duplicates")
        return value


class CustomerRiskSnapshot(StrictToolModel):
    customer_id: Identifier
    existing_risk_level: str
    is_politically_exposed: bool
    country_of_residence: str
    nationality_codes: list[str]
    occupation: str | None
    source: Literal["mock_case_context"] = "mock_case_context"
    is_mock: Literal[True] = True


class TransactionSummary(StrictToolModel):
    transaction_count: int = Field(ge=0)
    outbound_count: int = Field(ge=0)
    inbound_count: int = Field(ge=0)
    totals_by_currency: dict[str, Decimal]
    counterparty_countries: list[str]
    transaction_ids: list[Identifier]
    source: Literal["deterministic_case_calculation"] = "deterministic_case_calculation"


class ToolCallRequest(StrictToolModel):
    call_id: Identifier
    name: Identifier
    arguments: dict[str, Any]


class ToolCallOutcome(StrictToolModel):
    call_id: Identifier
    name: Identifier
    status: Literal["completed", "rejected"]
    arguments_summary: dict[str, str | int] = Field(default_factory=dict)
    output: dict[str, Any] | None = None
    error_code: str | None = None
    error_message: str | None = None
    duration_ms: int = Field(default=0, ge=0)


class ToolCallLimitExceededError(Exception):
    """The requested batch exceeds the application-owned tool-call boundary."""


class ToolExecutionFailure(Exception):
    """A safe, expected tool failure returned to the orchestration layer."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


ToolInput = LookupCustomerRiskInput | SummarizeTransactionsInput
ToolOutput = CustomerRiskSnapshot | TransactionSummary
ToolHandler = Callable[[InvestigationCase, ToolInput], ToolOutput]


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_model: type[StrictToolModel]
    handler: ToolHandler


def _lookup_customer_risk(
    case: InvestigationCase,
    arguments: ToolInput,
) -> CustomerRiskSnapshot:
    if not isinstance(arguments, LookupCustomerRiskInput):
        raise TypeError("lookup_customer_risk received the wrong validated input type")
    if arguments.customer_id != case.customer.customer_id:
        raise ToolExecutionFailure(
            "customer_not_in_case",
            "The requested customer is not available in this case context.",
        )
    customer = case.customer
    return CustomerRiskSnapshot(
        customer_id=customer.customer_id,
        existing_risk_level=customer.existing_risk_level,
        is_politically_exposed=customer.is_politically_exposed,
        country_of_residence=customer.country_of_residence,
        nationality_codes=customer.nationality_codes,
        occupation=customer.occupation,
    )


def _summarize_transactions(
    case: InvestigationCase,
    arguments: ToolInput,
) -> TransactionSummary:
    if not isinstance(arguments, SummarizeTransactionsInput):
        raise TypeError("summarize_transactions received the wrong validated input type")

    transactions_by_id = {
        transaction.transaction_id: transaction for transaction in case.transactions
    }
    selected_ids = arguments.transaction_ids or list(transactions_by_id)
    unknown_ids = set(selected_ids) - set(transactions_by_id)
    if unknown_ids:
        raise ToolExecutionFailure(
            "transaction_not_in_case",
            "One or more requested transactions are not available in this case context.",
        )

    selected = [transactions_by_id[transaction_id] for transaction_id in selected_ids]
    totals: dict[str, Decimal] = {}
    for transaction in selected:
        totals[transaction.currency] = totals.get(transaction.currency, Decimal("0.00")) + (
            transaction.amount
        )

    outbound_count = sum(transaction.direction == "outbound" for transaction in selected)
    return TransactionSummary(
        transaction_count=len(selected),
        outbound_count=outbound_count,
        inbound_count=len(selected) - outbound_count,
        totals_by_currency=dict(sorted(totals.items())),
        counterparty_countries=sorted(
            {transaction.counterparty_country for transaction in selected}
        ),
        transaction_ids=selected_ids,
    )


TOOL_REGISTRY: dict[str, ToolDefinition] = {
    LOOKUP_CUSTOMER_RISK: ToolDefinition(
        name=LOOKUP_CUSTOMER_RISK,
        description=(
            "Read the existing risk attributes for the customer already present in the case. "
            "This mock tool cannot search for other customers or modify risk."
        ),
        input_model=LookupCustomerRiskInput,
        handler=_lookup_customer_risk,
    ),
    SUMMARIZE_TRANSACTIONS: ToolDefinition(
        name=SUMMARIZE_TRANSACTIONS,
        description=(
            "Deterministically summarize transaction amounts and directions using only "
            "transaction IDs already present in the case."
        ),
        input_model=SummarizeTransactionsInput,
        handler=_summarize_transactions,
    ),
}


def chat_completion_tool_definitions() -> list[dict[str, Any]]:
    """Return the fixed tool schemas shown to a chat-completions model."""

    return [
        {
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description,
                "parameters": definition.input_model.model_json_schema(),
            },
        }
        for definition in TOOL_REGISTRY.values()
    ]


def execute_tool_calls(
    case: InvestigationCase,
    calls: list[ToolCallRequest],
    *,
    max_tool_calls: int,
) -> list[ToolCallOutcome]:
    """Execute allowlisted calls once each, within a hard application-owned limit."""

    if len(calls) > max_tool_calls:
        raise ToolCallLimitExceededError

    seen_fingerprints: set[str] = set()
    outcomes: list[ToolCallOutcome] = []
    for call in calls:
        started_at = perf_counter()
        fingerprint = _call_fingerprint(call)
        if fingerprint in seen_fingerprints:
            outcomes.append(
                _rejected_outcome(
                    call,
                    "duplicate_tool_call",
                    "The same tool call and arguments were already processed.",
                    started_at=started_at,
                )
            )
            continue
        seen_fingerprints.add(fingerprint)

        definition = TOOL_REGISTRY.get(call.name)
        if definition is None:
            outcomes.append(
                _rejected_outcome(
                    call,
                    "unknown_tool",
                    "The requested tool is not in the fixed allowlist.",
                    started_at=started_at,
                )
            )
            continue

        try:
            arguments = definition.input_model.model_validate(call.arguments)
            output = definition.handler(case, arguments)
        except ValidationError:
            outcomes.append(
                _rejected_outcome(
                    call,
                    "invalid_tool_arguments",
                    "The tool arguments did not match the declared schema.",
                    started_at=started_at,
                )
            )
        except ToolExecutionFailure as exc:
            outcomes.append(
                _rejected_outcome(
                    call,
                    exc.code,
                    exc.message,
                    started_at=started_at,
                )
            )
        else:
            outcomes.append(
                ToolCallOutcome(
                    call_id=call.call_id,
                    name=call.name,
                    status="completed",
                    arguments_summary=_safe_arguments_summary(call),
                    output=output.model_dump(mode="json"),
                    duration_ms=_duration_ms(started_at),
                )
            )
    return outcomes


def _call_fingerprint(call: ToolCallRequest) -> str:
    canonical = json.dumps(
        {"name": call.name, "arguments": call.arguments},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _rejected_outcome(
    call: ToolCallRequest,
    code: str,
    message: str,
    *,
    started_at: float,
) -> ToolCallOutcome:
    return ToolCallOutcome(
        call_id=call.call_id,
        name=call.name,
        status="rejected",
        arguments_summary=_safe_arguments_summary(call),
        error_code=code,
        error_message=message,
        duration_ms=_duration_ms(started_at),
    )


def _duration_ms(started_at: float) -> int:
    return max(0, round((perf_counter() - started_at) * 1_000))


def _safe_arguments_summary(call: ToolCallRequest) -> dict[str, str | int]:
    """Summarize scope without copying arbitrary model-produced values into logs."""

    if call.name == LOOKUP_CUSTOMER_RISK:
        customer_id = call.arguments.get("customer_id")
        return {"customer_id": customer_id} if isinstance(customer_id, str) else {}
    if call.name == SUMMARIZE_TRANSACTIONS:
        transaction_ids = call.arguments.get("transaction_ids")
        return {
            "requested_transaction_count": (
                len(transaction_ids) if isinstance(transaction_ids, list) else 0
            )
        }
    return {"provided_argument_count": len(call.arguments)}
