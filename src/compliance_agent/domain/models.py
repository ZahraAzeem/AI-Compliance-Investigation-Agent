"""Strict data contracts for compliance cases and AI recommendations."""

from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Annotated, Literal, Self

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    ),
]
CountryCode = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[A-Z]{2}$"),
]
CurrencyCode = Annotated[
    str,
    StringConstraints(strip_whitespace=True, pattern=r"^[A-Z]{3}$"),
]
ShortUntrustedText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
LongUntrustedText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=5_000),
]


class StrictDomainModel(BaseModel):
    """Base model that rejects misspelled or unexpected input fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CaseType(StrEnum):
    TRANSACTION_MONITORING = "transaction_monitoring"
    KYC_REVIEW = "kyc_review"
    SANCTIONS_SCREENING = "sanctions_screening"
    ADVERSE_MEDIA = "adverse_media"


class CasePriority(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class RiskLevel(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class TransactionDirection(StrEnum):
    INBOUND = "inbound"
    OUTBOUND = "outbound"


class TransactionChannel(StrEnum):
    ACH = "ach"
    CARD = "card"
    CASH = "cash"
    CRYPTO = "crypto"
    WIRE = "wire"
    OTHER = "other"


class RecommendationAction(StrEnum):
    CLOSE_NO_ACTION = "close_no_action"
    REQUEST_INFORMATION = "request_information"
    ENHANCED_DUE_DILIGENCE = "enhanced_due_diligence"
    ESCALATE_FOR_SAR_REVIEW = "escalate_for_sar_review"
    RESTRICT_ACCOUNT_REVIEW = "restrict_account_review"


class EvidenceSource(StrEnum):
    CASE_INPUT = "case_input"
    TOOL_RESULT = "tool_result"
    POLICY = "policy"


class CustomerProfile(StrictDomainModel):
    """A natural-person customer profile supplied as untrusted case data."""

    customer_id: Identifier
    full_name: ShortUntrustedText
    date_of_birth: date
    country_of_residence: CountryCode
    nationality_codes: list[CountryCode] = Field(min_length=1, max_length=5)
    occupation: ShortUntrustedText | None = None
    is_politically_exposed: bool = False
    account_opened_at: date
    existing_risk_level: RiskLevel = RiskLevel.UNKNOWN

    @field_validator("date_of_birth")
    @classmethod
    def date_of_birth_must_not_be_future(cls, value: date) -> date:
        if value > datetime.now(UTC).date():
            raise ValueError("date_of_birth must not be in the future")
        return value

    @field_validator("account_opened_at")
    @classmethod
    def account_opened_at_must_not_be_future(cls, value: date) -> date:
        if value > datetime.now(UTC).date():
            raise ValueError("account_opened_at must not be in the future")
        return value

    @model_validator(mode="after")
    def account_must_open_after_birth(self) -> Self:
        if self.account_opened_at <= self.date_of_birth:
            raise ValueError("account_opened_at must be after date_of_birth")
        if len(set(self.nationality_codes)) != len(self.nationality_codes):
            raise ValueError("nationality_codes must not contain duplicates")
        return self


class Transaction(StrictDomainModel):
    """A normalized financial transaction included in a compliance case."""

    transaction_id: Identifier
    occurred_at: AwareDatetime
    amount: Decimal = Field(gt=0, max_digits=18, decimal_places=2)
    currency: CurrencyCode
    direction: TransactionDirection
    channel: TransactionChannel
    counterparty_name: ShortUntrustedText
    counterparty_country: CountryCode
    description: ShortUntrustedText | None = None

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_must_not_be_future(cls, value: datetime) -> datetime:
        if value > datetime.now(UTC) + timedelta(minutes=5):
            raise ValueError("occurred_at must not be in the future")
        return value


class ComplianceCase(StrictDomainModel):
    """Validated input contract for one compliance investigation."""

    case_id: Identifier
    case_type: CaseType
    priority: CasePriority = CasePriority.NORMAL
    opened_at: AwareDatetime
    trigger_reason: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=2_000),
    ]
    investigator_notes: LongUntrustedText | None = Field(
        default=None,
        description="Untrusted case text. It must never be treated as model instructions.",
    )
    customer: CustomerProfile
    transactions: list[Transaction] = Field(min_length=1, max_length=500)

    @field_validator("opened_at")
    @classmethod
    def opened_at_must_not_be_future(cls, value: datetime) -> datetime:
        if value > datetime.now(UTC) + timedelta(minutes=5):
            raise ValueError("opened_at must not be in the future")
        return value

    @model_validator(mode="after")
    def transaction_ids_must_be_unique(self) -> Self:
        transaction_ids = [transaction.transaction_id for transaction in self.transactions]
        if len(set(transaction_ids)) != len(transaction_ids):
            raise ValueError("transactions must have unique transaction_id values")
        return self


class EvidenceItem(StrictDomainModel):
    source: EvidenceSource
    source_id: Identifier
    summary: ShortUntrustedText
    supports_recommendation: bool


class PolicyCitation(StrictDomainModel):
    policy_id: Identifier
    section_id: Identifier
    excerpt: ShortUntrustedText


class ComplianceRecommendation(StrictDomainModel):
    """Validated output contract for future model-generated recommendations."""

    case_id: Identifier
    risk_level: RiskLevel
    recommended_action: RecommendationAction
    confidence: float = Field(ge=0, le=1)
    rationale: LongUntrustedText
    evidence: list[EvidenceItem] = Field(min_length=1, max_length=50)
    policy_citations: list[PolicyCitation] = Field(default_factory=list, max_length=20)
    uncertainties: list[ShortUntrustedText] = Field(default_factory=list, max_length=20)
    missing_information: list[ShortUntrustedText] = Field(default_factory=list, max_length=20)
    human_review_required: Literal[True] = True
