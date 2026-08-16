"""Strict data contracts for alerts, investigation cases, and AI recommendations."""

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


class TransferPurpose(StrEnum):
    FAMILY_SUPPORT = "family_support"
    GOODS_SERVICES = "goods_services"
    EDUCATION = "education"
    MEDICAL = "medical"
    PERSONAL_SAVINGS = "personal_savings"
    OTHER = "other"


class RecipientRelationship(StrEnum):
    SELF = "self"
    IMMEDIATE_FAMILY = "immediate_family"
    EXTENDED_FAMILY = "extended_family"
    FRIEND = "friend"
    BUSINESS = "business"
    OTHER = "other"
    UNKNOWN = "unknown"


class ControlAction(StrEnum):
    NONE = "none"
    QUESTIONNAIRE_REQUIRED = "questionnaire_required"
    DOCUMENTS_REQUIRED = "documents_required"
    TEMPORARY_HOLD = "temporary_hold"
    MANUAL_REVIEW = "manual_review"


class ScreeningCategory(StrEnum):
    SANCTIONS = "sanctions"
    POLITICALLY_EXPOSED_PERSON = "politically_exposed_person"
    ADVERSE_MEDIA = "adverse_media"


class ScreeningSubjectType(StrEnum):
    SENDER = "sender"
    RECEIVER = "receiver"


class RequiredArtifact(StrEnum):
    QUESTIONNAIRE = "questionnaire"
    PASSPORT = "passport"
    BANK_STATEMENTS_3_MONTHS = "bank_statements_3_months"


class ArtifactStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    REQUESTED = "requested"
    PROVIDED = "provided"
    VERIFIED = "verified"
    REJECTED = "rejected"


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
    income_range: ShortUntrustedText | None = None
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
    """A normalized financial transaction included in monitoring."""

    transaction_id: Identifier
    occurred_at: AwareDatetime
    amount: Decimal = Field(gt=0, max_digits=18, decimal_places=2)
    currency: CurrencyCode
    direction: TransactionDirection
    channel: TransactionChannel
    counterparty_name: ShortUntrustedText
    counterparty_country: CountryCode
    recipient_relationship: RecipientRelationship
    transfer_purpose: TransferPurpose
    description: ShortUntrustedText | None = None

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_must_not_be_future(cls, value: datetime) -> datetime:
        if value > datetime.now(UTC) + timedelta(minutes=5):
            raise ValueError("occurred_at must not be in the future")
        return value


class TransactionAlertMetrics(StrictDomainModel):
    evaluation_month: date
    outbound_aed_total: Decimal = Field(ge=0, max_digits=18, decimal_places=2)
    non_family_outbound_count: int = Field(ge=0)


class TransactionMonitoringAlert(StrictDomainModel):
    alert_type: Literal["transaction_monitoring"] = "transaction_monitoring"
    alert_id: Identifier
    customer_id: Identifier
    created_at: AwareDatetime
    rule_id: Identifier
    rule_version: Identifier
    summary: ShortUntrustedText
    control_action: Literal[ControlAction.TEMPORARY_HOLD] = ControlAction.TEMPORARY_HOLD
    related_transaction_ids: list[Identifier] = Field(min_length=1, max_length=500)
    metrics: TransactionAlertMetrics
    is_mock: Literal[True] = True


class OnboardingRiskAlert(StrictDomainModel):
    alert_type: Literal["onboarding_risk"] = "onboarding_risk"
    alert_id: Identifier
    customer_id: Identifier
    created_at: AwareDatetime
    risk_level: Literal[RiskLevel.MEDIUM, RiskLevel.HIGH]
    risk_score: float = Field(ge=0, le=100)
    risk_factors: list[ShortUntrustedText] = Field(min_length=1, max_length=20)
    required_artifacts: list[RequiredArtifact] = Field(min_length=1, max_length=10)
    control_action: ControlAction
    summary: ShortUntrustedText
    is_mock: Literal[True] = True


class ScreeningAlert(StrictDomainModel):
    alert_type: Literal["screening_match"] = "screening_match"
    alert_id: Identifier
    customer_id: Identifier
    created_at: AwareDatetime
    subject_type: ScreeningSubjectType
    subject_id: Identifier
    provider: Literal["mock_screening_provider"] = "mock_screening_provider"
    match_categories: list[ScreeningCategory] = Field(min_length=1, max_length=3)
    match_score: float = Field(ge=0, le=1)
    matched_name: ShortUntrustedText
    control_action: Literal[ControlAction.MANUAL_REVIEW] = ControlAction.MANUAL_REVIEW
    summary: ShortUntrustedText
    is_mock: Literal[True] = True


class ArtifactReview(StrictDomainModel):
    artifact: RequiredArtifact
    status: ArtifactStatus


class BaseInvestigationCase(StrictDomainModel):
    """Fields shared by every case sent to the future AI investigation."""

    case_id: Identifier
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
    transactions: list[Transaction] = Field(default_factory=list, max_length=500)

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


class TransactionMonitoringCase(BaseInvestigationCase):
    case_type: Literal[CaseType.TRANSACTION_MONITORING] = CaseType.TRANSACTION_MONITORING
    alert: TransactionMonitoringAlert
    transactions: list[Transaction] = Field(min_length=1, max_length=500)

    @model_validator(mode="after")
    def alert_must_match_case_data(self) -> Self:
        if self.alert.customer_id != self.customer.customer_id:
            raise ValueError("alert customer_id must match case customer_id")
        transaction_ids = {transaction.transaction_id for transaction in self.transactions}
        if not set(self.alert.related_transaction_ids).issubset(transaction_ids):
            raise ValueError("alert transaction IDs must exist in case transactions")
        return self


class OnboardingInvestigationCase(BaseInvestigationCase):
    case_type: Literal[CaseType.KYC_REVIEW] = CaseType.KYC_REVIEW
    alert: OnboardingRiskAlert
    artifact_reviews: list[ArtifactReview] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def alert_must_match_customer(self) -> Self:
        if self.alert.customer_id != self.customer.customer_id:
            raise ValueError("alert customer_id must match case customer_id")
        return self


class SanctionsInvestigationCase(BaseInvestigationCase):
    case_type: Literal[CaseType.SANCTIONS_SCREENING] = CaseType.SANCTIONS_SCREENING
    alert: ScreeningAlert

    @model_validator(mode="after")
    def alert_must_include_sanctions(self) -> Self:
        if self.alert.customer_id != self.customer.customer_id:
            raise ValueError("alert customer_id must match case customer_id")
        if ScreeningCategory.SANCTIONS not in self.alert.match_categories:
            raise ValueError("sanctions case alert must include a sanctions match")
        return self


class AdverseMediaInvestigationCase(BaseInvestigationCase):
    case_type: Literal[CaseType.ADVERSE_MEDIA] = CaseType.ADVERSE_MEDIA
    alert: ScreeningAlert

    @model_validator(mode="after")
    def alert_must_include_adverse_media(self) -> Self:
        if self.alert.customer_id != self.customer.customer_id:
            raise ValueError("alert customer_id must match case customer_id")
        if ScreeningCategory.ADVERSE_MEDIA not in self.alert.match_categories:
            raise ValueError("adverse media case alert must include an adverse media match")
        return self


InvestigationCase = Annotated[
    TransactionMonitoringCase
    | OnboardingInvestigationCase
    | SanctionsInvestigationCase
    | AdverseMediaInvestigationCase,
    Field(discriminator="case_type"),
]

# Compatibility name retained while routes and clients move to the case union.
ComplianceCase = TransactionMonitoringCase


class TransactionRuleEvaluationRequest(StrictDomainModel):
    """Validated activity window for the synthetic transaction rules."""

    evaluation_month: date
    customer: CustomerProfile
    transactions: list[Transaction] = Field(min_length=1, max_length=500)

    @field_validator("evaluation_month")
    @classmethod
    def evaluation_month_must_be_first_day(cls, value: date) -> date:
        if value.day != 1:
            raise ValueError("evaluation_month must be the first day of a month")
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
