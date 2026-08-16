"""Provider-neutral recommendation orchestration and safety validation."""

from dataclasses import dataclass, replace
from typing import Protocol

from compliance_agent.domain import (
    ComplianceRecommendation,
    EvidenceSource,
    InvestigationCase,
)


class RecommendationError(Exception):
    """Base class for expected recommendation failures safe to classify publicly."""


class RecommendationConfigurationError(RecommendationError):
    """Required AI provider configuration is missing or unusable."""


class RecommendationAuthenticationError(RecommendationError):
    """The AI provider rejected configured credentials."""


class RecommendationRateLimitError(RecommendationError):
    """The AI provider temporarily rejected the request due to rate limits."""


class RecommendationQuotaError(RecommendationError):
    """The configured AI project has no available API quota."""


class RecommendationTimeoutError(RecommendationError):
    """The AI provider did not complete within the configured time boundary."""


class RecommendationUnavailableError(RecommendationError):
    """The AI provider could not complete the request."""


class RecommendationRefusalError(RecommendationError):
    """The AI provider declined to produce the requested recommendation."""


class RecommendationInvalidOutputError(RecommendationError):
    """The provider response failed application-owned output checks."""

    def __init__(self, reason: str = "unspecified") -> None:
        super().__init__()
        self.reason = reason


@dataclass(frozen=True)
class TokenUsage:
    """Safe token counts reported by the provider."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class ProviderRecommendation:
    """Validated provider result plus safe operational metadata."""

    recommendation: ComplianceRecommendation
    response_id: str
    model: str
    latency_ms: int
    retry_count: int
    usage: TokenUsage


class RecommendationProvider(Protocol):
    """Port implemented by an AI provider adapter and test fakes."""

    async def recommend(self, case: InvestigationCase) -> ProviderRecommendation:
        """Return a schema-validated recommendation for one validated case."""


class RecommendationService:
    """Apply business-owned checks after the provider validates the response shape."""

    def __init__(
        self,
        provider: RecommendationProvider,
        *,
        max_output_retries: int = 1,
    ) -> None:
        self._provider = provider
        self._max_output_retries = max_output_retries

    async def recommend(self, case: InvestigationCase) -> ProviderRecommendation:
        output_retry_count = 0
        while True:
            try:
                result = await self._provider.recommend(case)
                self._validate_case_grounding(case, result.recommendation)
            except RecommendationInvalidOutputError:
                if output_retry_count >= self._max_output_retries:
                    raise
                output_retry_count += 1
                continue

            return replace(
                result,
                retry_count=result.retry_count + output_retry_count,
            )

    @staticmethod
    def _validate_case_grounding(
        case: InvestigationCase,
        recommendation: ComplianceRecommendation,
    ) -> None:
        if recommendation.case_id != case.case_id:
            raise RecommendationInvalidOutputError("case_id_mismatch")

        allowed_case_source_ids = {
            case.case_id,
            case.alert.alert_id,
            case.customer.customer_id,
            *(transaction.transaction_id for transaction in case.transactions),
        }
        for evidence in recommendation.evidence:
            if evidence.source is not EvidenceSource.CASE_INPUT:
                raise RecommendationInvalidOutputError("unsupported_evidence_source")
            if evidence.source_id not in allowed_case_source_ids:
                raise RecommendationInvalidOutputError("unknown_evidence_id")

        # Policy retrieval is intentionally not implemented until the RAG milestone.
        if recommendation.policy_citations:
            raise RecommendationInvalidOutputError("policy_not_available")
