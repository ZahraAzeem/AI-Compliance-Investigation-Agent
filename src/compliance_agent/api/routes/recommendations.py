"""Structured AI recommendation route."""

from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from compliance_agent.config import Settings, get_settings
from compliance_agent.domain import ComplianceRecommendation, InvestigationCase
from compliance_agent.providers.openai_recommendations import (
    GroqRecommendationProvider,
    OpenAIRecommendationProvider,
)
from compliance_agent.services.recommendations import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


class UsageResponse(BaseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class RecommendationResponse(BaseModel):
    status: Literal["completed"]
    recommendation: ComplianceRecommendation
    response_id: str
    model: str
    latency_ms: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    usage: UsageResponse


async def get_recommendation_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[RecommendationService]:
    """Build the provider boundary from validated application settings."""

    provider_type = (
        GroqRecommendationProvider
        if settings.ai_provider == "groq"
        else OpenAIRecommendationProvider
    )
    provider = provider_type(settings)
    try:
        yield RecommendationService(
            provider,
            max_output_retries=settings.ai_max_output_retries,
        )
    finally:
        await provider.close()


RecommendationServiceDependency = Annotated[
    RecommendationService,
    Depends(get_recommendation_service),
]


@router.post("", response_model=RecommendationResponse)
async def create_recommendation(
    case: InvestigationCase,
    service: RecommendationServiceDependency,
) -> RecommendationResponse:
    """Generate a structured, human-review-only recommendation for a validated case."""

    result = await service.recommend(case)
    return RecommendationResponse(
        status="completed",
        recommendation=result.recommendation,
        response_id=result.response_id,
        model=result.model,
        latency_ms=result.latency_ms,
        retry_count=result.retry_count,
        usage=UsageResponse(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            total_tokens=result.usage.total_tokens,
        ),
    )
