"""Bounded AI investigation workflow route."""

from collections.abc import AsyncIterator
from typing import Annotated, Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from compliance_agent.api.routes.recommendations import RecommendationResponse, UsageResponse
from compliance_agent.config import Settings, get_settings
from compliance_agent.domain import InvestigationCase
from compliance_agent.providers.openai_recommendations import (
    GroqRecommendationProvider,
    OpenAIRecommendationProvider,
)
from compliance_agent.providers.tool_planner import CompatibleToolPlanner
from compliance_agent.services.agent import InvestigationAgentService
from compliance_agent.services.recommendations import RecommendationService
from compliance_agent.services.tools import ToolCallOutcome

router = APIRouter(prefix="/investigations", tags=["investigations"])


class PlanningResponse(BaseModel):
    response_id: str
    model: str
    latency_ms: int = Field(ge=0)
    requested_tool_count: int = Field(ge=0)
    usage: UsageResponse


class InvestigationResponse(BaseModel):
    status: Literal["completed", "partial"]
    planning: PlanningResponse
    tool_outcomes: list[ToolCallOutcome]
    recommendation: RecommendationResponse
    human_review_required: Literal[True] = True


async def get_investigation_agent_service(
    settings: Annotated[Settings, Depends(get_settings)],
) -> AsyncIterator[InvestigationAgentService]:
    planner = CompatibleToolPlanner(settings)
    provider_type = (
        GroqRecommendationProvider
        if settings.ai_provider == "groq"
        else OpenAIRecommendationProvider
    )
    recommendation_provider = provider_type(settings)
    recommendation_service = RecommendationService(
        recommendation_provider,
        max_output_retries=settings.ai_max_output_retries,
    )
    try:
        # This is the main entry point for the investigation agent service.
        # yield = “here is service for the route.” FastAPI pauses this function here, runs run_investigation, then comes back.
        # InvestigationAgentService is the conductor: planner + recommend + max tools + timeout.
        yield InvestigationAgentService(
            planner,
            recommendation_service,
            max_tool_calls=settings.agent_max_tool_calls,
            max_elapsed_seconds=settings.agent_max_elapsed_seconds,
        )
    finally:
        await planner.close()
        await recommendation_provider.close()


InvestigationServiceDependency = Annotated[
    InvestigationAgentService,
    Depends(get_investigation_agent_service),
]


@router.post("/run", response_model=InvestigationResponse)
async def run_investigation(
    case: InvestigationCase,
    service: InvestigationServiceDependency,
) -> InvestigationResponse:
    result = await service.run(case)
    planning = result.planning
    recommendation = result.recommendation
    return InvestigationResponse(
        status=result.status,
        planning=PlanningResponse(
            response_id=planning.response_id,
            model=planning.model,
            latency_ms=planning.latency_ms,
            requested_tool_count=len(planning.calls),
            usage=UsageResponse(
                input_tokens=planning.usage.input_tokens,
                output_tokens=planning.usage.output_tokens,
                total_tokens=planning.usage.total_tokens,
            ),
        ),
        tool_outcomes=result.tool_outcomes,
        recommendation=RecommendationResponse(
            status="completed",
            recommendation=recommendation.recommendation,
            response_id=recommendation.response_id,
            model=recommendation.model,
            latency_ms=recommendation.latency_ms,
            retry_count=recommendation.retry_count,
            usage=UsageResponse(
                input_tokens=recommendation.usage.input_tokens,
                output_tokens=recommendation.usage.output_tokens,
                total_tokens=recommendation.usage.total_tokens,
            ),
        ),
    )
