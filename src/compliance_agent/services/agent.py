"""Bounded investigation workflow connecting model requests to local tools."""

import asyncio
from dataclasses import dataclass
from typing import Literal, Protocol

from compliance_agent.domain import InvestigationCase
from compliance_agent.services.recommendations import (
    ProviderRecommendation,
    RecommendationInvalidOutputError,
    RecommendationService,
    RecommendationTimeoutError,
    TokenUsage,
)
from compliance_agent.services.tools import (
    ToolCallLimitExceededError,
    ToolCallOutcome,
    ToolCallRequest,
    execute_tool_calls,
)


@dataclass(frozen=True)
class ToolPlanningResult:
    calls: list[ToolCallRequest]
    response_id: str
    model: str
    latency_ms: int
    usage: TokenUsage


class ToolPlanner(Protocol):
    async def plan(self, case: InvestigationCase) -> ToolPlanningResult:
        """Request zero or more structured calls without executing them."""


@dataclass(frozen=True)
class InvestigationResult:
    status: Literal["completed", "partial"]
    planning: ToolPlanningResult
    tool_outcomes: list[ToolCallOutcome]
    recommendation: ProviderRecommendation


class InvestigationAgentService:
    """Run one tool-planning step, bounded execution, and a structured final step."""

    def __init__(
        self,
        planner: ToolPlanner,
        recommendation_service: RecommendationService,
        *,
        max_tool_calls: int,
        max_elapsed_seconds: float = 60.0,
    ) -> None:
        self._planner = planner
        self._recommendation_service = recommendation_service
        self._max_tool_calls = max_tool_calls
        self._max_elapsed_seconds = max_elapsed_seconds

    async def run(self, case: InvestigationCase) -> InvestigationResult:
        try:
            async with asyncio.timeout(self._max_elapsed_seconds):
                return await self._run_bounded(case)
        except TimeoutError as exc:
            raise RecommendationTimeoutError from exc

    async def _run_bounded(self, case: InvestigationCase) -> InvestigationResult:
        planning = await self._planner.plan(case)
        try:
            outcomes = execute_tool_calls(
                case,
                planning.calls,
                max_tool_calls=self._max_tool_calls,
            )
        except ToolCallLimitExceededError as exc:
            raise RecommendationInvalidOutputError("tool_call_limit_exceeded") from exc

        recommendation = await self._recommendation_service.recommend(
            case,
            tool_outcomes=outcomes,
        )
        return InvestigationResult(
            status=(
                "partial"
                if any(outcome.status == "rejected" for outcome in outcomes)
                else "completed"
            ),
            planning=planning,
            tool_outcomes=outcomes,
            recommendation=recommendation,
        )
