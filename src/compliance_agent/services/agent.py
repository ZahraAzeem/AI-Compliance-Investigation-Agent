"""Bounded investigation workflow connecting model requests to local tools."""

import asyncio
from collections.abc import Awaitable, Callable
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


ProgressStage = Literal[
    "planning_started",
    "planning_completed",
    "tool_call_completed",
    "recommendation_started",
]


@dataclass(frozen=True)
class InvestigationProgress:
    """Application-owned progress that is safe to expose without model reasoning."""

    stage: ProgressStage
    planning: ToolPlanningResult | None = None
    tool_outcome: ToolCallOutcome | None = None


ProgressCallback = Callable[[InvestigationProgress], Awaitable[None]]


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

    async def run(
        self,
        case: InvestigationCase,
        *,
        on_progress: ProgressCallback | None = None,
    ) -> InvestigationResult:
        try:
            async with asyncio.timeout(self._max_elapsed_seconds):
                return await self._run_bounded(case, on_progress=on_progress)
        except TimeoutError as exc:
            raise RecommendationTimeoutError from exc

    async def _run_bounded(
        self,
        case: InvestigationCase,
        *,
        on_progress: ProgressCallback | None,
    ) -> InvestigationResult:
        await _emit_progress(
            on_progress,
            InvestigationProgress(stage="planning_started"),
        )
        planning = await self._planner.plan(case)
        await _emit_progress(
            on_progress,
            InvestigationProgress(stage="planning_completed", planning=planning),
        )
        try:
            outcomes = execute_tool_calls(
                case,
                planning.calls,
                max_tool_calls=self._max_tool_calls,
            )
        except ToolCallLimitExceededError as exc:
            raise RecommendationInvalidOutputError("tool_call_limit_exceeded") from exc

        for outcome in outcomes:
            await _emit_progress(
                on_progress,
                InvestigationProgress(
                    stage="tool_call_completed",
                    tool_outcome=outcome,
                ),
            )

        await _emit_progress(
            on_progress,
            InvestigationProgress(stage="recommendation_started"),
        )
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


async def _emit_progress(
    callback: ProgressCallback | None,
    progress: InvestigationProgress,
) -> None:
    if callback is not None:
        await callback(progress)
