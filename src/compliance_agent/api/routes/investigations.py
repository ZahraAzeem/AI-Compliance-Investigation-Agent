"""Bounded AI investigation routes for complete and streamed responses."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import suppress
from datetime import UTC, datetime
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field
from sse_starlette import EventSourceResponse, JSONServerSentEvent

from compliance_agent.api.errors import recommendation_error_details
from compliance_agent.api.routes.recommendations import RecommendationResponse, UsageResponse
from compliance_agent.config import Settings, get_settings
from compliance_agent.domain import InvestigationCase
from compliance_agent.providers.openai_recommendations import (
    GroqRecommendationProvider,
    OpenAIRecommendationProvider,
)
from compliance_agent.providers.tool_planner import CompatibleToolPlanner
from compliance_agent.services.agent import (
    InvestigationAgentService,
    InvestigationProgress,
    InvestigationResult,
)
from compliance_agent.services.recommendations import RecommendationError, RecommendationService
from compliance_agent.services.tools import ToolCallOutcome

router = APIRouter(prefix="/investigations", tags=["investigations"])
logger = logging.getLogger(__name__)


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


class StreamEventData(BaseModel):
    """Fields included in every public investigation stream event."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    sequence: int = Field(ge=1)
    request_id: str
    case_id: str
    emitted_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StartedEventData(StreamEventData):
    message: Literal["Investigation started."] = "Investigation started."


class RetrievalEventData(StreamEventData):
    """Reserved for the policy-retrieval milestone; not emitted yet."""

    status: Literal["started", "completed", "skipped"]
    result_count: int | None = Field(default=None, ge=0)


class ToolCallEventData(StreamEventData):
    tool_call_id: str
    tool_name: str
    status: Literal["completed", "rejected"]
    duration_ms: int = Field(ge=0)
    error_code: str | None = None


class ProgressEventData(StreamEventData):
    stage: Literal[
        "planning_started",
        "planning_completed",
        "recommendation_started",
    ]
    message: str
    requested_tool_count: int | None = Field(default=None, ge=0)


class CompletedEventData(StreamEventData):
    result: InvestigationResponse


class FailedEventData(StreamEventData):
    code: str
    message: str
    retryable: bool


PublicStreamEventData = (
    StartedEventData
    | RetrievalEventData
    | ToolCallEventData
    | ProgressEventData
    | CompletedEventData
    | FailedEventData
)

_STREAM_END = object()


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
    return build_investigation_response(result)


def build_investigation_response(result: InvestigationResult) -> InvestigationResponse:
    """Build the one validated final contract shared by JSON and SSE endpoints."""

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


async def investigation_event_generator(
    request: Request,
    case: InvestigationCase,
    service: InvestigationAgentService,
) -> AsyncIterator[JSONServerSentEvent]:
    """Run the agent in a cancellable task and yield safe application-owned events."""

    queue: asyncio.Queue[tuple[str, PublicStreamEventData] | object] = asyncio.Queue()
    sequence = 0
    request_id = getattr(request.state, "request_id", "unknown")

    async def publish(
        event_name: str,
        event_type: type[PublicStreamEventData],
        **event_fields: object,
    ) -> None:
        nonlocal sequence
        sequence += 1
        payload = event_type(
            sequence=sequence,
            request_id=request_id,
            case_id=case.case_id,
            **event_fields,
        )
        await queue.put((event_name, payload))

    async def publish_progress(progress: InvestigationProgress) -> None:
        if progress.stage == "tool_call_completed":
            outcome = progress.tool_outcome
            if outcome is None:  # pragma: no cover - internal invariant guard.
                raise RuntimeError("Tool progress is missing its outcome")
            await publish(
                "tool_call",
                ToolCallEventData,
                tool_call_id=outcome.call_id,
                tool_name=outcome.name,
                status=outcome.status,
                duration_ms=outcome.duration_ms,
                error_code=outcome.error_code,
            )
            return

        messages = {
            "planning_started": "Selecting approved information sources.",
            "planning_completed": "Approved information requests selected.",
            "recommendation_started": "Creating the structured recommendation.",
        }
        requested_tool_count = (
            len(progress.planning.calls)
            if progress.stage == "planning_completed" and progress.planning is not None
            else None
        )
        await publish(
            "progress",
            ProgressEventData,
            stage=progress.stage,
            message=messages[progress.stage],
            requested_tool_count=requested_tool_count,
        )

    async def run_and_publish() -> None:
        await publish("started", StartedEventData)
        try:
            result = await service.run(case, on_progress=publish_progress)
        except asyncio.CancelledError:
            raise
        except RecommendationError as exc:
            status_code, code, message = recommendation_error_details(exc)
            await publish(
                "failed",
                FailedEventData,
                code=code,
                message=message,
                retryable=status_code in {429, 503, 504},
            )
        except Exception:
            logger.exception("Streamed investigation failed: request_id=%s", request_id)
            await publish(
                "failed",
                FailedEventData,
                code="internal_error",
                message="The service encountered an unexpected error.",
                retryable=False,
            )
        else:
            await publish(
                "completed",
                CompletedEventData,
                result=build_investigation_response(result),
            )
        finally:
            await queue.put(_STREAM_END)

    worker = asyncio.create_task(run_and_publish())
    try:
        while True:
            if await request.is_disconnected():
                break
            queued = await queue.get()
            if queued is _STREAM_END:
                break
            event_name, payload = queued
            yield JSONServerSentEvent(
                event=event_name,
                id=str(payload.sequence),
                data=payload.model_dump(mode="json"),
            )
    finally:
        worker.cancel()
        with suppress(asyncio.CancelledError):
            await worker


@router.post(
    "/stream",
    response_class=EventSourceResponse,
    responses={200: {"content": {"text/event-stream": {}}}},
)
async def stream_investigation(
    request: Request,
    case: InvestigationCase,
    service: InvestigationServiceDependency,
    settings: Annotated[Settings, Depends(get_settings)],
) -> EventSourceResponse:
    """Stream safe progress and finish with the same validated result as ``/run``."""

    return EventSourceResponse(
        investigation_event_generator(request, case, service),
        ping=settings.sse_ping_interval_seconds,
        send_timeout=settings.sse_send_timeout_seconds,
    )
