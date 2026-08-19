"""Teaching endpoint for bounded, read-only local tools."""

from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from compliance_agent.config import Settings, get_settings
from compliance_agent.domain import InvestigationCase
from compliance_agent.services.tools import (
    ToolCallLimitExceededError,
    ToolCallOutcome,
    ToolCallRequest,
    execute_tool_calls,
)

router = APIRouter(prefix="/tools", tags=["tools"])
SettingsDependency = Annotated[Settings, Depends(get_settings)]


class ToolExecutionRequest(BaseModel):
    case: InvestigationCase
    calls: list[ToolCallRequest] = Field(min_length=1, max_length=10)


class ToolExecutionResponse(BaseModel):
    status: Literal["completed"]
    is_mock: Literal[True] = True
    outcomes: list[ToolCallOutcome]


@router.post("/execute", response_model=ToolExecutionResponse)
async def execute_read_only_tools(
    request: ToolExecutionRequest,
    settings: SettingsDependency,
) -> ToolExecutionResponse:
    """Execute only fixed portfolio tools; no model call occurs on this endpoint."""

    try:
        outcomes = execute_tool_calls(
            request.case,
            request.calls,
            max_tool_calls=settings.agent_max_tool_calls,
        )
    except ToolCallLimitExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="The tool-call limit was exceeded.",
        ) from exc
    return ToolExecutionResponse(status="completed", outcomes=outcomes)
