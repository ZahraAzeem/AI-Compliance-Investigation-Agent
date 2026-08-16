"""Deterministic compliance-case validation routes."""

from typing import Literal

from fastapi import APIRouter, status
from pydantic import BaseModel

from compliance_agent.domain import InvestigationCase

router = APIRouter(prefix="/cases", tags=["cases"])


class CaseValidationResponse(BaseModel):
    status: Literal["valid"]
    case_id: str
    case_type: str
    alert_id: str
    transaction_count: int


@router.post(
    "/validate",
    response_model=CaseValidationResponse,
    status_code=status.HTTP_200_OK,
)
async def validate_case(case: InvestigationCase) -> CaseValidationResponse:
    """Validate and summarize a case without storing it or calling a model."""

    return CaseValidationResponse(
        status="valid",
        case_id=case.case_id,
        case_type=case.case_type,
        alert_id=case.alert.alert_id,
        transaction_count=len(case.transactions),
    )
