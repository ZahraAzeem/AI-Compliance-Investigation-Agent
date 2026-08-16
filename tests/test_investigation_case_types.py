"""Tests for typed investigation-case routing and alert coherence."""

from copy import deepcopy

import pytest
from pydantic import TypeAdapter, ValidationError

from compliance_agent.domain import InvestigationCase

CASE_ADAPTER = TypeAdapter(InvestigationCase)


def _base_case() -> dict[str, object]:
    return {
        "case_id": "CASE-TYPED-001",
        "priority": "high",
        "opened_at": "2026-08-15T10:00:00Z",
        "trigger_reason": "Synthetic portfolio alert requires review.",
        "customer": {
            "customer_id": "CUST-TYPED-001",
            "full_name": "Leila Noor",
            "date_of_birth": "1988-05-20",
            "country_of_residence": "AE",
            "nationality_codes": ["JO"],
            "occupation": "Consultant",
            "is_politically_exposed": False,
            "account_opened_at": "2023-01-12",
            "existing_risk_level": "high",
        },
        "transactions": [],
    }


def test_onboarding_case_routes_to_kyc_schema() -> None:
    case = _base_case()
    case.update(
        {
            "case_type": "kyc_review",
            "alert": {
                "alert_id": "ALT-KYC-001",
                "customer_id": "CUST-TYPED-001",
                "created_at": "2026-08-15T09:55:00Z",
                "risk_level": "high",
                "risk_score": 84,
                "risk_factors": ["Synthetic geography and occupation factors"],
                "required_artifacts": ["passport", "bank_statements_3_months"],
                "control_action": "documents_required",
                "summary": "Synthetic high-risk onboarding result.",
            },
            "artifact_reviews": [
                {"artifact": "passport", "status": "provided"},
                {"artifact": "bank_statements_3_months", "status": "requested"},
            ],
        }
    )

    validated = CASE_ADAPTER.validate_python(case)

    assert validated.case_type == "kyc_review"
    assert validated.alert.risk_level == "high"


def test_sanctions_case_requires_sanctions_category() -> None:
    case = _base_case()
    case.update(
        {
            "case_type": "sanctions_screening",
            "alert": {
                "alert_id": "ALT-SCREEN-001",
                "customer_id": "CUST-TYPED-001",
                "created_at": "2026-08-15T09:55:00Z",
                "subject_type": "sender",
                "subject_id": "CUST-TYPED-001",
                "match_categories": ["politically_exposed_person"],
                "match_score": 0.76,
                "matched_name": "Synthetic Candidate",
                "summary": "Synthetic screening candidate.",
            },
        }
    )

    with pytest.raises(ValidationError, match="must include a sanctions match"):
        CASE_ADAPTER.validate_python(case)


def test_adverse_media_case_accepts_matching_category() -> None:
    case = _base_case()
    case.update(
        {
            "case_type": "adverse_media",
            "alert": {
                "alert_id": "ALT-MEDIA-001",
                "customer_id": "CUST-TYPED-001",
                "created_at": "2026-08-15T09:55:00Z",
                "subject_type": "receiver",
                "subject_id": "RECIPIENT-001",
                "match_categories": ["adverse_media"],
                "match_score": 0.68,
                "matched_name": "Synthetic Candidate",
                "summary": "Synthetic adverse-media candidate.",
            },
        }
    )

    validated = CASE_ADAPTER.validate_python(case)

    assert validated.case_type == "adverse_media"
    assert validated.alert.provider == "mock_screening_provider"


def test_case_rejects_an_alert_for_another_customer() -> None:
    case = _base_case()
    case.update(
        {
            "case_type": "kyc_review",
            "alert": {
                "alert_id": "ALT-KYC-002",
                "customer_id": "DIFFERENT-CUSTOMER",
                "created_at": "2026-08-15T09:55:00Z",
                "risk_level": "medium",
                "risk_score": 55,
                "risk_factors": ["Synthetic risk factor"],
                "required_artifacts": ["questionnaire"],
                "control_action": "questionnaire_required",
                "summary": "Synthetic medium-risk onboarding result.",
            },
        }
    )
    mismatched = deepcopy(case)

    with pytest.raises(ValidationError, match="customer_id must match"):
        CASE_ADAPTER.validate_python(mismatched)
