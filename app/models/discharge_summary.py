"""
Discharge summary output schema.

Every field is Optional — missing fields are explicitly None or a
MISSING sentinel string rather than fabricated values.
"""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field
from .medications import ReconciliationReport
from .evidence import ConflictRecord, MissingField

MISSING = "MISSING — clinician review required"
PENDING = "PENDING — awaiting result"


class PatientDemographics(BaseModel):
    name: Optional[str] = MISSING
    date_of_birth: Optional[str] = MISSING
    mrn: Optional[str] = MISSING
    sex: Optional[str] = MISSING
    age: Optional[str] = MISSING


class DischargeSummary(BaseModel):
    """
    Structured discharge summary draft.

    This is ALWAYS a draft for clinician review — never auto-finalized.
    Fields without documentary evidence are marked MISSING or PENDING.
    """
    # Identity
    patient: PatientDemographics = Field(default_factory=PatientDemographics)
    admission_date: Optional[str] = MISSING
    discharge_date: Optional[str] = MISSING

    # Clinical
    principal_diagnosis: Optional[str] = MISSING
    secondary_diagnoses: list[str] = Field(default_factory=list)
    hospital_course: Optional[str] = MISSING
    procedures: list[str] = Field(default_factory=list)

    # Medications
    reconciliation_report: Optional[ReconciliationReport] = None
    allergies: list[str] = Field(default_factory=list)

    # Disposition
    discharge_condition: Optional[str] = MISSING
    follow_up_instructions: list[str] = Field(default_factory=list)
    pending_results: list[str] = Field(default_factory=list)

    # Safety metadata — always populated
    conflicts_detected: list[ConflictRecord] = Field(default_factory=list)
    missing_fields: list[MissingField] = Field(default_factory=list)
    clinician_review_flags: list[str] = Field(default_factory=list)
    hallucination_warnings: list[str] = Field(default_factory=list)

    # Provenance — maps field name -> list of evidence IDs
    evidence_map: dict[str, list[str]] = Field(default_factory=dict)

    # Processing metadata
    agent_iterations: int = 0
    escalated: bool = False
    draft_status: str = "draft_for_clinician_review"
