"""
AgentState — the single source of truth for the agent loop.

The agent reads from and writes to this object each iteration.
It is never mutated by tools directly; tools return results and
the agent orchestrator merges them in.
"""
from __future__ import annotations
from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field
import time
import uuid

from .evidence import Evidence, ConflictRecord, MissingField
from .medications import ReconciliationReport, Medication
from .discharge_summary import DischargeSummary


class AgentPhase(str, Enum):
    PLANNING = "planning"
    EXTRACTING = "extracting"
    RECONCILING = "reconciling"
    VALIDATING = "validating"
    COMPOSING = "composing"
    ESCALATING = "escalating"
    COMPLETE = "complete"
    FAILED = "failed"


class ToolCall(BaseModel):
    """Record of a single tool invocation."""
    call_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    tool_name: str
    inputs: dict[str, Any]
    output: Optional[Any] = None
    error: Optional[str] = None
    success: bool = False
    duration_ms: float = 0.0
    retry_count: int = 0
    timestamp: float = Field(default_factory=time.time)


class AgentStep(BaseModel):
    """One iteration of the agent loop — used for observability traces."""
    step_number: int
    phase: AgentPhase
    reasoning: str
    selected_tool: Optional[str] = None
    tool_inputs: Optional[dict[str, Any]] = None
    tool_output_summary: Optional[str] = None
    decision: str
    next_action: str
    warnings: list[str] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)


class AgentState(BaseModel):
    """
    The agent's complete working memory.

    Designed so the orchestrator can checkpoint and resume if needed.
    """
    # Session identity
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str = "unknown"
    source_documents: list[str] = Field(default_factory=list)  # file paths

    # Loop control
    iteration: int = 0
    max_iterations: int = 15
    phase: AgentPhase = AgentPhase.PLANNING
    finished: bool = False
    timed_out: bool = False

    # Evidence registry — immutable facts with provenance
    evidence_registry: list[Evidence] = Field(default_factory=list)

    # Extracted clinical data (raw, before composition)
    extracted_demographics: dict[str, Any] = Field(default_factory=dict)
    extracted_diagnoses: list[dict[str, Any]] = Field(default_factory=list)
    extracted_procedures: list[dict[str, Any]] = Field(default_factory=list)
    extracted_hospital_course: list[dict[str, Any]] = Field(default_factory=list)
    extracted_allergies: list[dict[str, Any]] = Field(default_factory=list)
    extracted_follow_up: list[dict[str, Any]] = Field(default_factory=list)
    extracted_discharge_condition: Optional[dict[str, Any]] = None
    extracted_pending_results: list[dict[str, Any]] = Field(default_factory=list)

    # Medications
    admission_medications: list[Medication] = Field(default_factory=list)
    discharge_medications: list[Medication] = Field(default_factory=list)
    reconciliation_report: Optional[ReconciliationReport] = None

    # Conflicts and missing data
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    missing_fields: list[MissingField] = Field(default_factory=list)

    # Safety
    hallucination_warnings: list[str] = Field(default_factory=list)
    clinician_review_flags: list[str] = Field(default_factory=list)
    drug_interaction_alerts: list[str] = Field(default_factory=list)
    escalated: bool = False

    # Which documents have been processed
    processed_documents: list[str] = Field(default_factory=list)
    pending_tasks: list[str] = Field(default_factory=list)

    # Tool call history
    tool_calls: list[ToolCall] = Field(default_factory=list)

    # Step trace — observability
    steps: list[AgentStep] = Field(default_factory=list)

    # Final output (set when phase == COMPLETE)
    final_summary: Optional[DischargeSummary] = None

    def add_evidence(self, ev: Evidence) -> None:
        self.evidence_registry.append(ev)

    def add_conflict(self, conflict: ConflictRecord) -> None:
        self.conflicts.append(conflict)
        self.clinician_review_flags.append(
            f"CONFLICT in '{conflict.field}': {conflict.source_1_doc} vs {conflict.source_2_doc}"
        )

    def add_missing(self, field: str, reason: str = "", tools: list[str] | None = None) -> None:
        self.missing_fields.append(MissingField(
            field_name=field,
            reason=reason or "No supporting evidence found",
            attempted_tools=tools or [],
        ))

    def add_flag(self, flag: str) -> None:
        if flag not in self.clinician_review_flags:
            self.clinician_review_flags.append(flag)

    def add_warning(self, warning: str) -> None:
        if warning not in self.hallucination_warnings:
            self.hallucination_warnings.append(warning)

    def unprocessed_documents(self) -> list[str]:
        return [d for d in self.source_documents if d not in self.processed_documents]

    def has_evidence_for(self, field: str) -> bool:
        return any(ev.fact.lower() == field.lower() for ev in self.evidence_registry)

    def evidence_for_field(self, field: str) -> list[Evidence]:
        return [ev for ev in self.evidence_registry if field.lower() in ev.fact.lower()]
