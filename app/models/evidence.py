"""
Evidence and provenance models.

Every clinical fact in the discharge summary must be backed by
one or more Evidence objects. If no evidence exists, the field
is marked MISSING — never fabricated.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field
import uuid


class ConfidenceLevel(str, Enum):
    HIGH = "high"        # Direct, unambiguous statement in source doc
    MEDIUM = "medium"    # Inferred from context, reasonable certainty
    LOW = "low"          # Uncertain, requires clinician verification


class Evidence(BaseModel):
    """Immutable provenance record for a single extracted clinical fact."""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    fact: str                         # The extracted clinical fact
    source_document: str              # Filename of source PDF
    page: int = 0                     # Page number (0-indexed)
    evidence_text: str                # Raw text supporting this fact
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    extraction_method: str = "llm"    # "llm", "regex", "ocr"
    section_in_doc: Optional[str] = None  # e.g. "Assessment", "Medications"

    model_config = {"frozen": True}  # Evidence is immutable once recorded


class ConflictRecord(BaseModel):
    """
    When two source documents disagree on a fact, we record the conflict
    rather than silently picking one source. This is escalated to the clinician.
    """
    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    field: str                          # Which discharge summary field conflicts
    source_1_doc: str
    source_1_value: str
    source_1_evidence_text: str
    source_2_doc: str
    source_2_value: str
    source_2_evidence_text: str
    status: str = "clinician_review_required"
    resolution: Optional[str] = None   # Set if clinician resolves it


class MissingField(BaseModel):
    """
    Represents a required discharge summary field that has no evidence.
    These are explicitly surfaced rather than filled with guesses.
    """
    field_name: str
    reason: str = "No supporting evidence found in source documents"
    attempted_tools: list[str] = Field(default_factory=list)
    escalated: bool = False
