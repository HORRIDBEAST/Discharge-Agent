"""
Medication models for admission vs discharge reconciliation.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class MedChangeType(str, Enum):
    ADDED = "added"
    DISCONTINUED = "discontinued"
    DOSE_CHANGED = "dose_changed"
    FREQUENCY_CHANGED = "frequency_changed"
    ROUTE_CHANGED = "route_changed"
    UNCHANGED = "unchanged"


class Medication(BaseModel):
    name: str
    dose: Optional[str] = None
    frequency: Optional[str] = None
    route: Optional[str] = None
    indication: Optional[str] = None
    source_document: str = "unknown"
    raw_text: str = ""


class MedicationChange(BaseModel):
    """
    Structured diff entry produced by medication reconciliation.
    If reason_documented is False, the change is flagged for clinician review.
    """
    medication_name: str
    change_type: MedChangeType
    admission_value: Optional[str] = None   # dose/freq before
    discharge_value: Optional[str] = None  # dose/freq after
    reason_documented: bool = False
    reason_text: Optional[str] = None
    flag_for_reconciliation: bool = False
    clinician_note: str = ""


class ReconciliationReport(BaseModel):
    """Complete medication reconciliation output."""
    admission_medications: list[Medication] = Field(default_factory=list)
    discharge_medications: list[Medication] = Field(default_factory=list)
    changes: list[MedicationChange] = Field(default_factory=list)
    unresolved_flags: list[str] = Field(default_factory=list)
    reconciliation_complete: bool = False
    notes: str = ""
