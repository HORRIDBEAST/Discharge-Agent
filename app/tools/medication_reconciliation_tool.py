"""
Medication Reconciliation Tool.

Compares admission vs discharge medication lists and produces a
structured diff. Any change without a documented reason is flagged
for clinician review — not silently resolved.
"""
from __future__ import annotations
from typing import Any

from app.tools.base import BaseTool, register_tool
from app.models.medications import (
    Medication, MedicationChange, MedChangeType, ReconciliationReport
)
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _normalise_name(name: str) -> str:
    """Lower-case, strip spaces — for fuzzy matching."""
    return name.lower().strip().replace("-", " ")


@register_tool
class MedicationReconciliationTool(BaseTool):
    name = "medication_reconciliation"
    description = "Compare admission vs discharge medications and flag changes needing clinician review."

    async def _execute(
        self,
        admission_meds: list[dict[str, Any]],
        discharge_meds: list[dict[str, Any]],
    ) -> dict[str, Any]:

        admit = [Medication(**m) if isinstance(m, dict) else m for m in admission_meds]
        discharge = [Medication(**m) if isinstance(m, dict) else m for m in discharge_meds]

        changes: list[MedicationChange] = []
        unresolved: list[str] = []

        admit_map = {_normalise_name(m.name): m for m in admit}
        discharge_map = {_normalise_name(m.name): m for m in discharge}

        all_names = set(admit_map.keys()) | set(discharge_map.keys())

        for name in all_names:
            in_admit = name in admit_map
            in_discharge = name in discharge_map

            if in_admit and in_discharge:
                am = admit_map[name]
                dm = discharge_map[name]

                # Check for dose change
                if am.dose and dm.dose and am.dose != dm.dose:
                    reason_present = bool(dm.indication)
                    change = MedicationChange(
                        medication_name=dm.name,
                        change_type=MedChangeType.DOSE_CHANGED,
                        admission_value=am.dose,
                        discharge_value=dm.dose,
                        reason_documented=reason_present,
                        reason_text=dm.indication,
                        flag_for_reconciliation=not reason_present,
                        clinician_note=(
                            "" if reason_present
                            else f"Dose changed from {am.dose} to {dm.dose} — reason not documented"
                        ),
                    )
                    changes.append(change)
                    if not reason_present:
                        unresolved.append(f"Dose change for {dm.name}: no reason documented")

                # Check for frequency change
                elif am.frequency and dm.frequency and am.frequency != dm.frequency:
                    reason_present = bool(dm.indication)
                    change = MedicationChange(
                        medication_name=dm.name,
                        change_type=MedChangeType.FREQUENCY_CHANGED,
                        admission_value=am.frequency,
                        discharge_value=dm.frequency,
                        reason_documented=reason_present,
                        reason_text=dm.indication,
                        flag_for_reconciliation=not reason_present,
                        clinician_note=(
                            "" if reason_present
                            else f"Frequency changed — reason not documented"
                        ),
                    )
                    changes.append(change)
                    if not reason_present:
                        unresolved.append(f"Frequency change for {dm.name}: no reason documented")

                else:
                    changes.append(MedicationChange(
                        medication_name=dm.name,
                        change_type=MedChangeType.UNCHANGED,
                    ))

            elif in_admit and not in_discharge:
                # Discontinued
                am = admit_map[name]
                reason_present = False  # No discharge info = no reason known
                change = MedicationChange(
                    medication_name=am.name,
                    change_type=MedChangeType.DISCONTINUED,
                    admission_value=f"{am.dose or ''} {am.frequency or ''}".strip(),
                    reason_documented=reason_present,
                    flag_for_reconciliation=True,
                    clinician_note=f"{am.name} discontinued — reason not documented in discharge record",
                )
                changes.append(change)
                unresolved.append(f"{am.name} discontinued without documented reason")

            elif not in_admit and in_discharge:
                # Added
                dm = discharge_map[name]
                reason_present = bool(dm.indication)
                change = MedicationChange(
                    medication_name=dm.name,
                    change_type=MedChangeType.ADDED,
                    discharge_value=f"{dm.dose or ''} {dm.frequency or ''}".strip(),
                    reason_documented=reason_present,
                    reason_text=dm.indication,
                    flag_for_reconciliation=not reason_present,
                    clinician_note=(
                        "" if reason_present
                        else f"{dm.name} added — indication not documented"
                    ),
                )
                changes.append(change)
                if not reason_present:
                    unresolved.append(f"{dm.name} added without documented indication")

        report = ReconciliationReport(
            admission_medications=admit,
            discharge_medications=discharge,
            changes=changes,
            unresolved_flags=unresolved,
            reconciliation_complete=len(unresolved) == 0,
            notes=f"{len(unresolved)} items require clinician review" if unresolved else "Reconciliation complete",
        )

        logger.info(
            "reconciliation_complete",
            total_changes=len(changes),
            unresolved=len(unresolved),
        )

        return report.model_dump()
