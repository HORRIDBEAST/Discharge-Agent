"""
Summary Composer Tool.

Takes the fully validated AgentState and assembles the final
DischargeSummary. This is the LAST step — it only runs after:

  1. All documents have been processed
  2. Conflicts are recorded (not resolved)
  3. Medications are reconciled
  4. Evidence is validated

The composer never infers missing data — it uses the MISSING sentinel.
"""
from __future__ import annotations
from typing import Any, Optional

from app.tools.base import BaseTool, register_tool
from app.models.discharge_summary import DischargeSummary, PatientDemographics, MISSING, PENDING
from app.models.evidence import ConflictRecord, MissingField
from app.models.medications import ReconciliationReport
from app.utils.logger import get_logger

logger = get_logger(__name__)


@register_tool
class SummaryComposerTool(BaseTool):
    name = "summary_composer"
    description = "Assemble the final structured discharge summary from validated agent state."

    async def _execute(
        self,
        patient_id: str,
        extracted_demographics: dict[str, Any],
        extracted_diagnoses: list[dict[str, Any]],
        extracted_procedures: list[dict[str, Any]],
        extracted_hospital_course: list[dict[str, Any]],
        extracted_allergies: list[dict[str, Any]],
        extracted_follow_up: list[dict[str, Any]],
        extracted_discharge_condition: Optional[dict[str, Any]],
        extracted_pending_results: list[dict[str, Any]],
        reconciliation_report: Optional[dict[str, Any]],
        conflicts: list[dict[str, Any]],
        missing_fields: list[dict[str, Any]],
        clinician_review_flags: list[str],
        hallucination_warnings: list[str],
        drug_interaction_alerts: list[str],
        evidence_registry: list[dict[str, Any]],
        agent_iterations: int,
        escalated: bool,
    ) -> dict[str, Any]:

        # --- Demographics ---
        demo_data = extracted_demographics
        patient = PatientDemographics(
            name=_get(demo_data, "name"),
            date_of_birth=_get(demo_data, "date_of_birth"),
            mrn=_get(demo_data, "mrn"),
            sex=_get(demo_data, "sex"),
            age=_get(demo_data, "age"),
        )

        # --- Admission / Discharge dates ---
        # Look in discharge info extractions first
        admit_date = MISSING
        discharge_date = MISSING
        for info in extracted_discharge_condition and [extracted_discharge_condition] or []:
            if info:
                if info.get("admission_date"):
                    admit_date = info["admission_date"]
                if info.get("discharge_date"):
                    discharge_date = info["discharge_date"]

        # --- Diagnoses ---
        principal_dx = MISSING
        secondary_dxs: list[str] = []

        # Check for conflict on principal diagnosis
        dx_conflict = any(c.get("field") == "principal_diagnosis" for c in conflicts)
        if dx_conflict:
            principal_dx = "CONFLICT DETECTED — clinician review required"
        else:
            principal_entries = [d for d in extracted_diagnoses if d.get("type") == "principal"]
            if principal_entries:
                # Pick highest confidence, warn if multiple
                principal_entries.sort(key=lambda x: x.get("confidence", 0), reverse=True)
                principal_dx = principal_entries[0]["diagnosis"]
                if len(principal_entries) > 1:
                    clinician_review_flags.append(
                        f"Multiple principal diagnoses found — using highest confidence: '{principal_dx}'"
                    )

        for d in extracted_diagnoses:
            if d.get("type") == "secondary" and d.get("diagnosis"):
                secondary_dxs.append(d["diagnosis"])

        # --- Hospital Course ---
        hospital_course = MISSING
        if extracted_hospital_course:
            course_entries = [h for h in extracted_hospital_course if h.get("hospital_course")]
            if course_entries:
                # Use the most detailed one
                course_entries.sort(key=lambda x: len(str(x.get("hospital_course", ""))), reverse=True)
                hospital_course = course_entries[0]["hospital_course"]

        # --- Procedures ---
        procedures = list({
            p["procedure"] for p in extracted_procedures if p.get("procedure")
        })

        # --- Allergies ---
        allergies = []
        for a in extracted_allergies:
            allergen = a.get("allergen", "")
            reaction = a.get("reaction", "")
            if allergen:
                entry = allergen
                if reaction:
                    entry += f" ({reaction})"
                allergies.append(entry)
        allergies = list(dict.fromkeys(allergies))  # dedup preserving order

        # --- Discharge condition ---
        discharge_condition = MISSING
        if extracted_discharge_condition:
            dc = extracted_discharge_condition.get("discharge_condition")
            if dc:
                discharge_condition = dc

        # --- Follow-up ---
        follow_up = []
        for fi in extracted_follow_up:
            instr = fi.get("instruction", "")
            if instr:
                follow_up.append(instr)
        follow_up = list(dict.fromkeys(follow_up))

        # --- Pending results ---
        pending: list[str] = []
        for pr in extracted_pending_results:
            result_text = pr.get("result") or pr.get("text", "")
            if result_text:
                pending.append(result_text)
        pending = list(dict.fromkeys(pending))

        # --- Add drug interaction alerts to flags ---
        all_flags = list(clinician_review_flags)
        for alert in drug_interaction_alerts:
            flag = f"DRUG INTERACTION ALERT: {alert}"
            if flag not in all_flags:
                all_flags.append(flag)

        # --- Reconciliation report ---
        recon = None
        if reconciliation_report:
            try:
                recon = ReconciliationReport(**reconciliation_report)
            except Exception:
                logger.warning("recon_deserialise_failed")

        # --- Evidence map (field -> evidence IDs) ---
        evidence_map: dict[str, list[str]] = {}
        for ev in evidence_registry:
            field = ev.get("fact", "unknown")
            ev_id = ev.get("evidence_id", "")
            evidence_map.setdefault(field, []).append(ev_id)

        summary = DischargeSummary(
            patient=patient,
            admission_date=admit_date,
            discharge_date=discharge_date,
            principal_diagnosis=principal_dx,
            secondary_diagnoses=secondary_dxs,
            hospital_course=hospital_course,
            procedures=procedures,
            reconciliation_report=recon,
            allergies=allergies,
            discharge_condition=discharge_condition,
            follow_up_instructions=follow_up,
            pending_results=pending,
            conflicts_detected=[ConflictRecord(**c) for c in conflicts],
            missing_fields=[MissingField(**m) for m in missing_fields],
            clinician_review_flags=all_flags,
            hallucination_warnings=hallucination_warnings,
            evidence_map=evidence_map,
            agent_iterations=agent_iterations,
            escalated=escalated,
        )

        logger.info(
            "summary_composed",
            patient_id=patient_id,
            conflicts=len(conflicts),
            missing=len(missing_fields),
            flags=len(all_flags),
            escalated=escalated,
        )

        return summary.model_dump()


def _get(d: dict[str, Any], key: str) -> Optional[str]:
    """Extract a field, returning MISSING sentinel if absent or null."""
    val = d.get(key)
    if not val or str(val).strip().lower() in ("null", "none", "missing", ""):
        return MISSING
    return str(val)
