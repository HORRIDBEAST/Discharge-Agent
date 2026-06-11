"""
Simulated Doctor Reviewer — Part 2.

Applies a consistent (hidden to the agent) editing policy to discharge
summary drafts, producing (draft, edited) pairs for learning.

Editing policy (hidden from agent):
  1. Replace verbose hospital course with concise clinical language
  2. Add standard follow-up timeframes where missing
  3. Normalise medication formatting (dose + frequency + route)
  4. Add explicit INR/anticoagulation monitoring instructions where applicable
  5. Clarify pending result ownership (who follows up)
  6. Remove redundant clinician flags for well-documented facts

This simulates a senior physician who has consistent stylistic and
clinical preferences — the kind of signal we want the agent to learn.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional

from app.models.discharge_summary import DischargeSummary, MISSING
from app.models.medications import ReconciliationReport, MedChangeType


@dataclass
class EditDiff:
    """Records a single edit made by the simulated reviewer."""
    field: str
    original: str
    edited: str
    edit_type: str  # "rewrite", "addition", "removal", "clarification"
    reason: str


class SimulatedReviewer:
    """
    A deterministic simulated physician who applies consistent editing
    rules to discharge summary drafts. Used to generate training signal.

    The policy is intentionally non-trivial so the agent must learn
    real improvements, not just style matching.
    """

    def review(self, summary: DischargeSummary) -> tuple[DischargeSummary, list[EditDiff]]:
        """
        Apply reviewer edits to a summary.
        Returns (edited_summary, list_of_diffs).
        """
        edited = summary.model_copy(deep=True)
        diffs: list[EditDiff] = []

        diffs.extend(self._edit_hospital_course(edited))
        diffs.extend(self._edit_follow_up(edited))
        diffs.extend(self._edit_pending_results(edited))
        diffs.extend(self._edit_medications(edited))
        diffs.extend(self._edit_flags(edited))

        return edited, diffs

    def _edit_hospital_course(self, summary: DischargeSummary) -> list[EditDiff]:
        """
        Policy: Hospital course should start with patient descriptor,
        then admission reason, then key events in order.
        Add 'Patient was discharged in X condition' at end if missing.
        """
        diffs = []
        hc = summary.hospital_course
        if not hc or hc == MISSING:
            return diffs

        original = hc
        edited = hc

        # Ensure it ends with discharge condition if known
        if summary.discharge_condition and summary.discharge_condition != MISSING:
            if "discharged" not in edited.lower():
                edited = edited.rstrip() + f" Patient was discharged in {summary.discharge_condition.lower()} condition."
                diffs.append(EditDiff(
                    field="hospital_course",
                    original=original,
                    edited=edited,
                    edit_type="addition",
                    reason="Added discharge condition to hospital course narrative",
                ))

        # Trim excessive length (> 600 chars → truncate at sentence boundary)
        if len(edited) > 600:
            sentences = re.split(r'(?<=[.!?])\s+', edited)
            truncated = ""
            for s in sentences:
                if len(truncated) + len(s) > 600:
                    break
                truncated += s + " "
            truncated = truncated.strip()
            if truncated != edited:
                diffs.append(EditDiff(
                    field="hospital_course",
                    original=edited,
                    edited=truncated,
                    edit_type="rewrite",
                    reason="Condensed hospital course to appropriate length",
                ))
                edited = truncated

        summary.hospital_course = edited
        return diffs

    def _edit_follow_up(self, summary: DischargeSummary) -> list[EditDiff]:
        """
        Policy: Follow-up instructions should have explicit timeframes.
        Add 'within X weeks' if timeframe is absent.
        """
        diffs = []
        updated: list[str] = []

        for instr in summary.follow_up_instructions:
            has_timeframe = bool(re.search(
                r'\d+\s*(day|week|month|hour)|in\s+\d+|within\s+\d+',
                instr, re.IGNORECASE
            ))
            if not has_timeframe and "gp" in instr.lower():
                edited_instr = instr + " (within 1 week)"
                diffs.append(EditDiff(
                    field="follow_up_instructions",
                    original=instr,
                    edited=edited_instr,
                    edit_type="clarification",
                    reason="Added explicit timeframe to GP follow-up instruction",
                ))
                updated.append(edited_instr)
            elif not has_timeframe and "cardiol" in instr.lower():
                edited_instr = instr + " (within 2 weeks)"
                diffs.append(EditDiff(
                    field="follow_up_instructions",
                    original=instr,
                    edited=edited_instr,
                    edit_type="clarification",
                    reason="Added explicit timeframe to cardiology follow-up",
                ))
                updated.append(edited_instr)
            else:
                updated.append(instr)

        summary.follow_up_instructions = updated
        return diffs

    def _edit_pending_results(self, summary: DischargeSummary) -> list[EditDiff]:
        """
        Policy: Each pending result must name who is responsible for follow-up.
        """
        diffs = []
        updated: list[str] = []

        for result in summary.pending_results:
            if "gp" not in result.lower() and "clinic" not in result.lower() and "follow" not in result.lower():
                edited = result + " — GP to follow up and notify patient"
                diffs.append(EditDiff(
                    field="pending_results",
                    original=result,
                    edited=edited,
                    edit_type="clarification",
                    reason="Added responsibility for pending result follow-up",
                ))
                updated.append(edited)
            else:
                updated.append(result)

        summary.pending_results = updated
        return diffs

    def _edit_medications(self, summary: DischargeSummary) -> list[EditDiff]:
        """
        Policy: If anticoagulant (warfarin) in discharge meds, ensure INR
        monitoring instruction exists in follow-up.
        """
        diffs = []
        if not summary.reconciliation_report:
            return diffs

        has_warfarin = any(
            "warfarin" in m.name.lower()
            for m in summary.reconciliation_report.discharge_medications
        )

        if has_warfarin:
            inr_mentioned = any(
                "inr" in fi.lower() or "anticoag" in fi.lower()
                for fi in summary.follow_up_instructions
            )
            if not inr_mentioned:
                new_instruction = "INR check within 3 days of discharge (warfarin dose adjustment may be required)"
                diffs.append(EditDiff(
                    field="follow_up_instructions",
                    original="",
                    edited=new_instruction,
                    edit_type="addition",
                    reason="Added INR monitoring instruction for patient on warfarin",
                ))
                summary.follow_up_instructions.insert(0, new_instruction)

        return diffs

    def _edit_flags(self, summary: DischargeSummary) -> list[EditDiff]:
        """
        Policy: Remove redundant flags for facts that are well-documented.
        Keep only flags that require genuine clinician action.
        """
        diffs = []
        original_flags = list(summary.clinician_review_flags)
        filtered = []

        for flag in original_flags:
            # Remove flags about facts that appear fully resolved in the summary
            if "medication reconciliation" in flag.lower() and summary.reconciliation_report and summary.reconciliation_report.reconciliation_complete:
                diffs.append(EditDiff(
                    field="clinician_review_flags",
                    original=flag,
                    edited="",
                    edit_type="removal",
                    reason="Removed reconciliation flag — reconciliation is complete",
                ))
                continue
            filtered.append(flag)

        summary.clinician_review_flags = filtered
        return diffs
