"""
Evaluation metrics for Part 2 — Learning from Doctor Edits.

Reward signal:
  1. Normalised edit distance per section (lower edit distance = better)
  2. Section accuracy (1.0 if no edits made, 0.0 if completely rewritten)
  3. Flag reduction rate (fewer unnecessary flags = better)
  4. Pending result ownership rate (fraction of pending results with named owner)

Overall score = weighted average across sections.
"""
from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional

from app.models.discharge_summary import DischargeSummary, MISSING


@dataclass
class SectionScore:
    section: str
    draft_text: str
    edited_text: str
    edit_distance: int
    normalised_edit_distance: float  # 0.0 (identical) to 1.0 (completely different)
    section_accuracy: float          # 1 - normalised_edit_distance


@dataclass
class EvaluationResult:
    patient_id: str
    iteration: int
    section_scores: list[SectionScore] = field(default_factory=list)
    overall_edit_distance: float = 0.0
    overall_accuracy: float = 0.0
    flag_count: int = 0
    missing_field_count: int = 0
    pending_with_owner: float = 0.0
    composite_score: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def levenshtein_distance(a: str, b: str) -> int:
    """Standard Levenshtein distance at character level."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # Use two-row DP for memory efficiency
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1] + [0] * len(b)
        for j, cb in enumerate(b):
            curr[j + 1] = min(
                prev[j + 1] + 1,    # deletion
                curr[j] + 1,        # insertion
                prev[j] + (0 if ca == cb else 1),  # substitution
            )
        prev = curr
    return prev[-1]


def normalised_edit_distance(a: str, b: str) -> float:
    """Edit distance normalised by max length. 0.0 = identical, 1.0 = completely different."""
    if not a and not b:
        return 0.0
    dist = levenshtein_distance(a, b)
    max_len = max(len(a), len(b))
    return dist / max_len if max_len > 0 else 0.0


def score_section(section: str, draft_val: str, edited_val: str) -> SectionScore:
    d_str = str(draft_val or "")
    e_str = str(edited_val or "")
    ned = normalised_edit_distance(d_str, e_str)
    return SectionScore(
        section=section,
        draft_text=d_str[:200],
        edited_text=e_str[:200],
        edit_distance=levenshtein_distance(d_str, e_str),
        normalised_edit_distance=ned,
        section_accuracy=1.0 - ned,
    )


def _join_list(lst: list) -> str:
    return " | ".join(str(x) for x in lst)


def evaluate(
    patient_id: str,
    iteration: int,
    draft: DischargeSummary,
    edited: DischargeSummary,
) -> EvaluationResult:
    """
    Compare draft vs edited summary and produce a scored EvaluationResult.
    """
    sections = [
        ("hospital_course", draft.hospital_course, edited.hospital_course),
        ("principal_diagnosis", draft.principal_diagnosis, edited.principal_diagnosis),
        ("follow_up", _join_list(draft.follow_up_instructions), _join_list(edited.follow_up_instructions)),
        ("pending_results", _join_list(draft.pending_results), _join_list(edited.pending_results)),
        ("discharge_condition", draft.discharge_condition, edited.discharge_condition),
        ("allergies", _join_list(draft.allergies), _join_list(edited.allergies)),
    ]

    section_scores = [score_section(name, d, e) for name, d, e in sections]

    # Section weights (hospital course and follow-up most important)
    weights = {"hospital_course": 0.30, "follow_up": 0.25, "pending_results": 0.20,
               "principal_diagnosis": 0.15, "discharge_condition": 0.05, "allergies": 0.05}

    weighted_accuracy = sum(
        weights.get(s.section, 0.1) * s.section_accuracy for s in section_scores
    )
    overall_ned = sum(s.normalised_edit_distance for s in section_scores) / len(section_scores)

    # Pending result ownership rate
    total_pending = len(edited.pending_results)
    with_owner = sum(
        1 for r in edited.pending_results
        if "gp" in r.lower() or "clinic" in r.lower() or "follow" in r.lower()
    )
    ownership_rate = with_owner / total_pending if total_pending > 0 else 1.0

    # Composite score
    composite = (
        weighted_accuracy * 0.6
        + (1 - overall_ned) * 0.2
        + ownership_rate * 0.1
        + (1 - min(len(edited.clinician_review_flags) / 10, 1.0)) * 0.1
    )

    return EvaluationResult(
        patient_id=patient_id,
        iteration=iteration,
        section_scores=section_scores,
        overall_edit_distance=overall_ned,
        overall_accuracy=weighted_accuracy,
        flag_count=len(edited.clinician_review_flags),
        missing_field_count=len(edited.missing_fields),
        pending_with_owner=ownership_rate,
        composite_score=composite,
    )


def average_score(results: list[EvaluationResult]) -> dict:
    if not results:
        return {}
    n = len(results)
    return {
        "n": n,
        "avg_composite_score": sum(r.composite_score for r in results) / n,
        "avg_overall_accuracy": sum(r.overall_accuracy for r in results) / n,
        "avg_edit_distance": sum(r.overall_edit_distance for r in results) / n,
        "avg_flags": sum(r.flag_count for r in results) / n,
        "avg_pending_ownership": sum(r.pending_with_owner for r in results) / n,
    }
