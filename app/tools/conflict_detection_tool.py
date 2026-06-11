"""
Conflict Detection Tool.

When the same clinical field has been extracted from multiple documents
with different values, this tool flags the conflict rather than picking
a winner. The conflict is surfaced in the discharge summary for clinician
resolution.
"""
from __future__ import annotations
from typing import Any

from app.tools.base import BaseTool, register_tool
from app.models.evidence import ConflictRecord
from app.utils.logger import get_logger

logger = get_logger(__name__)


@register_tool
class ConflictDetectionTool(BaseTool):
    name = "conflict_detection"
    description = "Detect contradictions between extracted facts from multiple source documents."

    async def _execute(
        self,
        field: str,
        extractions: list[dict[str, Any]],  # list of {value, source_doc, evidence_text}
    ) -> dict[str, Any]:
        """
        Given multiple extractions for the same field, determine if a conflict exists.
        Returns conflict records if values differ meaningfully.
        """
        if len(extractions) < 2:
            return {"conflicts": [], "has_conflict": False}

        conflicts: list[dict[str, Any]] = []

        # Compare all pairs
        seen_pairs: set[tuple[int, int]] = set()
        for i in range(len(extractions)):
            for j in range(i + 1, len(extractions)):
                if (i, j) in seen_pairs:
                    continue
                seen_pairs.add((i, j))

                val_i = str(extractions[i].get("value", "")).strip().lower()
                val_j = str(extractions[j].get("value", "")).strip().lower()

                # Skip if both are missing/null
                if not val_i or not val_j or val_i in ("null", "none", "missing") or val_j in ("null", "none", "missing"):
                    continue

                # Check for meaningful difference
                if not _values_agree(val_i, val_j):
                    conflict = ConflictRecord(
                        field=field,
                        source_1_doc=extractions[i].get("source_doc", "unknown"),
                        source_1_value=extractions[i].get("value", ""),
                        source_1_evidence_text=extractions[i].get("evidence_text", ""),
                        source_2_doc=extractions[j].get("source_doc", "unknown"),
                        source_2_value=extractions[j].get("value", ""),
                        source_2_evidence_text=extractions[j].get("evidence_text", ""),
                        status="clinician_review_required",
                    )
                    conflicts.append(conflict.model_dump())
                    logger.warning(
                        "conflict_detected",
                        field=field,
                        source_1=extractions[i].get("source_doc"),
                        source_2=extractions[j].get("source_doc"),
                        value_1=val_i[:80],
                        value_2=val_j[:80],
                    )

        return {
            "conflicts": conflicts,
            "has_conflict": len(conflicts) > 0,
            "field": field,
            "total_extractions": len(extractions),
        }


def _values_agree(val_a: str, val_b: str) -> bool:
    """
    Heuristic agreement check. Two values agree if one is a prefix/suffix
    of the other or they share all significant tokens.
    This avoids false conflicts from formatting differences.
    """
    if val_a == val_b:
        return True
    # Token overlap check
    tokens_a = set(val_a.split())
    tokens_b = set(val_b.split())
    if not tokens_a or not tokens_b:
        return True
    # Use min-length denominator: "type 2 diabetes" is 75% of "type 2 diabetes mellitus" tokens
    overlap = len(tokens_a & tokens_b) / min(len(tokens_a), len(tokens_b))
    return overlap >= 0.8
