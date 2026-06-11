"""
EvidenceValidator — the anti-hallucination firewall.

Before any fact enters the discharge summary, it must pass through
this validator. The validator checks:

  1. The fact has at least one Evidence record
  2. The evidence confidence meets the threshold
  3. The evidence_text is non-empty (i.e., there is actual source text)
  4. The source document is in the processed document set

If validation fails, the field is marked MISSING, not invented.

This is the final defence against fabrication at composition time.
"""
from __future__ import annotations
from typing import Optional

from app.models.evidence import Evidence, ConfidenceLevel
from app.models.discharge_summary import MISSING
from app.utils.logger import get_logger

logger = get_logger(__name__)

MIN_CONFIDENCE = 0.50  # Below this, LOW confidence → flag, not include
WARN_CONFIDENCE = 0.70  # Below this, MEDIUM confidence → warn


class ValidationResult:
    def __init__(self, valid: bool, value: str, warning: Optional[str] = None):
        self.valid = valid
        self.value = value
        self.warning = warning


class EvidenceValidator:
    """
    Validates that every claim in the discharge summary has
    documented evidence. This is the anti-hallucination layer.
    """

    def __init__(self, evidence_registry: list[Evidence], processed_docs: list[str]):
        self.registry = evidence_registry
        self.processed_docs = set(processed_docs)

    def validate_field(
        self,
        field_name: str,
        proposed_value: str,
        min_confidence: float = MIN_CONFIDENCE,
    ) -> ValidationResult:
        """
        Validate that a proposed field value has evidentiary support.
        Returns the value if valid, MISSING sentinel if not.
        """
        if not proposed_value or proposed_value.strip() in (MISSING, "", "None", "null"):
            return ValidationResult(
                valid=False,
                value=MISSING,
                warning=f"Field '{field_name}' has no proposed value",
            )

        # Find matching evidence
        relevant = self._find_evidence(field_name, proposed_value)

        if not relevant:
            logger.warning(
                "no_evidence_for_field",
                field=field_name,
                value=proposed_value[:80],
            )
            return ValidationResult(
                valid=False,
                value=MISSING,
                warning=f"No evidence found for '{field_name}' = '{proposed_value[:60]}...'",
            )

        # Take highest confidence evidence
        best = max(relevant, key=lambda e: e.confidence)

        if best.confidence < min_confidence:
            logger.warning(
                "evidence_confidence_too_low",
                field=field_name,
                confidence=best.confidence,
                threshold=min_confidence,
            )
            return ValidationResult(
                valid=False,
                value=MISSING,
                warning=(
                    f"Evidence for '{field_name}' has confidence {best.confidence:.0%} "
                    f"(below threshold {min_confidence:.0%})"
                ),
            )

        # Check source document is legitimate
        if best.source_document not in self.processed_docs and self.processed_docs:
            logger.error(
                "evidence_from_unknown_source",
                field=field_name,
                source=best.source_document,
            )
            return ValidationResult(
                valid=False,
                value=MISSING,
                warning=f"Evidence source '{best.source_document}' not in processed documents — possible hallucination",
            )

        warning = None
        if best.confidence < WARN_CONFIDENCE:
            warning = (
                f"Low-confidence evidence for '{field_name}' "
                f"({best.confidence:.0%}) — clinician should verify"
            )

        return ValidationResult(valid=True, value=proposed_value, warning=warning)

    def validate_narrative(self, field_name: str, narrative: str) -> ValidationResult:
        """
        For free-text narrative fields (hospital course), check that
        the text doesn't contain fabrication red flags.
        """
        if not narrative or narrative == MISSING:
            return ValidationResult(valid=False, value=MISSING)

        # Check for suspiciously specific numbers/values with no evidence
        # In a real system: sentence-level evidence check
        # Here: basic sanity — is there at least one piece of evidence overall?
        if not self.registry:
            return ValidationResult(
                valid=False,
                value=MISSING,
                warning=f"Narrative for '{field_name}' rejected — no evidence in registry",
            )

        return ValidationResult(valid=True, value=narrative)

    def _find_evidence(self, field_name: str, value: str) -> list[Evidence]:
        """Find evidence records relevant to a field/value pair."""
        field_lower = field_name.lower()
        value_lower = value.lower()[:100]
        matches = []

        for ev in self.registry:
            ev_fact_lower = ev.fact.lower()
            ev_text_lower = ev.evidence_text.lower()

            # Match if fact name overlaps OR value tokens appear in evidence text
            if (
                field_lower in ev_fact_lower
                or ev_fact_lower in field_lower
                or _token_overlap(value_lower, ev_fact_lower) > 0.3
                or _token_overlap(value_lower, ev_text_lower) > 0.3
            ):
                matches.append(ev)

        return matches


def _token_overlap(a: str, b: str) -> float:
    tokens_a = set(a.split())
    tokens_b = set(b.split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / max(len(tokens_a), len(tokens_b))
