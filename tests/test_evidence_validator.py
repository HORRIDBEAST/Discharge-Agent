"""
Tests for the EvidenceValidator — the anti-hallucination firewall.
"""
import pytest
from app.models.evidence import Evidence, ConfidenceLevel
from app.validators.evidence_validator import EvidenceValidator
from app.models.discharge_summary import MISSING


def _make_evidence(fact: str, text: str, confidence: float = 0.9, source: str = "test.pdf") -> Evidence:
    return Evidence(
        fact=fact,
        source_document=source,
        page=0,
        evidence_text=text,
        confidence=confidence,
        confidence_level=ConfidenceLevel.HIGH if confidence >= 0.85 else ConfidenceLevel.MEDIUM,
    )


class TestEvidenceValidator:
    def test_valid_field_with_evidence(self):
        registry = [_make_evidence("principal_diagnosis:NSTEMI", "Confirmed NSTEMI based on troponin rise")]
        validator = EvidenceValidator(registry, processed_docs=["test.pdf"])
        result = validator.validate_field("principal_diagnosis", "NSTEMI")
        assert result.valid
        assert result.value == "NSTEMI"
        assert result.warning is None

    def test_missing_field_returns_sentinel(self):
        registry = []  # No evidence
        validator = EvidenceValidator(registry, processed_docs=["test.pdf"])
        result = validator.validate_field("principal_diagnosis", "NSTEMI")
        assert not result.valid
        assert result.value == MISSING

    def test_low_confidence_below_threshold(self):
        registry = [_make_evidence("diagnosis:UTI", "possible UTI", confidence=0.3)]
        validator = EvidenceValidator(registry, processed_docs=["test.pdf"])
        result = validator.validate_field("diagnosis", "UTI", min_confidence=0.5)
        assert not result.valid
        assert result.value == MISSING

    def test_medium_confidence_warns_but_passes(self):
        registry = [_make_evidence("discharge_condition:stable", "patient appears stable", confidence=0.65)]
        validator = EvidenceValidator(registry, processed_docs=["test.pdf"])
        result = validator.validate_field("discharge_condition", "stable")
        assert result.valid
        assert result.warning is not None  # Low confidence warning

    def test_unknown_source_document_rejected(self):
        registry = [_make_evidence("diagnosis:pneumonia", "has pneumonia", source="unknown_doc.pdf")]
        validator = EvidenceValidator(registry, processed_docs=["test.pdf"])  # unknown_doc.pdf NOT in set
        result = validator.validate_field("diagnosis", "pneumonia")
        assert not result.valid
        assert result.value == MISSING

    def test_empty_proposed_value_returns_missing(self):
        registry = [_make_evidence("diagnosis:NSTEMI", "NSTEMI confirmed")]
        validator = EvidenceValidator(registry, processed_docs=["test.pdf"])
        result = validator.validate_field("diagnosis", "")
        assert not result.valid
        assert result.value == MISSING

    def test_missing_sentinel_value_returns_missing(self):
        registry = [_make_evidence("diagnosis:NSTEMI", "NSTEMI confirmed")]
        validator = EvidenceValidator(registry, processed_docs=["test.pdf"])
        result = validator.validate_field("diagnosis", MISSING)
        assert not result.valid

    def test_narrative_with_registry_passes(self):
        registry = [_make_evidence("hospital_course", "Patient admitted for chest pain")]
        validator = EvidenceValidator(registry, processed_docs=["test.pdf"])
        result = validator.validate_narrative("hospital_course", "Patient admitted with chest pain and underwent PCI.")
        assert result.valid

    def test_narrative_with_empty_registry_fails(self):
        validator = EvidenceValidator([], processed_docs=[])
        result = validator.validate_narrative("hospital_course", "Patient had a good stay.")
        assert not result.valid
