"""
Tests for evaluation metrics — edit distance and scoring.
"""
import pytest
from evaluation.metrics import (
    levenshtein_distance, normalised_edit_distance, score_section, evaluate
)
from app.models.discharge_summary import DischargeSummary, PatientDemographics


class TestLevenshtein:
    def test_identical_strings(self):
        assert levenshtein_distance("hello", "hello") == 0

    def test_empty_strings(self):
        assert levenshtein_distance("", "") == 0

    def test_one_empty(self):
        assert levenshtein_distance("abc", "") == 3
        assert levenshtein_distance("", "abc") == 3

    def test_single_substitution(self):
        assert levenshtein_distance("cat", "bat") == 1

    def test_insertion(self):
        assert levenshtein_distance("abc", "abcd") == 1

    def test_deletion(self):
        assert levenshtein_distance("abcd", "abc") == 1

    def test_complete_rewrite(self):
        dist = levenshtein_distance("NSTEMI", "pneumonia")
        assert dist > 0


class TestNormalisedEditDistance:
    def test_identical(self):
        assert normalised_edit_distance("hello", "hello") == 0.0

    def test_both_empty(self):
        assert normalised_edit_distance("", "") == 0.0

    def test_range(self):
        ned = normalised_edit_distance("abc", "xyz")
        assert 0.0 <= ned <= 1.0

    def test_complete_difference_high(self):
        ned = normalised_edit_distance("aaa", "bbb")
        assert ned > 0.5


class TestSectionScore:
    def test_identical_sections(self):
        score = score_section("hospital_course", "Patient admitted with chest pain.", "Patient admitted with chest pain.")
        assert score.section_accuracy == 1.0
        assert score.normalised_edit_distance == 0.0

    def test_different_sections(self):
        score = score_section("hospital_course", "Short text", "Completely different and much longer text here")
        assert score.section_accuracy < 1.0
        assert score.normalised_edit_distance > 0.0


class TestEvaluate:
    def _make_summary(self, hospital_course="Patient did well.", follow_up=None, pending=None, flags=None):
        return DischargeSummary(
            patient=PatientDemographics(name="Test Patient"),
            hospital_course=hospital_course,
            follow_up_instructions=follow_up or [],
            pending_results=pending or [],
            clinician_review_flags=flags or [],
        )

    def test_identical_draft_and_edited_high_score(self):
        draft = self._make_summary("Patient admitted with NSTEMI, underwent PCI. Discharged stable.")
        edited = self._make_summary("Patient admitted with NSTEMI, underwent PCI. Discharged stable.")
        result = evaluate("P001", 1, draft, edited)
        assert result.overall_accuracy > 0.9
        assert result.composite_score > 0.5

    def test_edited_with_changes_lower_score(self):
        draft = self._make_summary("Patient was admitted.")
        edited = self._make_summary(
            "Patient was admitted with NSTEMI and underwent PCI. "
            "Patient was discharged in stable condition.",
            follow_up=["GP follow-up within 1 week", "Cardiology within 2 weeks"],
            pending=["Blood culture — GP to follow up"],
        )
        result = evaluate("P001", 1, draft, edited)
        # Edited version differs significantly from draft
        assert result.overall_edit_distance > 0.0

    def test_pending_ownership_calculated(self):
        draft = self._make_summary(pending=["Blood culture pending", "Echo pending"])
        edited = self._make_summary(pending=[
            "Blood culture pending — GP to follow up",
            "Echo pending — Cardiology clinic to follow up",
        ])
        result = evaluate("P001", 1, draft, edited)
        assert result.pending_with_owner == 1.0  # Both have owner in edited
