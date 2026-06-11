"""
Tests for conflict detection tool.
"""
import pytest
import asyncio
from app.tools.conflict_detection_tool import ConflictDetectionTool, _values_agree


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestConflictDetection:
    def setup_method(self):
        self.tool = ConflictDetectionTool()

    def _extraction(self, value, source_doc, evidence_text=""):
        return {"value": value, "source_doc": source_doc, "evidence_text": evidence_text}

    def test_no_conflict_single_extraction(self):
        result = run(self.tool._execute(
            field="principal_diagnosis",
            extractions=[self._extraction("NSTEMI", "admission_note.pdf")],
        ))
        assert result["has_conflict"] is False
        assert result["conflicts"] == []

    def test_no_conflict_identical_values(self):
        result = run(self.tool._execute(
            field="principal_diagnosis",
            extractions=[
                self._extraction("NSTEMI", "admission_note.pdf"),
                self._extraction("NSTEMI", "progress_note.pdf"),
            ],
        ))
        assert result["has_conflict"] is False

    def test_conflict_detected_different_diagnoses(self):
        result = run(self.tool._execute(
            field="principal_diagnosis",
            extractions=[
                self._extraction("Urinary tract infection", "progress_note_1.pdf"),
                self._extraction("Community-acquired pneumonia", "progress_note_2.pdf"),
            ],
        ))
        assert result["has_conflict"] is True
        assert len(result["conflicts"]) == 1
        conflict = result["conflicts"][0]
        assert conflict["field"] == "principal_diagnosis"
        assert conflict["status"] == "clinician_review_required"

    def test_null_values_not_flagged_as_conflict(self):
        result = run(self.tool._execute(
            field="discharge_date",
            extractions=[
                self._extraction(None, "doc1.pdf"),
                self._extraction("2024-11-05", "doc2.pdf"),
            ],
        ))
        assert result["has_conflict"] is False

    def test_high_token_overlap_not_conflict(self):
        # "Type 2 diabetes mellitus" vs "Type II diabetes mellitus" — near-identical
        result = run(self.tool._execute(
            field="secondary_diagnosis",
            extractions=[
                self._extraction("Type 2 diabetes mellitus", "note1.pdf"),
                self._extraction("Type 2 diabetes mellitus known history", "note2.pdf"),
            ],
        ))
        # High overlap — should NOT be a conflict
        assert result["has_conflict"] is False

    def test_conflict_includes_source_documents(self):
        result = run(self.tool._execute(
            field="primary_diagnosis",
            extractions=[
                self._extraction("UTI", "dr_williams_note.pdf", "likely UTI"),
                self._extraction("Pneumonia", "dr_patel_note.pdf", "respiratory infection"),
            ],
        ))
        assert result["has_conflict"] is True
        c = result["conflicts"][0]
        assert "dr_williams_note.pdf" in (c["source_1_doc"], c["source_2_doc"])
        assert "dr_patel_note.pdf" in (c["source_1_doc"], c["source_2_doc"])


class TestValuesAgree:
    def test_identical(self):
        assert _values_agree("NSTEMI", "NSTEMI") is True

    def test_completely_different(self):
        assert _values_agree("pneumonia", "UTI") is False

    def test_high_overlap(self):
        assert _values_agree("type 2 diabetes mellitus", "type 2 diabetes") is True

    def test_empty_strings(self):
        assert _values_agree("", "") is True
