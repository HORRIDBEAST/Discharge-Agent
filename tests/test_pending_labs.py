"""
Tests for the pending lab detector tool.
"""
import pytest
import asyncio
from app.tools.pending_lab_tool import PendingLabDetectorTool


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestPendingLabDetector:
    def setup_method(self):
        self.tool = PendingLabDetectorTool()

    def test_detects_pending_culture(self):
        text = "Blood culture: PENDING — sent 2024-11-02, awaiting final results"
        result = run(self.tool._execute(text=text, source_doc="lab_results.pdf"))
        assert result["count"] >= 1
        assert any("culture" in item["text"].lower() for item in result["pending_items"])

    def test_detects_awaiting_echo(self):
        text = "Echocardiogram: ordered, awaiting results. Patient to follow up at clinic."
        result = run(self.tool._execute(text=text, source_doc="progress_note.pdf"))
        assert result["count"] >= 1

    def test_detects_pending_biopsy(self):
        text = "Biopsy sent to pathology — result pending, expected within 5 days."
        result = run(self.tool._execute(text=text, source_doc="note.pdf"))
        assert result["count"] >= 1

    def test_no_pending_in_clean_text(self):
        text = "Patient was discharged in stable condition. All labs resulted and reviewed."
        result = run(self.tool._execute(text=text, source_doc="discharge.pdf"))
        assert result["count"] == 0

    def test_deduplication(self):
        # Same line repeated
        text = "\n".join(["Blood culture pending"] * 5)
        result = run(self.tool._execute(text=text, source_doc="note.pdf"))
        assert result["count"] == 1  # Deduped

    def test_source_doc_recorded(self):
        text = "Urine culture: PENDING — sent yesterday"
        result = run(self.tool._execute(text=text, source_doc="progress_note_1.pdf"))
        assert result["source_doc"] == "progress_note_1.pdf"

    def test_multiline_document(self):
        text = """
        WBC: 11.2 (high)
        Haemoglobin: 13.4 (normal)
        Blood culture: PENDING — sent due to fever
        Echocardiogram: awaiting result
        Troponin: 2.8 (resulted)
        """
        result = run(self.tool._execute(text=text, source_doc="labs.pdf"))
        assert result["count"] >= 2
