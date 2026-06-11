"""
Tests for the PDF parser layer.
Covers text extraction, fallback behaviour, and error handling.
"""
import pytest
import tempfile
from pathlib import Path
from app.parsers.pdf_parser import parse_pdf, full_text, PDFParseError, ParsedDocument


def _write_simple_pdf(text: str) -> str:
    """Create a minimal text-based PDF using fitz for testing."""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), text, fontsize=12)
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp.close()
    doc.save(tmp.name)
    doc.close()
    return tmp.name


class TestPDFParser:
    def test_parse_simple_pdf(self):
        path = _write_simple_pdf("Patient: John Smith\nDiagnosis: NSTEMI")
        doc = parse_pdf(path)
        assert isinstance(doc, ParsedDocument)
        assert doc.total_pages >= 1
        text = full_text(doc)
        assert "John Smith" in text or "NSTEMI" in text

    def test_file_not_found_raises(self):
        with pytest.raises(PDFParseError):
            parse_pdf("/nonexistent/path/file.pdf")

    def test_full_text_includes_page_markers(self):
        path = _write_simple_pdf("Test content here")
        doc = parse_pdf(path)
        text = full_text(doc)
        assert "Page 1" in text

    def test_parsed_document_has_filename(self):
        path = _write_simple_pdf("Some text")
        doc = parse_pdf(path)
        assert doc.file_name.endswith(".pdf")

    def test_extraction_method_recorded(self):
        path = _write_simple_pdf("Test text content for extraction")
        doc = parse_pdf(path)
        assert doc.extraction_method in ("fitz", "pdfplumber", "ocr_stub")
