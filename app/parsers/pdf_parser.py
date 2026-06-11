"""
PDF parsing layer.

Strategy:
  1. Try PyMuPDF (fitz) for fast text extraction
  2. Fall back to pdfplumber for better table handling
  3. If both yield < MIN_CHARS, assume scanned — raise ParseError
     (OCR integration point clearly marked)

Every page is returned with its page number for evidence provenance.
"""
from __future__ import annotations
import io
from pathlib import Path
from typing import NamedTuple

from app.utils.logger import get_logger

logger = get_logger(__name__)

MIN_CHARS_PER_PAGE = 30  # Below this we suspect a scanned image page


class PageContent(NamedTuple):
    page_number: int       # 0-indexed
    text: str
    extraction_method: str  # "fitz" | "pdfplumber" | "ocr_stub"


class ParsedDocument(NamedTuple):
    file_path: str
    file_name: str
    pages: list[PageContent]
    total_pages: int
    extraction_method: str


class PDFParseError(Exception):
    pass


def parse_pdf(file_path: str) -> ParsedDocument:
    """
    Main entry point. Tries fitz first, falls back to pdfplumber.
    Returns a ParsedDocument with per-page text and provenance metadata.
    """
    path = Path(file_path)
    if not path.exists():
        raise PDFParseError(f"File not found: {file_path}")

    # Attempt 1: PyMuPDF (fitz)
    try:
        import fitz  # PyMuPDF
        pages = _extract_with_fitz(str(path))
        method = "fitz"
        # Check if extraction was meaningful
        total_text = "".join(p.text for p in pages)
        if len(total_text.strip()) < MIN_CHARS_PER_PAGE:
            raise PDFParseError("fitz yielded insufficient text")
        logger.info("pdf_parsed", file=path.name, method="fitz", pages=len(pages))
        return ParsedDocument(str(path), path.name, pages, len(pages), method)
    except PDFParseError:
        pass
    except Exception as e:
        logger.warning("fitz_failed", file=path.name, error=str(e))

    # Attempt 2: pdfplumber
    try:
        import pdfplumber
        pages = _extract_with_pdfplumber(str(path))
        method = "pdfplumber"
        total_text = "".join(p.text for p in pages)
        if len(total_text.strip()) < MIN_CHARS_PER_PAGE:
            raise PDFParseError("pdfplumber yielded insufficient text")
        logger.info("pdf_parsed", file=path.name, method="pdfplumber", pages=len(pages))
        return ParsedDocument(str(path), path.name, pages, len(pages), method)
    except PDFParseError:
        pass
    except Exception as e:
        logger.warning("pdfplumber_failed", file=path.name, error=str(e))

    # Attempt 3: OCR stub
    # In production: integrate pytesseract or cloud OCR here
    logger.error("pdf_parse_exhausted_fallbacks", file=path.name)
    stub_text = f"[OCR REQUIRED — scanned document detected: {path.name}]"
    pages = [PageContent(0, stub_text, "ocr_stub")]
    return ParsedDocument(str(path), path.name, pages, 1, "ocr_stub")


def _extract_with_fitz(file_path: str) -> list[PageContent]:
    import fitz
    doc = fitz.open(file_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append(PageContent(i, text.strip(), "fitz"))
    doc.close()
    return pages


def _extract_with_pdfplumber(file_path: str) -> list[PageContent]:
    import pdfplumber
    pages = []
    with pdfplumber.open(file_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Also extract tables as pipe-delimited text
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            text += "\n" + " | ".join(str(c) for c in row if c)
            pages.append(PageContent(i, text.strip(), "pdfplumber"))
    return pages


def full_text(doc: ParsedDocument) -> str:
    """Concatenate all pages with page markers for context."""
    parts = []
    for page in doc.pages:
        parts.append(f"\n--- Page {page.page_number + 1} ---\n{page.text}")
    return "\n".join(parts)


def parse_pdf_from_bytes(data: bytes, filename: str) -> ParsedDocument:
    """Parse a PDF from raw bytes (used by the API upload endpoint)."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(data)
        tmp_path = f.name
    try:
        result = parse_pdf(tmp_path)
        # Patch the filename
        return ParsedDocument(tmp_path, filename, result.pages, result.total_pages, result.extraction_method)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
