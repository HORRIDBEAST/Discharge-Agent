"""
PDF Reader Tool — reads a PDF file and returns extracted text per page.
The agent uses this to load source documents into working memory.
"""
from __future__ import annotations
from app.tools.base import BaseTool, register_tool
from app.parsers.pdf_parser import parse_pdf, ParsedDocument, full_text


@register_tool
class PDFReaderTool(BaseTool):
    name = "pdf_reader"
    description = "Read and extract text from a patient source note PDF."

    async def _execute(self, file_path: str) -> dict:
        doc: ParsedDocument = parse_pdf(file_path)
        return {
            "file_name": doc.file_name,
            "file_path": doc.file_path,
            "total_pages": doc.total_pages,
            "extraction_method": doc.extraction_method,
            "full_text": full_text(doc),
            "pages": [
                {"page": p.page_number, "text": p.text, "method": p.extraction_method}
                for p in doc.pages
            ],
        }
