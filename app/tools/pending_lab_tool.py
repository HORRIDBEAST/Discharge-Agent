"""
Pending Lab Detector Tool.

Scans extracted text for indicators of pending, outstanding, or
awaited laboratory results. These must be explicitly noted in the
discharge summary rather than omitted or fabricated.
"""
from __future__ import annotations
import re
from app.tools.base import BaseTool, register_tool
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Patterns that indicate a result is pending or outstanding
PENDING_PATTERNS = [
    r"pending",
    r"awaiting",
    r"outstanding",
    r"to follow",
    r"will follow",
    r"results? not yet",
    r"not resulted",
    r"sent for",
    r"ordered,?\s+not",
    r"follow[ -]?up\s+required",
    r"final report",
    r"preliminary",
    r"to be reported",
    r"culture\s+(?:pending|sent)",
    r"biopsy\s+(?:pending|sent)",
]

# Common lab/test names to anchor matches
LAB_KEYWORDS = [
    "culture", "sensitivity", "biopsy", "pathology", "cytology",
    "echocardiogram", "echo", "mri", "ct scan", "pet scan",
    "blood culture", "urine culture", "wound culture",
    "troponin", "d-dimer", "haemoglobin", "creatinine",
    "thyroid", "tsh", "hba1c", "glucose", "potassium",
    "sodium", "lipid panel", "lft", "lfts", "bnp", "nt-probnp",
]

COMPILED_PENDING = [re.compile(p, re.IGNORECASE) for p in PENDING_PATTERNS]


@register_tool
class PendingLabDetectorTool(BaseTool):
    name = "pending_lab_detector"
    description = "Identify pending or outstanding laboratory and test results in clinical notes."

    async def _execute(self, text: str, source_doc: str = "unknown") -> dict:
        pending_items: list[dict[str, str]] = []
        lines = text.split("\n")

        for line_num, line in enumerate(lines):
            line_stripped = line.strip()
            if not line_stripped:
                continue

            # Check if line contains pending indicator
            has_pending = any(p.search(line_stripped) for p in COMPILED_PENDING)
            if not has_pending:
                continue

            # Check if line also contains a lab/test keyword
            line_lower = line_stripped.lower()
            matched_labs = [kw for kw in LAB_KEYWORDS if kw in line_lower]

            if matched_labs or has_pending:
                pending_items.append({
                    "text": line_stripped[:300],
                    "line": line_num,
                    "matched_keywords": matched_labs,
                    "source_doc": source_doc,
                })

        # Deduplicate similar entries
        seen: set[str] = set()
        unique_items = []
        for item in pending_items:
            key = item["text"][:80].lower()
            if key not in seen:
                seen.add(key)
                unique_items.append(item)

        logger.info(
            "pending_labs_detected",
            source_doc=source_doc,
            count=len(unique_items),
        )

        return {
            "pending_items": unique_items,
            "count": len(unique_items),
            "source_doc": source_doc,
        }
