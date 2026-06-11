"""
Tool registry.

Import all tools here to trigger their @register_tool decorator,
which registers them for dynamic dispatch in the agent.
"""
from .base import BaseTool, ToolResult, get_tool_registry, register_tool
from .pdf_reader_tool import PDFReaderTool
from .medication_reconciliation_tool import MedicationReconciliationTool
from .conflict_detection_tool import ConflictDetectionTool
from .drug_interaction_tool import DrugInteractionTool
from .pending_lab_tool import PendingLabDetectorTool
from .escalation_tool import EscalationTool
from .summary_composer_tool import SummaryComposerTool

__all__ = [
    "BaseTool", "ToolResult", "get_tool_registry", "register_tool",
    "PDFReaderTool", "MedicationReconciliationTool", "ConflictDetectionTool",
    "DrugInteractionTool", "PendingLabDetectorTool", "EscalationTool",
    "SummaryComposerTool",
]
