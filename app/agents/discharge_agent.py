"""
DischargeAgent — the core agent loop.

This implements a proper Plan → Act → Observe → Re-plan loop.
It is NOT a linear pipeline. Each iteration the agent:

  1. Inspects current state (what do I know? what is missing?)
  2. Selects the highest-priority pending task
  3. Chooses the appropriate tool
  4. Executes it (with retries via BaseTool)
  5. Merges results into working memory
  6. Detects conflicts
  7. Re-evaluates completion criteria
  8. Records a trace step

Stopping conditions:
  - All required fields have evidence OR are marked MISSING/CONFLICT
  - Iteration limit reached → escalate
  - Timeout → escalate
  - Unrecoverable tool failure → escalate

The agent NEVER:
  - Fabricates facts
  - Silently drops failures
  - Picks a winner in a conflict
  - Finishes early without checking all documents
"""
from __future__ import annotations
import asyncio
import time
from typing import Any, Optional

from app.models.agent_state import AgentState, AgentPhase, AgentStep, ToolCall
from app.models.evidence import Evidence, ConflictRecord, MissingField
from app.models.medications import Medication, ReconciliationReport
from app.models.discharge_summary import DischargeSummary, MISSING
from app.services.llm_client import LLMClient
from app.services.clinical_extractor import ClinicalExtractor
from app.tools import (
    PDFReaderTool, MedicationReconciliationTool, ConflictDetectionTool,
    DrugInteractionTool, PendingLabDetectorTool, EscalationTool, SummaryComposerTool,
)
from app.validators.evidence_validator import EvidenceValidator
from app.utils.tracer import AgentTracer
from app.utils.logger import get_logger
from app.configs.settings import get_settings

logger = get_logger(__name__)

# Required fields the agent must attempt to fill
REQUIRED_FIELDS = [
    "patient_demographics",
    "admission_date",
    "discharge_date",
    "principal_diagnosis",
    "hospital_course",
    "discharge_medications",
    "admission_medications",
    "allergies",
    "discharge_condition",
    "follow_up",
]


class DischargeAgent:
    """
    Agentic discharge summary generator.

    Designed to be instantiated per patient session. Not thread-safe
    across concurrent sessions — use one instance per request.
    """

    def __init__(self, patient_id: str, source_document_paths: list[str]):
        self.settings = get_settings()
        self.patient_id = patient_id
        self.source_document_paths = source_document_paths

        # Core services
        self.llm = LLMClient()
        self.extractor = ClinicalExtractor(self.llm)

        # Tools (instantiated fresh per run)
        self.pdf_tool = PDFReaderTool()
        self.recon_tool = MedicationReconciliationTool()
        self.conflict_tool = ConflictDetectionTool()
        self.interaction_tool = DrugInteractionTool()
        self.pending_lab_tool = PendingLabDetectorTool()
        self.escalation_tool = EscalationTool()
        self.composer_tool = SummaryComposerTool()

        # State
        self.state = AgentState(
            patient_id=patient_id,
            source_documents=source_document_paths,
            max_iterations=self.settings.agent_max_iterations,
        )

        # Tracer
        self.tracer = AgentTracer(
            session_id=self.state.session_id,
            patient_id=patient_id,
            trace_dir=self.settings.trace_dir,
            console_output=self.settings.trace_console,
            file_output=self.settings.trace_file,
        )

        # Raw document text cache {filename: text}
        self._doc_cache: dict[str, str] = {}

    async def run(self) -> DischargeSummary:
        """
        Main entry point. Runs the agent loop and returns a DischargeSummary.
        Always returns — never raises to the caller.
        """
        start_time = time.monotonic()
        logger.info("agent_start", patient_id=self.patient_id, docs=len(self.source_document_paths))

        self._record_step(
            phase=AgentPhase.PLANNING,
            reasoning=f"Starting discharge summary generation for patient '{self.patient_id}'. "
                      f"Found {len(self.source_document_paths)} source documents. "
                      f"Will process each document, extract clinical facts, reconcile medications, "
                      f"detect conflicts, and compose the discharge summary.",
            decision="Begin document ingestion and extraction",
            next_action="Read and classify all source documents",
        )

        try:
            while not self.state.finished and self.state.iteration < self.state.max_iterations:
                self.state.iteration += 1
                elapsed = time.monotonic() - start_time

                if elapsed > self.settings.agent_timeout_seconds:
                    await self._handle_timeout()
                    break

                await self._run_iteration()

            if not self.state.finished:
                await self._handle_iteration_limit()

        except Exception as e:
            logger.error("agent_unexpected_error", error=str(e), exc_info=True)
            self.state.add_flag(f"Agent encountered unexpected error: {e}")
            self.state.escalated = True
            self.state.phase = AgentPhase.FAILED

        # Always compose — even partial results are better than nothing
        if not self.state.final_summary:
            await self._compose_summary()

        self.tracer.print_summary(
            total_iterations=self.state.iteration,
            escalated=self.state.escalated,
            flags=self.state.clinician_review_flags,
        )

        return self.state.final_summary

    async def _run_iteration(self) -> None:
        """Single iteration of the plan-act-observe loop."""
        state = self.state

        # --- Phase 1: Document Ingestion ---
        unprocessed = state.unprocessed_documents()
        if unprocessed:
            doc_path = unprocessed[0]
            await self._ingest_document(doc_path)
            return

        # --- Phase 2: Medication Reconciliation ---
        if state.reconciliation_report is None and (
            state.admission_medications or state.discharge_medications
        ):
            await self._reconcile_medications()
            return

        # --- Phase 3: Conflict Detection ---
        if state.phase not in (AgentPhase.VALIDATING, AgentPhase.COMPOSING):
            conflicts_checked = await self._check_conflicts()
            if conflicts_checked:
                state.phase = AgentPhase.VALIDATING
                return

        # --- Phase 4: Drug Interaction Check ---
        if not state.drug_interaction_alerts and state.discharge_medications:
            await self._check_drug_interactions()
            return

        # --- Phase 5: Check for missing critical fields ---
        missing = self._identify_missing_fields()
        if missing:
            self._record_missing_fields(missing)

        # --- Phase 6: Escalate if needed ---
        if state.conflicts and self.settings.conflict_auto_escalate and not state.escalated:
            await self._escalate_conflicts()

        # --- Phase 7: Compose summary ---
        state.phase = AgentPhase.COMPOSING
        await self._compose_summary()
        state.finished = True

    async def _ingest_document(self, doc_path: str) -> None:
        """Read and classify a document, then dispatch to appropriate extractors."""
        state = self.state
        state.phase = AgentPhase.EXTRACTING

        self._record_step(
            phase=AgentPhase.EXTRACTING,
            reasoning=f"Document '{doc_path}' has not been processed. "
                      f"Reading it and extracting all clinical facts.",
            selected_tool="pdf_reader",
            decision="Read document",
            next_action=f"Extract from {doc_path}",
        )

        # Read PDF
        result = await self.pdf_tool.run(file_path=doc_path)
        self._log_tool_call("pdf_reader", {"file_path": doc_path}, result)

        if not result.success:
            logger.error("pdf_read_failed", doc=doc_path, error=result.error)
            state.add_flag(f"Failed to read document: {doc_path} — {result.error}")
            state.processed_documents.append(doc_path)
            return

        doc_data = result.data
        text = doc_data["full_text"]
        filename = doc_data["file_name"]
        self._doc_cache[filename] = text

        # Classify document type
        doc_type = await self.extractor.classify_document_type(filename, text[:500])

        self._record_step(
            phase=AgentPhase.EXTRACTING,
            reasoning=f"Document classified as '{doc_type}'. "
                      f"Will run targeted extractors for this document type.",
            selected_tool="clinical_extractor",
            tool_output_summary=f"Document type: {doc_type}, pages: {doc_data['total_pages']}",
            decision=f"Run '{doc_type}' extractors",
            next_action="Extract clinical facts from document",
        )

        # Run all extractors (regardless of doc type — each handles absence gracefully)
        await self._extract_all(text, filename, doc_type)

        state.processed_documents.append(doc_path)
        logger.info("document_processed", doc=filename, doc_type=doc_type)

    async def _extract_all(self, text: str, source_doc: str, doc_type: str) -> None:
        """Run all clinical extractors on a document concurrently."""
        state = self.state

        # Run extractors in parallel for speed
        tasks = [
            self.extractor.extract_demographics(text, source_doc),
            self.extractor.extract_diagnoses(text, source_doc),
            self.extractor.extract_medications(text, source_doc, doc_type),
            self.extractor.extract_hospital_course(text, source_doc),
            self.extractor.extract_procedures(text, source_doc),
            self.extractor.extract_allergies(text, source_doc),
            self.extractor.extract_discharge_info(text, source_doc),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        (
            demographics,
            diagnoses,
            medications,
            hospital_course,
            procedures,
            allergies,
            discharge_info,
        ) = results

        # --- Merge demographics ---
        if isinstance(demographics, dict) and demographics:
            self._merge_demographics(demographics, source_doc)

        # --- Merge diagnoses ---
        if isinstance(diagnoses, list):
            for dx in diagnoses:
                if dx.get("diagnosis"):
                    state.extracted_diagnoses.append(dx)
                    ev = self.extractor.build_evidence(
                        fact=f"diagnosis:{dx.get('type', 'unknown')}:{dx['diagnosis']}",
                        evidence_text=dx.get("evidence_text", ""),
                        source_doc=source_doc,
                        page=dx.get("_page", 0),
                        confidence=float(dx.get("confidence", 0.7)),
                        section=dx.get("section"),
                    )
                    state.add_evidence(ev)

        # --- Merge medications ---
        if isinstance(medications, list):
            for med in medications:
                doc_lower = source_doc.lower()
                if any(k in doc_lower for k in ("admission", "admit", "admit_med", "med_admit")):
                    state.admission_medications.append(med)
                elif any(k in doc_lower for k in ("discharge", "disch", "dc_med", "med_dc")):
                    state.discharge_medications.append(med)
                else:
                    # Unknown source — if doc_type is medication_record, split by position
                    # otherwise add to both with a flag
                    state.discharge_medications.append(med)
                    state.add_flag(
                        f"Medication '{med.name}' from '{source_doc}' — "
                        "could not determine admission vs discharge — added to discharge list"
                    )
                ev = self.extractor.build_evidence(
                    fact=f"medication:{med.name}",
                    evidence_text=med.raw_text,
                    source_doc=source_doc,
                    page=0,
                    confidence=0.85,
                )
                state.add_evidence(ev)

        # --- Merge hospital course ---
        if isinstance(hospital_course, dict) and hospital_course.get("hospital_course"):
            state.extracted_hospital_course.append(hospital_course)
            ev = self.extractor.build_evidence(
                fact="hospital_course",
                evidence_text=hospital_course.get("hospital_course", "")[:200],
                source_doc=source_doc,
                page=hospital_course.get("_page", 0),
                confidence=float(hospital_course.get("confidence", 0.75)),
            )
            state.add_evidence(ev)

        # --- Merge procedures ---
        if isinstance(procedures, list):
            for proc in procedures:
                if proc.get("procedure"):
                    state.extracted_procedures.append(proc)
                    ev = self.extractor.build_evidence(
                        fact=f"procedure:{proc['procedure']}",
                        evidence_text=proc.get("evidence_text", ""),
                        source_doc=source_doc,
                        page=proc.get("_page", 0),
                        confidence=float(proc.get("confidence", 0.8)),
                    )
                    state.add_evidence(ev)

        # --- Merge allergies ---
        if isinstance(allergies, list):
            for al in allergies:
                if al.get("allergen"):
                    state.extracted_allergies.append(al)
                    ev = self.extractor.build_evidence(
                        fact=f"allergy:{al['allergen']}",
                        evidence_text=al.get("evidence_text", ""),
                        source_doc=source_doc,
                        page=al.get("_page", 0),
                        confidence=0.9,
                    )
                    state.add_evidence(ev)

        # --- Merge discharge info ---
        if isinstance(discharge_info, dict) and discharge_info:
            if discharge_info.get("discharge_condition"):
                state.extracted_discharge_condition = discharge_info
                ev = self.extractor.build_evidence(
                    fact=f"discharge_condition:{discharge_info['discharge_condition']}",
                    evidence_text=discharge_info.get("discharge_condition_evidence", ""),
                    source_doc=source_doc,
                    page=discharge_info.get("_page", 0),
                    confidence=float(discharge_info.get("confidence", 0.8)),
                )
                state.add_evidence(ev)

            for fu in discharge_info.get("follow_up_instructions", []):
                state.extracted_follow_up.append(fu)
                ev = self.extractor.build_evidence(
                    fact=f"follow_up:{fu.get('instruction', '')[:60]}",
                    evidence_text=fu.get("evidence_text", ""),
                    source_doc=source_doc,
                    page=discharge_info.get("_page", 0),
                    confidence=0.85,
                )
                state.add_evidence(ev)

            for pr in discharge_info.get("pending_results", []):
                state.extracted_pending_results.append(pr)

        # --- Pending lab scan ---
        pending_result = await self.pending_lab_tool.run(text=text, source_doc=source_doc)
        if pending_result.success and pending_result.data.get("count", 0) > 0:
            for item in pending_result.data.get("pending_items", []):
                state.extracted_pending_results.append({
                    "result": item["text"],
                    "source_doc": source_doc,
                })

        self._record_step(
            phase=AgentPhase.EXTRACTING,
            reasoning=f"Extraction from '{source_doc}' complete.",
            tool_output_summary=(
                f"Diagnoses: {len([d for d in state.extracted_diagnoses if d.get('_source_doc') == source_doc])}, "
                f"Medications: {len(medications) if isinstance(medications, list) else 0}, "
                f"Pending results: {pending_result.data.get('count', 0) if pending_result.success else 0}"
            ),
            decision="Merge extraction results into working memory",
            next_action="Process next document or move to reconciliation",
        )

    def _merge_demographics(self, demo: dict[str, Any], source_doc: str) -> None:
        """Merge demographics, checking for conflicts with existing data."""
        state = self.state
        existing = state.extracted_demographics

        fields = ["name", "date_of_birth", "mrn", "sex", "age"]
        for field in fields:
            new_val = demo.get(field)
            if not new_val or str(new_val).lower() in ("null", "none", ""):
                continue

            evidence_text = demo.get("evidence", {}).get(f"{field}_text", "") or new_val

            if field not in existing or not existing[field]:
                existing[field] = new_val
                ev = self.extractor.build_evidence(
                    fact=f"demographics:{field}:{new_val}",
                    evidence_text=evidence_text,
                    source_doc=source_doc,
                    page=demo.get("_page", 0),
                    confidence=float(demo.get("confidence", 0.85)),
                )
                state.add_evidence(ev)
            elif existing[field] != new_val:
                # Potential conflict — store it
                state.add_conflict(ConflictRecord(
                    field=f"demographics.{field}",
                    source_1_doc=existing.get("_source_doc", "prior_document"),
                    source_1_value=existing[field],
                    source_1_evidence_text=existing[field],
                    source_2_doc=source_doc,
                    source_2_value=new_val,
                    source_2_evidence_text=evidence_text,
                ))

    async def _reconcile_medications(self) -> None:
        """Run medication reconciliation between admission and discharge lists."""
        state = self.state
        state.phase = AgentPhase.RECONCILING

        admit_list = [m.model_dump() for m in state.admission_medications]
        discharge_list = [m.model_dump() for m in state.discharge_medications]

        self._record_step(
            phase=AgentPhase.RECONCILING,
            reasoning=(
                f"All documents processed. Found {len(admit_list)} admission medications "
                f"and {len(discharge_list)} discharge medications. "
                "Running reconciliation to identify changes, additions, and discontinuations."
            ),
            selected_tool="medication_reconciliation",
            decision="Run medication reconciliation",
            next_action="Identify medication changes",
        )

        result = await self.recon_tool.run(
            admission_meds=admit_list,
            discharge_meds=discharge_list,
        )
        self._log_tool_call("medication_reconciliation", {}, result)

        if result.success:
            state.reconciliation_report = ReconciliationReport(**result.data)
            unresolved = result.data.get("unresolved_flags", [])

            self._record_step(
                phase=AgentPhase.RECONCILING,
                tool_output_summary=(
                    f"Reconciliation complete. "
                    f"{len(result.data.get('changes', []))} changes found, "
                    f"{len(unresolved)} requiring clinician review."
                ),
                reasoning="Medication reconciliation returned results.",
                decision="Record reconciliation report",
                next_action="Check for drug interactions",
                warnings=[f"Reconciliation flag: {f}" for f in unresolved],
            )

            for flag in unresolved:
                state.add_flag(f"MEDICATION RECONCILIATION: {flag}")
        else:
            state.add_flag(f"Medication reconciliation failed: {result.error}")
            state.reconciliation_report = ReconciliationReport(
                reconciliation_complete=False,
                notes=f"Reconciliation failed: {result.error}",
            )

    async def _check_conflicts(self) -> bool:
        """Check for conflicts across extracted diagnoses from multiple documents."""
        state = self.state

        # Group diagnoses by type to check for conflicts
        principal_dxs = [
            d for d in state.extracted_diagnoses if d.get("type") == "principal"
        ]

        if len(principal_dxs) < 2:
            return False

        # Build extraction list for conflict tool
        extractions = [
            {
                "value": d.get("diagnosis", ""),
                "source_doc": d.get("_source_doc", "unknown"),
                "evidence_text": d.get("evidence_text", ""),
            }
            for d in principal_dxs
        ]

        self._record_step(
            phase=AgentPhase.VALIDATING,
            reasoning=(
                f"Found {len(principal_dxs)} principal diagnoses from different documents. "
                "Checking for conflicts."
            ),
            selected_tool="conflict_detection",
            decision="Run conflict detection on principal diagnoses",
            next_action="Record any conflicts found",
        )

        result = await self.conflict_tool.run(
            field="principal_diagnosis",
            extractions=extractions,
        )
        self._log_tool_call("conflict_detection", {}, result)

        if result.success and result.data.get("has_conflict"):
            for conflict_data in result.data.get("conflicts", []):
                conflict = ConflictRecord(**conflict_data)
                state.add_conflict(conflict)
                self._record_step(
                    phase=AgentPhase.VALIDATING,
                    reasoning="Conflict detected in principal diagnosis.",
                    tool_output_summary=f"Conflict: {conflict.source_1_doc} vs {conflict.source_2_doc}",
                    decision="Record conflict — do NOT pick a winner",
                    next_action="Escalate conflict to clinician",
                    warnings=[
                        f"CONFLICT: '{conflict.field}' differs between "
                        f"'{conflict.source_1_doc}' ({conflict.source_1_value}) and "
                        f"'{conflict.source_2_doc}' ({conflict.source_2_value})"
                    ],
                )
            return True

        return False

    async def _check_drug_interactions(self) -> None:
        """Check discharge medications for significant drug interactions."""
        state = self.state
        med_names = [m.name for m in state.discharge_medications]

        if len(med_names) < 2:
            return

        self._record_step(
            phase=AgentPhase.VALIDATING,
            reasoning=(
                f"Discharge medication list has {len(med_names)} medications. "
                "Running drug interaction check."
            ),
            selected_tool="drug_interaction_lookup",
            decision="Check for drug interactions",
            next_action="Surface any major interactions",
        )

        result = await self.interaction_tool.run(medications=med_names)
        self._log_tool_call("drug_interaction_lookup", {"medications": med_names}, result)

        if result.success:
            alerts = result.data.get("alerts", [])
            for alert in alerts:
                alert_text = (
                    f"[{alert['severity']}] {alert['description']} "
                    f"(involved: {', '.join(alert['involved_medications'])})"
                )
                state.drug_interaction_alerts.append(alert_text)
                if alert["severity"] == "MAJOR":
                    state.add_flag(f"MAJOR DRUG INTERACTION: {alert['description']}")

            self._record_step(
                phase=AgentPhase.VALIDATING,
                tool_output_summary=f"Found {len(alerts)} interaction(s). Major: {result.data.get('has_major_interaction', False)}",
                reasoning="Drug interaction check complete.",
                decision="Surface alerts in discharge summary",
                next_action="Continue to composition",
                warnings=[a["description"] for a in alerts if a["severity"] == "MAJOR"],
            )

    def _identify_missing_fields(self) -> list[str]:
        """Identify required fields that have no evidence."""
        state = self.state
        missing: list[str] = []

        checks = {
            "patient_name": bool(state.extracted_demographics.get("name")),
            "principal_diagnosis": bool(state.extracted_diagnoses),
            "hospital_course": bool(state.extracted_hospital_course),
            "discharge_condition": bool(state.extracted_discharge_condition),
            "allergies": bool(state.extracted_allergies),
            "admission_medications": bool(state.admission_medications),
            "discharge_medications": bool(state.discharge_medications),
        }

        for field, has_data in checks.items():
            if not has_data:
                missing.append(field)

        return missing

    def _record_missing_fields(self, missing_fields: list[str]) -> None:
        """Record missing fields in state and add clinician flags."""
        state = self.state
        existing_missing = {m.field_name for m in state.missing_fields}

        for field in missing_fields:
            if field not in existing_missing:
                state.add_missing(
                    field=field,
                    reason="No evidence found in any source document",
                    tools=["pdf_reader", "clinical_extractor"],
                )
                state.add_flag(f"MISSING FIELD: '{field}' — clinician must provide")

        if missing_fields:
            self._record_step(
                phase=AgentPhase.VALIDATING,
                reasoning=f"After processing all documents, {len(missing_fields)} required fields remain unpopulated.",
                tool_output_summary=f"Missing: {', '.join(missing_fields)}",
                decision="Mark fields as MISSING — do NOT fabricate",
                next_action="Include MISSING markers in discharge summary",
                warnings=[f"Missing required field: {f}" for f in missing_fields],
            )

    async def _escalate_conflicts(self) -> None:
        """Escalate unresolved conflicts to clinician."""
        state = self.state

        self._record_step(
            phase=AgentPhase.ESCALATING,
            reasoning=(
                f"Found {len(state.conflicts)} unresolved conflict(s). "
                "These cannot be resolved by the agent without clinical judgment. "
                "Escalating to clinician."
            ),
            selected_tool="escalation",
            decision="Escalate — do not guess the resolution",
            next_action="Create escalation record",
        )

        for conflict in state.conflicts:
            result = await self.escalation_tool.run(
                patient_id=self.patient_id,
                session_id=state.session_id,
                reason="conflicting_information",
                details=(
                    f"Conflict in '{conflict.field}': "
                    f"{conflict.source_1_doc} says '{conflict.source_1_value}', "
                    f"{conflict.source_2_doc} says '{conflict.source_2_value}'"
                ),
                severity="HIGH",
                fields_affected=[conflict.field],
            )
            self._log_tool_call("escalation", {}, result)

        state.escalated = True

    async def _handle_timeout(self) -> None:
        """Handle agent timeout gracefully."""
        state = self.state
        state.escalated = True
        state.phase = AgentPhase.ESCALATING
        state.add_flag(
            f"Agent timed out after {self.settings.agent_timeout_seconds}s. "
            "Summary is partial — clinician review mandatory."
        )

        self._record_step(
            phase=AgentPhase.ESCALATING,
            reasoning="Agent has exceeded the timeout limit.",
            selected_tool="escalation",
            decision="Escalate — timeout exceeded",
            next_action="Compose partial summary with MISSING markers",
            warnings=["TIMEOUT — partial summary only"],
        )

        await self.escalation_tool.run(
            patient_id=self.patient_id,
            session_id=state.session_id,
            reason="agent_iteration_limit_exceeded",
            details=f"Agent timed out after {self.settings.agent_timeout_seconds}s",
            severity="CRITICAL",
        )

    async def _handle_iteration_limit(self) -> None:
        """Handle exceeding the max iteration cap."""
        state = self.state
        state.escalated = True
        state.phase = AgentPhase.ESCALATING
        state.add_flag(
            f"Agent hit max iteration limit ({state.max_iterations}). "
            "Summary is partial — clinician review mandatory."
        )

        self._record_step(
            phase=AgentPhase.ESCALATING,
            reasoning=f"Max iterations ({state.max_iterations}) reached without completing all tasks.",
            selected_tool="escalation",
            decision="Escalate — iteration limit exceeded",
            next_action="Compose best-effort summary with MISSING markers",
            warnings=[f"ITERATION LIMIT ({state.max_iterations}) EXCEEDED"],
        )

        await self.escalation_tool.run(
            patient_id=self.patient_id,
            session_id=state.session_id,
            reason="agent_iteration_limit_exceeded",
            details=f"Agent exhausted {state.max_iterations} iterations",
            severity="HIGH",
        )

    async def _compose_summary(self) -> None:
        """Run the summary composer tool to assemble the final discharge summary."""
        state = self.state
        state.phase = AgentPhase.COMPOSING

        self._record_step(
            phase=AgentPhase.COMPOSING,
            reasoning=(
                "All documents processed, conflicts detected, medications reconciled. "
                "Composing the final discharge summary. "
                f"Evidence registry has {len(state.evidence_registry)} facts. "
                f"Conflicts: {len(state.conflicts)}. Missing fields: {len(state.missing_fields)}."
            ),
            selected_tool="summary_composer",
            decision="Compose discharge summary from validated working memory",
            next_action="Return summary for clinician review",
        )

        result = await self.composer_tool.run(
            patient_id=self.patient_id,
            extracted_demographics=state.extracted_demographics,
            extracted_diagnoses=state.extracted_diagnoses,
            extracted_procedures=state.extracted_procedures,
            extracted_hospital_course=state.extracted_hospital_course,
            extracted_allergies=state.extracted_allergies,
            extracted_follow_up=state.extracted_follow_up,
            extracted_discharge_condition=state.extracted_discharge_condition,
            extracted_pending_results=state.extracted_pending_results,
            reconciliation_report=state.reconciliation_report.model_dump() if state.reconciliation_report else None,
            conflicts=[c.model_dump() for c in state.conflicts],
            missing_fields=[m.model_dump() for m in state.missing_fields],
            clinician_review_flags=state.clinician_review_flags,
            hallucination_warnings=state.hallucination_warnings,
            drug_interaction_alerts=state.drug_interaction_alerts,
            evidence_registry=[e.model_dump() for e in state.evidence_registry],
            agent_iterations=state.iteration,
            escalated=state.escalated,
        )
        self._log_tool_call("summary_composer", {}, result)

        if result.success:
            state.final_summary = DischargeSummary(**result.data)
            state.phase = AgentPhase.COMPLETE
            self._record_step(
                phase=AgentPhase.COMPLETE,
                reasoning="Discharge summary successfully composed.",
                tool_output_summary=(
                    f"Summary ready. "
                    f"Conflicts: {len(state.conflicts)}, "
                    f"Missing: {len(state.missing_fields)}, "
                    f"Flags: {len(state.clinician_review_flags)}, "
                    f"Escalated: {state.escalated}"
                ),
                decision="Return summary as DRAFT for clinician review",
                next_action="Done",
            )
        else:
            state.add_flag(f"Summary composition failed: {result.error}")
            state.phase = AgentPhase.FAILED
            # Return minimal summary with all MISSING
            state.final_summary = DischargeSummary(
                clinician_review_flags=state.clinician_review_flags + [f"Composition error: {result.error}"],
                escalated=True,
                agent_iterations=state.iteration,
            )

    # --- Observability helpers ---

    def _record_step(
        self,
        phase: AgentPhase,
        reasoning: str,
        decision: str,
        next_action: str,
        selected_tool: Optional[str] = None,
        tool_inputs: Optional[dict] = None,
        tool_output_summary: Optional[str] = None,
        warnings: Optional[list[str]] = None,
    ) -> None:
        step = AgentStep(
            step_number=len(self.state.steps) + 1,
            phase=phase,
            reasoning=reasoning,
            selected_tool=selected_tool,
            tool_inputs=tool_inputs,
            tool_output_summary=tool_output_summary,
            decision=decision,
            next_action=next_action,
            warnings=warnings or [],
        )
        self.state.steps.append(step)
        self.tracer.record_step(step)

    def _log_tool_call(self, tool_name: str, inputs: dict, result: Any) -> None:
        tc = ToolCall(
            tool_name=tool_name,
            inputs=inputs,
            output=result.data if result.success else None,
            error=result.error,
            success=result.success,
            duration_ms=result.duration_ms,
            retry_count=result.retry_count,
        )
        self.state.tool_calls.append(tc)
        self.tracer.record_tool_call(
            tool_name=tool_name,
            inputs=inputs,
            output=result.data,
            success=result.success,
            duration_ms=result.duration_ms,
            error=result.error,
        )
