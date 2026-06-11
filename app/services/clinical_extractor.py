"""
Clinical fact extractor.

Takes raw document text and extracts structured clinical facts
with provenance. Every returned fact includes the exact supporting
text so the EvidenceValidator can verify it later.

This is NOT a single monolithic extraction — the agent calls
specific extractors per document type and per field, allowing
re-extraction when conflicts are found.
"""
from __future__ import annotations
import json
from typing import Any, Optional

from app.models.evidence import Evidence, ConfidenceLevel
from app.models.medications import Medication
from app.services.llm_client import LLMClient
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _confidence_from_raw(val: float | str | None) -> float:
    """Normalise confidence from LLM output (may be 0-1 or 0-100)."""
    if val is None:
        return 0.5
    try:
        v = float(val)
        return v / 100.0 if v > 1.0 else v
    except (ValueError, TypeError):
        return 0.5


class ClinicalExtractor:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def extract_demographics(
        self, text: str, source_doc: str, page: int = 0
    ) -> dict[str, Any]:
        """Extract patient demographics from document text."""
        prompt = f"""Extract patient demographics from the following clinical note.

Source document: {source_doc}

Text:
{text[:6000]}

Return JSON with this exact structure:
{{
  "name": "<string or null>",
  "date_of_birth": "<string or null>",
  "mrn": "<string or null>",
  "sex": "<string or null>",
  "age": "<string or null>",
  "evidence": {{
    "name_text": "<exact quote from source or null>",
    "dob_text": "<exact quote from source or null>",
    "mrn_text": "<exact quote from source or null>",
    "sex_text": "<exact quote from source or null>",
    "age_text": "<exact quote from source or null>"
  }},
  "confidence": <0.0 to 1.0>
}}

Return null for any field not found. Do NOT guess or fabricate."""

        result = await self.llm.extract_structured(prompt)
        if not result:
            logger.warning("demographics_extraction_failed", doc=source_doc)
            return {}
        result["_source_doc"] = source_doc
        result["_page"] = page
        return result

    async def extract_diagnoses(
        self, text: str, source_doc: str, page: int = 0
    ) -> list[dict[str, Any]]:
        """Extract principal and secondary diagnoses."""
        prompt = f"""Extract all diagnoses (principal and secondary) from the following clinical note.

Source document: {source_doc}

Text:
{text[:6000]}

Return JSON with this structure:
{{
  "diagnoses": [
    {{
      "type": "principal" or "secondary",
      "diagnosis": "<exact diagnosis string>",
      "evidence_text": "<exact quote from source that supports this>",
      "confidence": <0.0 to 1.0>,
      "section": "<section of note where found, e.g. Assessment>"
    }}
  ]
}}

Rules:
- Only include diagnoses explicitly stated in the text.
- Do NOT infer diagnoses from symptoms or lab values.
- If no diagnosis is stated, return empty list."""

        result = await self.llm.extract_structured(prompt)
        if not result:
            return []
        diagnoses = result.get("diagnoses", [])
        for d in diagnoses:
            d["_source_doc"] = source_doc
            d["_page"] = page
        return diagnoses

    async def extract_medications(
        self, text: str, source_doc: str, med_type: str = "unknown"
    ) -> list[Medication]:
        """Extract medication list from admission or discharge medication record."""
        prompt = f"""Extract all medications from the following clinical note.

Source document: {source_doc}
Medication type context: {med_type} (admission or discharge)

Text:
{text[:6000]}

Return JSON:
{{
  "medications": [
    {{
      "name": "<medication name>",
      "dose": "<dose or null>",
      "frequency": "<frequency or null>",
      "route": "<route or null>",
      "indication": "<indication if stated, or null>",
      "raw_text": "<exact text from source that mentions this medication>"
    }}
  ]
}}

- Extract only medications explicitly listed.
- Do NOT infer medications from diagnoses.
- Return empty list if no medications found."""

        result = await self.llm.extract_structured(prompt)
        if not result:
            return []

        meds = []
        for m in result.get("medications", []):
            try:
                meds.append(Medication(
                    name=m.get("name", "unknown"),
                    dose=m.get("dose"),
                    frequency=m.get("frequency"),
                    route=m.get("route"),
                    indication=m.get("indication"),
                    source_document=source_doc,
                    raw_text=m.get("raw_text", ""),
                ))
            except Exception as e:
                logger.warning("med_parse_error", error=str(e), med=m)
        return meds

    async def extract_hospital_course(
        self, text: str, source_doc: str, page: int = 0
    ) -> dict[str, Any]:
        """Extract the hospital course narrative."""
        prompt = f"""Extract the hospital course from the following clinical note.

Source document: {source_doc}

Text:
{text[:8000]}

Return JSON:
{{
  "hospital_course": "<concise narrative summarising the hospital stay, using only information stated in the text, or null if not found>",
  "evidence_sections": ["<list of section headers where this was found>"],
  "confidence": <0.0 to 1.0>
}}

- Do NOT add clinical interpretation or inference.
- Summarise only what is explicitly documented."""

        result = await self.llm.extract_structured(prompt)
        if not result:
            return {}
        result["_source_doc"] = source_doc
        result["_page"] = page
        return result

    async def extract_procedures(
        self, text: str, source_doc: str, page: int = 0
    ) -> list[dict[str, Any]]:
        """Extract procedures performed during admission."""
        prompt = f"""Extract all procedures performed during this hospital admission from the following text.

Source document: {source_doc}

Text:
{text[:6000]}

Return JSON:
{{
  "procedures": [
    {{
      "procedure": "<procedure name>",
      "date": "<date if stated or null>",
      "evidence_text": "<exact quote>",
      "confidence": <0.0 to 1.0>
    }}
  ]
}}

Return empty list if no procedures found. Do NOT infer procedures."""

        result = await self.llm.extract_structured(prompt)
        if not result:
            return []
        procs = result.get("procedures", [])
        for p in procs:
            p["_source_doc"] = source_doc
            p["_page"] = page
        return procs

    async def extract_allergies(
        self, text: str, source_doc: str, page: int = 0
    ) -> list[dict[str, Any]]:
        """Extract documented allergies."""
        prompt = f"""Extract all allergies documented in the following clinical note.

Source document: {source_doc}

Text:
{text[:4000]}

Return JSON:
{{
  "allergies": [
    {{
      "allergen": "<name>",
      "reaction": "<reaction type or null>",
      "evidence_text": "<exact quote>"
    }}
  ]
}}

Return empty list if no allergies documented. 'NKDA' (No Known Drug Allergies) should be recorded as a single entry."""

        result = await self.llm.extract_structured(prompt)
        if not result:
            return []
        allergies = result.get("allergies", [])
        for a in allergies:
            a["_source_doc"] = source_doc
            a["_page"] = page
        return allergies

    async def extract_discharge_info(
        self, text: str, source_doc: str, page: int = 0
    ) -> dict[str, Any]:
        """Extract discharge condition, follow-up, and pending results."""
        prompt = f"""Extract discharge information from the following clinical note.

Source document: {source_doc}

Text:
{text[:6000]}

Return JSON:
{{
  "discharge_condition": "<Good/Fair/Poor/Stable/Critical or null>",
  "discharge_condition_evidence": "<exact quote or null>",
  "follow_up_instructions": [
    {{"instruction": "<text>", "evidence_text": "<exact quote>"}}
  ],
  "pending_results": [
    {{"result": "<pending lab or test name>", "evidence_text": "<exact quote>"}}
  ],
  "admission_date": "<date string or null>",
  "discharge_date": "<date string or null>",
  "confidence": <0.0 to 1.0>
}}

Return null for any field not explicitly stated."""

        result = await self.llm.extract_structured(prompt)
        if not result:
            return {}
        result["_source_doc"] = source_doc
        result["_page"] = page
        return result

    async def classify_document_type(self, filename: str, text_sample: str) -> str:
        """
        Classify a document by type so the agent knows which extractor to prioritise.
        Returns one of: admission_note, progress_note, lab_results, medication_record, discharge_note, unknown
        """
        prompt = f"""Classify this clinical document into exactly one of these types:
- admission_note
- progress_note
- lab_results
- medication_record
- discharge_note
- unknown

Filename: {filename}
Text sample (first 500 chars):
{text_sample[:500]}

Return JSON: {{"document_type": "<type>", "confidence": <0.0-1.0>}}"""

        result = await self.llm.extract_structured(prompt)
        if result:
            return result.get("document_type", "unknown")
        return "unknown"

    def build_evidence(
        self,
        fact: str,
        evidence_text: str,
        source_doc: str,
        page: int,
        confidence: float,
        extraction_method: str = "llm",
        section: Optional[str] = None,
    ) -> Evidence:
        """Construct an Evidence object with proper provenance."""
        level = (
            ConfidenceLevel.HIGH if confidence >= 0.85
            else ConfidenceLevel.MEDIUM if confidence >= 0.6
            else ConfidenceLevel.LOW
        )
        return Evidence(
            fact=fact,
            source_document=source_doc,
            page=page,
            evidence_text=evidence_text or "extracted from document",
            confidence=confidence,
            confidence_level=level,
            extraction_method=extraction_method,
            section_in_doc=section,
        )
