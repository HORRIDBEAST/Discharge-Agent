"""
FastAPI routes for the discharge summary agent.

Endpoints:
  POST /api/v1/patients/{patient_id}/process
      Upload PDF files, run the agent, return the discharge summary.

  GET  /api/v1/patients/{patient_id}/summary
      Retrieve a cached summary (if available).

  GET  /api/v1/traces/{patient_id}
      Return the agent trace for a patient session.

  GET  /api/v1/health
      Health check.
"""
from __future__ import annotations
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.agents.discharge_agent import DischargeAgent
from app.models.discharge_summary import DischargeSummary
from app.configs.settings import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")
settings = get_settings()

# Simple in-memory cache for demo (production: use Redis/DB)
_summary_cache: dict[str, dict[str, Any]] = {}
_trace_cache: dict[str, str] = {}


@router.get("/health")
async def health_check():
    return {"status": "healthy", "model": settings.openai_model}


@router.post("/patients/{patient_id}/process")
async def process_patient(
    patient_id: str,
    files: list[UploadFile] = File(...),
):
    """
    Upload source note PDFs for a patient and run the discharge summary agent.
    Returns the structured discharge summary draft.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Save uploaded files to a temp directory
    tmp_dir = tempfile.mkdtemp(prefix=f"patient_{patient_id}_")
    saved_paths: list[str] = []

    try:
        for upload in files:
            if not upload.filename.endswith(".pdf"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Only PDF files accepted. Got: {upload.filename}"
                )
            content = await upload.read()
            dest = Path(tmp_dir) / upload.filename
            dest.write_bytes(content)
            saved_paths.append(str(dest))
            logger.info("file_saved", patient_id=patient_id, file=upload.filename, size=len(content))

        # Run agent
        agent = DischargeAgent(patient_id=patient_id, source_document_paths=saved_paths)
        summary = await agent.run()
        summary_dict = summary.model_dump()

        # Cache results
        _summary_cache[patient_id] = summary_dict
        trace_path = Path(settings.trace_dir) / f"{patient_id}_{agent.state.session_id[:8]}.json"
        if trace_path.exists():
            _trace_cache[patient_id] = str(trace_path)

        return JSONResponse(content=summary_dict)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("process_patient_error", patient_id=patient_id, error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")
    finally:
        # Clean up temp files
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/patients/{patient_id}/summary")
async def get_summary(patient_id: str):
    """Retrieve the most recent cached discharge summary for a patient."""
    summary = _summary_cache.get(patient_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"No summary found for patient '{patient_id}'")
    return JSONResponse(content=summary)


@router.get("/traces/{patient_id}")
async def get_trace(patient_id: str):
    """Return the agent trace for the most recent run of a patient."""
    trace_path = _trace_cache.get(patient_id)
    if not trace_path or not Path(trace_path).exists():
        # Try to find it in the trace directory
        trace_dir = Path(settings.trace_dir)
        matching = list(trace_dir.glob(f"{patient_id}_*.json"))
        if not matching:
            raise HTTPException(status_code=404, detail=f"No trace found for patient '{patient_id}'")
        trace_path = str(sorted(matching)[-1])  # Most recent

    with open(trace_path) as f:
        trace = json.load(f)
    return JSONResponse(content=trace)


@router.get("/patients")
async def list_patients():
    """List all patients with cached summaries."""
    return {"patients": list(_summary_cache.keys())}
