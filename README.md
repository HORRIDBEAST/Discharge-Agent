# Agentic AI for Discharge Summaries

> A clinically safe, evidence-grounded, agentic discharge summary generator with full observability traces, medication reconciliation, conflict detection, and a measurable doctor-edit learning loop.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DischargeAgent (Orchestrator)                │
│                                                                      │
│  AgentState (working memory)                                         │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  evidence_registry │ conflicts │ missing_fields │ tool_calls │    │
│  │  admission_meds    │ discharge_meds │ steps (trace)          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
│  Agent Loop (max 15 iterations, hard timeout 120s):                  │
│                                                                      │
│  ┌────────┐   ┌────────┐   ┌─────────┐   ┌──────────┐             │
│  │ PLAN   │──▶│  ACT   │──▶│ OBSERVE │──▶│ RE-PLAN  │──▶ loop     │
│  │ (what  │   │ (pick  │   │ (merge  │   │ (check   │             │
│  │  next?)│   │  tool) │   │  result)│   │  done?)  │             │
│  └────────┘   └────────┘   └─────────┘   └──────────┘             │
└─────────────────────────────────────────────────────────────────────┘

Layers:
  PDF Ingestion Layer    ──▶  PyMuPDF → pdfplumber → OCR stub
  Clinical Extractor     ──▶  LLM (gpt-4o, JSON mode, anti-fab prompt)
  Evidence Registry      ──▶  Every fact = {source, page, text, confidence}
  Tool Registry          ──▶  Dynamic dispatch via @register_tool
  Conflict Detector      ──▶  Pairwise comparison, never picks winner
  Med Reconciliation     ──▶  Admit vs discharge diff, flags undocumented changes
  Drug Interaction       ──▶  Mocked clinical DB (MAJOR/MODERATE/MINOR)
  Evidence Validator     ──▶  Anti-hallucination firewall at composition time
  Summary Composer       ──▶  Assembles final draft with MISSING sentinels
  Escalation Tool        ──▶  Formal escalation record for unresolvable issues
  Tracer                 ──▶  JSON + Markdown trace per session
  FastAPI Layer          ──▶  REST API for PDF upload and summary retrieval
```

---

## Quick Start

```bash
# 1. Clone and enter the project
cd discharge_agent

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env — set OPENAI_API_KEY

# 5. Generate synthetic patient data
python scripts/generate_sample_data.py

# 6. Run the agent on both patients
python scripts/run_agent.py --patient-id patient_001 --docs ./sample_data/patient_001/
python scripts/run_agent.py --patient-id patient_002 --docs ./sample_data/patient_002/

# 7. Run the demo (best for video recording)
python demo/demo_runner.py

# 8. Start the API server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 9. Run tests
pytest tests/ -v

# 10. Run Part 2 — Learning Loop
python evaluation/learning_loop.py \
    --patient-id patient_001 \
    --patient-dir ./sample_data/patient_001/ \
    --iterations 5
```

---

## API Usage

```bash
# Upload PDFs and process patient
curl -X POST http://localhost:8000/api/v1/patients/P001/process \
  -F "files=@sample_data/patient_001/admission_note.pdf" \
  -F "files=@sample_data/patient_001/progress_note_day2.pdf" \
  -F "files=@sample_data/patient_001/lab_results.pdf" \
  -F "files=@sample_data/patient_001/discharge_medications.pdf"

# Retrieve cached summary
curl http://localhost:8000/api/v1/patients/P001/summary

# Get agent trace
curl http://localhost:8000/api/v1/traces/P001

# Health check
curl http://localhost:8000/api/v1/health
```

---

## Agent Loop Design

The agent is **not a pipeline**. Each iteration it inspects its own state and dynamically decides what to do next:

```
while not finished and iteration < MAX_ITERATIONS:
    if unprocessed documents exist:
        → PDFReaderTool → ClinicalExtractor (parallel extractors)
        → merge into evidence_registry

    elif medication reconciliation not done:
        → MedicationReconciliationTool
        → flag undocumented changes

    elif conflicts not checked:
        → ConflictDetectionTool (pairwise across all docs)
        → store ConflictRecord — NEVER pick a winner

    elif drug interactions not checked:
        → DrugInteractionTool
        → escalate MAJOR interactions

    else:
        → identify missing required fields
        → escalate if conflicts exist and auto-escalate=true
        → SummaryComposerTool → DischargeSummary
        → finished = True
```

**Hard stops:**
- `MAX_ITERATIONS = 15` (configurable via env)
- `TIMEOUT = 120s`
- Either condition → escalate + compose best-effort summary

---

## No-Fabrication Guarantee

This is the most critical design constraint. Three layers enforce it:

### Layer 1 — Anti-Fabrication System Prompt
Every LLM call (via `LLMClient`) prepends `ANTI_FABRICATION_SYSTEM_PROMPT`:

> *"NEVER invent, infer, or guess any clinical fact. If information is not explicitly stated, return null. Return ONLY what is directly stated."*

The prompt explicitly instructs the model to return `null` or `"MISSING"` rather than fill gaps probabilistically.

### Layer 2 — Evidence Registry
Every extracted fact produces an `Evidence` object with:
```json
{
  "fact": "diagnosis:principal:NSTEMI",
  "source_document": "progress_note_day2.pdf",
  "page": 0,
  "evidence_text": "Troponin peak: 2.8 ng/mL — confirmed NSTEMI",
  "confidence": 0.94,
  "confidence_level": "high"
}
```
No evidence → field is `MISSING`, never invented.

### Layer 3 — EvidenceValidator (Composition Firewall)
Before any fact enters the discharge summary, `EvidenceValidator.validate_field()` checks:
1. At least one Evidence record exists for this field
2. Confidence ≥ 0.50 (configurable)
3. Source document is in the processed document set
4. Evidence text is non-empty

If any check fails → field value is replaced with `"MISSING — clinician review required"`.

---

## Conflict Handling

When two source documents disagree, the agent:

1. **Records a `ConflictRecord`** — never resolves it automatically
2. **Adds a clinician review flag** 
3. **Escalates the case** (if `CONFLICT_AUTO_ESCALATE=true`)

Example output:
```json
{
  "conflict_id": "a1b2c3d4",
  "field": "principal_diagnosis",
  "source_1_doc": "progress_note_1.pdf",
  "source_1_value": "Urinary tract infection",
  "source_2_doc": "progress_note_2.pdf",
  "source_2_value": "Community-acquired pneumonia",
  "status": "clinician_review_required"
}
```

The discharged summary will show: `"CONFLICT DETECTED — clinician review required"` for that field.

---

## Failure Handling

Every tool inherits from `BaseTool` which:
- Retries up to `max_retries` times (default 3) with exponential backoff
- **Never raises** — always returns a `ToolResult(success=True/False)`
- Logs every attempt (success, failure, error string)
- On total failure: the agent logs the failure, adds a clinician flag, and continues

The agent never crashes. Partial failures produce a partial-but-honest summary with explicit flags.

---

## Observability

Every agent step emits a trace entry:

```
┌─ STEP 4 [EXTRACTING] ──────────────────────────────────┐
│ 📋 Reasoning:                                           │
│    Document classified as 'progress_note'.              │
│    Will run targeted extractors for this document type. │
│                                                         │
│ 🔧 Tool Selected:                                       │
│    clinical_extractor                                   │
│                                                         │
│ 📤 Result:                                              │
│    Diagnoses: 1, Medications: 2, Pending results: 2     │
│                                                         │
│ ✅ Decision:                                            │
│    Merge extraction results into working memory         │
│                                                         │
│ ➡️  Next Action:                                         │
│    Process next document or move to reconciliation      │
└─────────────────────────────────────────────────────────┘
```

Traces are saved to `./traces/` as:
- **JSON**: `{patient_id}_{session_id[:8]}.json` — machine-readable
- **Markdown**: `{patient_id}_{session_id[:8]}.md` — human-readable

---

## Medication Reconciliation

Structured diff between admission and discharge medication lists:

| Medication | Change | Admission | Discharge | Reason | Flag |
|---|---|---|---|---|---|
| Aspirin | unchanged | 81mg daily | 81mg daily | - | ✓ |
| Atorvastatin | dose_changed | 40mg | 80mg | Post-ACS high-intensity | ✓ |
| Clopidogrel | added | - | 75mg daily | Post-PCI antiplatelet | ✓ |
| Ibuprofen | discontinued | 400mg PRN | - | Not documented | ⚠️ |

Any change **without a documented reason is flagged** for clinician reconciliation — never silently resolved.

---

## Part 2 — Learning from Doctor Edits

### Approach: Correction Memory + Prompt Injection

**Why this approach:**
- No GPU or fine-tuning required
- Immediate effect (no training loop lag)
- Interpretable — you can read the memory store as JSON
- Safe — memory is injected as hints, never as hard overrides
- Anti-fabrication rules always take precedence

**Loop design:**
```
Iteration N:
  1. Generate draft   → DischargeAgent
  2. Apply edits      → SimulatedReviewer (hidden, consistent policy)
  3. Measure reward   → normalised edit distance + section accuracy
  4. Store patterns   → CorrectionMemoryStore (JSON, indexed by field:edit_type)
  5. Inject memory    → next iteration's LLM prompts get few-shot correction examples

Iteration N+1:
  → Agent sees patterns like "GP follow-up should include timeframe"
  → Agent applies them where applicable, WITHOUT fabricating
```

**Reward signal:** `composite_score = 0.6 × section_accuracy + 0.2 × (1 − edit_distance) + 0.1 × pending_ownership + 0.1 × flag_efficiency`

### Limitations (honest)

1. **Cold start**: Memory is empty at iteration 1 — no benefit until patterns accumulate. With 2 patients and 5 iterations, improvement is modest but measurable.

2. **Reward hacking risk**: Optimising edit distance could be gamed by making text vaguer or mimicking reviewer style without clinical accuracy. Mitigation: `missing_field_count` is tracked as a safety counter-metric. Any increase in missing fields signals degradation.

3. **Distribution shift**: The simulated reviewer's policy is fixed and deterministic. Real reviewers vary. Learned patterns may overfit.

4. **Safety preserved**: Memory injection is additive. The anti-fabrication system prompt always prepends memory content and takes precedence. Even with memory injection, the agent returns `MISSING` rather than fabricate.

---

## Project Structure

```
discharge_agent/
├── app/
│   ├── agents/
│   │   └── discharge_agent.py      # Core agent loop
│   ├── tools/
│   │   ├── base.py                 # BaseTool with retry + never-crash guarantee
│   │   ├── pdf_reader_tool.py
│   │   ├── medication_reconciliation_tool.py
│   │   ├── conflict_detection_tool.py
│   │   ├── drug_interaction_tool.py
│   │   ├── pending_lab_tool.py
│   │   ├── escalation_tool.py
│   │   └── summary_composer_tool.py
│   ├── services/
│   │   ├── llm_client.py           # OpenAI wrapper (retry, anti-fab prompt)
│   │   └── clinical_extractor.py  # Field-specific LLM extractors
│   ├── parsers/
│   │   └── pdf_parser.py          # PyMuPDF → pdfplumber → OCR stub
│   ├── models/
│   │   ├── agent_state.py         # AgentState (working memory)
│   │   ├── evidence.py            # Evidence + ConflictRecord + MissingField
│   │   ├── medications.py         # Medication reconciliation models
│   │   └── discharge_summary.py   # Final output schema
│   ├── validators/
│   │   └── evidence_validator.py  # Anti-hallucination firewall
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints
│   ├── configs/
│   │   └── settings.py            # Pydantic settings
│   └── utils/
│       ├── logger.py              # structlog
│       └── tracer.py              # Rich console + JSON/MD file traces
├── evaluation/
│   ├── simulated_reviewer.py      # Deterministic doctor simulation
│   ├── metrics.py                 # Edit distance, section accuracy, composite score
│   ├── correction_memory.py       # Persistent learning store
│   └── learning_loop.py           # Part 2 end-to-end loop
├── tests/                         # pytest suite
├── scripts/
│   ├── run_agent.py               # CLI runner
│   └── generate_sample_data.py    # Synthetic patient PDF generator
├── demo/
│   └── demo_runner.py             # Video-demo-friendly runner
├── sample_data/                   # Generated patient PDFs
├── sample_outputs/                # Generated summaries
├── traces/                        # Agent traces (JSON + MD)
├── .env.example
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| From-scratch agent loop (not LangGraph) | Full control over iteration logic, clearer interview code, no framework magic |
| `BaseTool` with retry → `ToolResult` | Uniform never-crash contract; agent always gets a result object |
| `EvidenceValidator` as composition-time firewall | Last line of defence — even if LLM ignores system prompt, validator catches it |
| JSON mode for all extractions | Forces structured output, eliminates prose-parsing bugs |
| `MISSING` sentinel (not None/empty) | Explicit, visible, impossible to accidentally treat as valid data |
| Concurrent extractors per document | `asyncio.gather` cuts latency when multiple fields extracted from one doc |
| Pydantic v2 throughout | Schema validation at every boundary — models are self-documenting |
| Correction memory as JSON store | Simple, inspectable, version-controllable learning artefact |
| Simulated reviewer with deterministic policy | Reproducible evaluation; improvement curve is real not noisy |

---

## What I Would Do With More Time

1. **Sentence-level evidence validation** — validate individual sentences in the hospital course narrative against specific evidence records, not just at field level.

2. **Real OCR integration** — pytesseract or AWS Textract for genuinely scanned PDFs.

3. **Structured entity extraction** — dedicated NER model (e.g., Med7, scispaCy) to supplement LLM extraction for medications and diagnoses.

4. **Temporal reasoning** — build explicit patient timeline, detect when lab values are pre/post-procedure.

5. **Preference fine-tuning (DPO)** — with more (draft, edited) pairs, train a small adapter that internalises reviewer preferences without prompt injection.

6. **Production escalation** — real integration with Epic/Cerner FHIR APIs for workflow-native escalation.

7. **Multi-reviewer support** — different reviewers have different styles; the memory store should be keyed by reviewer identity.

8. **Human-in-the-loop UI** — a clinician review interface where edits are captured automatically as training signal.

---

## Known Limitations

- The LLM extraction quality depends on document clarity; very degraded scans fall back to an OCR stub that returns a placeholder.
- The drug interaction database is mocked with ~10 pairs; production needs a licensed database (DrFirst, Multum).
- Medication name matching uses normalised string overlap — misses brand/generic synonyms without a drug ontology.
- The learning loop improvement is bounded by how many iterations/patients are run; with 5 iterations on 1 patient, gains are modest but demonstrable.
- The simulated reviewer uses a fixed policy — improvement metrics on this set do not prove generalisation to real clinicians.
