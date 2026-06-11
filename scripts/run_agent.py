"""
CLI interface for the discharge summary agent.

Usage:
  python scripts/run_agent.py --patient-id P001 --docs ./sample_data/patient_001/

  python scripts/run_agent.py --patient-id P002 \
      --docs ./sample_data/patient_002/ \
      --output ./sample_outputs/

Runs the agent on a folder of PDFs and writes:
  - JSON discharge summary
  - Markdown discharge summary
  - Agent trace (JSON + Markdown)
"""
from __future__ import annotations
import asyncio
import json
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.markdown import Markdown

from app.agents.discharge_agent import DischargeAgent
from app.utils.logger import configure_logging
from app.configs.settings import get_settings

app = typer.Typer()
console = Console()


@app.command()
def run(
    patient_id: str = typer.Option(..., "--patient-id", "-p", help="Patient identifier"),
    docs_dir: str = typer.Option(..., "--docs", "-d", help="Directory containing source note PDFs"),
    output_dir: str = typer.Option("./sample_outputs", "--output", "-o", help="Output directory"),
    log_level: str = typer.Option("INFO", "--log-level", help="Log level"),
):
    """Run the discharge summary agent on a patient's source notes."""
    configure_logging(log_level)
    settings = get_settings()

    docs_path = Path(docs_dir)
    if not docs_path.exists():
        console.print(f"[red]Error: docs directory not found: {docs_dir}[/red]")
        raise typer.Exit(1)

    pdf_files = sorted(docs_path.glob("*.pdf"))
    if not pdf_files:
        console.print(f"[red]Error: No PDF files found in {docs_dir}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold cyan]Discharge Summary Agent[/bold cyan]")
    console.print(f"Patient: [bold]{patient_id}[/bold]")
    console.print(f"Documents: {len(pdf_files)} PDF(s) found")
    for f in pdf_files:
        console.print(f"  • {f.name}")
    console.print()

    # Run agent
    agent = DischargeAgent(
        patient_id=patient_id,
        source_document_paths=[str(f) for f in pdf_files],
    )

    summary = asyncio.run(agent.run())

    # Save outputs
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    json_out = out_path / f"{patient_id}_discharge_summary.json"
    md_out = out_path / f"{patient_id}_discharge_summary.md"

    with open(json_out, "w") as f:
        json.dump(summary.model_dump(), f, indent=2, default=str)

    _write_markdown_summary(summary, md_out, patient_id)

    console.print(f"\n[bold green]✓ Outputs saved:[/bold green]")
    console.print(f"  Summary JSON: {json_out}")
    console.print(f"  Summary MD:   {md_out}")
    console.print(f"  Trace dir:    {settings.trace_dir}/")


def _write_markdown_summary(summary, path: Path, patient_id: str) -> None:
    from app.models.discharge_summary import MISSING, PENDING
    from app.models.medications import MedChangeType

    lines = [
        f"# Discharge Summary — {patient_id}",
        "",
        "> **DRAFT FOR CLINICIAN REVIEW — NOT FOR CLINICAL USE WITHOUT VERIFICATION**",
        "",
        "---",
        "",
        "## Patient Demographics",
        f"- **Name:** {summary.patient.name}",
        f"- **MRN:** {summary.patient.mrn}",
        f"- **DOB:** {summary.patient.date_of_birth}",
        f"- **Sex:** {summary.patient.sex}",
        f"- **Age:** {summary.patient.age}",
        "",
        "## Admission / Discharge",
        f"- **Admission Date:** {summary.admission_date}",
        f"- **Discharge Date:** {summary.discharge_date}",
        "",
        "## Principal Diagnosis",
        f"{summary.principal_diagnosis}",
        "",
        "## Secondary Diagnoses",
    ]

    if summary.secondary_diagnoses:
        for dx in summary.secondary_diagnoses:
            lines.append(f"- {dx}")
    else:
        lines.append("_(None documented)_")

    lines += [
        "",
        "## Hospital Course",
        summary.hospital_course or MISSING,
        "",
        "## Procedures",
    ]

    if summary.procedures:
        for p in summary.procedures:
            lines.append(f"- {p}")
    else:
        lines.append("_(None documented)_")

    lines += ["", "## Medication Reconciliation"]

    if summary.reconciliation_report:
        rr = summary.reconciliation_report
        lines.append(f"\n**Reconciliation status:** {'Complete ✓' if rr.reconciliation_complete else '⚠️ Incomplete — clinician review required'}")
        lines.append("\n| Medication | Change | Admission | Discharge | Reason | Flag |")
        lines.append("|------------|--------|-----------|-----------|--------|------|")
        for ch in rr.changes:
            flag = "⚠️" if ch.flag_for_reconciliation else "✓"
            lines.append(
                f"| {ch.medication_name} | {ch.change_type.value} | "
                f"{ch.admission_value or '-'} | {ch.discharge_value or '-'} | "
                f"{ch.reason_text or 'Not documented'} | {flag} |"
            )
    else:
        lines.append(MISSING)

    lines += ["", "## Allergies"]
    if summary.allergies:
        for a in summary.allergies:
            lines.append(f"- {a}")
    else:
        lines.append("_(None documented)_")

    lines += [
        "",
        "## Discharge Condition",
        summary.discharge_condition or MISSING,
        "",
        "## Follow-up Instructions",
    ]
    if summary.follow_up_instructions:
        for fi in summary.follow_up_instructions:
            lines.append(f"- {fi}")
    else:
        lines.append(MISSING)

    lines += ["", "## Pending Results"]
    if summary.pending_results:
        for pr in summary.pending_results:
            lines.append(f"- {pr}")
    else:
        lines.append("_(None documented)_")

    if summary.conflicts_detected:
        lines += ["", "## ⚠️ Conflicts Detected"]
        for c in summary.conflicts_detected:
            lines += [
                f"- **Field:** `{c.field}`",
                f"  - {c.source_1_doc}: *{c.source_1_value}*",
                f"  - {c.source_2_doc}: *{c.source_2_value}*",
                f"  - **Status:** {c.status}",
            ]

    if summary.clinician_review_flags:
        lines += ["", "## 🔴 Clinician Review Flags"]
        for flag in summary.clinician_review_flags:
            lines.append(f"- {flag}")

    lines += [
        "",
        "---",
        "",
        f"*Generated by DischargeAgent | Iterations: {summary.agent_iterations} | Escalated: {summary.escalated}*",
        "",
        f"*Status: {summary.draft_status}*",
    ]

    path.write_text("\n".join(lines))


if __name__ == "__main__":
    app()
