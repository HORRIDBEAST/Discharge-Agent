"""
Demo runner for the discharge summary agent.

Runs both patients sequentially with rich console output,
ideal for the video demo requirement.

Usage:
  python demo/demo_runner.py
  python demo/demo_runner.py --patient patient_001
"""
from __future__ import annotations
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.text import Text

app = typer.Typer()
console = Console()


def print_summary_rich(summary: dict, patient_id: str) -> None:
    """Pretty-print a discharge summary dict using Rich."""
    console.print()
    console.print(Panel(
        f"[bold white]DISCHARGE SUMMARY DRAFT[/bold white]\n"
        f"[dim]Patient: {patient_id} | Status: {summary.get('draft_status', 'draft')}[/dim]",
        border_style="bold cyan",
        expand=False,
    ))

    # Demographics
    demo = summary.get("patient", {})
    table = Table(title="Patient Demographics", show_header=False, box=None)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    for field, label in [
        ("name", "Name"), ("mrn", "MRN"), ("date_of_birth", "DOB"),
        ("sex", "Sex"), ("age", "Age"),
    ]:
        val = demo.get(field, "MISSING")
        style = "yellow" if "MISSING" in str(val) else "white"
        table.add_row(label, Text(str(val), style=style))
    console.print(table)

    console.print(f"\n[bold]Admission Date:[/bold] {summary.get('admission_date', 'MISSING')}")
    console.print(f"[bold]Discharge Date:[/bold] {summary.get('discharge_date', 'MISSING')}")

    # Diagnoses
    dx = summary.get("principal_diagnosis", "MISSING")
    dx_style = "red bold" if "CONFLICT" in str(dx) else "yellow" if "MISSING" in str(dx) else "green"
    console.print(f"\n[bold]Principal Diagnosis:[/bold] [{dx_style}]{dx}[/{dx_style}]")

    secondary = summary.get("secondary_diagnoses", [])
    if secondary:
        console.print("[bold]Secondary Diagnoses:[/bold]")
        for d in secondary:
            console.print(f"  • {d}")

    # Hospital course
    hc = summary.get("hospital_course", "MISSING")
    hc_panel_style = "yellow" if "MISSING" in str(hc) else "white"
    console.print(Panel(
        str(hc),
        title="Hospital Course",
        border_style=hc_panel_style,
        expand=False,
    ))

    # Medication reconciliation
    recon = summary.get("reconciliation_report")
    if recon:
        changes = recon.get("changes", [])
        unresolved = recon.get("unresolved_flags", [])

        med_table = Table(title="Medication Reconciliation", show_header=True)
        med_table.add_column("Medication", style="bold")
        med_table.add_column("Change")
        med_table.add_column("Admit Dose")
        med_table.add_column("DC Dose")
        med_table.add_column("Flag", justify="center")

        for ch in changes:
            flag_text = "⚠️ " if ch.get("flag_for_reconciliation") else "✓"
            flag_style = "yellow" if ch.get("flag_for_reconciliation") else "green"
            med_table.add_row(
                ch.get("medication_name", ""),
                ch.get("change_type", ""),
                str(ch.get("admission_value") or "-"),
                str(ch.get("discharge_value") or "-"),
                Text(flag_text, style=flag_style),
            )
        console.print(med_table)

        if unresolved:
            console.print(f"[yellow]⚠️  {len(unresolved)} medication change(s) require reconciliation[/yellow]")

    # Allergies
    allergies = summary.get("allergies", [])
    console.print(f"\n[bold]Allergies:[/bold] {', '.join(allergies) if allergies else 'None documented'}")

    # Discharge condition
    dc = summary.get("discharge_condition", "MISSING")
    console.print(f"[bold]Discharge Condition:[/bold] {dc}")

    # Pending results
    pending = summary.get("pending_results", [])
    if pending:
        console.print("\n[bold yellow]⏳ Pending Results:[/bold yellow]")
        for p in pending:
            console.print(f"  • {p}", style="yellow")

    # Conflicts
    conflicts = summary.get("conflicts_detected", [])
    if conflicts:
        console.print(f"\n[bold red]⚠️  {len(conflicts)} CONFLICT(S) DETECTED:[/bold red]")
        for c in conflicts:
            console.print(
                f"  • Field: [bold]{c['field']}[/bold]\n"
                f"    {c['source_1_doc']}: '{c['source_1_value']}'\n"
                f"    {c['source_2_doc']}: '{c['source_2_value']}'\n"
                f"    Status: [red]{c['status']}[/red]",
                highlight=False,
            )

    # Clinician review flags
    flags = summary.get("clinician_review_flags", [])
    if flags:
        console.print(f"\n[bold red]🔴 Clinician Review Required ({len(flags)} flag(s)):[/bold red]")
        for flag in flags[:10]:  # Show max 10
            console.print(f"  • {flag}", style="red")

    # Agent metadata
    console.print(
        f"\n[dim]Agent iterations: {summary.get('agent_iterations', '?')} | "
        f"Escalated: {summary.get('escalated', False)} | "
        f"Evidence records: {len(summary.get('evidence_map', {}))}[/dim]"
    )


@app.command()
def demo(
    patient: str = typer.Option("all", "--patient", help="'all', 'patient_001', or 'patient_002'"),
    data_dir: str = typer.Option("./sample_data", "--data-dir"),
    output_dir: str = typer.Option("./sample_outputs", "--output"),
):
    """
    Run the discharge summary agent demo on one or both synthetic patients.
    """
    from app.agents.discharge_agent import DischargeAgent
    from app.utils.logger import configure_logging
    configure_logging("WARNING")

    data_path = Path(data_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    patients = []
    if patient == "all":
        patients = ["patient_001", "patient_002"]
    else:
        patients = [patient]

    for patient_id in patients:
        patient_dir = data_path / patient_id
        if not patient_dir.exists():
            console.print(f"[yellow]Warning: {patient_dir} not found. Run scripts/generate_sample_data.py first.[/yellow]")
            continue

        pdf_files = sorted(patient_dir.glob("*.pdf"))
        if not pdf_files:
            # Try txt fallback
            pdf_files = sorted(patient_dir.glob("*.txt"))

        if not pdf_files:
            console.print(f"[red]No source files found in {patient_dir}[/red]")
            continue

        console.print()
        console.print(Panel(
            f"[bold cyan]Processing Patient: {patient_id}[/bold cyan]\n"
            f"Source documents: {', '.join(f.name for f in pdf_files)}",
            border_style="cyan",
        ))

        start = time.time()
        agent = DischargeAgent(
            patient_id=patient_id,
            source_document_paths=[str(f) for f in pdf_files],
        )
        summary = asyncio.run(agent.run())
        elapsed = round(time.time() - start, 1)

        summary_dict = summary.model_dump()
        print_summary_rich(summary_dict, patient_id)

        # Save outputs
        json_path = out_path / f"{patient_id}_discharge_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary_dict, f, indent=2, default=str)

        console.print(f"\n[green]✓ Completed in {elapsed}s → {json_path}[/green]")
        console.print("─" * 80)

    console.print("\n[bold green]Demo complete![/bold green]")
    console.print(f"Trace files written to: {Path('./traces').absolute()}")


if __name__ == "__main__":
    app()
