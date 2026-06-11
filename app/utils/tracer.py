"""
Observability tracer.

Emits both:
  1. Human-readable console traces (Rich)
  2. JSON trace files saved to disk

Every agent step is recorded with:
  - step number
  - reasoning
  - tool selected
  - inputs / outputs
  - decision
  - next action
  - warnings

This makes the agent's behaviour fully auditable.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from app.models.agent_state import AgentStep, AgentPhase
from app.utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


class AgentTracer:
    """
    Records and emits traces for every step the agent takes.
    Designed to be demo-friendly — output is visually clear.
    """

    def __init__(
        self,
        session_id: str,
        patient_id: str,
        trace_dir: str = "./traces",
        console_output: bool = True,
        file_output: bool = True,
    ):
        self.session_id = session_id
        self.patient_id = patient_id
        self.trace_dir = Path(trace_dir)
        self.console_output = console_output
        self.file_output = file_output
        self.steps: list[AgentStep] = []
        self.start_time = time.time()

        if file_output:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.json_path = self.trace_dir / f"{patient_id}_{session_id[:8]}.json"
            self.md_path = self.trace_dir / f"{patient_id}_{session_id[:8]}.md"

    def record_step(self, step: AgentStep) -> None:
        """Record a single agent step and emit it immediately."""
        self.steps.append(step)

        if self.console_output:
            self._print_step(step)

        if self.file_output:
            self._flush_json()
            self._append_markdown(step)

    def _print_step(self, step: AgentStep) -> None:
        phase_colors = {
            AgentPhase.PLANNING: "cyan",
            AgentPhase.EXTRACTING: "blue",
            AgentPhase.RECONCILING: "yellow",
            AgentPhase.VALIDATING: "magenta",
            AgentPhase.COMPOSING: "green",
            AgentPhase.ESCALATING: "red",
            AgentPhase.COMPLETE: "bright_green",
            AgentPhase.FAILED: "bright_red",
        }
        color = phase_colors.get(step.phase, "white")

        title = Text(f"  STEP {step.step_number}  [{step.phase.value.upper()}]  ", style=f"bold {color}")

        body = Text()
        body.append("📋 Reasoning:\n", style="bold")
        body.append(f"   {step.reasoning}\n\n")

        if step.selected_tool:
            body.append("🔧 Tool Selected:\n", style="bold")
            body.append(f"   {step.selected_tool}\n\n")

        if step.tool_output_summary:
            body.append("📤 Result:\n", style="bold")
            body.append(f"   {step.tool_output_summary}\n\n")

        body.append("✅ Decision:\n", style="bold")
        body.append(f"   {step.decision}\n\n")

        body.append("➡️  Next Action:\n", style="bold")
        body.append(f"   {step.next_action}\n")

        if step.warnings:
            body.append("\n⚠️  Warnings:\n", style="bold yellow")
            for w in step.warnings:
                body.append(f"   • {w}\n", style="yellow")

        console.print(Panel(body, title=title, border_style=color, expand=False))

    def _flush_json(self) -> None:
        trace = {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "elapsed_seconds": round(time.time() - self.start_time, 2),
            "steps": [s.model_dump() for s in self.steps],
        }
        with open(self.json_path, "w") as f:
            json.dump(trace, f, indent=2, default=str)

    def _append_markdown(self, step: AgentStep) -> None:
        mode = "a" if self.md_path.exists() else "w"
        with open(self.md_path, mode) as f:
            if step.step_number == 1:
                f.write(f"# Agent Trace — Patient: {self.patient_id}\n\n")
                f.write(f"Session: `{self.session_id}`\n\n---\n\n")

            f.write(f"## Step {step.step_number} — {step.phase.value.upper()}\n\n")
            f.write(f"**Reasoning:** {step.reasoning}\n\n")
            if step.selected_tool:
                f.write(f"**Tool:** `{step.selected_tool}`\n\n")
            if step.tool_output_summary:
                f.write(f"**Result:** {step.tool_output_summary}\n\n")
            f.write(f"**Decision:** {step.decision}\n\n")
            f.write(f"**Next Action:** {step.next_action}\n\n")
            if step.warnings:
                f.write("**Warnings:**\n")
                for w in step.warnings:
                    f.write(f"- ⚠️ {w}\n")
                f.write("\n")
            f.write("---\n\n")

    def print_summary(self, total_iterations: int, escalated: bool, flags: list[str]) -> None:
        """Print a final summary table after the agent completes."""
        elapsed = round(time.time() - self.start_time, 2)

        table = Table(title="Agent Run Summary", show_header=True)
        table.add_column("Metric", style="bold")
        table.add_column("Value")

        table.add_row("Patient", self.patient_id)
        table.add_row("Session", self.session_id[:12])
        table.add_row("Total Iterations", str(total_iterations))
        table.add_row("Elapsed", f"{elapsed}s")
        table.add_row("Escalated", "YES ⚠️" if escalated else "No")
        table.add_row("Clinician Flags", str(len(flags)))

        console.print(table)

        if flags:
            console.print("\n[bold yellow]Clinician Review Flags:[/bold yellow]")
            for flag in flags:
                console.print(f"  • {flag}", style="yellow")

    def record_tool_call(
        self,
        tool_name: str,
        inputs: dict[str, Any],
        output: Any,
        success: bool,
        duration_ms: float,
        error: Optional[str] = None,
    ) -> None:
        logger.info(
            "tool_call",
            tool=tool_name,
            success=success,
            duration_ms=round(duration_ms, 1),
            error=error,
        )
