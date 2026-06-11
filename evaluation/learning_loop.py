"""
Part 2 — Learning Loop.

Runs N iterations of:
  1. Generate discharge summary draft (agent)
  2. Apply simulated reviewer edits
  3. Measure edit distance (reward signal)
  4. Store correction patterns in memory
  5. Inject learned patterns into next iteration's prompts

Produces:
  - Before/after metric table
  - Improvement curve (saved as JSON + optional matplotlib chart)

Usage:
  python evaluation/learning_loop.py \
      --patient-dir ./sample_data/patient_001 \
      --patient-id P001 \
      --iterations 5
"""
from __future__ import annotations
import asyncio
import json
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from app.agents.discharge_agent import DischargeAgent
from app.utils.logger import configure_logging
from evaluation.simulated_reviewer import SimulatedReviewer
from evaluation.metrics import evaluate, average_score, EvaluationResult
from evaluation.correction_memory import CorrectionMemoryStore

app = typer.Typer()
console = Console()


async def _run_learning_iteration(
    patient_id: str,
    doc_paths: list[str],
    iteration: int,
    memory_store: CorrectionMemoryStore,
    reviewer: SimulatedReviewer,
) -> EvaluationResult:
    """Run one iteration of generate → review → measure → learn."""

    # Inject learned patterns into agent's extractor prompts
    # (In a more sophisticated system, this would patch the LLM prompt builder)
    prompt_injection = memory_store.build_prompt_injection([
        "hospital_course", "follow_up_instructions", "pending_results"
    ])

    if prompt_injection:
        os.environ["CORRECTION_MEMORY_INJECTION"] = prompt_injection
    else:
        os.environ.pop("CORRECTION_MEMORY_INJECTION", None)

    # Generate summary
    agent = DischargeAgent(patient_id=patient_id, source_document_paths=doc_paths)
    draft = await agent.run()

    # Apply reviewer edits
    edited, diffs = reviewer.review(draft)

    # Measure improvement
    result = evaluate(
        patient_id=patient_id,
        iteration=iteration,
        draft=draft,
        edited=edited,
    )

    # Store corrections in memory
    if diffs:
        memory_store.store_diffs(diffs, improvement_delta=result.composite_score)

    console.print(
        f"  Iteration {iteration:2d}: "
        f"score={result.composite_score:.3f}  "
        f"accuracy={result.overall_accuracy:.3f}  "
        f"edit_dist={result.overall_edit_distance:.3f}  "
        f"flags={result.flag_count}  "
        f"edits_by_reviewer={len(diffs)}"
    )

    return result


def _save_results(results: list[EvaluationResult], output_path: Path) -> None:
    data = {
        "results": [r.to_dict() for r in results],
        "summary": average_score(results),
        "before": results[0].to_dict() if results else {},
        "after": results[-1].to_dict() if results else {},
        "improvement": {
            "composite_score": (results[-1].composite_score - results[0].composite_score) if len(results) > 1 else 0,
            "accuracy": (results[-1].overall_accuracy - results[0].overall_accuracy) if len(results) > 1 else 0,
            "edit_distance_reduction": (results[0].overall_edit_distance - results[-1].overall_edit_distance) if len(results) > 1 else 0,
        }
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    console.print(f"\n[green]Results saved to {output_path}[/green]")


def _plot_improvement(results: list[EvaluationResult], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        iterations = [r.iteration for r in results]
        scores = [r.composite_score for r in results]
        accuracies = [r.overall_accuracy for r in results]
        edit_dists = [r.overall_edit_distance for r in results]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        fig.suptitle("Learning Loop — Improvement Over Iterations", fontsize=14, fontweight='bold')

        axes[0].plot(iterations, scores, 'b-o', linewidth=2, markersize=8)
        axes[0].set_title("Composite Score")
        axes[0].set_xlabel("Iteration")
        axes[0].set_ylabel("Score (higher = better)")
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=scores[0], color='r', linestyle='--', alpha=0.5, label=f'Baseline: {scores[0]:.3f}')
        axes[0].legend()

        axes[1].plot(iterations, accuracies, 'g-o', linewidth=2, markersize=8)
        axes[1].set_title("Section Accuracy")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("Accuracy (higher = better)")
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=accuracies[0], color='r', linestyle='--', alpha=0.5, label=f'Baseline: {accuracies[0]:.3f}')
        axes[1].legend()

        axes[2].plot(iterations, edit_dists, 'r-o', linewidth=2, markersize=8)
        axes[2].set_title("Edit Distance (lower = better)")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("Normalised Edit Distance")
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].axhline(y=edit_dists[0], color='b', linestyle='--', alpha=0.5, label=f'Baseline: {edit_dists[0]:.3f}')
        axes[2].legend()

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
        console.print(f"[green]Chart saved to {output_path}[/green]")
    except ImportError:
        console.print("[yellow]matplotlib not available — skipping chart[/yellow]")
    except Exception as e:
        console.print(f"[yellow]Chart generation failed: {e}[/yellow]")


def _print_results_table(results: list[EvaluationResult]) -> None:
    table = Table(title="Learning Loop Results", show_header=True)
    table.add_column("Iter", style="bold", width=5)
    table.add_column("Composite ↑", width=12)
    table.add_column("Accuracy ↑", width=12)
    table.add_column("Edit Dist ↓", width=12)
    table.add_column("Flags", width=8)
    table.add_column("Pending Owner", width=14)

    for r in results:
        table.add_row(
            str(r.iteration),
            f"{r.composite_score:.3f}",
            f"{r.overall_accuracy:.3f}",
            f"{r.overall_edit_distance:.3f}",
            str(r.flag_count),
            f"{r.pending_with_owner:.1%}",
        )

    console.print(table)

    if len(results) > 1:
        first = results[0]
        last = results[-1]
        console.print(f"\n[bold]Improvement (Iteration 1 → {last.iteration}):[/bold]")
        delta_score = last.composite_score - first.composite_score
        delta_acc = last.overall_accuracy - first.overall_accuracy
        delta_ed = first.overall_edit_distance - last.overall_edit_distance

        color_s = "green" if delta_score > 0 else "red"
        color_a = "green" if delta_acc > 0 else "red"
        color_e = "green" if delta_ed > 0 else "red"

        console.print(f"  Composite Score:  [{color_s}]{delta_score:+.3f}[/{color_s}]")
        console.print(f"  Section Accuracy: [{color_a}]{delta_acc:+.3f}[/{color_a}]")
        console.print(f"  Edit Distance:    [{color_e}]{delta_ed:+.3f} reduction[/{color_e}]")


@app.command()
def run_learning_loop(
    patient_id: str = typer.Option(..., "--patient-id", "-p"),
    patient_dir: str = typer.Option(..., "--patient-dir", "-d"),
    iterations: int = typer.Option(5, "--iterations", "-n", help="Number of learning iterations"),
    output_dir: str = typer.Option("./evaluation/results", "--output", "-o"),
    memory_path: str = typer.Option("./evaluation/correction_memory.json", "--memory"),
    reset_memory: bool = typer.Option(False, "--reset-memory", help="Clear memory before running"),
):
    """
    Run the Part 2 learning loop: generate → review → measure → learn → repeat.
    """
    configure_logging("WARNING")

    docs_path = Path(patient_dir)
    pdf_files = sorted(docs_path.glob("*.pdf"))
    if not pdf_files:
        # Try txt fallback
        pdf_files = sorted(docs_path.glob("*.txt"))
    if not pdf_files:
        console.print(f"[red]No documents found in {patient_dir}[/red]")
        raise typer.Exit(1)

    doc_paths = [str(f) for f in pdf_files]

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    memory_store = CorrectionMemoryStore(store_path=memory_path)
    if reset_memory:
        memory_store.patterns.clear()
        console.print("[yellow]Correction memory cleared[/yellow]")

    reviewer = SimulatedReviewer()

    console.print(f"\n[bold cyan]Part 2 — Learning Loop[/bold cyan]")
    console.print(f"Patient: [bold]{patient_id}[/bold]")
    console.print(f"Documents: {len(doc_paths)}")
    console.print(f"Iterations: {iterations}")
    console.print(f"Memory: {memory_path}")
    console.print()

    results: list[EvaluationResult] = []

    for i in range(1, iterations + 1):
        console.print(f"[cyan]Running iteration {i}/{iterations}...[/cyan]")
        result = asyncio.run(_run_learning_iteration(
            patient_id=patient_id,
            doc_paths=doc_paths,
            iteration=i,
            memory_store=memory_store,
            reviewer=reviewer,
        ))
        results.append(result)

    console.print()
    _print_results_table(results)

    # Save results
    results_file = out_path / f"{patient_id}_learning_results.json"
    _save_results(results, results_file)

    # Plot chart
    chart_file = out_path / f"{patient_id}_improvement_curve.png"
    _plot_improvement(results, chart_file)

    # Print memory stats
    stats = memory_store.summary_stats()
    console.print(f"\n[bold]Correction Memory Stats:[/bold]")
    console.print(f"  Total patterns: {stats['total_patterns']}")
    console.print(f"  Total examples: {stats['total_examples']}")
    if stats["most_frequent"]:
        console.print("  Most frequent corrections:")
        for pattern_key, freq in stats["most_frequent"]:
            console.print(f"    • {pattern_key}: {freq}x")

    console.print("\n[bold]Limitations (as required by Part 2):[/bold]")
    console.print("""
  1. Cold-start: The memory store starts empty — first iterations get no benefit.
     With only 2 patients and 5 iterations, improvement may be modest.

  2. Reward hacking risk: Optimising edit distance could be gamed by making the
     agent produce vaguer text that matches the reviewer's style without being
     clinically accurate. We mitigate this by:
     a) Never allowing memory to override the anti-fabrication system prompt
     b) Only injecting stylistic hints (timeframes, ownership), not clinical facts
     c) Monitoring missing_field_count as a safety counter-metric

  3. Distribution shift: The simulated reviewer's policy is fixed. A real reviewer
     varies. The learned patterns may overfit to the simulation.

  4. Safety preserved: Memory injection is additive — it adds hints AFTER the
     anti-fabrication system prompt, which takes precedence. The agent will still
     return MISSING rather than fabricate, even if the memory suggests otherwise.
""")


if __name__ == "__main__":
    app()
