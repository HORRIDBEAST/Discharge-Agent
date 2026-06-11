"""
Correction Memory Store — Part 2 learning mechanism.

Design: Structured correction memory injected into future prompts.

How it works:
  1. When the simulated reviewer edits a draft, EditDiff objects are stored.
  2. Diffs are indexed by field name and edit type.
  3. When the agent next generates a summary, the most relevant corrections
     are retrieved and injected into the LLM prompt as examples.
  4. This steers the LLM toward patterns that the reviewer prefers.

Why this approach over fine-tuning:
  - No GPU required
  - Immediate effect (no training loop)
  - Interpretable (you can read the memory)
  - Safe: corrections are additive hints, not hard overrides
  - Easy to measure: before/after edit distance

Safety guarantee:
  - Memory never overrides the no-fabrication rules
  - Memory only provides stylistic/structural guidance
  - Anti-fabrication system prompt always takes precedence
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from evaluation.simulated_reviewer import EditDiff


@dataclass
class CorrectionPattern:
    """A generalised correction pattern learned from multiple edits."""
    field: str
    edit_type: str
    examples: list[dict] = field(default_factory=list)  # {original, edited, reason}
    frequency: int = 0
    avg_improvement: float = 0.0  # Improvement in edit distance after applying


class CorrectionMemoryStore:
    """
    Stores and retrieves correction patterns for prompt injection.

    Backed by a JSON file for persistence across runs.
    """

    def __init__(self, store_path: str = "./evaluation/correction_memory.json"):
        self.store_path = Path(store_path)
        self.patterns: dict[str, CorrectionPattern] = {}  # key: field:edit_type
        self._load()

    def store_diffs(self, diffs: list[EditDiff], improvement_delta: float = 0.0) -> None:
        """Store a batch of diffs from a reviewer session."""
        for diff in diffs:
            key = f"{diff.field}:{diff.edit_type}"
            if key not in self.patterns:
                self.patterns[key] = CorrectionPattern(
                    field=diff.field,
                    edit_type=diff.edit_type,
                )
            pattern = self.patterns[key]
            pattern.frequency += 1
            pattern.avg_improvement = (
                (pattern.avg_improvement * (pattern.frequency - 1) + improvement_delta)
                / pattern.frequency
            )
            # Keep up to 5 examples per pattern
            if len(pattern.examples) < 5:
                pattern.examples.append({
                    "original": diff.original[:200],
                    "edited": diff.edited[:200],
                    "reason": diff.reason,
                })
        self._save()

    def retrieve_relevant(self, field_name: str, top_k: int = 3) -> list[CorrectionPattern]:
        """Retrieve the most frequent/useful corrections for a given field."""
        relevant = [
            p for k, p in self.patterns.items()
            if p.field == field_name or field_name in p.field
        ]
        # Sort by frequency then avg_improvement
        relevant.sort(key=lambda p: (p.frequency, p.avg_improvement), reverse=True)
        return relevant[:top_k]

    def build_prompt_injection(self, fields: list[str]) -> str:
        """
        Build a prompt string that injects correction patterns as few-shot examples.
        This is prepended to the LLM prompt for affected fields.
        """
        lines = [
            "\n\n--- LEARNED CORRECTION PATTERNS (from prior clinician edits) ---",
            "The following patterns reflect how a clinician typically edits this type of content.",
            "Apply these patterns where appropriate WITHOUT fabricating information:\n",
        ]

        found_any = False
        for field in fields:
            patterns = self.retrieve_relevant(field)
            for pattern in patterns:
                if pattern.examples:
                    found_any = True
                    example = pattern.examples[-1]  # Most recent
                    lines.append(f"Field: {pattern.field} | Type: {pattern.edit_type}")
                    lines.append(f"Reason: {example['reason']}")
                    if example["original"]:
                        lines.append(f"Before: {example['original'][:150]}")
                    lines.append(f"After:  {example['edited'][:150]}")
                    lines.append("")

        if not found_any:
            return ""

        lines.append("--- END CORRECTION PATTERNS ---\n")
        return "\n".join(lines)

    def summary_stats(self) -> dict:
        return {
            "total_patterns": len(self.patterns),
            "total_examples": sum(len(p.examples) for p in self.patterns.values()),
            "most_frequent": sorted(
                [(k, p.frequency) for k, p in self.patterns.items()],
                key=lambda x: x[1], reverse=True
            )[:5],
        }

    def _save(self) -> None:
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        data = {k: {
            "field": p.field,
            "edit_type": p.edit_type,
            "examples": p.examples,
            "frequency": p.frequency,
            "avg_improvement": p.avg_improvement,
        } for k, p in self.patterns.items()}
        with open(self.store_path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        if not self.store_path.exists():
            return
        with open(self.store_path) as f:
            data = json.load(f)
        for k, v in data.items():
            self.patterns[k] = CorrectionPattern(
                field=v["field"],
                edit_type=v["edit_type"],
                examples=v["examples"],
                frequency=v["frequency"],
                avg_improvement=v["avg_improvement"],
            )
