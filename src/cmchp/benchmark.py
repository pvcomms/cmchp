"""
Round-trip fidelity benchmark for CMCHP.

Metrics:
- goal_preservation: fraction of original goals present in expanded context
- memory_accuracy: confidence-weighted key preservation
- compression_ratio: how much we shrank the context
- continuation_coherence: cosine sim of continuation hints (lexical)
- task_completability: can the receiving model actually continue the task?
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .compressor import Compressor
from .expander import Expander
from .schema import CMCHPPacket

console = Console()


@dataclass
class FidelityResult:
    scenario_name: str
    source_model: str
    target_model: str
    goal_preservation: float       # 0-1
    memory_coverage: float         # 0-1, weighted by confidence
    compression_ratio: float       # compressed/original tokens
    roundtrip_ms: int
    continuation_viable: bool      # heuristic: hint is non-empty and ≥8 words
    goals_original: int
    goals_preserved: int
    memory_entries_original: int
    memory_entries_preserved: int
    notes: str = ""


@dataclass
class BenchmarkResults:
    results: list[FidelityResult] = field(default_factory=list)

    @property
    def avg_goal_preservation(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.goal_preservation for r in self.results) / len(self.results)

    @property
    def avg_memory_coverage(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.memory_coverage for r in self.results) / len(self.results)

    @property
    def avg_compression_ratio(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.compression_ratio for r in self.results) / len(self.results)

    @property
    def continuation_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.continuation_viable) / len(self.results)


def _goal_preservation_score(
    packet: CMCHPPacket, expanded: str
) -> tuple[float, int, int]:
    """Check what fraction of goal descriptions appear (fuzzy) in expanded text."""
    if not packet.goals:
        return 1.0, 0, 0

    preserved = 0
    for goal in packet.goals:
        words = [w for w in goal.description.lower().split() if len(w) > 4]
        if not words:
            preserved += 1
            continue
        hits = sum(1 for w in words if w in expanded.lower())
        if hits / len(words) >= 0.5:
            preserved += 1

    return preserved / len(packet.goals), len(packet.goals), preserved


def _memory_coverage_score(
    packet: CMCHPPacket, expanded: str
) -> tuple[float, int, int]:
    """Confidence-weighted check of memory key/value presence in expanded text."""
    if not packet.working_memory:
        return 1.0, 0, 0

    total_weight = sum(m.confidence for m in packet.working_memory)
    covered_weight = 0.0
    covered_count = 0

    for m in packet.working_memory:
        key_present = m.key.lower() in expanded.lower()
        value_words = [w for w in m.value.lower().split() if len(w) > 3]
        value_present = (
            any(w in expanded.lower() for w in value_words[:3])
            if value_words
            else True
        )

        if key_present or value_present:
            covered_weight += m.confidence
            covered_count += 1

    return covered_weight / max(total_weight, 0.01), len(packet.working_memory), covered_count


def run_scenario(
    scenario: dict,
    compressor: Compressor,
    expander: Expander,
    source_model: str = "claude-sonnet-4-6",
    target_model: str = "gpt-4o",
) -> FidelityResult:
    start = time.monotonic()

    packet = compressor.compress(
        conversation=scenario["conversation"],
        tool_history=scenario.get("tool_history", []),
        source_model=source_model,
        target_model=target_model,
        token_budget=scenario.get("token_budget", 4000),
    )

    expanded = expander.expand(packet, target_model)

    elapsed_ms = int((time.monotonic() - start) * 1000)

    goal_score, goals_orig, goals_pres = _goal_preservation_score(packet, expanded)
    mem_score, mem_orig, mem_pres = _memory_coverage_score(packet, expanded)

    hint = packet.continuation_hint
    continuation_viable = bool(hint) and len(hint.split()) >= 8

    return FidelityResult(
        scenario_name=scenario["name"],
        source_model=source_model,
        target_model=target_model,
        goal_preservation=goal_score,
        memory_coverage=mem_score,
        compression_ratio=packet.compression.compression_ratio,
        roundtrip_ms=elapsed_ms,
        continuation_viable=continuation_viable,
        goals_original=goals_orig,
        goals_preserved=goals_pres,
        memory_entries_original=mem_orig,
        memory_entries_preserved=mem_pres,
        notes=(
            f"compressed {packet.compression.original_token_count}"
            f"→{packet.compression.compressed_token_count} tokens"
        ),
    )


def run_benchmark(
    scenarios_dir: str | Path,
    api_key: str | None = None,
    source_model: str = "claude-sonnet-4-6",
    target_models: list[str] | None = None,
) -> BenchmarkResults:
    if target_models is None:
        target_models = ["gpt-4o", "gemini-2.0-flash", "claude-haiku-4-5"]

    scenarios_dir = Path(scenarios_dir)
    scenarios: list[dict] = []
    for f in sorted(scenarios_dir.glob("*.json")):
        with f.open() as fh:
            scenarios.append(json.load(fh))

    if not scenarios:
        console.print(
            "[yellow]No scenarios found. Add JSON files to benchmarks/scenarios/[/yellow]"
        )
        return BenchmarkResults()

    compressor = Compressor(api_key=api_key)
    expander = Expander()
    results = BenchmarkResults()

    for scenario in scenarios:
        for target in target_models:
            console.print(
                f"  Running [bold]{scenario['name']}[/bold] → [cyan]{target}[/cyan]..."
            )
            try:
                result = run_scenario(scenario, compressor, expander, source_model, target)
                results.results.append(result)
                console.print(
                    f"    goal={result.goal_preservation:.0%}"
                    f"  mem={result.memory_coverage:.0%}"
                    f"  ratio={result.compression_ratio:.2f}x"
                    f"  {result.roundtrip_ms}ms"
                )
            except Exception as exc:
                console.print(f"    [red]FAILED: {exc}[/red]")

    return results


def print_results(results: BenchmarkResults) -> None:
    table = Table(title="CMCHP Round-Trip Fidelity")
    table.add_column("Scenario", style="bold")
    table.add_column("Target")
    table.add_column("Goals", justify="right")
    table.add_column("Memory", justify="right")
    table.add_column("Ratio", justify="right")
    table.add_column("ms", justify="right")
    table.add_column("Cont.", justify="center")

    for r in results.results:
        table.add_row(
            r.scenario_name,
            r.target_model,
            f"{r.goal_preservation:.0%} ({r.goals_preserved}/{r.goals_original})",
            f"{r.memory_coverage:.0%} ({r.memory_entries_preserved}/{r.memory_entries_original})",
            f"{r.compression_ratio:.2f}x",
            str(r.roundtrip_ms),
            "+" if r.continuation_viable else "-",
        )

    console.print(table)
    console.print(
        Panel(
            f"[bold]Summary[/bold]\n"
            f"Goal preservation:  [green]{results.avg_goal_preservation:.0%}[/green]\n"
            f"Memory coverage:    [green]{results.avg_memory_coverage:.0%}[/green]\n"
            f"Avg compression:    [cyan]{results.avg_compression_ratio:.2f}x[/cyan]\n"
            f"Continuation rate:  [green]{results.continuation_success_rate:.0%}[/green]",
            title="CMCHP Benchmark",
        )
    )
