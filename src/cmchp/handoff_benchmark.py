"""Does a receiving model actually know things after the handoff?

The benchmark this replaces asked whether the goals written into a packet
could be found in the text rendered from that same packet. They always
could, because the renderer's job is to write them. It scored the
renderer, not the protocol, and it scored it at 100%.

This asks a question that can come out wrong. A model that never saw the
source conversation answers probe questions from three different
contexts:

  full       the entire conversation and tool history
  packet     only the CMCHP packet, expanded
  truncated  the tail of the raw conversation, cut to the same number of
             characters the packet costs

`full` is the ceiling. `truncated` is the baseline that matters: keeping
the last N characters is what you would do if the protocol did not exist,
so a packet that does not beat it at equal cost has not earned anything.

Answers are graded by substring match against hand-written ground truth
in benchmarks/probes.json. No LLM judge — a judge would add a second
noisy model to a measurement whose whole point is to be checkable.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .compressor import Compressor
from .expander import Expander


@dataclass
class ProbeResult:
    probe_id: str
    correct: bool
    answer: str


@dataclass
class ConditionResult:
    scenario: str
    condition: str
    context_chars: int
    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def score(self) -> float:
        return sum(p.correct for p in self.probes) / len(self.probes) if self.probes else 0.0


def render_conversation(scenario: dict) -> str:
    """The full context, as a receiving model would be handed it."""
    parts = [f"{t['role'].upper()}: {t['content']}" for t in scenario["conversation"]]
    for tool in scenario.get("tool_history", []):
        parts.append(
            f"TOOL {tool.get('tool')}: input={tool.get('input')} output={tool.get('output')}"
        )
    return "\n\n".join(parts)


def truncate_tail(text: str, budget_chars: int) -> str:
    """Keep the end, which is where a naive truncation keeps.

    Cutting from the front is the stronger baseline: recency is what a
    sliding context window preserves, and it is genuinely informative.
    """
    if len(text) <= budget_chars:
        return text
    return text[-budget_chars:]


def _normalise(text: str) -> str:
    """Lowercase, drop thousands separators, collapse whitespace.

    Commas are deleted rather than replaced with a space, so that "50,000"
    and "50000" normalise to the same string. Replacing them with a space
    produces "50 000", which matches neither, and quietly turns every
    numeric probe into a formatting lottery.
    """
    return re.sub(r"\s+", " ", text.lower().replace(",", "")).strip()


def grade(answer: str, accept: list[str]) -> bool:
    haystack = _normalise(answer)
    return bool(haystack) and any(_normalise(a) in haystack for a in accept)


def ask(context: str, probes: list[dict], model: str, api_key: str) -> dict[str, str]:
    """One call per condition: all probes at once, JSON back.

    Batching keeps the comparison fair — every condition gets exactly one
    shot at the same questions — and keeps the run cheap enough to rerun.
    """
    import anthropic

    numbered = "\n".join(f"{i + 1}. [{p['id']}] {p['question']}" for i, p in enumerate(probes))
    prompt = (
        "Below is the context you have been handed from an earlier session with "
        "another model. Answer the questions using only that context.\n\n"
        "If the context does not contain the answer, reply exactly: UNKNOWN. "
        "Do not guess.\n\n"
        f"--- CONTEXT ---\n{context}\n--- END CONTEXT ---\n\n"
        f"Questions:\n{numbered}\n\n"
        'Reply with JSON only: {"probe_id": "answer", ...}'
    )

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    text = response.content[0].text.strip()

    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def run(
    scenarios_dir: str | Path,
    probes_path: str | Path,
    model: str = "claude-haiku-4-5-20251001",
    api_key: str | None = None,
) -> list[ConditionResult]:
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    all_probes = json.loads(Path(probes_path).read_text())
    compressor = Compressor(api_key=api_key)
    expander = Expander()
    results: list[ConditionResult] = []

    for path in sorted(Path(scenarios_dir).glob("*.json")):
        scenario = json.loads(path.read_text())
        name = scenario["name"]
        probes = all_probes.get(name)
        if not probes:
            continue

        full = render_conversation(scenario)

        packet = compressor.compress(
            conversation=scenario["conversation"],
            tool_history=scenario.get("tool_history", []),
            source_model="claude-sonnet-4-6",
            target_model=model,
            token_budget=scenario.get("token_budget", 4000),
        )
        packet_text = expander.expand(packet, model)

        contexts = {
            "full": full,
            "packet": packet_text,
            # Equal cost by construction — this is the comparison that counts.
            "truncated": truncate_tail(full, len(packet_text)),
        }

        for condition, context in contexts.items():
            answers = ask(context, probes, model, api_key)
            result = ConditionResult(
                scenario=name, condition=condition, context_chars=len(context)
            )
            for probe in probes:
                answer = str(answers.get(probe["id"], ""))
                result.probes.append(
                    ProbeResult(probe["id"], grade(answer, probe["accept"]), answer)
                )
            results.append(result)

    return results


def summarise(results: list[ConditionResult]) -> str:
    lines = ["| scenario | condition | context chars | probes correct | score |",
             "| --- | --- | --- | --- | --- |"]
    for r in results:
        hits = sum(p.correct for p in r.probes)
        lines.append(
            f"| {r.scenario} | {r.condition} | {r.context_chars:,} | "
            f"{hits}/{len(r.probes)} | {r.score:.0%} |"
        )

    lines.append("")
    for condition in ("full", "packet", "truncated"):
        subset = [r for r in results if r.condition == condition]
        if subset:
            mean = sum(r.score for r in subset) / len(subset)
            chars = sum(r.context_chars for r in subset) // len(subset)
            lines.append(f"- **{condition}**: {mean:.0%} mean, {chars:,} chars mean")
    return "\n".join(lines)
