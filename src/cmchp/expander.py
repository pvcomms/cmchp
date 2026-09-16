"""
Expander: CMCHPPacket → model-specific context injection
"""
import json

from .schema import CMCHPPacket

TEMPLATES = {
    "claude": """<cmchp_handoff>
You are receiving a mid-task handoff from {source_model}. Resume exactly where they left off.

<active_goals>
{goals}
</active_goals>

<working_memory>
{memory}
</working_memory>

{tool_calls_section}

<agent_context>
Persona: {persona}
Constraints: {constraints}
</agent_context>

<decision_history>
{trace}
</decision_history>

<next_action>
{continuation_hint}
</next_action>
</cmchp_handoff>""",

    "openai": """[CMCHP HANDOFF — from {source_model}]

ACTIVE GOALS:
{goals}

WORKING MEMORY:
{memory}

{tool_calls_section}

CONSTRAINTS: {constraints}
PERSONA: {persona}

RECENT DECISIONS:
{trace}

→ NEXT: {continuation_hint}""",

    "gemini": """Context handoff from {source_model}:

Goals (priority-ordered):
{goals}

Known facts (with confidence):
{memory}

{tool_calls_section}

Operating as: {persona}
Constraints: {constraints}

Next: {continuation_hint}""",
}


def _format_goals(goals: list) -> str:
    lines = []
    for g in goals:
        lines.append(f"[{g.status.upper()} | p={g.priority:.1f}] {g.description}")
        for sg in g.subgoals:
            lines.append(f"  [{sg.status.upper()} | p={sg.priority:.1f}] {sg.description}")
    return "\n".join(lines) if lines else "(none)"


def _format_memory(memory: list) -> str:
    lines = []
    for m in memory:
        conf = f"{m.confidence:.0%}"
        lines.append(f"- {m.key} [{conf} | {m.source}]: {m.value}")
    return "\n".join(lines) if lines else "(none)"


def _format_tool_calls(calls: list) -> str:
    if not calls:
        return ""
    lines = ["PENDING TOOL CALLS:"]
    for t in calls:
        lines.append(f"- {t.tool_name}({json.dumps(t.arguments)}) [{t.status}]: {t.reasoning}")
    return "\n".join(lines)


def _format_trace(trace: list) -> str:
    lines = []
    for d in trace[-5:]:
        lines.append(f"{d.step}. {d.decision}")
        if d.outcome:
            lines.append(f"   → {d.outcome}")
    return "\n".join(lines) if lines else "(none)"


class Expander:
    def expand(self, packet: CMCHPPacket, target_model: str | None = None) -> str:
        model_family = self._detect_family(target_model or packet.target_model)
        template = TEMPLATES.get(model_family, TEMPLATES["openai"])

        tool_section = _format_tool_calls(packet.open_tool_calls)

        return template.format(
            source_model=packet.source_model,
            goals=_format_goals(packet.goals),
            memory=_format_memory(packet.working_memory),
            tool_calls_section=tool_section,
            persona=packet.agent_state.persona or "general assistant",
            constraints=", ".join(packet.agent_state.constraints) or "none",
            trace=_format_trace(packet.decision_trace),
            continuation_hint=packet.continuation_hint,
        )

    def _detect_family(self, model_str: str) -> str:
        s = model_str.lower()
        if "claude" in s:
            return "claude"
        if "gpt" in s or "openai" in s or "o1" in s or "o3" in s or "o4" in s:
            return "openai"
        if "gemini" in s or "google" in s:
            return "gemini"
        return "openai"
