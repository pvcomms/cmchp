"""
Compressor: raw agent state → CMCHPPacket
Uses Claude to do semantic extraction.
"""
import json
import uuid
from datetime import datetime, timezone

from anthropic import Anthropic

from .schema import (
    AgentState,
    CMCHPPacket,
    CompressionMeta,
    DecisionStep,
    Goal,
    MemoryEntry,
    OpenToolCall,
)

COMPRESSION_SYSTEM = """You are a semantic state compressor for AI agent handoffs.

Given a conversation history and tool call log, extract the agent's current state into structured JSON.

Output ONLY valid JSON matching this exact schema:
{
  "goals": [{"id": "g1", "description": "...", "priority": 0.9, "status": "active", "subgoals": []}],
  "working_memory": [{"key": "...", "value": "...", "confidence": 0.9, "source": "inference", "ttl_tokens": null}],
  "open_tool_calls": [{"tool_name": "...", "arguments": {}, "status": "pending", "reasoning": "..."}],
  "agent_state": {"persona": "...", "constraints": [], "behavioral_priors": []},
  "decision_trace": [{"step": 1, "decision": "...", "reasoning": "...", "outcome": null}],
  "continuation_hint": "One sentence: exactly what the receiving agent should do next."
}

Rules:
- goals: hierarchical, prioritized (1.0 = critical), only active/blocked goals
- working_memory: only facts the next agent NEEDS. Be ruthless — cut anything derivable from tool results
- open_tool_calls: only truly pending calls (not completed ones)
- decision_trace: only decisions that constrain future choices
- continuation_hint: crisp, actionable, model-agnostic
- confidence: 0.0-1.0, be honest about uncertainty
"""


class Compressor:
    def __init__(self, api_key: str | None = None):
        self.client = Anthropic(api_key=api_key)
        self.model = "claude-opus-4-7"

    def compress(
        self,
        conversation: list[dict],
        tool_history: list[dict] | None = None,
        source_model: str = "unknown",
        target_model: str = "unknown",
        token_budget: int = 4000,
    ) -> CMCHPPacket:
        original_text = json.dumps(conversation) + json.dumps(tool_history or [])
        original_tokens = len(original_text) // 4

        user_content = f"""<conversation>
{json.dumps(conversation, indent=2)}
</conversation>

<tool_history>
{json.dumps(tool_history or [], indent=2)}
</tool_history>

<target_model>{target_model}</target_model>
<token_budget>{token_budget}</token_budget>

Extract the semantic state. Output only JSON."""

        response = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            system=COMPRESSION_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        data = json.loads(raw)

        compressed_text = json.dumps(data)
        compressed_tokens = len(compressed_text) // 4

        goals = [Goal(**g) for g in data.get("goals", [])]
        memory = [MemoryEntry(**m) for m in data.get("working_memory", [])]
        tool_calls = [OpenToolCall(**t) for t in data.get("open_tool_calls", [])]
        trace = [DecisionStep(**d) for d in data.get("decision_trace", [])]
        agent_state = AgentState(
            **data.get(
                "agent_state",
                {"persona": "", "constraints": [], "behavioral_priors": []},
            )
        )

        meta = CompressionMeta(
            method="semantic_llm",
            original_token_count=original_tokens,
            compressed_token_count=compressed_tokens,
            compression_ratio=compressed_tokens / max(original_tokens, 1),
            model_used=self.model,
        )

        return CMCHPPacket(
            session_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc),
            source_model=source_model,
            target_model=target_model,
            compression=meta,
            goals=goals,
            working_memory=memory,
            open_tool_calls=tool_calls,
            agent_state=agent_state,
            decision_trace=trace,
            continuation_hint=data.get("continuation_hint", ""),
            raw_token_budget=token_budget,
        )
