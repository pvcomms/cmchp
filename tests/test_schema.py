"""Basic schema round-trip tests — no API calls."""
import json
import uuid
from datetime import datetime, timezone

import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cmchp.schema import (
    CMCHPPacket, Goal, MemoryEntry, OpenToolCall,
    DecisionStep, AgentState, CompressionMeta,
)


def make_packet(**overrides) -> CMCHPPacket:
    defaults = dict(
        session_id=str(uuid.uuid4()),
        timestamp=datetime.now(timezone.utc),
        source_model="claude-sonnet-4-6",
        target_model="gpt-4o",
        compression=CompressionMeta(
            method="semantic_llm",
            original_token_count=5000,
            compressed_token_count=300,
            compression_ratio=0.06,
            model_used="claude-opus-4-7",
        ),
        goals=[Goal(id="g1", description="Fix memory leak", priority=0.9, status="active")],
        working_memory=[
            MemoryEntry(key="rss_mb", value="1843", confidence=1.0, source="tool_result")
        ],
        open_tool_calls=[],
        agent_state=AgentState(persona="senior SRE", constraints=["no downtime"]),
        decision_trace=[DecisionStep(step=1, decision="Use psutil profiling", reasoning="fast baseline")],
        continuation_hint="Run pool_pre_ping check on SQLAlchemy config.",
    )
    defaults.update(overrides)
    return CMCHPPacket(**defaults)


def test_packet_roundtrip():
    p = make_packet()
    dumped = p.model_dump_json()
    restored = CMCHPPacket.model_validate_json(dumped)
    assert restored.session_id == p.session_id
    assert restored.goals[0].description == "Fix memory leak"
    assert restored.compression.compression_ratio == pytest.approx(0.06)


def test_subgoals():
    parent = Goal(
        id="g1",
        description="Deploy new auth service",
        priority=0.9,
        status="active",
        subgoals=[
            Goal(id="g1a", description="Write middleware", priority=0.8, status="completed"),
            Goal(id="g1b", description="Add env vars", priority=0.7, status="active"),
        ],
    )
    p = make_packet(goals=[parent])
    dumped = p.model_dump_json()
    restored = CMCHPPacket.model_validate_json(dumped)
    assert len(restored.goals[0].subgoals) == 2
    assert restored.goals[0].subgoals[1].status == "active"


def test_expander_returns_string():
    from cmchp.expander import Expander
    p = make_packet()
    e = Expander()
    for model in ["claude-sonnet-4-6", "gpt-4o", "gemini-2.0-flash", "unknown-model-xyz"]:
        result = e.expand(p, model)
        assert isinstance(result, str)
        assert len(result) > 50
        assert "Fix memory leak" in result


def test_claude_expansion_uses_xml():
    from cmchp.expander import Expander
    p = make_packet()
    result = Expander().expand(p, "claude-sonnet-4-6")
    assert "<cmchp_handoff>" in result


def test_compression_ratio():
    p = make_packet()
    assert p.compression.compression_ratio == pytest.approx(0.06)
    assert p.compression.original_token_count == 5000
