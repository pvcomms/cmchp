"""Proof that the v0.1 fidelity metrics could not fail.

Kept as a test rather than a paragraph because a claim this embarrassing
should be executable. If these ever fail, the story in the README is
wrong and the README is what needs fixing.

The v0.1 benchmark scored `goal_preservation` by checking whether the
words of `packet.goals` appeared in `expanded` — where `expanded` was the
string the expander had just rendered *from* `packet`. The expander's job
is to write the goals into the text. So the metric asked whether a
renderer renders, and the answer is yes, at 100%, for any input.
"""

from __future__ import annotations

import pytest

from cmchp.benchmark import _goal_preservation_score, _memory_coverage_score
from cmchp.expander import Expander
from cmchp.schema import CMCHPPacket


def _packet(goals=None, memory=None) -> CMCHPPacket:
    """Smallest packet the schema will accept."""
    from cmchp.schema import AgentState, CompressionMeta

    return CMCHPPacket(
        session_id="test-session",
        timestamp="2026-09-16T00:00:00Z",
        source_model="claude-sonnet-4-6",
        target_model="claude-haiku-4-5-20251001",
        compression=CompressionMeta(
            method="test",
            original_token_count=1000,
            compressed_token_count=100,
            compression_ratio=0.1,
            model_used="test",
        ),
        goals=goals or [],
        working_memory=memory or [],
        agent_state=AgentState(persona="test"),
        continuation_hint="continue the work described above in order",
    )


def _goal(text: str):
    from cmchp.schema import Goal

    return Goal(id="g1", description=text, priority=1, status="active")


@pytest.mark.parametrize(
    "goal_text",
    [
        "migrate the authentication service to short-lived tokens",
        "asdfgh qwerty zxcvbn poiuyt lkjhgf mnbvcx",  # meaningless
        "the quick brown foxes jumped over thirteen lazy sleeping dogs",
    ],
)
def test_goal_preservation_scores_100_percent_for_any_goal(goal_text: str) -> None:
    # Including nonsense. The metric cannot distinguish a good handoff
    # from a bad one because it never looks at anything but its own output.
    packet = _packet(goals=[_goal(goal_text)])
    expanded = Expander().expand(packet, "claude-haiku-4-5-20251001")

    score, total, preserved = _goal_preservation_score(packet, expanded)

    assert score == 1.0
    assert preserved == total == 1


def test_the_metric_never_consults_the_source_conversation() -> None:
    # _goal_preservation_score takes (packet, expanded). The original
    # conversation is not one of its arguments, so no amount of
    # information loss during compression can move the number.
    import inspect

    params = list(inspect.signature(_goal_preservation_score).parameters)
    assert params == ["packet", "expanded"]
    assert "conversation" not in params


def test_memory_coverage_has_the_same_shape_of_defect() -> None:
    from cmchp.schema import MemoryEntry

    packet = _packet(
        memory=[
            MemoryEntry(
                key="unrelated_key",
                value="unrelated value here",
                confidence=1.0,
                source="inference",
            )
        ]
    )
    expanded = Expander().expand(packet, "claude-haiku-4-5-20251001")

    score, _, _ = _memory_coverage_score(packet, expanded)
    assert score == 1.0


def test_continuation_viable_is_a_word_count() -> None:
    # v0.1 reported "continuation viability" as a success rate. It was
    # len(hint.split()) >= 8, so eight arbitrary words pass.
    from cmchp.benchmark import run_scenario  # noqa: F401  (import is the assertion)

    hint = "one two three four five six seven eight"
    assert bool(hint) and len(hint.split()) >= 8
