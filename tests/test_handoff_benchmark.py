"""The parts of the handoff benchmark that do not need a network.

The grading and the budget maths are where a benchmark quietly cheats, so
they are the parts worth pinning. The one API call per condition is not
mocked — mocking it would test the mock.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cmchp.handoff_benchmark import grade, render_conversation, truncate_tail

SCENARIOS = Path(__file__).parent.parent / "benchmarks" / "scenarios"
PROBES = Path(__file__).parent.parent / "benchmarks" / "probes.json"


# --- grading -------------------------------------------------------------

def test_grade_accepts_any_listed_variant():
    assert grade("about 847 threads", ["847"])
    assert grade("2,341 open files", ["2341", "2,341"])


def test_grade_is_whitespace_and_comma_insensitive():
    # "50,000" in the answer must match "50000" in the accept list and
    # vice versa, or every numeric probe becomes a formatting lottery.
    assert grade("roughly 50,000 users", ["50000"])
    assert grade("50000 users", ["50,000"])
    assert grade("50,000 users", ["50,000"])


def test_grade_is_case_insensitive():
    assert grade("It was SQLAlchemy", ["sqlalchemy"])


def test_grade_rejects_a_wrong_answer():
    assert not grade("about 400 threads", ["847"])


def test_grade_rejects_unknown():
    assert not grade("UNKNOWN", ["847"])


def test_grade_rejects_an_empty_answer():
    assert not grade("", ["847"])


# --- the budget, which is the whole comparison ---------------------------

def test_truncation_keeps_the_tail_not_the_head():
    # Recency is what a sliding window preserves, so cutting from the
    # front is the stronger baseline and therefore the fair one.
    text = "START" + ("x" * 100) + "END"
    assert truncate_tail(text, 10).endswith("END")
    assert "START" not in truncate_tail(text, 10)


def test_truncation_is_exact_so_the_conditions_cost_the_same():
    assert len(truncate_tail("y" * 500, 120)) == 120


def test_truncation_leaves_short_text_alone():
    assert truncate_tail("short", 1000) == "short"


# --- scenarios and probes agree -----------------------------------------

def test_every_scenario_has_probes():
    probes = json.loads(PROBES.read_text())
    for path in SCENARIOS.glob("*.json"):
        name = json.loads(path.read_text())["name"]
        assert name in probes, f"{name} has no probes"
        assert len(probes[name]) >= 4


def test_probe_ids_are_unique_within_a_scenario():
    for name, probes in json.loads(PROBES.read_text()).items():
        if name.startswith("_"):
            continue
        ids = [p["id"] for p in probes]
        assert len(ids) == len(set(ids)), f"duplicate probe id in {name}"


def test_every_probe_answer_is_actually_present_in_its_source_conversation():
    # If a probe cannot be answered from the full conversation, the `full`
    # condition is not a ceiling and the whole comparison is meaningless.
    probes = json.loads(PROBES.read_text())
    for path in sorted(SCENARIOS.glob("*.json")):
        scenario = json.loads(path.read_text())
        context = render_conversation(scenario).lower()
        for probe in probes.get(scenario["name"], []):
            assert any(
                a.lower().replace(",", "") in context.replace(",", "")
                for a in probe["accept"]
            ), f"{scenario['name']}/{probe['id']} is unanswerable from the source"


def test_render_conversation_includes_tool_history():
    scenario = json.loads((SCENARIOS / "01_code_debug.json").read_text())
    rendered = render_conversation(scenario)
    assert "TOOL" in rendered
    assert len(rendered) > 500


@pytest.mark.parametrize("path", sorted(SCENARIOS.glob("*.json")))
def test_scenarios_parse(path: Path):
    scenario = json.loads(path.read_text())
    assert scenario["conversation"]
    assert scenario["name"]
