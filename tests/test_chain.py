"""Chain explorer tests — no API calls (monkeypatch dispatch)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cmchp.chain as chain_mod
from cmchp.chain import (
    Chain,
    ChainTree,
    drift_score,
    export_recipe,
    fork_chain,
    load_tree,
    new_chain,
    run_chain,
    save_tree,
    undo,
)


def test_drift_score_bounds():
    assert drift_score("", "") == 0.0
    assert drift_score("hello world", "") == 1.0
    assert drift_score("abcd efgh ijkl", "abcd efgh ijkl") == 0.0
    assert 0.0 < drift_score("alpha beta gamma", "alpha delta epsilon") < 1.0


def test_fork_creates_sibling(monkeypatch):
    tree = ChainTree(root_prompt="seed")
    ch = new_chain("seed", ["model-a", "model-b", "model-c"])
    tree.add(ch)

    # Fake the dispatch so run_chain doesn't hit an API.
    monkeypatch.setattr(chain_mod, "_dispatch", lambda m, s, u: f"{m}-output-{u[:10]}")

    for _ in run_chain(ch, ["model-a", "model-b", "model-c"]):
        pass
    assert len(ch.hops) == 3
    assert all(h.status == "done" for h in ch.hops)

    child = fork_chain(tree, ch.id, at_hop=1, swap_model="model-z")
    assert child.parent_id == ch.id
    assert child.fork_hop == 1
    assert len(child.hops) == 2
    assert child.hops[1].model == "model-z"
    assert child.hops[1].status == "pending"


def test_undo_roundtrip(tmp_path, monkeypatch):
    # Redirect the store so we don't pollute the user's home.
    monkeypatch.setattr(chain_mod, "STORE_DIR", tmp_path / "chains")

    tree = ChainTree(root_prompt="seed")
    tree.add(Chain(id="aaa", seed_prompt="seed"))
    save_tree(tree, slug="t")

    tree2 = load_tree("t")
    tree2.add(Chain(id="bbb", seed_prompt="seed"))
    save_tree(tree2, slug="t")

    loaded = load_tree("t")
    assert set(loaded.chains.keys()) == {"aaa", "bbb"}

    reverted = undo("t")
    assert reverted is not None
    assert set(reverted.chains.keys()) == {"aaa"}


def test_export_recipe_shape(monkeypatch):
    ch = new_chain("seed", ["m1", "m2"])
    monkeypatch.setattr(chain_mod, "_dispatch", lambda m, s, u: f"out-{m}")
    for _ in run_chain(ch, ["m1", "m2"]):
        pass

    recipe = export_recipe(ch)
    assert recipe["cmchp_recipe_version"] == "0.1"
    assert recipe["models"] == ["m1", "m2"]
    assert len(recipe["hops"]) == 2
    assert recipe["final_drift"] == ch.hops[-1].drift_score


def test_drift_alert_emitted(monkeypatch):
    """Seed output has unique words; hop 1 output has none → drift≈1.0 → alert."""
    ch = new_chain("seed", ["m1", "m2"])
    outputs = iter(["quantum helical thermodynamic cascade", "zebra"])
    monkeypatch.setattr(chain_mod, "_dispatch", lambda m, s, u: next(outputs))

    events: list[dict] = []
    for _ in run_chain(ch, ["m1", "m2"], emit=events.append, drift_threshold=0.5):
        pass

    kinds = [e["event"] for e in events]
    assert "drift_alert" in kinds
