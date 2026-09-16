"""
Handoff chain explorer — non-linear tree of CMCHP handoffs across models.

A Chain is an ordered list of Hops. A ChainTree is a forest where chains
can fork from any hop (spawning a sibling that shares the prefix).

Runtime emits NDJSON events per hop so a CLI timeline view can render
progress live.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from .schema import CMCHPPacket


# Model family routing — keeps this module free of adapter imports at top
# so the CLI can run `chain tree`/`fork` without API keys present.
KNOWN_MODELS = {
    "claude-opus-4-7": "claude",
    "claude-sonnet-4-6": "claude",
    "claude-haiku-4-5": "claude",
    "gpt-5": "openai",
    "gpt-5-mini": "openai",
    "gpt-4o": "openai",
    "gemini-2.5-pro": "gemini",
    "gemini-2.0-flash": "gemini",
}


@dataclass
class Hop:
    index: int
    model: str
    input_excerpt: str
    output_excerpt: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    drift_score: float = 0.0          # 0=no drift from seed, 1=total drift
    status: str = "pending"            # pending|running|done|error
    error: str | None = None


@dataclass
class Chain:
    id: str
    seed_prompt: str
    seed_output: str = ""
    parent_id: str | None = None
    fork_hop: int | None = None        # hop index at parent where fork happened
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    hops: list[Hop] = field(default_factory=list)


@dataclass
class ChainTree:
    root_prompt: str
    chains: dict[str, Chain] = field(default_factory=dict)
    order: list[str] = field(default_factory=list)   # insertion order for render

    def add(self, chain: Chain) -> None:
        self.chains[chain.id] = chain
        if chain.id not in self.order:
            self.order.append(chain.id)

    def children_of(self, chain_id: str) -> list[Chain]:
        return [c for c in self.chains.values() if c.parent_id == chain_id]


# --- Drift score ------------------------------------------------------------
# Lexical only on purpose: zero extra deps, deterministic, matches how
# benchmark.py already scores goal/memory coverage. Drift = 1 - Jaccard over
# content-word sets (len>3, lowercased) against the seed output.

def _content_words(text: str) -> set[str]:
    return {w.lower() for w in text.split() if len(w) > 3}


def drift_score(seed_output: str, hop_output: str) -> float:
    a, b = _content_words(seed_output), _content_words(hop_output)
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    jacc = len(a & b) / len(a | b)
    return round(1.0 - jacc, 4)


# --- Storage / undo ---------------------------------------------------------

STORE_DIR = Path(os.path.expanduser("~/.cmchp/chains"))
UNDO_LIMIT = 10


def _store() -> Path:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return STORE_DIR


def _tree_to_dict(tree: ChainTree) -> dict:
    return {
        "root_prompt": tree.root_prompt,
        "order": tree.order,
        "chains": {cid: {**asdict(c)} for cid, c in tree.chains.items()},
    }


def _tree_from_dict(data: dict) -> ChainTree:
    tree = ChainTree(root_prompt=data["root_prompt"])
    tree.order = list(data.get("order", []))
    for cid, cd in data.get("chains", {}).items():
        hops = [Hop(**h) for h in cd.get("hops", [])]
        cd2 = {**cd, "hops": hops}
        tree.chains[cid] = Chain(**cd2)
    return tree


def save_tree(tree: ChainTree, slug: str = "tree") -> Path:
    """Snapshot tree + keep last UNDO_LIMIT historical snapshots."""
    d = _store()
    path = d / f"{slug}.json"
    history = d / f"{slug}.history"
    history.mkdir(exist_ok=True)

    if path.exists():
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
        (history / f"{ts}.json").write_text(path.read_text())
        snaps = sorted(history.glob("*.json"))
        for old in snaps[:-UNDO_LIMIT]:
            old.unlink()

    path.write_text(json.dumps(_tree_to_dict(tree), indent=2))
    return path


def load_tree(slug: str = "tree") -> ChainTree | None:
    path = _store() / f"{slug}.json"
    if not path.exists():
        return None
    return _tree_from_dict(json.loads(path.read_text()))


def undo(slug: str = "tree") -> ChainTree | None:
    """Roll back to the most recent pre-edit snapshot."""
    d = _store()
    history = d / f"{slug}.history"
    if not history.exists():
        return None
    snaps = sorted(history.glob("*.json"))
    if not snaps:
        return None
    latest = snaps[-1]
    target = d / f"{slug}.json"
    target.write_text(latest.read_text())
    latest.unlink()
    return _tree_from_dict(json.loads(target.read_text()))


# --- Execution --------------------------------------------------------------

def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _dispatch(model: str, system_prompt: str, user_message: str) -> str:
    """Route to the right adapter. Raises if API key absent."""
    family = KNOWN_MODELS.get(model) or _fallback_family(model)
    if family == "claude":
        from .models.claude import ClaudeAdapter
        adapter = ClaudeAdapter(api_key=os.environ.get("ANTHROPIC_API_KEY"), model=model)
    elif family == "openai":
        from .models.openai_adapter import OpenAIAdapter
        adapter = OpenAIAdapter(api_key=os.environ.get("OPENAI_API_KEY"), model=model)
    elif family == "gemini":
        from .models.gemini_adapter import GeminiAdapter
        adapter = GeminiAdapter(api_key=os.environ.get("GEMINI_API_KEY"), model=model)
    else:
        raise ValueError(f"Unknown model family for '{model}'")

    # Build a thin pseudo-packet by routing the system through the adapter's
    # expander path would require a real CMCHPPacket; instead we hand the
    # adapter's underlying client directly so chain execution works without
    # the full compress/expand trip.
    if family == "claude":
        resp = adapter.client.messages.create(
            model=model, max_tokens=2048,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        return resp.content[0].text
    if family == "openai":
        resp = adapter.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content
    if family == "gemini":
        import google.generativeai as genai
        m = genai.GenerativeModel(model, system_instruction=system_prompt)
        return m.generate_content(user_message).text
    raise RuntimeError("unreachable")


def _fallback_family(model: str) -> str:
    s = model.lower()
    if "claude" in s:
        return "claude"
    if "gpt" in s or s.startswith("o"):
        return "openai"
    if "gemini" in s:
        return "gemini"
    return "openai"


SYSTEM_HANDOFF = (
    "You are receiving a handoff from the previous model in a chain. "
    "Continue the task. Be concise."
)


def run_chain(
    chain: Chain,
    models: list[str],
    emit: Callable[[dict], None] | None = None,
    drift_threshold: float = 0.6,
) -> Iterator[Hop]:
    """Execute each hop sequentially. Yields each Hop as it completes.

    emit(event) receives NDJSON-style events: hop_start, hop_delta, hop_done,
    drift_alert. The CLI timeline view consumes these.
    """
    emit = emit or (lambda _e: None)

    prior_output = chain.seed_prompt
    seed_reference = chain.seed_output or chain.seed_prompt

    for idx, model in enumerate(models):
        hop = Hop(
            index=idx,
            model=model,
            input_excerpt=prior_output[:240],
            status="running",
            tokens_in=_approx_tokens(prior_output),
        )
        chain.hops.append(hop)
        emit({
            "event": "hop_start",
            "chain_id": chain.id,
            "hop": idx,
            "model": model,
            "tokens_in": hop.tokens_in,
        })

        t0 = time.monotonic()
        try:
            out = _dispatch(model, SYSTEM_HANDOFF, prior_output)
        except Exception as exc:
            hop.status = "error"
            hop.error = f"{type(exc).__name__}: {exc}"
            hop.latency_ms = int((time.monotonic() - t0) * 1000)
            emit({"event": "hop_done", "chain_id": chain.id, "hop": idx, "error": hop.error})
            yield hop
            return

        hop.output_excerpt = out[:240]
        hop.tokens_out = _approx_tokens(out)
        hop.latency_ms = int((time.monotonic() - t0) * 1000)
        hop.drift_score = drift_score(seed_reference, out)
        hop.status = "done"

        emit({
            "event": "hop_done",
            "chain_id": chain.id,
            "hop": idx,
            "model": model,
            "tokens_out": hop.tokens_out,
            "latency_ms": hop.latency_ms,
            "drift": hop.drift_score,
            "output_excerpt": hop.output_excerpt,
        })

        if idx == 0 and not chain.seed_output:
            chain.seed_output = out
            seed_reference = out

        if hop.drift_score >= drift_threshold:
            emit({
                "event": "drift_alert",
                "chain_id": chain.id,
                "hop": idx,
                "drift": hop.drift_score,
                "threshold": drift_threshold,
            })

        yield hop
        prior_output = out


def fork_chain(tree: ChainTree, parent_id: str, at_hop: int, swap_model: str) -> Chain:
    """Create a sibling chain that shares hops[:at_hop] with the parent and
    will re-run starting at at_hop using swap_model instead of the original.
    The new chain starts unexecuted past the shared prefix."""
    parent = tree.chains[parent_id]
    if at_hop < 0 or at_hop >= len(parent.hops):
        raise ValueError(f"fork hop {at_hop} out of range for chain {parent_id}")

    new_id = uuid.uuid4().hex[:8]
    shared = [Hop(**asdict(h)) for h in parent.hops[:at_hop]]
    # The hop at `at_hop` is replaced with a pending hop using swap_model.
    shared.append(Hop(
        index=at_hop,
        model=swap_model,
        input_excerpt=parent.hops[at_hop].input_excerpt,
        status="pending",
    ))

    child = Chain(
        id=new_id,
        seed_prompt=parent.seed_prompt,
        seed_output=parent.seed_output,
        parent_id=parent_id,
        fork_hop=at_hop,
        hops=shared,
    )
    tree.add(child)
    return child


def export_recipe(chain: Chain) -> dict:
    """Emit a minimal JSON recipe — models, seed, drift scores. Reproducible."""
    return {
        "cmchp_recipe_version": "0.1",
        "seed_prompt": chain.seed_prompt,
        "models": [h.model for h in chain.hops],
        "hops": [
            {
                "model": h.model,
                "drift_score": h.drift_score,
                "latency_ms": h.latency_ms,
                "tokens_in": h.tokens_in,
                "tokens_out": h.tokens_out,
            }
            for h in chain.hops
        ],
        "final_drift": chain.hops[-1].drift_score if chain.hops else None,
    }


def new_chain(seed_prompt: str, models: list[str]) -> Chain:
    chain = Chain(id=uuid.uuid4().hex[:8], seed_prompt=seed_prompt)
    # Pre-fill pending hops so the timeline can render the skeleton immediately.
    for idx, m in enumerate(models):
        chain.hops.append(Hop(index=idx, model=m, input_excerpt="", status="pending"))
    # Clear them — run_chain appends as it goes.
    chain.hops.clear()
    return chain
