"""
Live timeline TUI for chain execution and a static tree renderer.

No new deps — uses `rich` which is already in pyproject.toml.

Palette intent (terminal-constrained — we pick from 256-color space):
  charcoal ink + warm off-white surface + single terracotta accent on drift.
"""
from __future__ import annotations

from typing import Iterable

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree as RichTree

from .chain import Chain, ChainTree, Hop


INK = "#111111"
PAPER = "#f7f6f3"
MUTED = "#8a8680"
LINE = "#e7e4dd"
ACCENT_DRIFT = "#b84a2f"       # terracotta — used ONCE on drift alert
GOOD = "#3d5a3a"
WARN = "#c7a96a"


def _hop_cell(hop: Hop, highlight_drift: bool) -> Panel:
    status_map = {
        "pending": ("·", MUTED),
        "running": ("▸", WARN),
        "done":    ("●", GOOD),
        "error":   ("✕", ACCENT_DRIFT),
    }
    glyph, color = status_map.get(hop.status, ("·", MUTED))

    if hop.status == "done" and highlight_drift:
        color = ACCENT_DRIFT

    header = Text()
    header.append(f"{glyph} ", style=color)
    header.append(f"hop {hop.index} ", style=f"{INK} bold")
    header.append(hop.model, style=MUTED)

    body = Text()
    if hop.status == "done":
        body.append(f"in {hop.tokens_in}  out {hop.tokens_out}\n", style=MUTED)
        body.append(f"{hop.latency_ms} ms\n", style=MUTED)
        body.append("drift ", style=MUTED)
        drift_style = ACCENT_DRIFT if highlight_drift else GOOD
        body.append(f"{hop.drift_score:.2f}", style=f"{drift_style} bold")
    elif hop.status == "running":
        body.append("streaming…", style=WARN)
    elif hop.status == "error":
        body.append(hop.error or "error", style=ACCENT_DRIFT)
    else:
        body.append("queued", style=MUTED)

    return Panel(
        Group(header, body),
        border_style=color,
        padding=(0, 1),
        width=26,
        title=None,
    )


def _timeline(chain: Chain, drift_threshold: float, alerted_hop: int | None) -> Panel:
    if not chain.hops:
        body: Text | Table = Text("initializing…", style=MUTED)
    else:
        row = Table.grid(padding=(0, 0))
        for _ in chain.hops:
            row.add_column()
        cells = []
        for h in chain.hops:
            highlight = h.status == "done" and (
                h.drift_score >= drift_threshold or alerted_hop == h.index
            )
            cells.append(_hop_cell(h, highlight))
        row.add_row(*cells)
        body = row

    title = Text()
    title.append("chain ", style=MUTED)
    title.append(chain.id, style=f"{INK} bold")
    title.append(f"  threshold {drift_threshold:.2f}", style=MUTED)
    return Panel(body, title=title, border_style=LINE, padding=(1, 2))


def run_with_live_view(
    chain: Chain,
    models: list[str],
    drift_threshold: float = 0.6,
    console: Console | None = None,
) -> Chain:
    """Execute chain and render a live horizontal timeline."""
    from .chain import run_chain

    console = console or Console()
    last_alert: int | None = None
    events: list[dict] = []

    def emit(ev: dict) -> None:
        events.append(ev)
        if ev.get("event") == "drift_alert":
            nonlocal_holder["alert"] = ev["hop"]

    nonlocal_holder = {"alert": None}

    def emit_bridge(ev: dict) -> None:
        events.append(ev)
        if ev.get("event") == "drift_alert":
            nonlocal_holder["alert"] = ev["hop"]

    with Live(_timeline(chain, drift_threshold, None), console=console, refresh_per_second=8) as live:
        for _hop in run_chain(chain, models, emit=emit_bridge, drift_threshold=drift_threshold):
            live.update(_timeline(chain, drift_threshold, nonlocal_holder["alert"]))

    # Final render with any late alerts
    console.print(_timeline(chain, drift_threshold, nonlocal_holder["alert"]))
    return chain


def render_tree(tree: ChainTree, console: Console | None = None) -> None:
    console = console or Console()
    if not tree.chains:
        console.print(Text("(tree empty)", style=MUTED))
        return

    roots = [c for c in tree.chains.values() if c.parent_id is None]
    rich_tree = RichTree(
        Text(f"seed  ", style=MUTED) + Text(tree.root_prompt[:80], style=INK),
        guide_style=LINE,
    )
    for root in roots:
        _attach(rich_tree, tree, root)
    console.print(rich_tree)


def _attach(parent_node, tree: ChainTree, chain: Chain) -> None:
    label = Text()
    label.append(f"chain {chain.id}", style=f"{INK} bold")
    if chain.fork_hop is not None:
        label.append(f"  (fork @ hop {chain.fork_hop})", style=MUTED)
    node = parent_node.add(label)
    for h in chain.hops:
        hop_label = Text()
        hop_label.append(f"hop {h.index}  ", style=MUTED)
        hop_label.append(h.model, style=INK)
        if h.status == "done":
            color = ACCENT_DRIFT if h.drift_score >= 0.6 else GOOD
            hop_label.append(f"   drift {h.drift_score:.2f}", style=color)
        else:
            hop_label.append(f"   {h.status}", style=MUTED)
        node.add(hop_label)
    for child in tree.children_of(chain.id):
        _attach(node, tree, child)
