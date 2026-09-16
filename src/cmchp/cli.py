"""CMCHP CLI — compress, expand, benchmark, demo."""
from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.syntax import Syntax

app = typer.Typer(help="Cross-Model Context Handoff Protocol CLI", no_args_is_help=True)
chain_app = typer.Typer(help="Handoff chain explorer — build, fork, and export non-linear chains", no_args_is_help=True)
app.add_typer(chain_app, name="chain")
console = Console()


@app.command()
def compress(
    input_file: Path = typer.Argument(..., help="JSON file with {conversation, tool_history}"),
    target_model: str = typer.Option("gpt-4o", help="Target model name"),
    source_model: str = typer.Option("claude-sonnet-4-6", help="Source model name"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Output JSON file (default: stdout)"),
) -> None:
    """Compress conversation state into a CMCHPPacket."""
    from .compressor import Compressor

    if not input_file.exists():
        console.print(f"[red]File not found: {input_file}[/red]")
        raise typer.Exit(1)

    data = json.loads(input_file.read_text())
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    compressor = Compressor(api_key=api_key)

    with console.status("Compressing..."):
        packet = compressor.compress(
            conversation=data.get("conversation", []),
            tool_history=data.get("tool_history", []),
            source_model=source_model,
            target_model=target_model,
        )

    result = packet.model_dump_json(indent=2)

    if output:
        output.write_text(result)
        console.print(f"[green]Packet written to {output}[/green]")
        console.print(
            f"Compression: {packet.compression.original_token_count}"
            f" → {packet.compression.compressed_token_count} tokens"
            f" ({packet.compression.compression_ratio:.2f}x)"
        )
    else:
        console.print(Syntax(result, "json", theme="monokai"))


@app.command()
def expand(
    packet_file: Path = typer.Argument(..., help="CMCHPPacket JSON file"),
    target_model: str | None = typer.Option(None, help="Override target model"),
) -> None:
    """Expand a CMCHPPacket into a model-specific context injection."""
    from .expander import Expander
    from .schema import CMCHPPacket

    if not packet_file.exists():
        console.print(f"[red]File not found: {packet_file}[/red]")
        raise typer.Exit(1)

    packet = CMCHPPacket.model_validate_json(packet_file.read_text())
    expander = Expander()
    expanded = expander.expand(packet, target_model)
    console.print(expanded)


@app.command()
def benchmark(
    scenarios_dir: Path = typer.Option(
        Path("benchmarks/scenarios"),
        "--scenarios-dir",
        "-s",
        help="Directory of scenario JSON files",
    ),
    source_model: str = typer.Option("claude-sonnet-4-6", help="Source model"),
    target_models: str = typer.Option(
        "gpt-4o,gemini-2.0-flash,claude-haiku-4-5",
        help="Comma-separated target models",
    ),
) -> None:
    """Run round-trip fidelity benchmark across scenarios and target models."""
    from .benchmark import print_results, run_benchmark

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    targets = [t.strip() for t in target_models.split(",") if t.strip()]

    console.print(f"[bold]CMCHP Benchmark[/bold] — {len(targets)} target model(s)")
    results = run_benchmark(
        scenarios_dir,
        api_key=api_key,
        source_model=source_model,
        target_models=targets,
    )
    print_results(results)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8000, help="Bind port"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload"),
) -> None:
    """Start the CMCHP demo server."""
    try:
        import uvicorn
    except ImportError:
        console.print("[red]uvicorn not installed. Run: pip install uvicorn[/red]")
        raise typer.Exit(1)

    uvicorn.run("demo.server:app", host=host, port=port, reload=reload)


@chain_app.command("run")
def chain_run(
    prompt: str = typer.Argument(..., help="Seed prompt"),
    models: str = typer.Option(
        "claude-opus-4-7,claude-sonnet-4-6,claude-haiku-4-5",
        "--models",
        "-m",
        help="Comma-separated handoff chain (in order)",
    ),
    slug: str = typer.Option("tree", "--slug", help="Persistence slug under ~/.cmchp/chains"),
    drift_threshold: float = typer.Option(0.6, "--drift", help="Drift alert threshold (0-1)"),
    ndjson: bool = typer.Option(False, "--ndjson", help="Stream NDJSON events to stdout (no TUI)"),
) -> None:
    """Run a new chain. Persists to the tree under --slug."""
    from .chain import load_tree, new_chain, run_chain, save_tree, ChainTree
    from .chain_view import run_with_live_view

    model_list = [m.strip() for m in models.split(",") if m.strip()]
    if not model_list:
        console.print("[red]No models specified[/red]")
        raise typer.Exit(1)

    tree = load_tree(slug) or ChainTree(root_prompt=prompt)
    chain = new_chain(seed_prompt=prompt, models=model_list)
    tree.add(chain)

    if ndjson:
        import json as _json
        for _hop in run_chain(
            chain, model_list,
            emit=lambda ev: print(_json.dumps(ev), flush=True),
            drift_threshold=drift_threshold,
        ):
            pass
    else:
        run_with_live_view(chain, model_list, drift_threshold=drift_threshold, console=console)

    save_tree(tree, slug)
    console.print(f"[dim]saved chain {chain.id} to tree '{slug}'[/dim]")


@chain_app.command("fork")
def chain_fork(
    chain_id: str = typer.Argument(..., help="Parent chain id"),
    at_hop: int = typer.Option(..., "--at", help="Hop index to fork at"),
    swap_model: str = typer.Option(..., "--swap", help="Model to substitute at that hop"),
    slug: str = typer.Option("tree", "--slug"),
    execute: bool = typer.Option(True, "--execute/--no-execute", help="Run the forked chain now"),
    drift_threshold: float = typer.Option(0.6, "--drift"),
) -> None:
    """Fork an existing chain at --at, swapping in --swap, and (by default) run it."""
    from .chain import fork_chain, load_tree, run_chain, save_tree
    from .chain_view import run_with_live_view

    tree = load_tree(slug)
    if not tree or chain_id not in tree.chains:
        console.print(f"[red]chain {chain_id} not found in tree '{slug}'[/red]")
        raise typer.Exit(1)

    parent = tree.chains[chain_id]
    child = fork_chain(tree, chain_id, at_hop, swap_model)
    save_tree(tree, slug)
    console.print(f"[green]forked[/green] {chain_id} → {child.id} (at hop {at_hop}, swap → {swap_model})")

    if execute:
        # Re-run from the fork point onward, using swap_model + remaining parent models.
        remaining = [swap_model] + [h.model for h in parent.hops[at_hop + 1 :]]
        # Wipe the prepared placeholder — run_chain appends from scratch.
        child.hops = child.hops[:at_hop]
        run_with_live_view(child, remaining, drift_threshold=drift_threshold, console=console)
        save_tree(tree, slug)


@chain_app.command("tree")
def chain_tree(
    slug: str = typer.Option("tree", "--slug"),
) -> None:
    """Print the whole chain tree with drift scores."""
    from .chain import load_tree
    from .chain_view import render_tree

    tree = load_tree(slug)
    if not tree:
        console.print(f"[yellow]no tree at slug '{slug}'[/yellow]")
        raise typer.Exit(0)
    render_tree(tree, console=console)


@chain_app.command("export")
def chain_export(
    chain_id: str = typer.Argument(..., help="Chain id to export"),
    slug: str = typer.Option("tree", "--slug"),
    output: Path | None = typer.Option(None, "--output", "-o"),
) -> None:
    """Export a chain as a reproducible JSON recipe."""
    from .chain import export_recipe, load_tree

    tree = load_tree(slug)
    if not tree or chain_id not in tree.chains:
        console.print(f"[red]chain {chain_id} not found[/red]")
        raise typer.Exit(1)
    recipe = export_recipe(tree.chains[chain_id])
    out = json.dumps(recipe, indent=2)
    if output:
        output.write_text(out)
        console.print(f"[green]recipe written to {output}[/green]")
    else:
        console.print(Syntax(out, "json", theme="monokai"))


@chain_app.command("undo")
def chain_undo(
    slug: str = typer.Option("tree", "--slug"),
) -> None:
    """Roll back to the previous snapshot (up to 10 held)."""
    from .chain import undo

    restored = undo(slug)
    if restored is None:
        console.print(f"[yellow]nothing to undo for slug '{slug}'[/yellow]")
        raise typer.Exit(0)
    console.print(f"[green]reverted '{slug}' to previous snapshot[/green] ({len(restored.chains)} chains)")


def main() -> None:
    app()
