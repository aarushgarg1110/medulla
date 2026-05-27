"""Medulla CLI — three-layer memory for Claude Code and Kiro."""
from __future__ import annotations

import subprocess
from typing import Annotated, Optional
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="medulla",
    help="Three-layer memory for Claude Code and Kiro.",
    no_args_is_help=True,
)
console = Console()


# ── scan ──────────────────────────────────────────────────────────────────────

@app.command()
def scan(
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-index all sessions, not just new ones")] = False,
    source: Annotated[Optional[str], typer.Option("--source", help="Only scan: claude | kiro")] = None,
):
    """Index new/changed Claude and Kiro sessions."""
    from medulla.db.database import connect
    from medulla.episodic.scanner import scan as do_scan

    with console.status("[bold green]Scanning sessions..."):
        conn = connect()
        counts = do_scan(conn, force=force, source=source)

    empty_note = f", {counts['empty']} empty/stub" if counts.get('empty') else ""
    console.print(f"[green]✓[/green] Sessions: {counts['indexed']} indexed, {counts['skipped']} unchanged{empty_note}, {counts['errors']} errors")
    console.print(f"[green]✓[/green] Agents:   {counts['agents_indexed']} indexed, {counts['agents_skipped']} unchanged")


# ── search ─────────────────────────────────────────────────────────────────────

@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Search query")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    layer: Annotated[Optional[str], typer.Option("--layer", help="episodic | semantic | code")] = None,
):
    """Search across all memory layers."""
    from medulla.db.database import connect
    from medulla.search import search as do_search

    conn = connect()
    results = do_search(conn, query, limit=limit, layer=layer)

    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit()

    for r in results:
        if r.layer == "semantic":
            label = f"[magenta]{r.title}[/magenta]"
        else:
            date_str = r.date[:10] if r.date else ""
            proj = r.project_dir.split("/")[-1] if r.project_dir else ""
            label = f"[bold cyan]{r.id[:8]}[/bold cyan]  [dim]{date_str}  {proj}[/dim]"
        console.print(f"\n{label}")
        console.print(f"  [italic]{r.excerpt}[/italic]")


# ── list ───────────────────────────────────────────────────────────────────────

@app.command(name="list")
def list_sessions(
    project: Annotated[Optional[str], typer.Option("--project", "-p")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 20,
):
    """List recent sessions."""
    from medulla.db.database import connect
    from medulla.episodic.store import list_sessions as do_list

    conn = connect()
    rows = do_list(conn, project=project, limit=limit)

    if not rows:
        console.print("[yellow]No sessions found.[/yellow]")
        raise typer.Exit()

    table = Table(show_header=True, header_style="bold")
    table.add_column("Date", style="dim", width=12)
    table.add_column("Session ID", width=10)
    table.add_column("Project", width=25)
    table.add_column("Turns", justify="right", width=6)
    table.add_column("First message", width=50)

    for row in rows:
        date = (row["started_at"] or "")[:10]
        sid = (row["session_id"] or "")[:8]
        proj = (row["project_dir"] or "").split("/")[-1][:25]
        turns = str(row["turn_count"] or 0)
        msg = (row["first_message"] or "")[:80].replace("\n", " ")
        table.add_row(date, sid, proj, turns, msg)

    console.print(table)


# ── stats ──────────────────────────────────────────────────────────────────────

@app.command()
def stats():
    """Show aggregate statistics."""
    from medulla.db.database import connect
    from medulla.episodic.store import get_stats

    from medulla.semantic.store import get_wiki_stats
    conn = connect()
    s = get_stats(conn)
    ws = get_wiki_stats(conn)

    console.print(f"\n[bold]Medulla stats[/bold]")
    console.print(f"\n  [bold]Episodic[/bold]")
    console.print(f"    Sessions:      {s['sessions']:,}")
    console.print(f"    Chunks:        {s['chunks']:,}")
    console.print(f"    Agent sessions:{s['agent_sessions']:,}")
    console.print(f"    Turns:         {s['turns']:,}")
    console.print(f"    Tool calls:    {s['tool_calls']:,}")
    if s['oldest']:
        console.print(f"    Date range:    {s['oldest'][:10]} → {s['newest'][:10]}")

    console.print(f"\n  [bold]Semantic (wiki)[/bold]")
    console.print(f"    Pages:         {ws['total']:,}")
    _PLURAL = {"source": "sources", "concept": "concepts", "entity": "entities"}
    for page_type, count in ws.get("by_type", {}).items():
        console.print(f"    {_PLURAL.get(page_type, page_type + 's'):<14} {count:,}")

    if s['top_tools']:
        console.print(f"\n  [bold]Top tools:[/bold]")
        for name, count in s['top_tools'][:10]:
            console.print(f"    {name:<30} {count:>6}")


# ── session-detail ────────────────────────────────────────────────────────────

@app.command(name="session-detail")
def session_detail(
    session_id: Annotated[str, typer.Argument(help="Session ID (full or 8-char prefix)")],
):
    """Show full detail for a session — chunks, agents, files touched."""
    from medulla.db.database import connect
    from medulla.episodic.store import get_session_detail

    conn = connect()

    # Allow 8-char prefix lookup
    if len(session_id) < 36:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id LIKE ?",
            (f"{session_id}%",)
        ).fetchone()
        if not row:
            console.print(f"[red]Session not found: {session_id}[/red]")
            raise typer.Exit(1)
        session_id = row["session_id"]

    detail = get_session_detail(conn, session_id)
    if not detail:
        console.print(f"[red]Session not found: {session_id}[/red]")
        raise typer.Exit(1)

    s = detail["session"]
    console.print(f"\n[bold]Session:[/bold] {s['session_id']}")
    console.print(f"  Project:    {s.get('project_dir', '')}")
    console.print(f"  Model:      {s.get('model', '')}")
    console.print(f"  Date:       {(s.get('started_at') or '')[:10]} → {(s.get('ended_at') or '')[:10]}")
    console.print(f"  Turns:      {s.get('turn_count', 0)}   Tool calls: {s.get('tool_call_count', 0)}")

    if detail["agents"]:
        console.print(f"\n  [bold]Subagents ({len(detail['agents'])}):[/bold]")
        for a in detail["agents"]:
            console.print(f"    {a['agent_id'][:8]}  {a.get('first_message', '')[:60]}")

    console.print(f"\n  [bold]Chunks ({len(detail['chunks'])}):[/bold]")
    for c in detail["chunks"]:
        console.print(f"\n[dim]── Chunk {c['chunk_index']} (turns {c['turn_start']}–{c['turn_end']}) ──[/dim]")
        console.print(c["chunk_text"][:400])


# ── mcp ────────────────────────────────────────────────────────────────────────

@app.command()
def mcp():
    """Start the MCP stdio server (for: claude mcp add medulla ...)."""
    from medulla.mcp import serve
    serve()


# ── use ───────────────────────────────────────────────────────────────────────

@app.command()
def use(
    provider: Annotated[str, typer.Argument(help="bedrock | anthropic | ollama")],
    model: Annotated[Optional[str], typer.Option("--model", "-m", help="Model name override")] = None,
):
    """Switch the active LLM provider."""
    import os, subprocess
    valid = {"bedrock", "anthropic", "ollama"}
    if provider not in valid:
        console.print(f"[red]Unknown provider: {provider}. Choose: {', '.join(sorted(valid))}[/red]")
        raise typer.Exit(1)

    from medulla.config import set_active_provider, get_config, save_config

    if provider == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[yellow]Warning:[/yellow] ANTHROPIC_API_KEY is not set.")
            console.print("  Add to your shell profile:  [bold]export ANTHROPIC_API_KEY=sk-ant-...[/bold]")
            console.print("  Medulla will use it at ingest time (never stored in config.toml).")

    if provider == "ollama":
        # Check server is reachable
        try:
            import httpx
            httpx.get("http://localhost:11434/api/tags", timeout=3.0).raise_for_status()
        except Exception:
            console.print("[yellow]Warning:[/yellow] Ollama server not reachable at http://localhost:11434")
            console.print("  Start it with:  [bold]ollama serve[/bold]")
            console.print("  Install models: [bold]ollama pull llama3.2[/bold]")

        # Show available models if no --model given
        if not model:
            try:
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                lines = [l for l in result.stdout.strip().splitlines() if l and not l.startswith("NAME")]
                if lines:
                    console.print("\n  [bold]Available Ollama models:[/bold]")
                    for line in lines:
                        name = line.split()[0]
                        console.print(f"    {name}")
                    console.print(f"\n  Using default: [cyan]{get_config().llm.ollama.model}[/cyan]")
                    console.print("  Override with: [bold]medulla use ollama --model <name>[/bold]")
            except Exception:
                pass

    set_active_provider(provider)

    if model:
        cfg = get_config()
        if provider == "ollama":
            cfg.llm.ollama.model = model
        elif provider == "anthropic":
            cfg.llm.anthropic.model = model
        elif provider == "bedrock":
            cfg.llm.bedrock.model = model
        save_config(cfg)
        console.print(f"[green]✓[/green] Provider: [bold]{provider}[/bold]  Model: [bold]{model}[/bold]")
    else:
        console.print(f"[green]✓[/green] Active provider set to [bold]{provider}[/bold]")
    console.print("  Run [bold]medulla status[/bold] to verify.")


# ── status ─────────────────────────────────────────────────────────────────────

@app.command()
def status():
    """Show provider, model, pending sources, and unindexed sessions."""
    from medulla.config import get_config
    from medulla.db.database import connect
    from medulla.semantic.store import get_pending_count, get_wiki_stats
    import glob, os

    cfg = get_config()
    active = cfg.llm.active

    console.print(f"\n[bold]Medulla status[/bold]")
    console.print(f"\n  [bold]LLM Provider[/bold]")
    console.print(f"    Active:   [cyan]{active}[/cyan]")
    if active == "bedrock":
        console.print(f"    Model:    {cfg.llm.bedrock.model}")
        console.print(f"    Profile:  {cfg.llm.bedrock.aws_profile}  Region: {cfg.llm.bedrock.aws_region}")
    elif active == "anthropic":
        console.print(f"    Model:    {cfg.llm.anthropic.model}")
        key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
        console.print(f"    API key:  {'[green]set[/green]' if key_set else '[red]not set[/red]'}")
    elif active == "ollama":
        console.print(f"    Model:    {cfg.llm.ollama.model}")
        console.print(f"    Host:     {cfg.llm.ollama.host}")

    conn = connect()

    # Wiki stats
    wiki_stats = get_wiki_stats(conn)
    console.print(f"\n  [bold]Wiki (semantic layer)[/bold]")
    console.print(f"    Pages:    {wiki_stats['total']} total")
    for page_type, count in wiki_stats.get("by_type", {}).items():
        console.print(f"              {count} {page_type}s")

    # Pending ingests (raw/ files not yet processed)
    pending = get_pending_count(conn)
    raw_dir = cfg.wiki_path / "raw"
    raw_files = [f for f in raw_dir.iterdir() if f.is_file() and f.name != "url-references.md"] if raw_dir.exists() else []
    # Cross-check: only count pending entries where the raw/ file still exists
    valid_pending = conn.execute("""
        SELECT COUNT(*) FROM pending_ingests WHERE status = 'queued'
    """).fetchone()[0]
    console.print(f"\n  [bold]raw/ (intake queue)[/bold]")
    console.print(f"    Files:    {len(raw_files)} in raw/")
    if valid_pending:
        console.print(f"    Queued:   [yellow]{valid_pending} awaiting processing[/yellow]")
        console.print("    Run [bold]medulla ingest[/bold] to process them.")
    else:
        console.print(f"    Queued:   none — all processed")

    # Sessions on disk vs indexed
    from medulla.episodic.scanner import CLAUDE_PROJECTS_DIR, is_subagent_file
    all_jsonl = list(CLAUDE_PROJECTS_DIR.rglob("*.jsonl")) if CLAUDE_PROJECTS_DIR.exists() else []
    session_files = [f for f in all_jsonl if not is_subagent_file(f)]
    indexed = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    agents_indexed = conn.execute("SELECT COUNT(*) FROM agent_sessions").fetchone()[0]
    console.print(f"\n  [bold]Sessions[/bold]")
    console.print(f"    Indexed:  {indexed} sessions, {agents_indexed} subagents")
    unindexed = len(session_files) - indexed
    if unindexed > 0:
        console.print(f"    On disk:  {len(session_files)} session files ({unindexed} not yet indexed — run [bold]medulla scan[/bold])")
    else:
        console.print(f"    On disk:  {len(session_files)} session files — all indexed ✓")


# ── ingest ─────────────────────────────────────────────────────────────────────

@app.command()
def ingest(
    source: Annotated[Optional[str], typer.Argument(help="File path or URL. Omit to process everything in raw/")] = None,
    title: Annotated[Optional[str], typer.Option("--title", "-t")] = None,
    scope: Annotated[str, typer.Option("--scope")] = "personal",
    force: Annotated[bool, typer.Option("--force", "-f", help="Re-ingest even if already processed")] = False,
):
    """Ingest sources into the semantic wiki via raw/.

    No args: discover new files in raw/ + process all queued.
    With path/URL: copy/fetch to raw/ then process immediately.
    --force: re-ingest even if this source was previously processed.
    """
    from pathlib import Path
    from medulla.db.database import connect
    from medulla.config import get_config
    from medulla.semantic.ingest import intake_to_raw, discover_raw, process_pending

    conn = connect()
    cfg = get_config()
    wiki_path = cfg.wiki_path

    def _get_provider():
        try:
            from medulla.llm import get_provider
            return get_provider()
        except Exception as e:
            return None, str(e)

    provider_result = _get_provider()
    has_provider = not isinstance(provider_result, tuple)

    if source:
        # Intake to raw/ first (copy/fetch)
        with console.status(f"Copying to raw/: {source}"):
            try:
                raw_path = intake_to_raw(conn, wiki_path, source, title, force=force)
                console.print(f"[dim]→ raw/{raw_path.name}[/dim]")
            except Exception as e:
                console.print(f"[red]✗ Failed: {e}[/red]")
                raise typer.Exit(1)

        if not has_provider:
            console.print(f"[yellow]⚠[/yellow]  No LLM provider — queued in raw/")
            console.print(f"  Configure: [bold]medulla use bedrock[/bold]  then run [bold]medulla ingest[/bold]")
            return
    else:
        # Discover any files dropped in raw/ (Obsidian Clipper, manual)
        new = discover_raw(wiki_path, conn)
        if new:
            console.print(f"[dim]Discovered {len(new)} new file(s) in raw/[/dim]")

        if not has_provider:
            from medulla.semantic.store import get_pending_count
            n = get_pending_count(conn)
            console.print(f"[yellow]⚠[/yellow]  No LLM provider — {n} file(s) queued in raw/")
            console.print(f"  Configure: [bold]medulla use bedrock[/bold]  then run [bold]medulla ingest[/bold]")
            return

    # Process all queued files with live streaming output
    from medulla.semantic.store import get_pending_count
    n = get_pending_count(conn)
    if n == 0:
        console.print("[dim]Nothing queued to process.[/dim]")
        return

    console.print(f"Processing {n} queued file(s)...\n")

    def _stream_token(text: str) -> None:
        """Print LLM tokens as they arrive so the CLI doesn't look hung."""
        print(text, end="", flush=True)

    results = process_pending(wiki_path, conn, provider_result, scope=scope, on_token=_stream_token)
    print()  # newline after streaming output
    for r in results:
        name = Path(r["source_path"]).name
        if "error" in r:
            console.print(f"\n[red]✗[/red] {name}: {r['error']}")
        else:
            console.print(f"\n[green]✓[/green] {name} → {r['total_pages']} pages "
                          f"({', '.join(r['concepts'] + r['entities']) or 'source only'})")


# ── wiki subcommands ───────────────────────────────────────────────────────────

wiki_app = typer.Typer(help="Wiki management commands.")
app.add_typer(wiki_app, name="wiki")


@wiki_app.command(name="list")
def wiki_list(
    page_type: Annotated[Optional[str], typer.Option("--type", "-t", help="source | concept | entity")] = None,
    limit: Annotated[int, typer.Option("--limit", "-n")] = 30,
):
    """List wiki pages."""
    from medulla.db.database import connect
    from medulla.semantic.store import list_wiki_pages

    conn = connect()
    rows = list_wiki_pages(conn, page_type=page_type, limit=limit)
    if not rows:
        console.print("[yellow]No wiki pages found. Run medulla ingest <source>[/yellow]")
        return

    table = Table(show_header=True, header_style="bold")
    table.add_column("Type", width=8)
    table.add_column("Slug", width=30)
    table.add_column("Title", width=40)
    table.add_column("Date", width=12, style="dim")
    for row in rows:
        table.add_row(row["type"], row["slug"][:30], row["title"][:40], (row["ingested_at"] or "")[:10])
    console.print(table)


@wiki_app.command(name="open")
def wiki_open():
    """Open the wiki vault in Obsidian."""
    from medulla.config import get_config
    wiki = get_config().wiki_path
    if not wiki.exists():
        console.print("[red]Wiki not found. Run medulla ingest first.[/red]")
        raise typer.Exit(1)
    subprocess.run(["open", "-a", "Obsidian", str(wiki)], check=False)
    console.print(f"[green]✓[/green] Opening {wiki} in Obsidian")


@wiki_app.command(name="lint")
def wiki_lint():
    """Check for broken links and orphaned pages."""
    from medulla.config import get_config
    from medulla.semantic.wiki import lint_wiki

    wiki_path = get_config().wiki_path
    report = lint_wiki(wiki_path)

    if "error" in report:
        console.print(f"[red]{report['error']}[/red]")
        return

    console.print(f"\n[bold]Wiki lint[/bold] — {report['total_pages']} pages")

    suggested = report.get("suggested_pages", [])
    if suggested:
        console.print(f"\n[dim]Suggested pages ({len(suggested)} linked but not yet created — ingest more sources to fill them):[/dim]")
        for link in suggested[:20]:
            console.print(f"  {link}", markup=False)
    else:
        console.print("[green]✓[/green] No forward references")

    if report["orphaned_pages"]:
        console.print(f"\n[yellow]Orphaned pages ({len(report['orphaned_pages'])}):[/yellow]")
        for slug in report["orphaned_pages"][:20]:
            console.print(f"  {slug}")
    else:
        console.print("[green]✓[/green] No orphaned pages")


@app.command()
def reset(
    all_data: Annotated[bool, typer.Option("--all", help="Also wipe raw/ and episodic sessions")] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation")] = False,
):
    """Reset wiki state for a clean slate.

    Default: clears wiki pages (sources/concepts/entities), log, index.
    Keeps raw/ so your source files aren't lost.
    --all: also clears raw/, episodic sessions, and agent sessions.
    """
    from medulla.config import get_config
    from medulla.db.database import connect
    import shutil

    cfg = get_config()
    wiki = cfg.wiki_path

    what = "wiki pages, index.md, log.md (raw/ preserved)"
    if all_data:
        what = "ALL wiki data including raw/, episodic sessions, agent sessions"

    if not yes:
        console.print(f"[yellow]⚠[/yellow]  This will delete: {what}")
        console.print(f"  Wiki path: [dim]{wiki}[/dim]")
        typer.confirm("Continue?", abort=True)

    conn = connect()

    # Always clear wiki layer
    conn.execute("DELETE FROM wiki_pages")
    conn.execute("DELETE FROM pending_ingests")
    conn.commit()

    for subdir in ["sources", "concepts", "entities"]:
        d = wiki / subdir
        if d.exists():
            shutil.rmtree(d)

    for f in ["index.md", "log.md"]:
        p = wiki / f
        if p.exists():
            p.unlink()

    if all_data:
        raw = wiki / "raw"
        if raw.exists():
            shutil.rmtree(raw)
        conn.execute("DELETE FROM sessions")
        conn.execute("DELETE FROM session_chunks")
        conn.execute("DELETE FROM agent_sessions")
        conn.execute("DELETE FROM tool_events")
        conn.commit()

    console.print("[green]✓[/green] Reset complete.")
    if not all_data:
        console.print("  raw/ preserved — run [bold]medulla ingest[/bold] to re-process sources.")


def main():
    app()
