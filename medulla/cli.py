"""Medulla CLI — three-layer memory for Claude Code and Kiro."""
from __future__ import annotations

from typing import Annotated, Optional
import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

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

    console.print(f"[green]✓[/green] Sessions: {counts['indexed']} indexed, {counts['skipped']} skipped, {counts['errors']} errors")
    console.print(f"[green]✓[/green] Agents:   {counts['agents_indexed']} indexed, {counts['agents_skipped']} skipped")


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
        date_str = r.date[:10] if r.date else "unknown"
        proj = r.project_dir.split("/")[-1] if r.project_dir else ""
        console.print(f"\n[bold cyan]{r.id[:8]}[/bold cyan]  [dim]{date_str}  {proj}[/dim]")
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

    conn = connect()
    s = get_stats(conn)

    console.print(f"\n[bold]Medulla stats[/bold]")
    console.print(f"  Sessions:      {s['sessions']:,}")
    console.print(f"  Chunks:        {s['chunks']:,}")
    console.print(f"  Agent sessions:{s['agent_sessions']:,}")
    console.print(f"  Turns:         {s['turns']:,}")
    console.print(f"  Tool calls:    {s['tool_calls']:,}")
    if s['oldest']:
        console.print(f"  Date range:    {s['oldest'][:10]} → {s['newest'][:10]}")
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


def main():
    app()
