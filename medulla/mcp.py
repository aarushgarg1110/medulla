"""Medulla MCP stdio server — Sprint 2.

Exposes 11 tools across the episodic memory layer (semantic + codebase in later sprints).
Register with Claude Code:
    claude mcp add medulla uv -- --project /path/to/medulla run medulla mcp
"""
from __future__ import annotations

import json
import sys
from typing import Any

import anyio
import mcp.types as types
from mcp.server import Server
from mcp.server.stdio import stdio_server

from medulla.db.database import connect
from medulla.episodic.store import (
    get_project_context,
    get_session_detail,
    get_session_tree,
    get_stats,
    list_sessions,
    search_events,
)
from medulla.search import search as fts_search

_server = Server("medulla")

# ── Tool registry ──────────────────────────────────────────────────────────────

_TOOLS = [
    types.Tool(
        name="medulla_search",
        description="Search past Claude/Kiro sessions and wiki pages by keyword. Returns matched excerpts with session ID, date, and project.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "default": 10},
                "layer": {"type": "string", "enum": ["episodic", "semantic", "code"], "description": "Restrict to a single layer (omit for all)"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="medulla_session_detail",
        description="Retrieve full content of a specific session — chunks, files touched, tools used, and child agents.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID from medulla_search or medulla_list"},
            },
            "required": ["session_id"],
        },
    ),
    types.Tool(
        name="medulla_session_tree",
        description="Show a session and all its child subagent sessions.",
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        },
    ),
    types.Tool(
        name="medulla_project_context",
        description="Recent sessions and tool events for a project directory. Defaults to current working directory.",
        inputSchema={
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project directory substring (default: current dir)"},
                "session_limit": {"type": "integer", "default": 5},
                "event_limit": {"type": "integer", "default": 20},
            },
        },
    ),
    types.Tool(
        name="medulla_list",
        description="List recent Claude/Kiro sessions, optionally filtered by project.",
        inputSchema={
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
        },
    ),
    types.Tool(
        name="medulla_stats",
        description="Aggregate statistics: session count, turns, tool calls, top tools, date range.",
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="medulla_events_search",
        description="Search tool-call events (Bash commands, file reads, etc.) by keyword.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="medulla_wiki_search",
        description="Search the semantic wiki (source/concept/entity pages). Available after Sprint 3.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "type": {"type": "string", "enum": ["source", "concept", "entity"]},
                "limit": {"type": "integer", "default": 10},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="medulla_wiki_page",
        description="Get full content of a wiki page by slug. Available after Sprint 3.",
        inputSchema={
            "type": "object",
            "properties": {
                "slug": {"type": "string"},
            },
            "required": ["slug"],
        },
    ),
    types.Tool(
        name="medulla_ingest",
        description="Write findings or notes directly into the wiki as a new source page. Available after Sprint 3.",
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "content"],
        },
    ),
    types.Tool(
        name="medulla_analyze",
        description="Manifest quality analysis: retry rates, help-followup rates, error rates per tool manifest.",
        inputSchema={
            "type": "object",
            "properties": {
                "since_days": {"type": "integer", "default": 30},
                "top": {"type": "integer", "default": 20},
            },
        },
    ),
]


@_server.list_tools()
async def list_tools() -> list[types.Tool]:
    return _TOOLS


@_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:
    text = _dispatch(name, arguments)
    return [types.TextContent(type="text", text=text)]


# ── Tool dispatch ──────────────────────────────────────────────────────────────

def _dispatch(name: str, args: dict) -> str:
    conn = connect()
    match name:
        case "medulla_search":
            return _tool_search(conn, args)
        case "medulla_session_detail":
            return _tool_session_detail(conn, args)
        case "medulla_session_tree":
            return _tool_session_tree(conn, args)
        case "medulla_project_context":
            return _tool_project_context(conn, args)
        case "medulla_list":
            return _tool_list(conn, args)
        case "medulla_stats":
            return _tool_stats(conn)
        case "medulla_events_search":
            return _tool_events_search(conn, args)
        case "medulla_wiki_search" | "medulla_wiki_page" | "medulla_ingest":
            return "Semantic wiki layer available in Sprint 3. Run `medulla ingest <file>` to add documents."
        case "medulla_analyze":
            return _tool_analyze(conn, args)
        case _:
            return f"Unknown tool: {name}"


def _tool_search(conn, args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: query is required"
    results = fts_search(
        conn, query,
        limit=args.get("limit", 10),
        layer=args.get("layer"),
    )
    if not results:
        return f"No results found for: {query}"

    lines = [f"{len(results)} result(s) for \"{query}\":\n"]
    for r in results:
        date = (r.date or "")[:10]
        proj = (r.project_dir or "").split("/")[-1]
        lines.append(f"[{r.id[:8]}]  {date}  {proj}")
        lines.append(f"  {r.excerpt}\n")
    return "\n".join(lines)


def _tool_session_detail(conn, args: dict) -> str:
    session_id = args.get("session_id", "").strip()
    if not session_id:
        return "Error: session_id is required"
    detail = get_session_detail(conn, session_id)
    if not detail:
        return f"Session not found: {session_id}"

    s = detail["session"]
    lines = [
        f"Session: {s['session_id']}",
        f"Project: {s.get('project_dir', '')}",
        f"Model:   {s.get('model', '')}",
        f"Date:    {(s.get('started_at') or '')[:10]} → {(s.get('ended_at') or '')[:10]}",
        f"Turns:   {s.get('turn_count', 0)}   Tool calls: {s.get('tool_call_count', 0)}",
        "",
    ]
    if detail["agents"]:
        lines.append(f"Subagents ({len(detail['agents'])}):")
        for a in detail["agents"]:
            lines.append(f"  {a['agent_id'][:8]}  {a.get('first_message', '')[:60]}")
        lines.append("")

    lines.append(f"Chunks ({len(detail['chunks'])}):")
    for c in detail["chunks"]:
        lines.append(f"\n── Chunk {c['chunk_index']} (turns {c['turn_start']}–{c['turn_end']}) ──")
        lines.append(c["chunk_text"][:600])

    return "\n".join(lines)


def _tool_session_tree(conn, args: dict) -> str:
    session_id = args.get("session_id", "").strip()
    if not session_id:
        return "Error: session_id is required"
    tree = get_session_tree(conn, session_id)
    if not tree:
        return f"Session not found: {session_id}"

    s = tree["session"]
    lines = [
        f"Session: {s['session_id']}",
        f"  {(s.get('started_at') or '')[:10]}  {s.get('project_dir', '')}",
        f"  Turns: {s.get('turn_count', 0)}  Tools: {s.get('tool_call_count', 0)}",
        f"  {s.get('first_message', '')[:80]}",
    ]
    if tree["agents"]:
        lines.append(f"\nSubagents ({len(tree['agents'])}):")
        for a in tree["agents"]:
            lines.append(
                f"  └─ {a['agent_id'][:8]}  [{a.get('agent_slug') or 'agent'}]"
                f"  turns={a.get('turn_count', 0)}"
                f"  {a.get('first_message', '')[:60]}"
            )
    else:
        lines.append("\nNo subagents.")
    return "\n".join(lines)


def _tool_project_context(conn, args: dict) -> str:
    import os
    project = args.get("project") or os.getcwd()
    ctx = get_project_context(
        conn, project,
        session_limit=args.get("session_limit", 5),
        event_limit=args.get("event_limit", 20),
    )
    lines = [f"Project context: {project}\n"]

    if ctx["sessions"]:
        lines.append(f"Recent sessions ({len(ctx['sessions'])}):")
        for s in ctx["sessions"]:
            lines.append(
                f"  {s['session_id'][:8]}  {(s.get('started_at') or '')[:10]}"
                f"  {s.get('first_message', '')[:60]}"
            )
    else:
        lines.append("No sessions found for this project.")

    if ctx["events"]:
        lines.append(f"\nRecent tool events ({len(ctx['events'])}):")
        for e in ctx["events"]:
            lines.append(f"  {(e.get('event_ts') or '')[:16]}  {e.get('tool', '')}  {(e.get('command') or '')[:60]}")
    return "\n".join(lines)


def _tool_list(conn, args: dict) -> str:
    rows = list_sessions(conn, project=args.get("project"), limit=args.get("limit", 20))
    if not rows:
        return "No sessions found."
    lines = [f"{len(rows)} session(s):\n"]
    for r in rows:
        date = (r["started_at"] or "")[:10]
        proj = (r["project_dir"] or "").split("/")[-1]
        msg = (r["first_message"] or "")[:60].replace("\n", " ")
        lines.append(f"  {r['session_id'][:8]}  {date}  {proj}")
        lines.append(f"    turns={r['turn_count']}  {msg}")
    return "\n".join(lines)


def _tool_stats(conn) -> str:
    s = get_stats(conn)
    lines = [
        f"Sessions:       {s['sessions']:,}",
        f"Chunks:         {s['chunks']:,}",
        f"Agent sessions: {s['agent_sessions']:,}",
        f"Turns:          {s['turns']:,}",
        f"Tool calls:     {s['tool_calls']:,}",
    ]
    if s["oldest"]:
        lines.append(f"Date range:     {s['oldest'][:10]} → {s['newest'][:10]}")
    if s["top_tools"]:
        lines.append("\nTop tools:")
        for name, count in s["top_tools"]:
            lines.append(f"  {name:<35} {count:>5}")
    return "\n".join(lines)


def _tool_events_search(conn, args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: query is required"
    rows = search_events(conn, query, limit=args.get("limit", 20))
    if not rows:
        return f"No tool events found for: {query}"
    lines = [f"{len(rows)} event(s):\n"]
    for r in rows:
        lines.append(f"  {(r['event_ts'] or '')[:16]}  {r['tool']}  {(r['command'] or '')[:80]}")
        if r["output_preview"]:
            lines.append(f"    → {r['output_preview'][:60]}")
    return "\n".join(lines)


def _tool_analyze(conn, args: dict) -> str:
    """Manifest quality — only meaningful once tool_events is populated."""
    count = conn.execute("SELECT COUNT(*) FROM tool_events").fetchone()[0]
    if count == 0:
        return (
            "No tool events indexed yet.\n"
            "Wire up the PostToolUse hook to populate tool_events:\n"
            '  Add to ~/.claude/settings.json hooks → PostToolUse:\n'
            '  {"type": "command", "command": "medulla scan"}'
        )
    # Future: compute retry/help/error rates per manifest_key
    return f"tool_events count: {count}. Full manifest quality analysis coming in Sprint 5."


# ── Entry point ────────────────────────────────────────────────────────────────

def serve() -> None:
    anyio.run(_serve)


async def _serve() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await _server.run(
            read_stream,
            write_stream,
            _server.create_initialization_options(),
        )
