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
        description=(
            "Retrieve content of a specific session. "
            "Without chunk_index: returns session metadata + first 5 chunks. "
            "With chunk_index: returns that specific chunk in full (use to page through a long session). "
            "Call repeatedly with increasing chunk_index to read the full session."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID (full UUID or 8-char prefix)"},
                "chunk_index": {"type": "integer", "description": "Specific chunk to fetch (0-based). Omit for session overview + first 5 chunks."},
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
        description=(
            "Store a wiki page you have synthesized. Call this tool MULTIPLE TIMES to build a fully connected graph — "
            "one call per page type. Skipping concept/entity pages leaves orphaned nodes in Obsidian.\n\n"
            "REQUIRED WORKFLOW for a complete graph:\n"
            "0. Call medulla_wiki_schema FIRST to get existing page slugs — use ONLY those slugs for [[wikilinks]]\n"
            "1. page_type='source': the full source summary\n"
            "2. page_type='concept': once per significant concept — NO LIMIT, create as many as are meaningful\n"
            "3. page_type='entity': once per significant entity — NO LIMIT\n\n"
            "CRITICAL: Concept and entity pages MUST include 'sources: [source-slug]' in their frontmatter. "
            "This creates bidirectional edges in the Obsidian graph. Without it, nodes are isolated.\n\n"
            "SOURCE PAGE format:\n"
            "---\\ntitle: Full Title\\nsource: <url or path>\\ndate_ingested: YYYY-MM-DD\\n"
            "tags: [tag1, tag2, ...]\\n---\\n"
            "## Summary\\n## Key Points\\n## Concepts Introduced or Updated\\n"
            "## Entities Mentioned\\n## Connections\\n## Gaps / Open Questions\n\n"
            "CONCEPT PAGE format:\n"
            "---\\ntitle: Concept Name\\ntags: [tag1, tag2]\\nsources: [source-slug]\\n---\\n"
            "## Definition\\n## How It Works\\n## Why It Matters\\n## Nuances & Caveats\\n"
            "## Evidence & Examples\\n## Connections\\n## Open Questions\n\n"
            "ENTITY PAGE format:\n"
            "---\\ntitle: Entity Name\\ntype: person|org|tool|project|database\\n"
            "tags: [tag1]\\nsources: [source-slug]\\n---\\n"
            "## Who / What\\n## Relevance\\n## Key Contributions / Features\\n## Connections\n\n"
            "Use [[slug]] wikilinks for ALL cross-references (slugs = lowercase-hyphenated). "
            "Be generous with tags — they power the Obsidian graph filter. "
            "The 3-concept/2-entity cap applies ONLY to CLI ingest (Bedrock token limits). MCP has NO limit. "
            "ALWAYS pass source_path when you read a local file (PDF, markdown) — the file gets copied to wiki/raw/ as an immutable archive. "
            "ALWAYS pass source_url when you fetched via WebFetch — the URL gets logged to url-references.md. "
            "Both can be provided for a PDF downloaded from a URL."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Page title"},
                "content": {"type": "string", "description": "Full markdown content you have synthesized"},
                "page_type": {"type": "string", "enum": ["source", "concept", "entity"], "default": "source"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "source_url": {"type": "string", "description": "Original URL if you fetched via WebFetch (appended to url-references.md log)"},
                "source_path": {"type": "string", "description": "Local file path if you read a PDF/file (e.g. /Users/agarg/Downloads/paper.pdf) — file is copied to wiki/raw/ for immutable archive"},
            },
            "required": ["title", "content"],
        },
    ),
    types.Tool(
        name="medulla_ingest_url",
        description=(
            "Fetch a URL, synthesize it using the configured LLM provider, and store wiki pages. "
            "Use this ONLY if you cannot fetch URLs yourself (no WebFetch tool). "
            "If you have WebFetch, fetch the URL yourself, synthesize the content, and call medulla_ingest instead — "
            "that keeps the synthesis in your context and uses the model you are already running."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to fetch and ingest"},
                "title": {"type": "string", "description": "Optional title override"},
            },
            "required": ["url"],
        },
    ),
    types.Tool(
        name="medulla_wiki_schema",
        description=(
            "Get all existing wiki page slugs and titles. "
            "CALL THIS FIRST before calling medulla_ingest for any source. "
            "Use the returned slugs to write accurate [[wikilinks]] — only link to slugs that exist "
            "or that you are about to create in this ingest session. "
            "Linking to non-existent slugs fragments the Obsidian graph."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
    types.Tool(
        name="medulla_list_raw",
        description=(
            "List files in wiki/raw/ that haven't been processed yet. "
            "Use this to see what's available for ingestion — files dropped by Obsidian Clipper, "
            "PDFs placed manually, or URL text fetched by medulla. "
            "You can then read a file from raw/ and call medulla_ingest with your synthesis."
        ),
        inputSchema={"type": "object", "properties": {}},
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
async def list_tools() -> list[types.Tool]:  # pragma: no cover
    return _TOOLS


@_server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.TextContent]:  # pragma: no cover
    text = _dispatch(name, arguments)
    return [types.TextContent(type="text", text=text)]


# ── Tool dispatch ──────────────────────────────────────────────────────────────

_HANDLERS: dict[str, Any] = {
    "medulla_search": lambda conn, args: _tool_search(conn, args),
    "medulla_session_detail": lambda conn, args: _tool_session_detail(conn, args),
    "medulla_session_tree": lambda conn, args: _tool_session_tree(conn, args),
    "medulla_project_context": lambda conn, args: _tool_project_context(conn, args),
    "medulla_list": lambda conn, args: _tool_list(conn, args),
    "medulla_stats": lambda conn, args: _tool_stats(conn),
    "medulla_events_search": lambda conn, args: _tool_events_search(conn, args),
    "medulla_wiki_schema": lambda conn, args: _tool_wiki_schema(conn, args),
    "medulla_wiki_search": lambda conn, args: _tool_wiki_search(conn, args),
    "medulla_wiki_page": lambda conn, args: _tool_wiki_page(conn, args),
    "medulla_ingest": lambda conn, args: _tool_ingest(conn, args),
    "medulla_ingest_url": lambda conn, args: _tool_ingest_url(conn, args),
    "medulla_list_raw": lambda conn, args: _tool_list_raw(conn, args),
    "medulla_analyze": lambda conn, args: _tool_analyze(conn, args),
}


def _dispatch(name: str, args: dict) -> str:
    conn = connect()
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    return handler(conn, args)


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
    # Resolve 8-char prefix to full UUID (Claude passes short IDs from medulla_list)
    if len(session_id) < 36:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id LIKE ?",
            (f"{session_id}%",)
        ).fetchone()
        if not row:
            return f"Session not found: {session_id}"
        session_id = row["session_id"]
    detail = get_session_detail(conn, session_id)
    if not detail:
        return f"Session not found: {session_id}"

    s = detail["session"]
    total_chunks = len(detail["chunks"])

    # Specific chunk requested — return it in full
    chunk_index = args.get("chunk_index")
    if chunk_index is not None:
        matches = [c for c in detail["chunks"] if c["chunk_index"] == chunk_index]
        if not matches:
            return f"Chunk {chunk_index} not found. Session has {total_chunks} chunks (0–{total_chunks - 1})."
        c = matches[0]
        lines = [
            f"Session {s['session_id'][:8]} — Chunk {chunk_index} of {total_chunks - 1} "
            f"(turns {c['turn_start']}–{c['turn_end']})",
            "",
            c["chunk_text"],  # full text, no truncation
            "",
            f"[Next: chunk_index={chunk_index + 1}]" if chunk_index + 1 < total_chunks else "[End of session]",
        ]
        return "\n".join(lines)

    # Overview: metadata + first 5 chunks
    lines = []
    lines.append(f"Session: {s['session_id']}")
    lines.append(f"Project: {s.get('project_dir', '')}")
    lines.append(f"Date:    {(s.get('started_at') or '')[:10]} → {(s.get('ended_at') or '')[:10]}")
    lines.append(f"Turns:   {s.get('turn_count', 0)}   Tool calls: {s.get('tool_call_count', 0)}")
    lines.append(f"Chunks:  {total_chunks} total (use chunk_index=N to fetch any chunk in full)")
    lines.append("")
    if detail["agents"]:
        lines.append(f"Subagents ({len(detail['agents'])}):")
        for a in detail["agents"]:
            lines.append(f"  {a['agent_id'][:8]}  {a.get('first_message', '')[:60]}")
        lines.append("")
    lines.append("First 5 chunks (call with chunk_index=N for any specific chunk):")
    for c in detail["chunks"][:5]:
        lines.append(f"\n── Chunk {c['chunk_index']} (turns {c['turn_start']}–{c['turn_end']}) ──")
        lines.append(c["chunk_text"][:1500])
    if total_chunks > 5:
        lines.append(f"\n... {total_chunks - 5} more chunks. Use chunk_index=5 through chunk_index={total_chunks - 1} to read them.")
    return "\n".join(lines)


def _tool_session_tree(conn, args: dict) -> str:
    session_id = args.get("session_id", "").strip()
    if not session_id:
        return "Error: session_id is required"
    if len(session_id) < 36:
        row = conn.execute(
            "SELECT session_id FROM sessions WHERE session_id LIKE ?",
            (f"{session_id}%",)
        ).fetchone()
        if not row:
            return f"Session not found: {session_id}"
        session_id = row["session_id"]
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
    from medulla.semantic.store import get_wiki_stats
    s = get_stats(conn)
    ws = get_wiki_stats(conn)
    lines = [
        "Episodic:",
        f"  Sessions:       {s['sessions']:,}",
        f"  Chunks:         {s['chunks']:,}",
        f"  Agent sessions: {s['agent_sessions']:,}",
        f"  Turns:          {s['turns']:,}",
        f"  Tool calls:     {s['tool_calls']:,}",
    ]
    if s["oldest"]:
        lines.append(f"  Date range:     {s['oldest'][:10]} → {s['newest'][:10]}")
    lines.append(f"\nSemantic (wiki):")
    lines.append(f"  Pages:          {ws['total']:,}")
    _PLURAL = {"source": "sources", "concept": "concepts", "entity": "entities"}
    for pt, count in ws.get("by_type", {}).items():
        lines.append(f"  {_PLURAL.get(pt, pt + 's'):<15} {count:,}")
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


def _tool_wiki_schema(conn, args: dict) -> str:
    from medulla.config import get_config
    from medulla.semantic.ingest import _build_wiki_schema
    wiki_path = get_config().wiki_path
    schema = _build_wiki_schema(wiki_path)
    return f"Current wiki pages (use these exact slugs for [[wikilinks]]):\n\n{schema}"


def _tool_wiki_search(conn, args: dict) -> str:
    from medulla.semantic.store import search_wiki
    from medulla.search import _snippet
    query = args.get("query", "").strip()
    if not query:
        return "Error: query is required"
    page_type = args.get("type")
    rows = search_wiki(conn, query, page_type=page_type, limit=args.get("limit", 10))
    if not rows:
        return f"No wiki pages found for: {query}"
    lines = [f"{len(rows)} wiki page(s) for \"{query}\":\n"]
    for row in rows:
        lines.append(f"[{row['type']}] {row['slug']} — {row['title']}")
        lines.append(f"  {_snippet(row['content'], 150)}\n")
    return "\n".join(lines)


def _tool_wiki_page(conn, args: dict) -> str:
    from medulla.semantic.store import get_wiki_page
    slug = args.get("slug", "").strip()
    if not slug:
        return "Error: slug is required"
    row = get_wiki_page(conn, slug)
    if not row:
        return f"Wiki page not found: {slug}"
    return f"# {row['title']} [{row['type']}]\n\n{row['content']}"


def _tool_ingest(conn, args: dict) -> str:
    """Pure storage — Claude has already synthesized the content."""
    title = args.get("title", "").strip()
    content = args.get("content", "").strip()
    if not title or not content:
        return "Error: title and content are required"
    try:
        from medulla.semantic.ingest import store_wiki_page
        from medulla.config import get_config
        wiki_path = get_config().wiki_path
        result = store_wiki_page(
            conn, wiki_path, title, content,
            page_type=args.get("page_type", "source"),
            tags=args.get("tags", []),
            source_url=args.get("source_url"),
            source_path=args.get("source_path"),
        )
        extras = []
        if args.get("source_path"):
            extras.append("PDF copied to raw/")
        if args.get("source_url"):
            extras.append("URL logged to url-references.md")
        note = f" ({', '.join(extras)})" if extras else ""
        return f"Stored: {result['slug']} ({result['type']}){note}"
    except Exception as e:
        return f"Store failed: {e}"


def _tool_ingest_url(conn, args: dict) -> str:
    """Fetch URL → raw/ → LLM → wiki pages. For clients without WebFetch."""
    url = args.get("url", "").strip()
    if not url:
        return "Error: url is required"
    try:
        from medulla.llm import get_provider
        from medulla.semantic.ingest import ingest_url_mcp
        from medulla.config import get_config
        provider = get_provider()
        wiki_path = get_config().wiki_path
        result = ingest_url_mcp(conn, url, wiki_path, provider, title=args.get("title"))
        return (
            f"Ingested: {result.get('source', '?')} ({result.get('total_pages', 0)} pages)\n"
            f"Raw text saved to wiki/raw/ for backtrace."
        )
    except Exception as e:
        return f"Ingest URL failed: {e}"


def _tool_list_raw(conn, args: dict) -> str:
    """List unprocessed files in wiki/raw/."""
    from medulla.config import get_config
    from medulla.semantic.store import get_pending
    wiki_path = get_config().wiki_path
    raw_dir = wiki_path / "raw"
    if not raw_dir.exists():
        return "wiki/raw/ is empty — no files to ingest yet."
    skip = {"url-references.md"}
    all_files = [f for f in sorted(raw_dir.iterdir()) if f.is_file() and f.name not in skip]
    if not all_files:
        return "wiki/raw/ is empty — no files to ingest yet."
    pending = {row["source_path"] for row in get_pending(conn)}
    lines = [f"{len(all_files)} file(s) in wiki/raw/:\n"]
    for f in all_files:
        status = "⏳ queued" if str(f) in pending else "✓ processed"
        lines.append(f"  {status}  {f.name}")
    lines.append("\nTo ingest a file: read it and call medulla_ingest with your synthesis.")
    lines.append("Or call medulla_ingest_url(url) for URL sources if you lack WebFetch.")
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

def serve() -> None:  # pragma: no cover
    anyio.run(_serve)


async def _serve() -> None:  # pragma: no cover
    async with stdio_server() as (read_stream, write_stream):
        await _server.run(
            read_stream,
            write_stream,
            _server.create_initialization_options(),
        )
