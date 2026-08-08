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
from mcp.server import Server, ServerRequestContext
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
        description="Search past Claude/Kiro sessions, wiki pages, and harvested tool/command history (SQL/duckdb/bash you've run). Uses hybrid BM25+vector search when embeddings exist, falls back to keyword search otherwise.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms"},
                "limit": {"type": "integer", "default": 10},
                "layer": {"type": "string", "enum": ["episodic", "semantic", "code", "events"], "description": "Restrict to a single layer (omit for all). 'events' = harvested tool/command history only."},
                "bm25_only": {"type": "boolean", "default": False, "description": "Force keyword-only search, skip vector reranking"},
            },
            "required": ["query"],
        },
    ),
    types.Tool(
        name="medulla_session_detail",
        description=(
            "Retrieve content of a specific session. "
            "Without chunk args: returns session metadata + first 3 chunks. "
            "With chunk_index: returns that one chunk in full. "
            "With chunk_start/chunk_end: returns that whole range in one call (preferred for "
            "reading a topic that spans several chunks — do this instead of paging one chunk at a time)."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Session ID (full UUID or 8-char prefix)"},
                "chunk_index": {"type": "integer", "description": "Single chunk to fetch (0-based). Omit for overview or use chunk_start/chunk_end for a range."},
                "chunk_start": {"type": "integer", "description": "First chunk of a range (0-based, inclusive). Defaults to 0 if only chunk_end is given."},
                "chunk_end": {"type": "integer", "description": "Last chunk of a range (0-based, inclusive). Defaults to the final chunk if only chunk_start is given. Clamped to the session length."},
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
            "1. page_type='source': ONE call for the full source summary (must complete before concepts)\n"
            "2. page_type='concept': issue ALL concept calls IN PARALLEL in a single response — they are independent of each other. DO NOT call them one at a time.\n"
            "3. page_type='entity': issue ALL entity calls IN PARALLEL in a single response — after all concepts complete. DO NOT call them one at a time.\n\n"
            "PARALLELISM IS REQUIRED: concepts do not depend on each other, entities do not depend on each other. "
            "Issuing them sequentially one-per-turn is unnecessary and slow. "
            "Batch all concept tool calls together, wait for all to complete, then batch all entity tool calls together.\n\n"
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
            "Be generous with tags — they power the Obsidian graph filter. "
            "No limit on concepts or entities — create as many as are genuinely meaningful. "
            "ALWAYS pass source_path when you read a local file (PDF, markdown) — the file gets copied to wiki/raw/ as an immutable archive. "
            "ALWAYS pass source_url when you fetched via WebFetch — the URL gets logged to url-references.md. "
            "Both can be provided for a PDF downloaded from a URL.\n\n"
            "WIKILINK PATH CONVENTION — this is critical for Obsidian graph correctness:\n"
            "- Concepts: [[concepts/slug]] e.g. [[concepts/adam-optimizer]]\n"
            "- Entities: [[entities/slug]] e.g. [[entities/andrej-karpathy]]\n"
            "- Sources: [[sources/slug]] e.g. [[sources/microgpt-karpathy-2026]]\n"
            "NEVER use bare [[slug]] — always include the folder prefix. "
            "medulla_wiki_schema returns slugs in this exact format — copy them verbatim. "
            "When the schema is empty (first ingest), plan ALL slugs for this session upfront, "
            "then write every wikilink as [[concepts/slug]] or [[entities/slug]] consistently throughout.\n\n"
            "SLUG CONSISTENCY — wikilinks must resolve to real pages.\n"
            "The stored slug is: slug param if provided, else slugify(title).\n"
            "Your wikilinks MUST match the stored slug exactly.\n"
            "BEST PRACTICE: always pass slug= explicitly to decouple title from slug:\n"
            "  title='MA-RAE: Macro-Averaged Relative Absolute Error', slug='ma-rae'\n"
            "  → stored as concepts/ma-rae, wikilink [[concepts/ma-rae]] resolves correctly\n"
            "WITHOUT slug param: slugify(title) must equal your wikilink slug:\n"
            "  title='Adam Optimizer' → slug='adam-optimizer' → [[concepts/adam-optimizer]] ✓\n"
            "  title='MA-RAE: Macro-Averaged...' → slug='ma-rae-macro-averaged-...' → [[concepts/ma-rae]] ✗\n"
            "WORKFLOW: plan your slugs first, then pass slug= on every concept/entity call."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Page title — can be as descriptive as needed. Use slug param to control the wikilink slug explicitly."},
                "slug": {"type": "string", "description": "Explicit slug override (lowercase-hyphenated, e.g. 'ma-rae'). Use this to decouple the wikilink slug from the title. If omitted, slugify(title) is used. ALWAYS provide slug when your wikilinks use a short slug but the title is long/descriptive."},
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
            "Linking to non-existent slugs fragments the Obsidian graph.\n\n"
            "WIKILINK FORMAT: slugs are returned as [[concepts/slug]], [[entities/slug]], [[sources/slug]]. "
            "Always include the folder prefix in wikilinks — never use bare [[slug]]. "
            "When this returns empty (fresh wiki), you must still use the folder-prefixed format "
            "for all pages you create: [[concepts/your-slug]], [[entities/your-slug]], [[sources/your-slug]]."
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
    types.Tool(
        name="medulla_remove",
        description=(
            "Remove a wiki page or raw file. "
            "target format: 'sources/slug', 'concepts/slug', 'entities/slug', or 'raw/filename'. "
            "cascade=true also deletes concepts/entities that become orphaned after removing a source."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "sources/slug, concepts/slug, entities/slug, or raw/filename"},
                "cascade": {"type": "boolean", "default": False},
            },
            "required": ["target"],
        },
    ),
    types.Tool(
        name="medulla_reindex_edges",
        description=(
            "Recompute cosine-similarity related: wikilinks for all embedded wiki pages. "
            "Call this after a batch of medulla_ingest calls to wire up semantic connections "
            "between all newly created pages. Returns count of pages updated."
        ),
        inputSchema={"type": "object", "properties": {}},
    ),
]


async def list_tools(
    ctx: ServerRequestContext[Any],
    params: types.PaginatedRequestParams | None = None,
) -> types.ListToolsResult:
    return types.ListToolsResult(tools=_TOOLS)


async def call_tool(
    ctx: ServerRequestContext[Any],
    params: types.CallToolRequestParams,
) -> types.CallToolResult:
    # A raised exception would become a JSON-RPC *protocol* error, which Claude
    # surfaces as a dead tool rather than a readable message. Keep every failure
    # inside a normal result with is_error, so one bad tool never looks like a
    # broken server.
    try:
        text = _dispatch(params.name, params.arguments or {})
        is_error = False
    except Exception as e:
        text = f"{params.name} failed: {type(e).__name__}: {e}"
        is_error = True
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=text)],
        is_error=is_error,
    )


# SDK 2.0 removed the @server.list_tools()/@server.call_tool() decorators in
# favour of explicit registration; handlers now take (ctx, params) and return a
# full Result object instead of a bare list.
_server.add_request_handler("tools/list", types.PaginatedRequestParams, list_tools)
_server.add_request_handler("tools/call", types.CallToolRequestParams, call_tool)


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
    "medulla_reindex_edges": lambda conn, args: _tool_reindex_edges(conn, args),
    "medulla_remove": lambda conn, args: _tool_remove(conn, args),
}


# One SQLite connection is reused for the life of the stdio server. Opening a
# fresh connect() per tool call re-loaded the sqlite-vec extension and re-scanned
# the migrations dir every time; the stdio server is single-threaded (anyio) so a
# shared connection is safe and lets the page cache stay warm across calls.
_conn = None


def _get_conn():
    global _conn
    if _conn is None:
        _conn = connect()
    return _conn


def _dispatch(name: str, args: dict) -> str:
    handler = _HANDLERS.get(name)
    if handler is None:
        return f"Unknown tool: {name}"   # no connection needed for an unknown tool
    return handler(_get_conn(), args)


def _tool_search(conn, args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: query is required"
    results = fts_search(
        conn, query,
        limit=args.get("limit", 10),
        layer=args.get("layer"),
        bm25_only=args.get("bm25_only", False),
    )
    if not results:
        return f"No results found for: {query}"

    lines = [f"{len(results)} result(s) for \"{query}\":\n"]
    for r in results:
        date = (r.date or "")[:10]
        proj = (r.project_dir or "").split("/")[-1]
        if r.result_type == "tool_event":
            sid = r.id.split("#")[0][:8]     # id is "{session_id}#evt{rowid}"
            err = " ✗" if r.is_error else ""
            lines.append(f"[{sid}]{err}  {date}  command  {proj}")
            lines.append(f"  $ {_command_preview(r.excerpt)[:100]}\n")
            continue
        lines.append(f"[{r.id[:8]}]  {date}  {r.layer}  {proj}")
        if r.chunk_index is not None:
            lines.append(f"  → medulla_session_detail(session_id=\"{r.id[:8]}\", chunk_index={r.chunk_index})")
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

    # Range requested — return chunk_start..chunk_end concatenated in one call.
    chunk_start = args.get("chunk_start")
    chunk_end = args.get("chunk_end")
    if chunk_start is not None or chunk_end is not None:
        start = 0 if chunk_start is None else max(0, chunk_start)
        end = total_chunks - 1 if chunk_end is None else min(total_chunks - 1, chunk_end)
        if start > end:
            return (f"Empty range: chunk_start={chunk_start}, chunk_end={chunk_end}. "
                    f"Session has {total_chunks} chunks (0–{total_chunks - 1}).")
        selected = [c for c in detail["chunks"] if start <= c["chunk_index"] <= end]
        lines = [
            f"Session {s['session_id'][:8]} — chunks {start}–{end} of {total_chunks - 1}",
            "",
        ]
        for c in selected:
            lines.append(f"── Chunk {c['chunk_index']} (turns {c['turn_start']}–{c['turn_end']}) ──")
            lines.append(c["chunk_text"])
            lines.append("")
        lines.append(
            f"[Next: chunk_start={end + 1}]" if end + 1 < total_chunks else "[End of session]"
        )
        return "\n".join(lines)

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

    # Overview: metadata + first 3 chunks
    lines = []
    lines.append(f"Session: {s['session_id']}")
    lines.append(f"Project: {s.get('project_dir', '')}")
    lines.append(f"Date:    {(s.get('started_at') or '')[:10]} → {(s.get('ended_at') or '')[:10]}")
    lines.append(f"Turns:   {s.get('turn_count', 0)}   Tool calls: {s.get('tool_call_count', 0)}")
    lines.append(f"Chunks:  {total_chunks} total (chunk_index=N for one, or chunk_start/chunk_end for a range)")
    lines.append("")
    if detail["agents"]:
        lines.append(f"Subagents ({len(detail['agents'])}):")
        for a in detail["agents"]:
            lines.append(f"  {a['agent_id'][:8]}  {a.get('first_message', '')[:60]}")
        lines.append("")
    preview = min(3, total_chunks)
    lines.append(f"First {preview} chunk(s) (use chunk_start/chunk_end to read a range in one call):")
    for c in detail["chunks"][:preview]:
        lines.append(f"\n── Chunk {c['chunk_index']} (turns {c['turn_start']}–{c['turn_end']}) ──")
        lines.append(c["chunk_text"][:1500])
    if total_chunks > preview:
        lines.append(f"\n... {total_chunks - preview} more chunks. "
                     f"Use chunk_start={preview}, chunk_end={total_chunks - 1} to read the rest.")
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


_TRIVIAL_LINE_RE = __import__("re").compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")
_RUNNERS = {"uv", "run", "python", "python3", "bash", "sh", "time", "sudo", "env",
            "poetry", "npx", "node"}
_ABORT_FOLLOWUP_WINDOW_S = 300   # a fix-after-abort follows quickly


def _command_family(command: str) -> str:
    """Deterministic 'family' key: the invoked script/binary after stripping runner
    prefixes (uv run, python, …). '' if none. Used only to gate adjacency display —
    never to assert intent."""
    toks = [t for t in _command_preview(command).split() if t]
    i = 0
    while i < len(toks) and toks[i].lower() in _RUNNERS:
        i += 1
    return toks[i].lower() if i < len(toks) else ""


def _within_seconds(ts_a: str, ts_b: str, secs: int) -> bool:
    from datetime import datetime
    try:
        a = datetime.fromisoformat((ts_a or "").replace("Z", "+00:00"))
        b = datetime.fromisoformat((ts_b or "").replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return False
    return 0 <= (b - a).total_seconds() <= secs


_HEREDOC_OPEN_RE = __import__("re").compile(r"<<-?\s*['\"]?[A-Za-z_]\w*['\"]?\s*$")


def _command_preview(command: str) -> str:
    """First substantive line of a (possibly multi-line) command — skip blanks,
    comments (# and --), leading `cd`, bare VAR=value setup, and heredoc openers
    (so `duckdb << 'EOF'` shows the actual SQL line, not the opener)."""
    for ln in (command or "").splitlines():
        s = ln.strip()
        if not s or s.startswith("#") or s.startswith("--"):
            continue
        if s.startswith("cd ") and "&&" not in s:
            continue
        if _TRIVIAL_LINE_RE.match(s) or _HEREDOC_OPEN_RE.search(s):
            continue
        return s
    return (command or "").strip().splitlines()[0] if (command or "").strip() else ""


def _tool_events_search(conn, args: dict) -> str:
    query = args.get("query", "").strip()
    if not query:
        return "Error: query is required"
    rows = search_events(conn, query, limit=args.get("limit", 20))
    if not rows:
        return f"No tool events found for: {query}"
    from medulla.episodic.store import get_next_command
    lines = [f"{len(rows)} event(s):\n"]
    for r in rows:
        err = " ✗" if r["is_error"] else ""
        lines.append(f"  {(r['event_ts'] or '')[:16]}{err}  {r['tool']}  {_command_preview(r['command'])[:80]}")
        if r["output_preview"]:
            lines.append(f"    → {r['output_preview'][:60]}")
        # Failure → what ran next (context, not an asserted fix).
        if r["is_error"]:
            nxt = get_next_command(conn, r["session_id"], r["event_ts"], limit=1)
            if nxt:
                lines.append(f"    ↳ next in session: {_command_preview(nxt[0]['command'])[:70]}")
        # Abort → what ran INSTEAD, but only when the next command is the SAME family
        # within a short window (deterministic gates). Adjacency, not a fix claim.
        elif r["interrupted"]:
            nxt = get_next_command(conn, r["session_id"], r["event_ts"], limit=1)
            if (nxt and _command_family(r["command"])
                    and _command_family(nxt[0]["command"]) == _command_family(r["command"])
                    and _within_seconds(r["event_ts"], nxt[0]["event_ts"], _ABORT_FOLLOWUP_WINDOW_S)):
                lines.append(f"    ↳ ran instead: {_command_preview(nxt[0]['command'])[:70]}")
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
            slug=args.get("slug") or None,
        )
        extras = []
        if args.get("source_path"):
            extras.append("PDF copied to raw/")
        if args.get("source_url"):
            extras.append("URL logged to url-references.md")
        note = f" ({', '.join(extras)})" if extras else ""
        msg = f"Stored: {result['slug']} ({result['type']}){note}"
        broken = result.get("broken_wikilinks", [])
        if broken:
            msg += "\n⚠ Broken wikilinks in this page (these pages don't exist yet):\n"
            msg += "\n".join(f"  {b}" for b in broken)
            msg += "\nCreate the missing pages or fix the wikilinks before finishing."
        return msg
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


def _tool_remove(conn, args: dict) -> str:
    target = args.get("target", "").strip()
    if not target:
        return "Error: target is required (e.g. 'concepts/adam-optimizer')"
    from medulla.config import get_config
    from medulla.semantic.remove import plan_remove, execute_remove
    wiki_path = get_config().wiki_path
    plan = plan_remove(conn, target, wiki_path=wiki_path)
    if "error" in plan:
        return plan["error"]
    result = execute_remove(conn, target, wiki_path=wiki_path, cascade=args.get("cascade", False))
    removed = result.get("removed", [])
    cleaned = result.get("cleaned", [])
    return (f"Removed {len(removed)} file(s), cleaned references in {len(cleaned)} page(s). "
            f"Removed: {', '.join(removed[:3])}{'...' if len(removed) > 3 else ''}")


def _tool_reindex_edges(conn, args: dict) -> str:
    from medulla.config import get_config
    from medulla.semantic.wiki import reindex_wiki_edges
    wiki_path = get_config().wiki_path
    updated = reindex_wiki_edges(conn, top_k=5)
    return f"Reindexed related: edges for {updated} wiki pages."


# ── Entry point ────────────────────────────────────────────────────────────────

def _prewarm_embeddings() -> "Any":
    """Load the search embedding model in a background daemon thread.

    The e5 model takes ~11s to load and is loaded lazily on the first search,
    so without this the first medulla_search of a session blocks for ~11s. By
    kicking the load off at server startup it is usually resident by the time
    the first query arrives; a query that races the load just blocks on the
    same lock (no worse than today). Returns the thread (for tests). Never
    raises — embeddings are optional.
    """
    import threading
    import time

    def _load() -> None:
        start = time.monotonic()
        try:
            from medulla.search import _get_search_embedding_provider
            _get_search_embedding_provider().embed(["warmup"])
            # stderr only — stdout is the MCP protocol channel. Shows in server logs.
            print(f"medulla: embedding model warmed in {time.monotonic() - start:.1f}s",
                  file=sys.stderr, flush=True)
        except Exception as e:  # embeddings unavailable → search falls back to BM25
            print(f"medulla: embedding pre-warm skipped ({type(e).__name__}) — search uses BM25",
                  file=sys.stderr, flush=True)

    t = threading.Thread(target=_load, name="medulla-embed-prewarm", daemon=True)
    t.start()
    return t


def serve() -> None:  # pragma: no cover
    anyio.run(_serve)


async def _serve() -> None:  # pragma: no cover
    _prewarm_embeddings()
    async with stdio_server() as (read_stream, write_stream):
        await _server.run(
            read_stream,
            write_stream,
            _server.create_initialization_options(),
        )
