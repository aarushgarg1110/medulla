# Sprint 2 Reflection — MCP Server + Topic-Shift Chunking

**Date:** 2026-05-20  
**Issue:** #1 — MCP server + topic-shift chunking  
**PR:** #8 (merged via squash to main)  
**Commit:** 54850eb  
**Tests:** 213 passed · 95.78% statement · 93% statement+branch  

---

## What was shipped

### MCP stdio server — 11 tools
Full JSON-RPC 2.0 MCP server (`medulla mcp`) compatible with Claude Code and Kiro. All 11 tools working end-to-end:

- `medulla_search` — FTS5 search with BM25 ranking across all chunks
- `medulla_session_detail` — full session with chunk paging (chunk_index param)
- `medulla_session_tree` — parent + all child subagents
- `medulla_project_context` — recent sessions + tool events per project
- `medulla_list` — session listing with filter
- `medulla_stats` — aggregate counts
- `medulla_events_search` — tool-call event FTS
- `medulla_analyze` — manifest quality (needs PostToolUse hook)
- `medulla_wiki_search/page/ingest` — stubs for Sprint 3

### Topic-shift chunker
Replaced fixed 20-turn windows with Jaccard vocabulary-divergence detection. Sessions with clear topic changes now produce separate chunks per topic. Effect: 89 fixed chunks → **2,681 topic-coherent chunks** across 22 sessions. Dramatically improves search precision.

### CLI additions
- `medulla session-detail <id>` — full session detail with 8-char prefix lookup, shows chunks and subagents
- `medulla mcp` — now actually starts the server (was a stub in Sprint 1)

---

## Bugs caught during live testing (before merge)

### Bug 1 — 8-char prefix not resolved in MCP tools
`medulla_list` returns 8-char IDs (e.g. `dc6ae946`). Claude passed these directly to `medulla_session_detail` which did exact UUID match — returning "not found" for every call. Fixed: both `medulla_session_detail` and `medulla_session_tree` now resolve short prefixes via LIKE query, same as the CLI.

**Lesson:** Test with real Claude invocations, not just unit tests. The real client passes short IDs; unit tests used full UUIDs.

### Bug 2 — Chunks truncated to 600 chars with no paging
Claude was hitting truncated chunk text and falling back to raw filesystem search (`fd -e jsonl ...`) to find the continuation. This defeats the entire purpose of medulla.

Fixed: `medulla_session_detail` now has two modes:
- **Overview** (no chunk_index): metadata + first 5 chunks at 1500 chars + total chunk count
- **Paging** (chunk_index=N): returns chunk N in full (no truncation) + "Next: chunk_index=N+1" hint

### Bug 3 — Kiro MCP config had args as single string
`"args": ["--project /path run medulla mcp"]` passes everything as one argument to `uv`. Fixed to `["--project", "/path", "run", "medulla", "mcp"]` (separate array elements).

---

## Manual verification

### Protocol (stdio pipe)
```bash
# initialize → responds with protocol version
echo '{"jsonrpc":"2.0","id":1,"method":"initialize",...}' | uv run medulla mcp
→ {"result": {"protocolVersion": "2024-11-05", "serverInfo": {"name": "medulla"}}}

# tools/list → 11 tools returned
# tools/call medulla_stats → real DB data
# tools/call medulla_search "logD outlier CompoundX" → finds correct session chunks
```

### Claude Code (`~/code/personal/medulla`)
- `/mcp` → medulla: ✓ Connected
- "What sessions do I have indexed?" → called medulla_list + medulla_stats, returned full table of 22 sessions
- "What are suspect compounds from logD/pKa sessions?" → called medulla_search 11 times, surfaced correct internal compound analysis with NDI identifiers
- Session detail paging working after bug fix

### Kiro
- Configured in `~/.kiro/settings/mcp.json` with correct split args
- `/mcp list` → medulla ✓ Connected
- Tools work identically to Claude Code

---

## Architecture note — dispatch pattern
`match/case` statement coverage differs between Python 3.12.3 (Linux CI) and 3.12.13 (macOS). Replaced with dict dispatch (`_HANDLERS`) which coverage.py tracks consistently across versions. Added `# pragma: no cover` to async MCP wire-protocol handlers (`list_tools`, `call_tool`, `serve`, `_serve`) — these are integration-tested via stdio pipe, not unit-testable.

---

## What's next

Sprint 3 (Issue #2): Semantic ingest — PDF, URL, markdown → LLM wiki pages → indexed in wiki_fts. The three wiki stubs (`medulla_wiki_search`, `medulla_wiki_page`, `medulla_ingest`) become real tools.
