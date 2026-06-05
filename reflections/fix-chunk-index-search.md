# Fix: chunk_index in medulla_search results

**Branch:** `fix/chunk-index-search`  
**Issue:** #16  
**Date:** 2026-06-05  
**Tests:** 431 passing · 96% statement coverage

---

## What Was Wrong

`SearchResult` stored `chunk_index` embedded in the `title` string:
```
title = "Session dc6ae946 chunk 140"
```

When MCP clients called `medulla_search`, they received a text blob with the chunk number
buried in a human-readable label. The model had to parse the string to extract `140`, and
often didn't — calling `medulla_session_detail` without `chunk_index`, receiving chunks 0–4,
then paging forward one by one. Result: 20 tool calls to reach what should take 2.

## What Changed

**`medulla/search.py`**
- Added `chunk_index: int | None = None` as a dedicated field on `SearchResult`
- `_search_chunks()` now sets `chunk_index=row["chunk_index"]` from the DB row
- Session-level and wiki results leave `chunk_index=None`
- Title cleaned up to just `"Session {id[:8]}"` — no embedded number

**`medulla/mcp.py`**
- `_tool_search()` now emits a direct navigation hint for chunk results:
  ```
  → medulla_session_detail(session_id="dc6ae946", chunk_index=140)
  ```
  Models read this and jump straight to the right chunk. Wiki and session results omit the hint.

**`medulla/cli.py`**
- CLI search output now shows `chunk N` inline for chunk results:
  ```
  dc6ae946  2026-05-19  medulla  chunk 140
    Multi-call pipeline decision…
  ```
  Users know exactly which `--chunk` flag to pass to `medulla session-detail` without guessing.

## Tests Added

`tests/test_search.py`:
- `test_chunk_result_has_chunk_index` — chunk results have integer chunk_index
- `test_session_result_has_no_chunk_index` — session results have None
- `test_wiki_result_has_no_chunk_index` — wiki results have None
- `test_chunk_index_is_not_always_zero` — chunk_index reflects actual matched chunk
- `test_chunk_title_no_longer_embeds_chunk_index` — title is clean

`tests/test_mcp_tools.py`:
- `test_tool_search_chunk_result_includes_chunk_index_hint` — hint present for chunks
- `test_tool_search_wiki_result_no_chunk_index_hint` — hint absent for wiki results

`tests/test_cli.py`:
- `test_search_command_shows_chunk_index_for_chunk_results` — CLI shows `chunk N` label inline
