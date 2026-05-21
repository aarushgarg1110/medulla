# Medulla CLI — All Commands & Variations

Reference for the sanity check script. Commands marked ✅ are covered in
`cli-sanity-check.sh`. Commands marked ❌ are verified manually but not yet
in the script. Commands marked ⚠️ have known issues.

---

## medulla scan

```bash
medulla scan                    # ✅ incremental — skips unchanged (mtime)
medulla scan --force            # ✅ re-indexes everything regardless of mtime
medulla scan --source claude    # ✅ Claude sessions only
medulla scan --source kiro      # ⚠️  finds Kiro files but all empty/stub
                                #     Kiro uses different JSONL format (kind:Prompt/Completion)
                                #     not Claude format — needs Kiro parser (Sprint 2 backlog)
```

**Output terminology:**
- `N indexed` — new or changed files successfully parsed and stored
- `N unchanged` — skipped because file mtime ≤ last scanned_at
- `N empty/stub` — file found but parse_session() returned None
  (empty files, no user messages, or unsupported format like Kiro)

---

## medulla search

```bash
medulla search "query"                    # ✅ all layers (episodic + semantic)
medulla search "query" --layer episodic   # ✅ session chunks only
medulla search "query" --layer semantic   # ✅ wiki pages only
medulla search "query" --layer code       # (Sprint 6 — codebase layer, not yet)
medulla search "query" --limit 5          # ✅ verified manually, limit results
```

---

## medulla list

```bash
medulla list                        # ❌ not in sanity script — works, shows recent sessions
medulla list --limit 5              # ❌ not in sanity script
medulla list --project mlops        # ❌ not in sanity script — filters by project name substring
```

---

## medulla stats

```bash
medulla stats                       # ✅ shows Episodic + Semantic sections
```

---

## medulla session-detail

```bash
medulla session-detail <session-id>   # ❌ not in sanity script — works
medulla session-detail bff7439b       # ❌ 8-char prefix lookup — works
# Shows: metadata, turns, tool calls, subagents (17 for bff7439b), chunks
```

---

## medulla use

```bash
medulla use bedrock     # ✅ switches active provider, persists to config.toml
medulla use anthropic   # ❌ not in sanity (needs ANTHROPIC_API_KEY)
medulla use ollama      # ✅ in sanity script
```

---

## medulla status

```bash
medulla status          # ✅ shows: provider, model, profile, wiki pages,
                        #          raw/ files, queued count, sessions indexed
```

---

## medulla ingest

```bash
medulla ingest https://url          # ✅ fetches → raw/<slug>.md → LLM → wiki pages
medulla ingest path/to/file.pdf     # ✅ copies PDF to raw/ → LLM → wiki pages
medulla ingest path/to/notes.md     # tested via Obsidian Clipper simulation
medulla ingest                      # ✅ (no args) discover new raw/ files + process all queued
medulla ingest https://url --title "Custom Title"  # ❌ not in sanity — works (slug uses title)
medulla ingest https://url --scope org             # ❌ not tested (org sync Sprint 5)
```

**Flow:** all paths go through `raw/` first. `pending_ingests` tracks state.
- `queued` → file in raw/, not yet processed
- `done` → processed into wiki pages
- `error` → failed (re-queued automatically on next ingest)

---

## medulla wiki list

```bash
medulla wiki list                   # ✅ all page types, most recent first
medulla wiki list --type source     # ❌ not in sanity — works when wiki has pages
medulla wiki list --type concept    # ❌ not in sanity — works when wiki has pages
medulla wiki list --type entity     # ❌ not in sanity — works when wiki has pages
medulla wiki list --limit 10        # ❌ not in sanity
```

---

## medulla wiki lint

```bash
medulla wiki lint       # ✅ structural check: suggested pages, orphaned pages
                        # "Suggested pages" = [[wikilinks]] to pages not yet created (normal)
                        # "Orphaned pages" = pages with no inbound links (worth investigating)
                        # Source pages are excluded from orphan check (they're leaf nodes)
```

---

## medulla reset

```bash
medulla reset --yes           # ✅ clears wiki pages, index, log, pending queue. Keeps raw/
medulla reset --all --yes     # ✅ also clears raw/, sessions, agents, tool events
# Without --yes: prompts for confirmation
```

---

## medulla mcp

```bash
medulla mcp             # ✅ starts MCP stdio server (protocol verified manually)
# Register with Claude Code:
#   claude mcp add medulla medulla -- mcp
# Register with Kiro (in ~/.kiro/settings/mcp.json):
#   "command": "medulla", "args": ["mcp"]
# MCP protocol version: 2024-11-05
# Tools exposed: 12 (medulla_search, session_detail, session_tree,
#   project_context, list, stats, events_search, wiki_search, wiki_page,
#   ingest, ingest_url, list_raw, analyze)
```

---

## Known gaps / future work

| Gap | Sprint |
|---|---|
| Kiro JSONL format parser | Sprint 2 backlog |
| `medulla list` + `session-detail` in sanity check | Next update |
| `medulla scan --force` in sanity check | Next update |
| `medulla wiki list --type` in sanity check | Next update |
| `medulla mcp` protocol test in sanity check | Next update |
| `medulla use anthropic` test (needs key) | When key available |
| `--layer code` search (codebase layer) | Sprint 6 |
| `--scope org` ingest (org sync) | Sprint 5 |
