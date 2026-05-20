# Sprint 1 Checkpoint Reflection

**Date:** 2026-05-20  
**Commit:** `8fc2609` — feat: Sprint 1 — episodic foundation  
**Branch:** main  

---

## What we built

Full episodic memory foundation — a working drop-in replacement for kcp-memory, solving the 8KB cap problem that made buried session content unsearchable.

### Modules shipped

| Module | Purpose |
|---|---|
| `db/migrations/V1__episodic.sql` | Schema: sessions, session_chunks, agent_sessions, tool_events + FTS5 virtual tables + sync triggers |
| `db/database.py` | SQLite connection (WAL mode), versioned migration runner |
| `config.py` | Paths: `~/.medulla/medulla.db`, `~/.medulla/wiki/` |
| `episodic/parser.py` | Claude JSONL parsing — full text extraction, tool/file detection, subagent detection |
| `episodic/chunker.py` | Fixed 20-turn window chunking — each chunk gets its own FTS5 row |
| `episodic/store.py` | Session/chunk/agent upserts, list, stats queries |
| `episodic/scanner.py` | File discovery, mtime-based incremental scan, `--force` re-index |
| `search.py` | Unified FTS5 BM25 search — chunk-preferred results, per-session dedup, matched excerpt |
| `cli.py` | `scan`, `search`, `list`, `stats`, `mcp` (stub) |

### Test suite
- **132 tests, 96% line coverage, 0 failures**
- No mocks for DB (real SQLite in tmp_path), no mocks for file I/O (real JSONL fixtures)
- Covers: parser edge cases (bad JSON, empty files, subagent detection, content as list), chunker boundaries, store upsert/update/conflict, scanner incremental logic, search dedup and FTS error handling

### Algorithms used
- **FTS5 + BM25** (SQLite built-in) — inverted index, BM25 ranking. No ML, no models, pure keyword.
- **Fixed-window chunking** (20 turns per chunk) — each slice gets its own FTS row so buried content is findable.
- **mtime-based incremental scan** — skip files whose modification time ≤ last indexed timestamp.

---

## CLI manual verification (real data, 2026-05-20)

```
medulla scan --force
→ Sessions: 22 indexed, 5 skipped, 0 errors | Agents: 49 indexed, 0 skipped

medulla stats
→ 22 sessions, 89 chunks, 49 agents, 9,028 turns, 4,749 tool calls
→ Date range: 2026-04-13 → 2026-05-20

medulla search "logD outliers CompoundX NDI"
→ bff7439b (2026-04-13, mlops) — chunk from deep in 46MB session, invisible to kcp-memory
  Excerpt: "did we join the ndi? i want project by project insights on what got better or worse"

medulla search "pKa basic acidic site selection"
→ bff7439b — different chunk, pKa probability output debugging turn
  Excerpt: "(3.253, 0.9704), (1.58, 0.9629), it has a lower prob..."

medulla search "MMP contrastive learning"
→ 3 results across 3 different sessions (brain, LogD-Model-Build, bff7439b) — cross-session search working
```

---

## Gaps identified

### 1. User-only indexing (critical — addressed in Sprint 1.5)

**What happened:** `medulla search "CHEMBL12345 batch effect"` returned no results. The detailed analysis — "CHEMBL12345 has Δ = +6.11, 4σ above batch mean, almost certainly a measurement error" — was in *Claude's response*, not in the user's message. The user said things like "look at that CompoundX outlier", which doesn't contain "CHEMBL12345".

**Root cause:** Parser only indexes `role == "user"` turns. Assistant turns are silently skipped.

**Why kcp-memory did this:** Unclear. Possibly to limit index size, possibly assuming user messages carry intent. But the most valuable content in a Claude session — analysis, code, findings, explanations — lives in assistant turns.

**Fix:** Sprint 1.5 — index both user AND assistant `type: "text"` content, interleaved in conversation order. No opt-in flag needed; always-on makes more sense since assistant messages are the primary value.

### 2. Table column truncation in `medulla list`

Rich auto-fits columns to terminal width, so session IDs show as `b…` and dates as `202…`. Minor UX issue — either fix column widths or switch to a plain list format in Sprint 2.

### 3. Topic-shift chunking is naive

Fixed 20-turn windows don't respect topic boundaries. If pKa discussion spans turns 15-35, it straddles two chunks, degrading recall for pKa queries. Sprint 2 will add vocabulary-divergence detection to cut at actual topic shifts.

### 4. `tool_events` table is empty

The table exists and is indexed, but population requires a PostToolUse hook wired into `~/.claude/settings.json`. Not yet configured. Low priority — events search is a Sprint 2 MCP tool feature.

---

## Design decisions made

| Decision | Choice | Rationale |
|---|---|---|
| Embedding model | Deferred to Sprint 4 | MCP server (Sprint 2) delivers more immediate value; embeddings improve search quality, not access |
| Embedding eval framework | MLflow + NDCG@5/MRR/Recall@10 | Objective model comparison across e5-base-v2, nomic, MedCPT, BioBERT, BGE, Gemma-300M |
| Wiki location | `~/.medulla/wiki/` | Global, single Obsidian vault |
| Assistant message indexing | Always-on (Sprint 1.5) | Most valuable content is in assistant turns |
| DB schema | SQLite + FTS5, no vector columns yet | Add sqlite-vec in Sprint 4 via migration V3 |

---

## What's next

- **Sprint 1.5:** Index assistant messages + GitHub Actions CI
- **Sprint 2:** MCP server (11 tools) + topic-shift chunker
- **Sprint 3:** Semantic ingest (PDF, URL, markdown → LLM wiki pages)
- **Sprint 4:** Embeddings (pluggable, MLflow eval) + Canvas + `medulla update`
