# Medulla

> Three-layer memory for Claude Code and Kiro — episodic, semantic, codebase.

The medulla oblongata is the relay between brain and body — routing signals between thinking and doing. Medulla the tool does the same: it routes everything your AI assistant has seen, learned, and built into a persistent, searchable memory layer that travels with every new session.

---

## The problem with starting blank

Claude Code starts blank every time. You re-explain your codebase, re-paste context, re-summarize what you covered last week. The model has no memory of the session where you debugged that pipeline, no recall of the paper you ingested, no awareness that this function was refactored three sprints ago.

Existing solutions are partial:

- **kcp-memory** indexes sessions but caps content at 8KB — anything discussed after the first ~15 messages is invisible to search
- **RAG / vector DBs** retrieve documents but rediscover knowledge from scratch on every query — nothing accumulates, no synthesis happens
- **Neo4j / graph DBs** are powerful but require manual schema design, deployment infrastructure, and don't integrate natively with the LLM wiki pattern or MCP

Medulla takes a different approach: **compile knowledge once, search it forever**. Sessions are chunked and indexed with no cap. Documents are synthesized into a curated wiki of source/concept/entity pages that compound over time. Everything lives in a single local SQLite file — no infrastructure, no server, no setup beyond `uv tool install`.

---

## Three layers

| Layer | What it is | Source |
|---|---|---|
| **Episodic** | Every Claude + Kiro session, chunked by topic, FTS5-indexed | `~/.claude/projects/**/*.jsonl`, `~/.kiro/sessions/` |
| **Semantic** | LLM-synthesized wiki — source, concept, entity pages, Obsidian-compatible | PDFs, URLs, markdown, Perplexity exports |
| **Codebase** *(v2)* | Repo intelligence via tree-sitter AST, blast radius on commit | Git repos |

All three layers share one SQLite DB (`~/.medulla/medulla.db`) with FTS5 full-text search. One MCP server exposes tools across all active layers.

---

## Prerequisites

- **Python 3.12** and **[uv](https://docs.astral.sh/uv/getting-started/installation/)** (required)
- **[Obsidian](https://obsidian.md/)** (optional — recommended for the semantic wiki layer)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Obsidian (macOS)
brew install --cask obsidian
```

---

## Install

**Via uv (recommended):**
```bash
uv tool install git+https://github.com/aarushgarg1110/medulla
medulla --help
```

**From source:**
```bash
git clone https://github.com/aarushgarg1110/medulla
cd medulla
uv run medulla --help
```

**Add to Claude Code:**
```bash
claude mcp add medulla medulla -- mcp
```

**Add to Kiro** — add to your MCP config:
```json
{
  "mcpServers": {
    "medulla": {
      "command": "medulla",
      "args": ["mcp"]
    }
  }
}
```

---

## Configuration

### LLM Provider

```bash
medulla use bedrock       # AWS Bedrock (default, cross-region inference profile)
medulla use anthropic     # Anthropic API (requires ANTHROPIC_API_KEY env var)
medulla use ollama        # Local Ollama (shows available downloaded models)

medulla use bedrock --model us.anthropic.claude-sonnet-4-6   # override model
medulla status            # verify active provider + connectivity
```

### Corporate SSL (Zscaler etc.)

If your organization uses a custom CA bundle, add to `~/.zshenv` (not `.zshrc` — needs to apply to all subprocesses):

```bash
export SSL_CERT_FILE=/path/to/your/ca-bundle.pem
export REQUESTS_CA_BUNDLE=/path/to/your/ca-bundle.pem
```

### Custom data directory

```bash
export MEDULLA_DIR=~/.medulla-dev   # point at a dev/test directory
```

---

## CLI Commands

### Episodic

| Command | Description |
|---|---|
| `medulla scan` | Index new/changed Claude + Kiro sessions (incremental by mtime) |
| `medulla scan --force` | Re-index all sessions |
| `medulla scan --source claude\|kiro` | Scan one source only |
| `medulla search "<query>"` | FTS5 search across session chunks + wiki pages |
| `medulla search "<query>" --layer episodic\|semantic` | Layer-specific search |
| `medulla list` | List recent sessions |
| `medulla list --project <path>` | Filter by project directory |
| `medulla session-detail <id>` | Full session transcript with chunks (8-char prefix OK) |
| `medulla stats` | Episodic + semantic stats, top tools, pending queue |

### Semantic (Wiki)

| Command | Description |
|---|---|
| `medulla ingest <file\|url>` | Ingest source → LLM generates wiki pages |
| `medulla ingest` | Discover raw/ + process all queued sources |
| `medulla ingest --force` | Re-ingest even if previously processed |
| `medulla ingest --streaming` | Show raw tokens as they arrive (caps at 4096 — for debugging) |
| `medulla wiki list` | Table of all wiki pages |
| `medulla wiki lint` | Check for broken wikilinks + orphaned pages |
| `medulla wiki open` | Open wiki vault in Obsidian (first time: Obsidian prompts "Open as vault?" — click yes) |

### Config & Utility

| Command | Description |
|---|---|
| `medulla use bedrock\|anthropic\|ollama` | Switch active LLM provider |
| `medulla use <provider> --model <name>` | Switch provider and override model |
| `medulla status` | Provider, model, pending queue, session counts |
| `medulla config` | Show current config path and values |
| `medulla reset` | Wipe DB + wiki pages (preserves raw/) |
| `medulla reset --all` | Wipe everything including raw/ and episodic sessions |

---

## MCP Tools (14)

| Tool | Description |
|---|---|
| `medulla_search` | FTS5 search across session chunks + wiki pages |
| `medulla_session_detail` | Full session transcript + chunks (accepts 8-char prefix) |
| `medulla_session_tree` | Parent + all subagent sessions linked together |
| `medulla_project_context` | Recent sessions + tool events for a project directory |
| `medulla_list` | Recent sessions with metadata |
| `medulla_stats` | Aggregate stats across all layers |
| `medulla_events_search` | Search tool-call events by tool name or output preview |
| `medulla_wiki_search` | FTS5 search within wiki pages only |
| `medulla_wiki_page` | Full content of a wiki page by slug |
| `medulla_wiki_schema` | All existing page slugs — **call before `medulla_ingest`** |
| `medulla_ingest` | Store a wiki page you synthesized (pure storage — you are the LLM) |
| `medulla_ingest_url` | Fetch + synthesize a URL (for clients without WebFetch) |
| `medulla_list_raw` | Raw/ files and pending queue status |
| `medulla_analyze` | Session quality metrics — retry rates, error rates per tool |

---

## Tutorial

### CLI: ingest a paper, search from the terminal

```bash
# 1. Scan your Claude + Kiro sessions
medulla scan
# ✓ Sessions: 31 indexed, 0 unchanged, 3 empty/stub, 0 errors
# ✓ Agents:   43 indexed, 0 unchanged

# 2. Set your provider
medulla use bedrock
medulla status

# 3. Ingest a PDF
medulla ingest ~/Downloads/Adam-Optimizer.pdf
```

```
→ raw/Adam-Optimizer.pdf
Processing 1 queued file(s)...

  ✓ Source page ready — plan: 11 concepts, 6 entities
    ✓ adam-optimizer (1/11)
    ✓ adaptive-learning-rate (2/11)
    ✓ bias-correction-moments (3/11)
    ✓ adamax (4/11)
    ✓ rmsprop (5/11)
    ✓ adagrad (6/11)
    ✓ online-convex-optimization (7/11)
    ✓ regret-bound (8/11)
    ✓ stochastic-gradient-descent (9/11)
    ✓ temporal-averaging (10/11)
    ✓ variational-autoencoder (11/11)
    ✓ diederik-kingma (1/6)
    ✓ jimmy-lei-ba (2/6)
    ✓ iclr-2015 (3/6)
    ✓ mnist-dataset (4/6)
    ✓ cifar-10-dataset (5/6)
    ✓ imdb-dataset (6/6)

✓ Adam-Optimizer.pdf → 18 pages (adam-optimizer, adaptive-learning-rate, ...)
```

```bash
# 4. Ingest a second source — shared concepts are updated automatically
medulla ingest https://karpathy.github.io/2026/02/12/microgpt/
# ↻ Updated existing: adam-optimizer, bias-correction-moments, rmsprop

# 5. Same source again — deduplication prevents re-ingestion
medulla ingest ~/Downloads/Adam-Optimizer.pdf
# → Nothing queued to process.

# 6. Search
medulla search "bias correction moments"

# 7. Open wiki in Obsidian
medulla wiki open
```

---

### MCP: ingest via Claude Code

In a Claude Code or Kiro session, ask your assistant to ingest a source. The recommended workflow:

1. Call `medulla_wiki_schema` to get all existing page slugs
2. Fetch the source (via `WebFetch` or `Read`)
3. Call `medulla_ingest` once per page — source, then each concept, then each entity

**Example prompt to Claude:**
> Use the medulla MCP to ingest https://karpathy.github.io/2026/02/12/microgpt/

**What happens:**

```
→ Called medulla_wiki_schema
  [[concepts/adam-optimizer]], [[entities/diederik-kingma]], ...

→ Fetched https://karpathy.github.io/2026/02/12/microgpt/ (93.8KB)

→ Called medulla_ingest  (source page)
  Stored: microgpt-karpathy-2026 (source)

→ Called medulla_ingest  (concept: transformer-architecture)
  Stored: transformer-architecture (concept)

→ Called medulla_ingest  (concept: autograd-engine)
  Stored: autograd-engine (concept)

... (one call per concept and entity)

→ All 8 pages ingested. Graph fully connected.
```

**Then search within the same session:**
> What does Karpathy say about KV cache in microGPT?

Claude calls `medulla_wiki_search` and answers from the synthesized wiki — no re-reading the full 93KB post.

---

## Tech Stack

| Component | Choice | Why |
|---|---|---|
| Language | Python 3.12 | ML ecosystem, uv tooling |
| Package manager | uv | Fast, lockfile-based, `uv tool install` for CLI distribution |
| CLI | Typer + Rich | Clean API, auto-generates `--help`, beautiful output |
| DB | SQLite (stdlib) | Zero infra, single portable file |
| Full-text search | FTS5 (built into SQLite) | BM25 ranking, no dependencies |
| Vector search | sqlite-vec *(Sprint 4)* | Embedded in SQLite, no separate server |
| MCP server | `mcp` Python SDK | Official Anthropic SDK, stdio transport |
| PDF parsing | PyMuPDF (`fitz`) | Fast, accurate text + metadata extraction |
| URL extraction | httpx + trafilatura | Clean article text from any web page |
| Code parsing | tree-sitter *(Sprint 6)* | 12+ languages, AST extraction |

---

## Roadmap

| Sprint | Feature |
|---|---|
| **4** | Embeddings (sqlite-vec, pluggable models, NDCG eval), Canvas LMS ingestion, `medulla update` |
| **5** | Org sync — tag sessions public/private, push to shared S3, pull coworkers' curated summaries |
| **6** | Codebase layer — tree-sitter AST, function/class graph, blast radius on commit |

Smaller improvements in [Issues](https://github.com/aarushgarg1110/medulla/issues): parallel ingest calls (#13), `medulla remove` (#14), fuzzy search (#11), additional LLM providers (#9).

---

## Development

```bash
git clone https://github.com/aarushgarg1110/medulla
cd medulla
uv sync --dev
uv run pytest tests/ --cov=medulla   # 96%+ statement coverage required
```

- `main` is always green — CI enforces passing tests and ≥95% statement coverage
- Features tracked as GitHub Issues, worked on `feat/<name>` branches
- PRs merged to `main` after CI passes and manual approval

---

## Acknowledgements

Medulla builds on ideas from:

- **[Andrej Karpathy](https://x.com/karpathy/status/2039805659525644595)** — the LLM Wiki pattern: source, concept, and entity pages maintained by an LLM, accumulating knowledge over time into an Obsidian-compatible vault
- **[kcp-memory](https://github.com/Cantara/kcp-memory)** (Cantara) — the episodic session indexing architecture: SQLite schema, JSONL parsing, FTS5 search, incremental mtime-based scanning
- **[OpenKB](https://github.com/VectifyAI/OpenKB)** (VectifyAI) — the planning step pattern, schema injection for accurate wikilinks, and ghost wikilink detection
- **[Tirth Kanani](https://tirthkanani18.medium.com/i-built-a-knowledge-graph-that-cuts-claude-codes-token-usage-by-49x-ca73ef078981)** — the codebase intelligence graph design: AST parsing pipeline, blast radius, call graph edges (inspiration for Sprint 6)

---

## License

MIT
