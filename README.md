# Medulla

> Three-layer memory for Claude Code and Kiro — episodic, semantic, codebase.

Medulla gives your AI assistant persistent memory across sessions. It indexes everything you've done, everything you've learned, and (soon) everything in your codebase — then exposes it through a CLI and an MCP server that any MCP-compatible client can query mid-conversation.

## The problem

Claude Code starts blank every time. [kcp-memory](https://github.com/Cantara/kcp-memory) helps, but caps indexed session text at 8KB — any topic discussed after the first ~15 messages is invisible to search. Medulla removes that cap entirely, chunks sessions into topic-coherent slices, and adds a semantic wiki layer for ingested documents.

## Three layers

| Layer | What it is | Source |
|---|---|---|
| **Episodic** | Past Claude/Kiro session history | `~/.claude/projects/**/*.jsonl` |
| **Semantic** | Curated knowledge wiki (Karpathy LLM-wiki style) | PDFs, URLs, markdown, Canvas |
| **Codebase** *(v2)* | Repo intelligence via tree-sitter AST | Git repos |

All layers share one SQLite DB (`~/.medulla/medulla.db`) with FTS5 full-text search and (soon) sqlite-vec embeddings.

## Install

```bash
uv tool install git+https://github.com/aarushgarg1110/medulla
```

Or clone and run locally:

```bash
git clone https://github.com/aarushgarg1110/medulla
cd medulla
uv run medulla --help
```

## Usage

```bash
# Episodic — index your Claude sessions
medulla scan                          # index new/changed sessions
medulla scan --force                  # re-index all
medulla search "logD outliers"        # search across all layers
medulla list                          # list recent sessions
medulla stats                         # aggregate stats

# Semantic (v1.x) — ingest knowledge
medulla ingest paper.pdf
medulla ingest https://some-blog.com/post
medulla ingest --canvas               # pull all Canvas LMS courses
medulla wiki lint                     # check for orphaned pages

# MCP server — works with Claude Code and Kiro
claude mcp add medulla uv -- run --project /path/to/medulla medulla mcp
```

## Development

```bash
uv sync --dev
uv run pytest tests/ --cov=medulla    # 96%+ coverage required
```

### Workflow

- `main` is always green (CI enforces passing tests)
- Features and sprints tracked as GitHub Issues
- Work on `feat/<name>` branches, open PRs against `main`
- PRs must pass `pytest` before merge

## Roadmap

See [GitHub Issues](https://github.com/aarushgarg1110/medulla/issues) for the full sprint backlog.

- **Sprint 2** — MCP server (11 tools) + smarter topic-shift chunking
- **Sprint 3** — Semantic ingest: PDF, URL, markdown → LLM wiki pages
- **Sprint 4** — Embeddings (pluggable, evaluated) + Canvas API + `medulla update`
- **Sprint 5** — Org sync: public/private session tagging, S3 push, shared wiki
- **Sprint 6** — Codebase layer: tree-sitter, blast radius, git hooks

## License

MIT
