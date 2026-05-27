# Sprint 3 Reflection — Semantic Ingest Pipeline

**Branch:** `feat/sprint-3-semantic-ingest`  
**PR:** #10  
**Date:** 2026-05-27  
**Tests:** 395 passing · 95% statement+branch coverage

---

## What We Built

Sprint 3 delivered the semantic layer — the LLM wiki half of Medulla. A source (PDF, URL, markdown, Perplexity export) goes in; source/concept/entity wiki pages come out in `~/.medulla/wiki/`, Obsidian-compatible and FTS5-indexed.

### Core pipeline
- `medulla ingest <source>` — PDF (PyMuPDF), URL (httpx + trafilatura), markdown
- `intake_to_raw()` → queued in `pending_ingests` → `process_pending()` → LLM → wiki pages
- `discover_raw()` — picks up files dropped in `raw/` by Obsidian Clipper automatically
- `medulla ingest` with no args — discover + process all pending

### LLM provider abstraction
- Bedrock (cross-region inference profile), Anthropic API, Ollama — all in `config.toml`
- `medulla use bedrock|anthropic|ollama` switches provider; `medulla status` verifies
- Pluggable `LLMProvider` protocol — add new providers without touching ingest logic

### MCP tools added
- `medulla_ingest` — pure storage: Claude synthesizes, tool stores (no second LLM call)
- `medulla_ingest_url` — for MCP clients without WebFetch
- `medulla_wiki_search`, `medulla_wiki_page` — query the semantic layer
- `medulla_wiki_schema` (14th tool) — returns all existing slugs for accurate wikilinks

### CLI additions
- `medulla wiki list`, `medulla wiki lint`, `medulla wiki open`
- `medulla status` — provider + model + pending count + wiki stats

---

## Bugs Fixed Mid-Sprint

### `entitys/` folder (pluralization bug)
`f"{page_type}s"` on `"entity"` → `"entitys"`. Fixed with `_DIR = {"entity": "entities", ...}` dict throughout cli.py, mcp.py, ingest.py, wiki.py.

### `software-20` vs `software-2-0` (slugify version numbers)
`slugify()` stripped `.` from "Software 2.0" → `software-20`. Fixed: `re.sub(r"(\d)\.(\d)", r"\1-\2", s)` before stripping punctuation.

### Ghost wikilinks (LLM inventing non-existent slugs)
LLM was creating `[[multi-head-attenshun]]`-style invented slugs. Fixed with schema injection: `_build_wiki_schema()` loads all existing page slugs and injects them into every prompt. LLM can only link to slugs it can see.

### `entitys/` path in DB (existing pages had wrong `file_path`)
After the folder rename, DB rows still pointed to `entitys/`. Fixed: migrated file_path values in DB + removed stale `entitys/` directory.

### URL markdown files in `raw/` (policy violation)
`intake_to_raw()` was writing `.md` files to `raw/` for URLs. Policy: `raw/` is for binary files only. URLs go to `url-references.md` log. Fixed: URL intake writes temp file for processing only, appends to `url-references.md`.

### Duplicate source pages (`software-20-andrej-karpathy` + `software-20-andrej-karpathy-2017`)
Caused by the slugify bug creating two different slugs for the same source on successive ingests. Fixed by the version number regex; old duplicates manually cleaned from DB and filesystem.

### Index.md pipe-in-table Obsidian rendering bug
`[[sources/slug|slug]]` inside a Markdown table — Obsidian treats `|` as column separator, rendering the link as literal text with broken display. Fixed: use `[[sources/slug]]` (no alias) in table cells.

### Source slug from programmatic extraction vs LLM title
For Perplexity exports (`title:` is a truncated user question, H1 is `# You`), programmatic extraction returned `"You"` → slug `you`. Fixed: slug is now derived from `source_data["title"]` **after** the plan stage returns, using the LLM's synthesized title.

### Markdown `_extract_title` lost its Perplexity skip logic
At some point `_extract_title` was simplified to a single H1 regex, losing the `_SKIP` set (`{"you", "assistant", ...}`) and frontmatter truncation detection. Restored: skips speaker-label H1s, skips truncated frontmatter titles (ending in `...`).

### Bedrock streaming 4096 token limit (dense PDFs truncated)
`invoke_model_with_response_stream` on cross-region inference profiles caps output at 4096 tokens. Dense PDFs (Attention Is All You Need) hit this mid-JSON → parse error or missing concepts.

**Fix: multi-call pipeline**
- Stage 1 (plan): source page + list of concepts/entities to create (~800 tokens)
- Stage 2 (per-concept): one focused call per concept (~600 tokens each)
- Stage 3 (per-entity): one focused call per entity (~400 tokens each)
- Stage 4 (updates): no LLM call — `_add_source_to_page()` just edits frontmatter

Each call fits in the streaming cap. CLI shows checkpoint output after each page.

### Concept UPDATE pathway (Nimbus-Brain parity, Sprint 3.5)
Ingesting paper #2 needed to add itself to existing concept pages' `sources:` list. Was noted as a gap. Implemented: `update_concepts`/`update_entities` in the plan JSON; `_add_source_to_page()` parses frontmatter, appends source slug, rewrites file without touching content.

### Tag vocabulary control (Nimbus-Brain parity, Sprint 3.5)
LLM was inventing new tags on every ingest (e.g., `ml-research`, `machine-learning-research`, `ml`). Fixed: `TAG_VOCABULARY` list of ~45 canonical tags injected into every prompt. LLM instructed to reuse before inventing.

---

## Architecture Decisions

**Multi-call vs single-call:** Staying with streaming (not switching to non-streaming) preserves the live-output UX. Multi-call gives checkpoints per page which is better than raw JSON stream anyway.

**Slug from LLM title:** The LLM always produces better source titles than programmatic extraction for messy real-world files. Cost: one extra `slugify()` call after the plan stage, and test MockProvider needs stage-detection logic.

**`raw/` policy:** Binary files only in `raw/`. URLs in `url-references.md`. Keeps `raw/` as a clean, importable archive — not a collection of near-empty stubs.

**MCP = pure storage:** When called from Claude Code, Claude IS the LLM. `medulla_ingest` stores Claude's synthesis directly. `medulla_ingest_url` is only for clients without WebFetch. This avoids paying twice for synthesis.

---

## What's Next (Open Issues)

| Issue | Sprint | Status |
|---|---|---|
| #3 | Sprint 4 | Embeddings (sqlite-vec), Canvas ingestion, `medulla update` |
| #9 | Sprint 3.5 | Additional LLM providers: Gemini, OpenAI, Vertex |
| #11 | — | Fuzzy/typo-tolerant search |
| #12 | — | Session parser registry (route by source directory) |
| #4 | Sprint 5 | Org sync |
| #5 | Sprint 6 | Codebase layer (tree-sitter) |

Sprint 4 (embeddings) is the natural next step — it adds vector re-ranking on top of the FTS5 that exists now.
