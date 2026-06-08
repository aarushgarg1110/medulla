# Feat: sqlite-vec Integration — Embedding Layer Foundation

**Branch:** `feat/sqlite-vec-embeddings`  
**Issues:** #18 + #19 (combined)  
**Date:** 2026-06-08  
**Tests:** 474 passing · 96% statement coverage

---

## What Was Built

Sprint 4 foundation: vector storage for all session chunks and wiki pages,
`medulla embed` CLI command, and auto-embedding on scan and ingest.

### New dependencies
- `sqlite-vec` — SQLite extension providing `vec_distance_cosine()` for cosine similarity
- `sentence-transformers` — embedding model library; model lazy-loaded on first `embed()` call

### DB migration V4 (`medulla/db/migrations/V4__embeddings.sql`)
Two regular SQLite tables with BLOB columns:
- `vec_chunks(session_id, chunk_index, embedding BLOB)` — one row per session chunk
- `vec_wiki(slug, embedding BLOB)` — one row per wiki page

Used regular tables + `vec_distance_cosine()` (exact cosine) rather than vec0 virtual
tables — simpler, no rowid-lookup complexity, fast enough at medulla's scale.

### `medulla/embeddings.py`
- `EmbeddingProvider` Protocol — `embed(texts) -> list[list[float]]`, `dimension`, `model_name`
- `SentenceTransformersProvider` — default local model (`intfloat/e5-base-v2`, 768-dim).
  Lazy-loaded: model downloads to `~/.cache/huggingface/hub/` on first `embed()` call only.
- `get_embedding_provider()` — returns configured provider

### `medulla/db/embedding_store.py`
- `upsert_chunk_embedding` / `get_chunk_embedding` / `get_chunks_without_embeddings`
- `upsert_wiki_embedding` / `get_wiki_embedding` / `get_wiki_pages_without_embeddings`
- `find_similar_chunks` / `find_similar_wiki_pages` — cosine similarity via `vec_distance_cosine()`

### Auto-embedding on scan and ingest
- `scanner._embed_session_chunks()` — called after `upsert_session`
- `ingest._embed_new_wiki_pages()` — called after `process_pending`
- Both silently skip on failure — embedding failures never break indexing or ingest

### `medulla embed` CLI command
- Backfills embeddings for all chunks and wiki pages without embeddings
- `--force` clears and re-embeds everything
- Shows counts and download hint on first run

---

## Bugs Fixed During Live Testing

### SSL: corporate CA bundle not reaching huggingface_hub
`huggingface_hub` creates `httpx.Client` without a `verify=` argument, bypassing
`SSL_CERT_FILE`. Fixed by monkey-patching `httpx.Client.__init__` to inject
`verify=ssl_cert` before importing sentence_transformers.

### Python 3.14 rejects corporate CA certs at ssl.c level
Python 3.14 added stricter `Basic Constraints` enforcement that rejects certain
corporate CA certs regardless of the cert bundle passed. `requires-python = ">=3.12,<3.14"`
was already in pyproject.toml but uv was ignoring it for tool installs.

Fixed by uv 0.11.19 which added inference of Python version from `requires-python`
in source trees for `uv tool install` — no `--python 3.12` flag needed anymore.
Verified: `uv tool install .` now installs into Python 3.12 automatically.

### Model loaded once per session, not once per scan
`_get_embedding_provider()` created a new `SentenceTransformersProvider()` instance
on every call → model weights reloaded from disk for every session scanned.
Fixed by making each module's `_get_embedding_provider()` a module-level singleton.
Before: "Loading weights" bar appeared N times during scan. After: once.

### Test suite slow after adding auto-embedding
`scanner._embed_session_chunks()` and `ingest._embed_new_wiki_pages()` were loading
the real sentence-transformers model in tests (~4-8s per test file on first load).
Fixed by adding `autouse` fixtures in test_scanner.py, test_ingest_pipeline.py,
test_cli.py, test_mcp_tools.py, and test_coverage_gaps_sprint3.py that patch
`_get_embedding_provider` to return a `MockEmbeddingProvider`. Full suite now runs
in ~40s (no model loading in tests).

---

## Live Demo

After scanning 44 sessions (3537 chunks embedded), semantic search works now:

```
Query: "logD outlier batch effects chemistry"
#1 [dc6ae946] dist=0.1550 — logD deep dive, NDI batch outlier analysis
#2 [bff7439b] dist=0.1569 — same batch, 4σ outlier discussion
#3 [bff7439b] dist=0.1624 — insoluble series / salacia outliers
```

No keyword overlap required. Purely meaning-based. This is what #20 hybrid search
plugs into `medulla search` automatically.

---

## What's Next

- `#20` — hybrid search: BM25 + cosine rerank with RRF (uses `find_similar_chunks`)
- `#21` — cosine-similarity automatic wikilink edges (uses `find_similar_wiki_pages`)
- `#19` remaining — config.toml provider switching, Bedrock embeddings, `medulla use-embeddings`
