"""Unified FTS5 + hybrid (BM25 + cosine) search across all layers."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any


@dataclass
class SearchResult:
    layer: str          # episodic | semantic | codebase
    result_type: str    # session | chunk | wiki_page
    id: str             # session_id or wiki slug
    title: str
    excerpt: str        # matched text snippet
    project_dir: str | None
    date: str | None
    rank: float
    chunk_index: int | None = None  # set for result_type="chunk"; use with medulla_session_detail


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    layer: str | None = None,
    bm25_only: bool = False,
) -> list[SearchResult]:  # noqa: C901
    """Search FTS5 indexes, upgrading to hybrid (BM25+cosine+RRF) when embeddings exist.

    Pass bm25_only=True to force pure keyword search (e.g. for debugging).
    """
    if not query.strip():
        return []

    if not bm25_only:
        try:
            provider = _get_search_embedding_provider()
            # Only use hybrid if at least one embedding exists
            has_embeddings = conn.execute(
                "SELECT 1 FROM vec_chunks LIMIT 1"
            ).fetchone()
            if has_embeddings:
                return hybrid_search(conn, query, limit=limit, layer=layer, provider=provider)
        except Exception:
            pass  # fall through to BM25

    fts_query = _to_fts_query(query)
    results: list[SearchResult] = []

    if layer is None or layer == "episodic":
        results.extend(_search_chunks(conn, fts_query, limit))
        results.extend(_search_sessions(conn, fts_query, limit))

    if layer is None or layer == "semantic":
        results.extend(_search_wiki(conn, fts_query, limit))

    # Deduplicate: one result per session_id / wiki slug, prefer best-ranked chunk over session-level match
    seen: dict[str, SearchResult] = {}
    for r in sorted(results, key=lambda x: x.rank):
        if r.id not in seen:
            seen[r.id] = r
        elif r.result_type == "chunk" and seen[r.id].result_type == "session":
            # Upgrade session-level match to a more specific chunk match
            seen[r.id] = r

    return sorted(seen.values(), key=lambda x: x.rank)[:limit]


def _search_chunks(conn: sqlite3.Connection, fts_query: str, limit: int) -> list[SearchResult]:
    try:
        rows = conn.execute("""
            SELECT
                sc.session_id,
                sc.chunk_index,
                snippet(session_chunks_fts, 2, '', '', '…', 24) AS snip,
                s.project_dir,
                s.started_at,
                scf.rank
            FROM session_chunks_fts scf
            JOIN session_chunks sc ON sc.rowid = scf.rowid
            JOIN sessions s ON s.session_id = sc.session_id
            WHERE session_chunks_fts MATCH ?
            ORDER BY scf.rank
            LIMIT ?
        """, (fts_query, limit)).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        excerpt = row["snip"] or ""
        results.append(SearchResult(
            layer="episodic",
            result_type="chunk",
            id=row["session_id"],
            title=f"Session {row['session_id'][:8]}",
            excerpt=excerpt,
            project_dir=row["project_dir"],
            date=row["started_at"],
            rank=row["rank"],
            chunk_index=row["chunk_index"],
        ))
    return results


def _search_sessions(conn: sqlite3.Connection, fts_query: str, limit: int) -> list[SearchResult]:
    """Fallback: search session-level FTS for sessions not caught by chunks."""
    try:
        rows = conn.execute("""
            SELECT
                s.session_id,
                s.first_message,
                snippet(sessions_fts, 2, '', '', '…', 24) AS snip,
                s.project_dir,
                s.started_at,
                sf.rank
            FROM sessions_fts sf
            JOIN sessions s ON s.rowid = sf.rowid
            WHERE sessions_fts MATCH ?
            ORDER BY sf.rank
            LIMIT ?
        """, (fts_query, limit)).fetchall()
    except sqlite3.OperationalError:
        return []

    results = []
    for row in rows:
        # snippet() over all_user_text centers on the match; fall back to first_message.
        excerpt = row["snip"] or _snippet(row["first_message"] or "", 200)
        results.append(SearchResult(
            layer="episodic",
            result_type="session",
            id=row["session_id"],
            title=f"Session {row['session_id'][:8]}",
            excerpt=excerpt,
            project_dir=row["project_dir"],
            date=row["started_at"],
            rank=row["rank"],
        ))
    return results


def _search_wiki(conn: sqlite3.Connection, fts_query: str, limit: int) -> list[SearchResult]:
    try:
        rows = conn.execute("""
            SELECT wp.slug, wp.type, wp.title,
                   snippet(wiki_fts, 3, '', '', '…', 24) AS snip,
                   wf.rank
            FROM wiki_fts wf
            JOIN wiki_pages wp ON wp.rowid = wf.rowid
            WHERE wiki_fts MATCH ?
            ORDER BY wf.rank LIMIT ?
        """, (fts_query, limit)).fetchall()
    except sqlite3.OperationalError:
        return []
    results = []
    for row in rows:
        results.append(SearchResult(
            layer="semantic",
            result_type="wiki_page",
            id=row["slug"],
            title=f"[{row['type']}] {row['title']}",
            excerpt=row["snip"] or "",
            project_dir=None,
            date=None,
            rank=row["rank"],
        ))
    return results


# ── hybrid search ─────────────────────────────────────────────────────────────

# Module-level singleton so the embedding model loads once per process.
# Guarded by a lock so a background pre-warm thread and the first real search
# can't both construct/load the model concurrently.
import threading as _threading
_search_embedding_provider = None
_search_provider_lock = _threading.Lock()


def _get_search_embedding_provider():
    global _search_embedding_provider
    if _search_embedding_provider is None:
        with _search_provider_lock:
            if _search_embedding_provider is None:
                from medulla.embeddings import get_embedding_provider
                _search_embedding_provider = get_embedding_provider()
    return _search_embedding_provider


def _rrf_score(rank_a: int | None, rank_b: int | None, k: int = 60) -> float:
    """Reciprocal Rank Fusion score across two ranked lists (1-based ranks)."""
    score = 0.0
    if rank_a is not None:
        score += 1.0 / (k + rank_a)
    if rank_b is not None:
        score += 1.0 / (k + rank_b)
    return score


def _rrf_fuse(
    bm25_ids: list[tuple[str, int]],   # [(id, 0-based rank), ...]
    vec_ids: list[tuple[str, int]],
) -> list[tuple[str, float]]:
    """Merge two ranked lists with RRF. Returns [(id, score)] sorted descending."""
    bm25_rank = {id_: rank + 1 for id_, rank in bm25_ids}
    vec_rank  = {id_: rank + 1 for id_, rank in vec_ids}
    all_ids = set(bm25_rank) | set(vec_rank)
    scored = [
        (id_, _rrf_score(bm25_rank.get(id_), vec_rank.get(id_)))
        for id_ in all_ids
    ]
    return sorted(scored, key=lambda x: x[1], reverse=True)


def hybrid_search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    layer: str | None = None,
    provider: Any = None,
) -> list[SearchResult]:
    """BM25 + cosine similarity search fused with RRF.

    Falls back to pure BM25 if no embeddings exist or provider unavailable.
    """
    if not query.strip():
        return []

    # ── BM25 leg ──────────────────────────────────────────────────────────────
    # Use bm25_only=True to avoid recursive call back into hybrid_search
    bm25_results = search(conn, query, limit=50, layer=layer, bm25_only=True)
    if not bm25_results:
        # BM25 found nothing — try vector-only path
        bm25_results = []

    # ── Vector leg ────────────────────────────────────────────────────────────
    try:
        if provider is None:
            provider = _get_search_embedding_provider()
        query_vec = provider.embed([query])[0]
        from medulla.db.embedding_store import find_similar_chunks, find_similar_wiki_pages
        vec_hits: list[dict] = []
        if layer is None or layer == "episodic":
            vec_hits.extend(find_similar_chunks(conn, query_vec, top_k=50))
        if layer is None or layer == "semantic":
            vec_hits.extend(find_similar_wiki_pages(conn, query_vec, top_k=50))
    except Exception:
        return bm25_results[:limit] if bm25_results else []

    if not vec_hits:
        return bm25_results[:limit]

    # ── Vector-only path (BM25 found nothing) ────────────────────────────────
    if not bm25_results:
        return _vec_hits_to_results(conn, vec_hits, limit)

    # ── RRF fusion ────────────────────────────────────────────────────────────
    # Build id keys: chunks use "session_id:chunk_index", wiki uses slug
    def _chunk_key(h: dict) -> str:
        return f"{h['session_id']}:{h['chunk_index']}"

    def _result_key(r: SearchResult) -> str:
        if r.result_type == "chunk":
            return f"{r.id}:{r.chunk_index}"
        return r.id  # wiki slug or session id

    bm25_ranked = [(_result_key(r), i) for i, r in enumerate(bm25_results)]
    vec_ranked = [
        (_chunk_key(h) if "session_id" in h else h["slug"], i)
        for i, h in enumerate(vec_hits)
    ]
    fused = _rrf_fuse(bm25_ranked, vec_ranked)

    # Re-order bm25_results by fused ranking, drop items not in bm25 (vec-only hits
    # don't have excerpts yet — BM25 guarantees text extraction)
    key_to_result = {_result_key(r): r for r in bm25_results}
    ordered: list[SearchResult] = []
    seen: set[str] = set()
    for key, _ in fused:
        if key in key_to_result and key not in seen:
            ordered.append(key_to_result[key])
            seen.add(key)
        if len(ordered) >= limit:
            break

    # Append any BM25 results not reached by fused ordering
    for r in bm25_results:
        k = _result_key(r)
        if k not in seen and len(ordered) < limit:
            ordered.append(r)
            seen.add(k)

    return ordered[:limit]


def _vec_hits_to_results(
    conn: sqlite3.Connection,
    vec_hits: list[dict],
    limit: int,
) -> list[SearchResult]:
    """Convert vec similarity hits to SearchResult objects (used when BM25 finds nothing)."""
    results = []
    for hit in vec_hits[:limit]:
        if "session_id" in hit:
            row = conn.execute(
                "SELECT project_dir, started_at FROM sessions WHERE session_id = ?",
                (hit["session_id"],)
            ).fetchone()
            results.append(SearchResult(
                layer="episodic",
                result_type="chunk",
                id=hit["session_id"],
                title=f"Session {hit['session_id'][:8]}",
                excerpt=_snippet(hit.get("chunk_text", ""), 200),
                project_dir=row["project_dir"] if row else None,
                date=row["started_at"] if row else None,
                rank=float(hit.get("distance", 0)),
                chunk_index=hit["chunk_index"],
            ))
        elif "slug" in hit:
            results.append(SearchResult(
                layer="semantic",
                result_type="wiki_page",
                id=hit["slug"],
                title=hit.get("title", hit["slug"]),
                excerpt=_snippet(_strip_frontmatter(hit.get("content", "")), 200),
                project_dir=None,
                date=None,
                rank=float(hit.get("distance", 0)),
            ))
    return results


def _strip_frontmatter(content: str) -> str:
    """Remove YAML frontmatter (--- ... ---) before excerpting."""
    if content.startswith("---"):
        end = content.find("\n---", 3)
        if end != -1:
            return content[end + 4:].lstrip()
    return content


def _to_fts_query(query: str) -> str:
    """Convert user query to FTS5 query — quote each token for literal match."""
    tokens = query.split()
    return " ".join(f'"{t}"' for t in tokens if t)


def _snippet(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"
