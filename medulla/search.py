"""Unified FTS5 search across all layers."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass


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


def search(
    conn: sqlite3.Connection,
    query: str,
    limit: int = 10,
    layer: str | None = None,
) -> list[SearchResult]:
    """Search FTS5 indexes. Returns results ranked by BM25."""
    if not query.strip():
        return []

    fts_query = _to_fts_query(query)
    results: list[SearchResult] = []

    if layer is None or layer == "episodic":
        results.extend(_search_chunks(conn, fts_query, limit))
        results.extend(_search_sessions(conn, fts_query, limit))

    # Deduplicate: one result per session_id, prefer best-ranked chunk over session-level match
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
                sc.chunk_text,
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
        excerpt = _snippet(row["chunk_text"], 200)
        results.append(SearchResult(
            layer="episodic",
            result_type="chunk",
            id=row["session_id"],
            title=f"Session {row['session_id'][:8]} chunk {row['chunk_index']}",
            excerpt=excerpt,
            project_dir=row["project_dir"],
            date=row["started_at"],
            rank=row["rank"],
        ))
    return results


def _search_sessions(conn: sqlite3.Connection, fts_query: str, limit: int) -> list[SearchResult]:
    """Fallback: search session-level FTS for sessions not caught by chunks."""
    try:
        rows = conn.execute("""
            SELECT
                s.session_id,
                s.first_message,
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
        results.append(SearchResult(
            layer="episodic",
            result_type="session",
            id=row["session_id"],
            title=f"Session {row['session_id'][:8]}",
            excerpt=_snippet(row["first_message"] or "", 200),
            project_dir=row["project_dir"],
            date=row["started_at"],
            rank=row["rank"],
        ))
    return results


def _to_fts_query(query: str) -> str:
    """Convert user query to FTS5 query — quote each token for literal match."""
    tokens = query.split()
    return " ".join(f'"{t}"' for t in tokens if t)


def _snippet(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len].rsplit(" ", 1)[0] + "…"
