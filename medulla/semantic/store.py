"""DB reads/writes for semantic (wiki) layer."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, UTC
from pathlib import Path


def upsert_wiki_page(
    conn: sqlite3.Connection,
    slug: str,
    page_type: str,
    title: str,
    content: str,
    file_path: Path,
    tags: list[str] | None = None,
    sources: list[str] | None = None,
    scope: str = "personal",
    session_id: str | None = None,
    raw_path: "Path | None" = None,
) -> None:
    now = datetime.now(UTC).isoformat()
    conn.execute("""
        INSERT INTO wiki_pages (slug, type, title, tags, sources, content, file_path, scope, session_id, raw_path, ingested_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(slug) DO UPDATE SET
            title=excluded.title,
            tags=excluded.tags,
            sources=excluded.sources,
            content=excluded.content,
            file_path=excluded.file_path,
            raw_path=excluded.raw_path,
            updated_at=excluded.updated_at
    """, (
        slug, page_type, title,
        json.dumps(tags or []),
        json.dumps(sources or []),
        content,
        str(file_path),
        scope,
        session_id,
        str(raw_path) if raw_path else None,
        now, now,
    ))
    conn.commit()


def list_wiki_pages(conn: sqlite3.Connection, page_type: str | None = None, limit: int = 50) -> list[sqlite3.Row]:
    if page_type:
        return conn.execute("""
            SELECT slug, type, title, ingested_at FROM wiki_pages
            WHERE type = ? ORDER BY ingested_at DESC LIMIT ?
        """, (page_type, limit)).fetchall()
    return conn.execute("""
        SELECT slug, type, title, ingested_at FROM wiki_pages
        ORDER BY ingested_at DESC LIMIT ?
    """, (limit,)).fetchall()


def get_wiki_page(conn: sqlite3.Connection, slug: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM wiki_pages WHERE slug = ?", (slug,)).fetchone()


def search_wiki(conn: sqlite3.Connection, query: str, page_type: str | None = None, limit: int = 10) -> list[sqlite3.Row]:
    if not query.strip():
        return []
    fts_q = " ".join(f'"{t}"' for t in query.split() if t)
    try:
        if page_type:
            return conn.execute("""
                SELECT wp.slug, wp.type, wp.title, wp.content, wf.rank
                FROM wiki_fts wf
                JOIN wiki_pages wp ON wp.rowid = wf.rowid
                WHERE wiki_fts MATCH ? AND wp.type = ?
                ORDER BY wf.rank LIMIT ?
            """, (fts_q, page_type, limit)).fetchall()
        return conn.execute("""
            SELECT wp.slug, wp.type, wp.title, wp.content, wf.rank
            FROM wiki_fts wf
            JOIN wiki_pages wp ON wp.rowid = wf.rowid
            WHERE wiki_fts MATCH ?
            ORDER BY wf.rank LIMIT ?
        """, (fts_q, limit)).fetchall()
    except sqlite3.OperationalError:
        return []


def get_wiki_stats(conn: sqlite3.Connection) -> dict:
    total = conn.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()[0]
    by_type = conn.execute("""
        SELECT type, COUNT(*) as cnt FROM wiki_pages GROUP BY type
    """).fetchall()
    return {
        "total": total,
        "by_type": {row["type"]: row["cnt"] for row in by_type},
    }


# ── Pending ingest queue ───────────────────────────────────────────────────────

def queue_pending(
    conn: sqlite3.Connection,
    source_path: str,
    source_type: str,
    title: str | None = None,
    force: bool = False,
    processing_path: str | None = None,
) -> int:
    """Add source to pending queue. Idempotent.

    source_path: dedup key — URL string or 'sha256:<hash>' for binary files.
    processing_path: actual file path passed to the LLM pipeline (temp file for
                     URLs, raw/ path for binaries). Falls back to source_path if None.

    States:
    - already queued → no-op
    - previously errored → reset to queued (retry)
    - already done + dedup key is a file that still exists → skip
    - already done + dedup key is URL or missing file → re-queue
    - force=True → always re-queue
    """
    import os
    effective_path = processing_path or source_path

    existing = conn.execute(
        "SELECT id, status FROM pending_ingests WHERE source_path = ? ORDER BY id DESC LIMIT 1",
        (source_path,)
    ).fetchone()

    if existing:
        if existing["status"] == "queued":
            return existing["id"]
        if existing["status"] == "done" and not force:
            # For URL/hash keys, treat as already processed
            is_file_key = not source_path.startswith("http") and not source_path.startswith("sha256:")
            if not is_file_key or os.path.exists(effective_path):
                return existing["id"]
        # error, done+missing, or force → reset
        conn.execute(
            "UPDATE pending_ingests SET status='queued', error=NULL, queued_at=datetime('now'), "
            "processing_path=? WHERE id=?",
            (effective_path, existing["id"])
        )
        conn.commit()
        return existing["id"]

    cur = conn.execute("""
        INSERT INTO pending_ingests (source_path, source_type, title, status, queued_at, processing_path)
        VALUES (?, ?, ?, 'queued', datetime('now'), ?)
    """, (source_path, source_type, title, effective_path))
    conn.commit()
    return cur.lastrowid


def get_pending(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute("""
        SELECT * FROM pending_ingests WHERE status = 'queued' ORDER BY queued_at
    """).fetchall()


def get_pending_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM pending_ingests WHERE status = 'queued'").fetchone()[0]


def mark_pending_done(conn: sqlite3.Connection, pending_id: int) -> None:
    conn.execute("""
        UPDATE pending_ingests SET status='done', processed_at=datetime('now') WHERE id=?
    """, (pending_id,))
    conn.commit()


def mark_pending_error(conn: sqlite3.Connection, pending_id: int, error: str) -> None:
    conn.execute("""
        UPDATE pending_ingests SET status='error', error=?, processed_at=datetime('now') WHERE id=?
    """, (error, pending_id))
    conn.commit()
