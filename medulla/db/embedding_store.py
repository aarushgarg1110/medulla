"""Vector storage and retrieval for session chunks and wiki pages.

Embeddings are stored as packed float32 BLOBs in regular SQLite tables.
Similarity search uses vec_distance_cosine() from the sqlite-vec extension.
This gives exact cosine search — appropriate for medulla's scale (thousands
of rows) without the complexity of ANN virtual tables.
"""
from __future__ import annotations

import struct
import sqlite3
from typing import Any


def _pack(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack(blob: bytes) -> list[float]:
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


# ── chunk embeddings ──────────────────────────────────────────────────────────

def upsert_chunk_embedding(
    conn: sqlite3.Connection,
    session_id: str,
    chunk_index: int,
    embedding: list[float],
) -> None:
    conn.execute(
        """INSERT INTO vec_chunks(session_id, chunk_index, embedding)
           VALUES (?, ?, ?)
           ON CONFLICT(session_id, chunk_index) DO UPDATE SET embedding = excluded.embedding""",
        (session_id, chunk_index, _pack(embedding)),
    )
    conn.commit()


def get_chunk_embedding(
    conn: sqlite3.Connection,
    session_id: str,
    chunk_index: int,
) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding FROM vec_chunks WHERE session_id = ? AND chunk_index = ?",
        (session_id, chunk_index),
    ).fetchone()
    return _unpack(row[0]) if row else None


def get_chunks_without_embeddings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return session_chunks rows that have no embedding yet."""
    return conn.execute("""
        SELECT sc.session_id, sc.chunk_index, sc.chunk_text
        FROM session_chunks sc
        LEFT JOIN vec_chunks vc
            ON vc.session_id = sc.session_id AND vc.chunk_index = sc.chunk_index
        WHERE vc.session_id IS NULL
    """).fetchall()


def find_similar_chunks(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return top-k session chunks by cosine similarity to query_embedding."""
    blob = _pack(query_embedding)
    rows = conn.execute("""
        SELECT sc.session_id, sc.chunk_index, sc.chunk_text,
               vec_distance_cosine(vc.embedding, ?) AS distance
        FROM vec_chunks vc
        JOIN session_chunks sc
            ON sc.session_id = vc.session_id AND sc.chunk_index = vc.chunk_index
        ORDER BY distance
        LIMIT ?
    """, (blob, top_k)).fetchall()
    return [dict(r) for r in rows]


# ── wiki embeddings ───────────────────────────────────────────────────────────

def upsert_wiki_embedding(
    conn: sqlite3.Connection,
    slug: str,
    embedding: list[float],
) -> None:
    conn.execute(
        """INSERT INTO vec_wiki(slug, embedding) VALUES (?, ?)
           ON CONFLICT(slug) DO UPDATE SET embedding = excluded.embedding""",
        (slug, _pack(embedding)),
    )
    conn.commit()


def get_wiki_embedding(
    conn: sqlite3.Connection,
    slug: str,
) -> list[float] | None:
    row = conn.execute(
        "SELECT embedding FROM vec_wiki WHERE slug = ?", (slug,)
    ).fetchone()
    return _unpack(row[0]) if row else None


def get_wiki_pages_without_embeddings(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return wiki_pages rows that have no embedding yet."""
    return conn.execute("""
        SELECT wp.slug, wp.content
        FROM wiki_pages wp
        LEFT JOIN vec_wiki vw ON vw.slug = wp.slug
        WHERE vw.slug IS NULL
    """).fetchall()


def find_similar_wiki_pages(
    conn: sqlite3.Connection,
    query_embedding: list[float],
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Return top-k wiki pages by cosine similarity to query_embedding."""
    blob = _pack(query_embedding)
    rows = conn.execute("""
        SELECT wp.slug, wp.type, wp.title,
               vec_distance_cosine(vw.embedding, ?) AS distance
        FROM vec_wiki vw
        JOIN wiki_pages wp ON wp.slug = vw.slug
        ORDER BY distance
        LIMIT ?
    """, (blob, top_k)).fetchall()
    return [dict(r) for r in rows]
