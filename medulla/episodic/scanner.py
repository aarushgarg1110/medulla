"""Discover and scan Claude Code session files into the DB."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from medulla.episodic.parser import (
    parse_session, parse_agent_session, is_subagent_file
)
from medulla.episodic.store import (
    upsert_session, upsert_agent_session,
    get_session_scanned_at, get_agent_scanned_at,
)


_embedding_provider = None

def _get_embedding_provider():
    global _embedding_provider
    if _embedding_provider is None:
        from medulla.embeddings import get_embedding_provider
        _embedding_provider = get_embedding_provider()
    return _embedding_provider

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
KIRO_SESSIONS_DIR = Path.home() / ".kiro" / "sessions"


def scan(conn: sqlite3.Connection, force: bool = False, source: str | None = None) -> dict:
    """Scan session files and index new/changed ones.

    Returns counts: {indexed, skipped, errors, agents_indexed, agents_skipped}
    """
    indexed = skipped = errors = empty = 0
    agents_indexed = agents_skipped = 0

    session_files, agent_files = _discover_files(source)

    for path in session_files:
        try:
            result = _process_session(conn, path, force)
            if result == "indexed":
                indexed += 1
            elif result == "skipped_mtime":
                skipped += 1
            elif result == "skipped_empty":
                empty += 1
        except Exception:
            errors += 1

    for path in agent_files:
        try:
            result = _process_agent(conn, path, force)
            if result == "indexed":
                agents_indexed += 1
            elif result == "skipped_mtime":
                agents_skipped += 1
        except Exception:
            pass

    return {
        "indexed": indexed,
        "skipped": skipped,
        "empty": empty,
        "errors": errors,
        "agents_indexed": agents_indexed,
        "agents_skipped": agents_skipped,
    }


def _discover_files(source: str | None) -> tuple[list[Path], list[Path]]:
    session_files: list[Path] = []
    agent_files: list[Path] = []

    if source is None or source == "claude":
        if CLAUDE_PROJECTS_DIR.exists():
            for jsonl in CLAUDE_PROJECTS_DIR.rglob("*.jsonl"):
                if is_subagent_file(jsonl):
                    agent_files.append(jsonl)
                else:
                    session_files.append(jsonl)

    if source is None or source == "kiro":
        if KIRO_SESSIONS_DIR.exists():
            for jsonl in KIRO_SESSIONS_DIR.rglob("*.jsonl"):
                session_files.append(jsonl)

    return session_files, agent_files


def _process_session(conn: sqlite3.Connection, path: Path, force: bool) -> str:
    if not force:
        scanned_at = get_session_scanned_at(conn, path.stem)
        if scanned_at:
            file_mtime = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            if file_mtime <= scanned_at:
                return "skipped_mtime"

    session = parse_session(path)
    if session is None:
        return "skipped_empty"  # empty/stub/no-user-messages

    upsert_session(conn, session)
    _embed_session_chunks(conn, session.session_id)
    return "indexed"


def _embed_session_chunks(conn: sqlite3.Connection, session_id: str) -> None:
    """Embed all chunks for a session that don't have embeddings yet."""
    try:
        from medulla.db.embedding_store import get_chunks_without_embeddings, upsert_chunk_embedding
        provider = _get_embedding_provider()
        missing = [r for r in get_chunks_without_embeddings(conn)
                   if r["session_id"] == session_id]
        if not missing:
            return
        texts = [r["chunk_text"] for r in missing]
        embeddings = provider.embed(texts)
        for row, emb in zip(missing, embeddings):
            upsert_chunk_embedding(conn, row["session_id"], row["chunk_index"], emb)
    except Exception:
        pass  # embedding failures must never break indexing


def _process_agent(conn: sqlite3.Connection, path: Path, force: bool) -> str:
    agent_id = path.stem.removeprefix("agent-")
    if not force:
        scanned_at = get_agent_scanned_at(conn, agent_id)
        if scanned_at:
            file_mtime = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            if file_mtime <= scanned_at:
                return "skipped_mtime"

    agent = parse_agent_session(path)
    if agent is None:
        return "skipped"

    upsert_agent_session(conn, agent)
    return "indexed"
