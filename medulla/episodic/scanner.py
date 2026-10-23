"""Discover and scan Claude Code session files into the DB."""
from __future__ import annotations

import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

from medulla.episodic.parser import (
    parse_session, parse_agent_session, is_subagent_file, extract_tool_events
)
from medulla.episodic.store import (
    upsert_session, upsert_agent_session, upsert_tool_events,
    get_session_scanned_at, get_agent_scanned_at, is_forgotten, forgotten_ids,
    delete_session_rows, get_empty_mtime, mark_empty, get_session_title,
)

_USER_LINE = re.compile(r'"type":\s*"user"')


def _get_embedding_provider():
    """The process's one embedding model, shared with search. Query and chunk vectors must come from
    the same model, and the MCP server pre-loads it at startup (~11 s), so indexing from the server
    reuses that copy instead of loading a second one."""
    from medulla.search import _get_search_embedding_provider
    return _get_search_embedding_provider()

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
KIRO_SESSIONS_DIR = Path.home() / ".kiro" / "sessions"


class ScanProgress(NamedTuple):
    """How far a scan is: sessions for the count, transcript bytes for the bar and the ETA, since a few big
    sessions hold most of the work. The label is the session's title from the index."""
    done: int
    total: int
    bytes_done: int
    bytes_total: int
    label: str


def scan(conn: sqlite3.Connection, force: bool = False, source: str | None = None,
         progress: Callable[[ScanProgress], None] | None = None) -> dict:
    """Scan session files and index new/changed ones. `progress` gets a ScanProgress before each session and
    once when they are done, for the CLI to draw; the subagents follow uncounted (192 took 1 s on 2026-10-23).

    Returns counts: {indexed, skipped, empty, removed, forgotten, errors, agents_indexed, agents_skipped}
    """
    indexed = skipped = errors = empty = removed = forgotten = 0
    agents_indexed = agents_skipped = 0

    session_files, agent_files = _discover_files(source)
    sizes = [_size(p) for p in session_files]
    bytes_total, bytes_done, total = sum(sizes), 0, len(session_files)
    report = progress or (lambda p: None)

    for done, (path, size) in enumerate(zip(session_files, sizes)):
        report(ScanProgress(done, total, bytes_done, bytes_total, _label(conn, path.stem)))
        try:
            result = _process_session(conn, path, force)
            if result == "indexed":
                indexed += 1
            elif result == "skipped_mtime":
                skipped += 1
            elif result == "skipped_empty":
                empty += 1
            elif result == "removed_empty":
                removed += 1
            elif result == "skipped_forgotten":
                forgotten += 1
        except Exception:
            errors += 1
        bytes_done += size
    report(ScanProgress(total, total, bytes_total, bytes_total, ""))

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
        "removed": removed,
        "forgotten": forgotten,
        "errors": errors,
        "agents_indexed": agents_indexed,
        "agents_skipped": agents_skipped,
    }


def _size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:     # gone since it was listed: the scan counts it as an error and goes on
        return 0


def _label(conn: sqlite3.Connection, session_id: str) -> str:
    return get_session_title(conn, session_id) or session_id


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


def needs_index(conn: sqlite3.Connection, path: Path) -> bool:
    """True when the file changed after its session was last indexed or last parsed as no conversation, or
    was never looked at."""
    mtime = _mtime(path)
    scanned_at = get_session_scanned_at(conn, path.stem)
    if scanned_at:
        return mtime > scanned_at
    empty_at = get_empty_mtime(conn, path.stem)
    return empty_at is None or mtime > empty_at


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _has_user_message(path: Path) -> bool:
    with open(path, errors="replace") as fh:
        return any(_USER_LINE.search(line) for line in fh)


def title_of(path: Path) -> str | None:
    """The session's /rename title (last write wins), read without parsing the whole file."""
    title = None
    with open(path, errors="replace") as fh:
        for line in fh:
            if '"custom-title"' not in line:
                continue
            try:
                node = json.loads(line)
            except ValueError:
                continue
            if node.get("type") == "custom-title" and node.get("customTitle"):
                title = node["customTitle"]
    return title


def stale_sessions(conn: sqlite3.Connection, settle_s: float = 60) -> list[Path]:
    """Session files that changed since they were indexed, oldest change first.

    A file modified in the last `settle_s` seconds is a live session, not a stale index, and is left
    out. A never-indexed file counts only if it holds a user message: the parser skips stubs, so they
    would otherwise read as stale forever."""
    files, _ = _discover_files("claude")
    cutoff = time.time() - settle_s
    gone = forgotten_ids(conn)
    out = []
    for p in files:
        if p.stem in gone or p.stat().st_mtime > cutoff or not needs_index(conn, p):
            continue
        if get_session_scanned_at(conn, p.stem) or _has_user_message(p):
            out.append(p)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def index_session(conn: sqlite3.Connection, path: Path) -> str:
    """Index one session file now, whatever its mtime: what a resume needs for a session that ended
    after the last scan, without paying for a scan of every changed file."""
    return _process_session(conn, path, force=True)


def _process_session(conn: sqlite3.Connection, path: Path, force: bool) -> str:
    if is_forgotten(conn, path.stem):
        return "skipped_forgotten"
    if not force and not needs_index(conn, path):
        return "skipped_mtime"

    session = parse_session(path)
    if session is None:
        return _not_a_conversation(conn, path)
    if is_forgotten(conn, session.session_id):
        return "skipped_forgotten"  # a copy of a forgotten transcript under another name

    mark_empty(conn, path.stem, None)
    upsert_session(conn, session)
    _embed_session_chunks(conn, session.session_id)
    upsert_tool_events(conn, session.session_id, extract_tool_events(path))
    return "indexed"


def _not_a_conversation(conn: sqlite3.Connection, path: Path) -> str:
    """Remember a transcript that holds no conversation (no reply, only commands) at its mtime, and drop the
    rows an older parser wrote for it. Not when it cannot be read or is blank: then the rows may be the only copy."""
    try:
        if not path.read_text(encoding="utf-8", errors="replace").strip():
            return "skipped_empty"
    except OSError:
        return "skipped_empty"
    had_rows = get_session_scanned_at(conn, path.stem) is not None
    if had_rows:
        delete_session_rows(conn, path.stem)
    mark_empty(conn, path.stem, _mtime(path))
    conn.commit()
    return "removed_empty" if had_rows else "skipped_empty"


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
    if agent is None or is_forgotten(conn, agent.parent_session_id):
        return "skipped"

    upsert_agent_session(conn, agent)
    return "indexed"
