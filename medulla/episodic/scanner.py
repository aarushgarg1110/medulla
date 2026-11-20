"""Discover and scan session files (Claude Code, Kiro, Antigravity/Gemini) into the DB.

Episodic sources are registered in `SOURCES` so discovery, session-id extraction, title lookup, and
live staleness checks share one interface instead of branching on source strings (planned refactor #12).
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, NamedTuple

from medulla.episodic.parser import (
    parse_session, parse_agent_session, is_subagent_file, extract_tool_events,
    is_gemini_transcript, gemini_session_id, gemini_summary_meta,
)
from medulla.episodic.store import (
    upsert_session, upsert_agent_session, upsert_tool_events,
    get_session_scanned_at, get_agent_scanned_at, is_forgotten, forgotten_ids,
    delete_session_rows, get_empty_mtime, mark_empty, get_session_title,
)

_USER_LINE = re.compile(r'"type":\s*"user"')
_GEMINI_USER_LINE = re.compile(r'"type":\s*"USER_INPUT"')


def _get_embedding_provider():
    """The process's one embedding model, shared with search. Query and chunk vectors must come from
    the same model, and the MCP server pre-loads it at startup (~11 s), so indexing from the server
    reuses that copy instead of loading a second one."""
    from medulla.search import _get_search_embedding_provider
    return _get_search_embedding_provider()


CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"
KIRO_SESSIONS_DIR = Path.home() / ".kiro" / "sessions"
GEMINI_BRAIN_DIR = Path.home() / ".gemini" / "antigravity-cli" / "brain"


@dataclass(frozen=True)
class EpisodicSource:
    name: str
    discover: Callable[[], tuple[list[Path], list[Path]]]
    matches: Callable[[Path], bool]
    session_id: Callable[[Path], str]
    title_of: Callable[[Path], str | None]
    has_user_message: Callable[[Path], bool]
    check_stale: bool = True


# ── Per-source adapters ───────────────────────────────────────────────────────

def _discover_claude() -> tuple[list[Path], list[Path]]:
    sessions, agents = [], []
    if CLAUDE_PROJECTS_DIR.exists():
        for jsonl in CLAUDE_PROJECTS_DIR.rglob("*.jsonl"):
            (agents if is_subagent_file(jsonl) else sessions).append(jsonl)
    return sessions, agents


def _claude_title_of(path: Path) -> str | None:
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


def _file_matches(path: Path, pattern: re.Pattern) -> bool:
    with open(path, errors="replace") as fh:
        return any(pattern.search(line) for line in fh)


def _discover_kiro() -> tuple[list[Path], list[Path]]:
    if not KIRO_SESSIONS_DIR.exists():
        return [], []
    return list(KIRO_SESSIONS_DIR.rglob("*.jsonl")), []


def _gemini_subagent_ids() -> set[str]:
    """Conversation IDs with nesting_depth > 0 or a parent_conversation_id in conversation_summaries.db."""
    db = GEMINI_BRAIN_DIR.parent / "conversation_summaries.db"
    try:
        if not db.exists():
            return set()
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            rows = conn.execute(
                "SELECT conversation_id FROM conversation_summaries "
                "WHERE nesting_depth > 0 OR parent_conversation_id != ''"
            ).fetchall()
            return {r[0] for r in rows}
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return set()


def _discover_gemini() -> tuple[list[Path], list[Path]]:
    if not GEMINI_BRAIN_DIR.exists():
        return [], []
    subagents = _gemini_subagent_ids()
    sessions, agents = [], []
    for jsonl in GEMINI_BRAIN_DIR.glob("*/.system_generated/logs/transcript.jsonl"):
        sid = gemini_session_id(jsonl)
        (agents if sid in subagents else sessions).append(jsonl)
    return sessions, agents


def _gemini_title_of(path: Path) -> str | None:
    return gemini_summary_meta(path, gemini_session_id(path)).get("title")


SOURCES: dict[str, EpisodicSource] = {
    "claude": EpisodicSource(
        name="claude",
        discover=_discover_claude,
        matches=lambda p: not is_gemini_transcript(p),
        session_id=lambda p: p.stem,
        title_of=_claude_title_of,
        has_user_message=lambda p: _file_matches(p, _USER_LINE),
        check_stale=True,
    ),
    "kiro": EpisodicSource(
        name="kiro",
        discover=_discover_kiro,
        matches=lambda p: str(KIRO_SESSIONS_DIR) in str(p),
        session_id=lambda p: p.stem,
        title_of=lambda p: None,
        has_user_message=lambda p: False,
        check_stale=False,
    ),
    "gemini": EpisodicSource(
        name="gemini",
        discover=_discover_gemini,
        matches=is_gemini_transcript,
        session_id=gemini_session_id,
        title_of=_gemini_title_of,
        has_user_message=lambda p: _file_matches(p, _GEMINI_USER_LINE),
        check_stale=True,
    ),
}


def get_source(name: str) -> EpisodicSource:
    if name not in SOURCES:
        raise ValueError(f"unknown source: {name!r} (expected {', '.join(SOURCES)})")
    return SOURCES[name]


def source_for(path: Path) -> EpisodicSource:
    if is_gemini_transcript(path):
        return SOURCES["gemini"]
    return SOURCES["claude"]


def session_id_for(path: Path) -> str:
    return source_for(path).session_id(path)


def title_of(path: Path) -> str | None:
    """The session's title (Claude custom-title or Gemini conversation_summaries.db title)."""
    return source_for(path).title_of(path)


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
        report(ScanProgress(done, total, bytes_done, bytes_total, _label(conn, session_id_for(path))))
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
    targets = [get_source(source)] if source is not None else list(SOURCES.values())
    for src in targets:
        s, a = src.discover()
        session_files.extend(s)
        agent_files.extend(a)
    return session_files, agent_files


def needs_index(conn: sqlite3.Connection, path: Path) -> bool:
    """True when the file changed after its session was last indexed or last parsed as no conversation, or
    was never looked at."""
    sid = session_id_for(path)
    mtime = _mtime(path)
    scanned_at = get_session_scanned_at(conn, sid)
    if scanned_at:
        return mtime > scanned_at
    empty_at = get_empty_mtime(conn, sid)
    return empty_at is None or mtime > empty_at


def _mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _has_user_message(path: Path) -> bool:
    return source_for(path).has_user_message(path)


def stale_sessions(conn: sqlite3.Connection, settle_s: float = 60) -> list[Path]:
    """Session files that changed since they were indexed, oldest change first.

    A file modified in the last `settle_s` seconds is a live session, not a stale index, and is left
    out. A never-indexed file counts only if it holds a user message: the parser skips stubs, so they
    would otherwise read as stale forever."""
    cutoff = time.time() - settle_s
    gone = forgotten_ids(conn)
    out = []
    for src in SOURCES.values():
        if not src.check_stale:
            continue
        files, _ = src.discover()
        for p in files:
            sid = src.session_id(p)
            if sid in gone or p.stat().st_mtime > cutoff or not needs_index(conn, p):
                continue
            if get_session_scanned_at(conn, sid) or src.has_user_message(p):
                out.append(p)
    return sorted(out, key=lambda p: p.stat().st_mtime)


def index_session(conn: sqlite3.Connection, path: Path) -> str:
    """Index one session file now, whatever its mtime: what a resume needs for a session that ended
    after the last scan, without paying for a scan of every changed file."""
    return _process_session(conn, path, force=True)


def _process_session(conn: sqlite3.Connection, path: Path, force: bool) -> str:
    sid = session_id_for(path)
    if is_forgotten(conn, sid):
        return "skipped_forgotten"
    if not force and not needs_index(conn, path):
        return "skipped_mtime"

    session = parse_session(path)
    if session is None:
        return _not_a_conversation(conn, path)
    if is_forgotten(conn, session.session_id):
        return "skipped_forgotten"  # a copy of a forgotten transcript under another name

    mark_empty(conn, sid, None)
    upsert_session(conn, session)
    _embed_session_chunks(conn, session.session_id)
    upsert_tool_events(conn, session.session_id, extract_tool_events(path))
    return "indexed"


def _not_a_conversation(conn: sqlite3.Connection, path: Path) -> str:
    """Remember a transcript that holds no conversation (no reply, only commands) at its mtime, and drop the
    rows an older parser wrote for it. Not when it cannot be read or is blank: then the rows may be the only copy."""
    sid = session_id_for(path)
    try:
        if not path.read_text(encoding="utf-8", errors="replace").strip():
            return "skipped_empty"
    except OSError:
        return "skipped_empty"
    had_rows = get_session_scanned_at(conn, sid) is not None
    if had_rows:
        delete_session_rows(conn, sid)
    mark_empty(conn, sid, _mtime(path))
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
    agent_id = gemini_session_id(path) if is_gemini_transcript(path) else path.stem.removeprefix("agent-")
    if not force:
        scanned_at = get_agent_scanned_at(conn, agent_id)
        if scanned_at:
            file_mtime = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
            if file_mtime <= scanned_at:
                return "skipped_mtime"

    agent = parse_agent_session(path)
    if agent is None or (agent.parent_session_id and is_forgotten(conn, agent.parent_session_id)):
        return "skipped"

    upsert_agent_session(conn, agent)
    return "indexed"
