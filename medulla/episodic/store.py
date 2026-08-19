"""DB reads/writes for episodic layer."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, UTC

from medulla.episodic.parser import ParsedSession, ParsedAgentSession, ToolEvent
from medulla.episodic.chunker import chunk_messages, Chunk


def upsert_session(conn: sqlite3.Connection, session: ParsedSession) -> None:
    conn.execute("""
        INSERT INTO sessions (
            session_id, source, project_dir, git_branch, slug, model,
            started_at, ended_at, turn_count, tool_call_count,
            tool_names, files_json, first_message, all_user_text,
            scope, scanned_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(session_id) DO UPDATE SET
            source=excluded.source,
            project_dir=excluded.project_dir,
            git_branch=excluded.git_branch,
            slug=excluded.slug,
            model=excluded.model,
            started_at=excluded.started_at,
            ended_at=excluded.ended_at,
            turn_count=excluded.turn_count,
            tool_call_count=excluded.tool_call_count,
            tool_names=excluded.tool_names,
            files_json=excluded.files_json,
            first_message=excluded.first_message,
            all_user_text=excluded.all_user_text,
            scanned_at=excluded.scanned_at
    """, (
        session.session_id,
        session.source,
        session.project_dir,
        session.git_branch,
        session.slug,
        session.model,
        session.started_at,
        session.ended_at,
        session.turn_count,
        session.tool_call_count,
        json.dumps(session.tool_names),
        json.dumps(session.files),
        session.first_message,
        session.all_user_text,
        "private",
        datetime.now(UTC).isoformat(),
    ))

    # Replace chunks. Also drop stale embeddings — chunk_index space changes when
    # re-chunking, so old vectors would mismatch the new chunk text. They are
    # re-embedded by the scanner after upsert.
    conn.execute("DELETE FROM session_chunks WHERE session_id = ?", (session.session_id,))
    conn.execute("DELETE FROM vec_chunks WHERE session_id = ?", (session.session_id,))
    chunks = chunk_messages(session.messages)
    for chunk in chunks:
        conn.execute("""
            INSERT INTO session_chunks (session_id, chunk_index, chunk_text, turn_start, turn_end)
            VALUES (?,?,?,?,?)
        """, (session.session_id, chunk.chunk_index, chunk.chunk_text, chunk.turn_start, chunk.turn_end))

    conn.commit()


def upsert_agent_session(conn: sqlite3.Connection, agent: ParsedAgentSession) -> None:
    conn.execute("""
        INSERT INTO agent_sessions (
            agent_id, parent_session_id, agent_slug, project_dir, cwd, model,
            turn_count, tool_call_count, tool_names, first_message, all_user_text,
            message_count, first_seen_at, last_updated_at, scanned_at
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(agent_id) DO UPDATE SET
            parent_session_id=excluded.parent_session_id,
            agent_slug=excluded.agent_slug,
            project_dir=excluded.project_dir,
            cwd=excluded.cwd,
            model=excluded.model,
            turn_count=excluded.turn_count,
            tool_call_count=excluded.tool_call_count,
            tool_names=excluded.tool_names,
            first_message=excluded.first_message,
            all_user_text=excluded.all_user_text,
            message_count=excluded.message_count,
            last_updated_at=excluded.last_updated_at,
            scanned_at=excluded.scanned_at
    """, (
        agent.agent_id,
        agent.parent_session_id,
        agent.agent_slug,
        agent.project_dir,
        agent.cwd,
        agent.model,
        agent.turn_count,
        agent.tool_call_count,
        json.dumps(agent.tool_names),
        agent.first_message,
        agent.all_user_text,
        agent.message_count,
        agent.first_seen_at,
        agent.last_updated_at,
        datetime.now(UTC).isoformat(),
    ))
    conn.commit()


def upsert_tool_events(
    conn: sqlite3.Connection, session_id: str, events: list[ToolEvent]
) -> None:
    """Replace a session's harvested tool_events (idempotent on re-scan).

    FTS stays in sync via the tool_events_ai/ad triggers. event_hash is UNIQUE so
    duplicate calls within a batch are ignored.
    """
    conn.execute("DELETE FROM tool_events WHERE session_id = ?", (session_id,))
    for e in events:
        conn.execute("""
            INSERT OR IGNORE INTO tool_events
                (event_ts, session_id, project_dir, tool, command,
                 output_preview, is_error, interrupted, event_hash)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            e.event_ts, e.session_id, e.project_dir, e.tool, e.command,
            e.output_preview, 1 if e.is_error else 0, 1 if e.interrupted else 0,
            e.event_hash,
        ))
    conn.commit()


def get_session_scanned_at(conn: sqlite3.Connection, session_id: str) -> str | None:
    row = conn.execute(
        "SELECT scanned_at FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    return row["scanned_at"] if row else None


def get_agent_scanned_at(conn: sqlite3.Connection, agent_id: str) -> str | None:
    row = conn.execute(
        "SELECT scanned_at FROM agent_sessions WHERE agent_id = ?", (agent_id,)
    ).fetchone()
    return row["scanned_at"] if row else None


def list_sessions(conn: sqlite3.Connection, project: str | None = None, limit: int = 20) -> list[sqlite3.Row]:
    if project:
        return conn.execute("""
            SELECT session_id, source, project_dir, slug, model, started_at, ended_at,
                   turn_count, tool_call_count, first_message
            FROM sessions
            WHERE project_dir LIKE ?
            ORDER BY started_at DESC
            LIMIT ?
        """, (f"%{project}%", limit)).fetchall()
    return conn.execute("""
        SELECT session_id, source, project_dir, slug, model, started_at, ended_at,
               turn_count, tool_call_count, first_message
        FROM sessions
        ORDER BY started_at DESC
        LIMIT ?
    """, (limit,)).fetchall()


def get_session_detail(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Return full session info plus ordered chunks."""
    row = conn.execute("""
        SELECT session_id, source, project_dir, git_branch, slug, model,
               started_at, ended_at, turn_count, tool_call_count,
               tool_names, files_json, first_message
        FROM sessions WHERE session_id = ?
    """, (session_id,)).fetchone()
    if not row:
        return None

    chunks = conn.execute("""
        SELECT chunk_index, chunk_text, turn_start, turn_end
        FROM session_chunks WHERE session_id = ?
        ORDER BY chunk_index
    """, (session_id,)).fetchall()

    agents = conn.execute("""
        SELECT agent_id, agent_slug, turn_count, tool_call_count, first_message
        FROM agent_sessions WHERE parent_session_id = ?
        ORDER BY first_seen_at
    """, (session_id,)).fetchall()

    return {
        "session": dict(row),
        "chunks": [dict(c) for c in chunks],
        "agents": [dict(a) for a in agents],
    }


def get_session_tree(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Return parent session header plus all child agent sessions."""
    row = conn.execute("""
        SELECT session_id, project_dir, model, started_at, turn_count,
               tool_call_count, first_message
        FROM sessions WHERE session_id = ?
    """, (session_id,)).fetchone()
    if not row:
        return None

    agents = conn.execute("""
        SELECT agent_id, agent_slug, turn_count, tool_call_count,
               first_message, first_seen_at
        FROM agent_sessions WHERE parent_session_id = ?
        ORDER BY first_seen_at
        LIMIT 100
    """, (session_id,)).fetchall()

    return {"session": dict(row), "agents": [dict(a) for a in agents]}


def get_project_context(
    conn: sqlite3.Connection,
    project: str,
    session_limit: int = 5,
    event_limit: int = 20,
) -> dict:
    """Recent sessions and tool events for a project directory."""
    sessions = conn.execute("""
        SELECT session_id, source, project_dir, model, started_at,
               turn_count, tool_call_count, first_message
        FROM sessions WHERE project_dir LIKE ?
        ORDER BY started_at DESC LIMIT ?
    """, (f"%{project}%", session_limit)).fetchall()

    events = conn.execute("""
        SELECT event_ts, session_id, tool, command, output_preview
        FROM tool_events WHERE project_dir LIKE ?
        ORDER BY event_ts DESC LIMIT ?
    """, (f"%{project}%", event_limit)).fetchall()

    return {
        "project": project,
        "sessions": [dict(s) for s in sessions],
        "events": [dict(e) for e in events],
    }


def search_events(
    conn: sqlite3.Connection, query: str, limit: int = 20
) -> list[sqlite3.Row]:
    """FTS5 search over tool_events."""
    try:
        fts_q = " ".join(f'"{t}"' for t in query.split() if t)
        return conn.execute("""
            SELECT te.event_ts, te.session_id, te.tool, te.command,
                   te.output_preview, te.is_error, te.interrupted, tef.rank
            FROM tool_events_fts tef
            JOIN tool_events te ON te.rowid = tef.rowid
            WHERE tool_events_fts MATCH ?
            ORDER BY tef.rank LIMIT ?
        """, (fts_q, limit)).fetchall()
    except Exception:
        return []


def get_next_command(
    conn: sqlite3.Connection, session_id: str, after_ts: str, limit: int = 1
) -> list[sqlite3.Row]:
    """The next non-aborted command(s) in a session after a timestamp.

    Used to show 'what ran next' after a failed command — surfaced as context,
    NOT asserted as the fix (the reader judges).
    """
    return conn.execute("""
        SELECT command, is_error, event_ts FROM tool_events
        WHERE session_id = ? AND event_ts > ? AND interrupted = 0
        ORDER BY event_ts LIMIT ?
    """, (session_id, after_ts, limit)).fetchall()


def get_stats(conn: sqlite3.Connection) -> dict:
    sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    turns = conn.execute("SELECT SUM(turn_count) FROM sessions").fetchone()[0] or 0
    tools = conn.execute("SELECT SUM(tool_call_count) FROM sessions").fetchone()[0] or 0
    chunks = conn.execute("SELECT COUNT(*) FROM session_chunks").fetchone()[0]
    agents = conn.execute("SELECT COUNT(*) FROM agent_sessions").fetchone()[0]
    oldest = conn.execute("SELECT MIN(started_at) FROM sessions").fetchone()[0]
    newest = conn.execute("SELECT MAX(started_at) FROM sessions").fetchone()[0]

    top_tools_raw = conn.execute("""
        SELECT tool_names FROM sessions WHERE tool_names IS NOT NULL
    """).fetchall()
    tool_counts: dict[str, int] = {}
    for row in top_tools_raw:
        try:
            for t in json.loads(row[0]):
                tool_counts[t] = tool_counts.get(t, 0) + 1
        except Exception:
            pass
    top_tools = sorted(tool_counts.items(), key=lambda x: -x[1])[:10]

    return {
        "sessions": sessions,
        "turns": turns,
        "tool_calls": tools,
        "chunks": chunks,
        "agent_sessions": agents,
        "oldest": oldest,
        "newest": newest,
        "top_tools": top_tools,
    }
