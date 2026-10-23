"""Forget a session: remove it from the index and keep it out of every later scan (#65).

Shared by the MCP tool and the CLI, like resume. Transcripts are Claude Code's and only ever read, so a
forget records the id in forgotten_sessions, which the scanner skips, and leaves the file alone. Deleting the
transcript is a separate step only the CLI takes, after asking. Wiki pages that cite the session are
authored content and stay.
"""
from __future__ import annotations

import re
import shutil
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, UTC
from pathlib import Path

from medulla.episodic import scanner
from medulla.episodic.resume import NotFound, not_found

LIVE_S = 60   # a transcript written this recently belongs to a running session (the stale note's settle time)

_PLAIN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_TABLES = ("sessions", "forgotten_sessions")


class Ambiguous(LookupError):
    """A title or id prefix matched several sessions; the message lists them."""


class LiveTranscript(RuntimeError):
    """The transcript was written in the last minute: it belongs to a running session."""


@dataclass
class Target:
    session_id: str
    source: str | None
    slug: str | None
    project_dir: str | None
    started_at: str | None
    turn_count: int | None
    forgotten_at: str | None = None


def find(conn: sqlite3.Connection, name: str, tables: tuple[str, ...] = _TABLES) -> Target:
    """The one session `name` means, by title and then id prefix, indexed first and then forgotten. Several
    matches raise Ambiguous, where resume would take the newest: a forget must not guess."""
    for table in tables:
        extra = ", forgotten_at" if table == "forgotten_sessions" else ""
        for where, arg in (("lower(slug) = lower(?)", name), ("session_id LIKE ?", name + "%")):
            found = conn.execute(f"SELECT session_id, source, slug, project_dir, started_at, turn_count{extra} "
                                 f"FROM {table} WHERE {where} ORDER BY started_at DESC", (arg,)).fetchall()
            if len(found) > 1:
                listed = "\n".join(f"  {r['session_id'][:8]}  {(r['started_at'] or '')[:10]}  "
                                   f"{r['turn_count']} turns  {_project(r['project_dir'])}" for r in found)
                raise Ambiguous(f"'{name}' matches {len(found)} sessions, so nothing was forgotten. "
                                f"Rerun with an id prefix:\n{listed}")
            if found:
                return Target(**dict(found[0]))
    if tables == _TABLES:
        raise not_found(conn, name)
    raise NotFound(f"'{name}' is not on the forgotten list.")


def forget(conn: sqlite3.Connection, t: Target) -> str:
    """Delete every row of the session and put it on the forgotten list, in one transaction."""
    paths = transcript(t)
    if t.forgotten_at:
        return f"{_describe(t)} was already forgotten on {t.forgotten_at[:10]}.\n{_transcript_line(paths)}"
    sid = t.session_id
    chunks = conn.execute("SELECT COUNT(*) FROM session_chunks WHERE session_id = ?", (sid,)).fetchone()[0]
    agents = conn.execute("SELECT COUNT(*) FROM agent_sessions WHERE parent_session_id = ?", (sid,)).fetchone()[0]
    with conn:
        conn.execute("INSERT INTO forgotten_sessions (session_id, source, slug, project_dir, started_at, turn_count, "
                     "forgotten_at) VALUES (?,?,?,?,?,?,?)", (sid, t.source, t.slug, t.project_dir, t.started_at,
                                                              t.turn_count, datetime.now(UTC).isoformat()))
        conn.execute("DELETE FROM vec_chunks WHERE session_id = ?", (sid,))   # no delete trigger
        conn.execute("DELETE FROM session_chunks WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM tool_events WHERE session_id = ?", (sid,))
        conn.execute("DELETE FROM agent_sessions WHERE parent_session_id = ?", (sid,))
        conn.execute("DELETE FROM sessions WHERE session_id = ?", (sid,))
    return (f"Forgot {_describe(t, _n(chunks, 'chunk'), _n(agents, 'subagent'))}: removed from the index, "
            f"and every scan now skips it.\n{_transcript_line(paths)}")


def transcript(t: Target) -> list[Path]:
    """The session's own files: <id>.jsonl and the <id>/ folder beside it (subagents, tool results). Nothing
    else in the project folder is the session's: memory/ sits there too."""
    if t.source != "claude" or not _PLAIN_ID.match(t.session_id):
        return []
    out = []
    for f in sorted(scanner.CLAUDE_PROJECTS_DIR.glob(f"*/{t.session_id}.jsonl")):
        out.append(f)
        if f.with_suffix("").is_dir():
            out.append(f.with_suffix(""))
    return out


def delete_transcript(conn: sqlite3.Connection, t: Target) -> str:
    paths = transcript(t)
    if not paths:
        return _transcript_line(paths)
    live = [p for p in paths if p.is_file() and time.time() - p.stat().st_mtime < LIVE_S]
    if live:
        raise LiveTranscript(f"{live[0]} was written in the last minute, so it belongs to a running session "
                             "and was not deleted.")
    done = []
    for p in paths:
        if p.is_dir():
            shutil.rmtree(p)
            done.append(f"Deleted {p}/")
        else:
            p.unlink()
            done.append(f"Deleted {p}")
    conn.execute("UPDATE forgotten_sessions SET transcript_deleted = 1 WHERE session_id = ?", (t.session_id,))
    conn.commit()
    return "\n".join(done)


def undo(conn: sqlite3.Connection, name: str) -> str:
    t = find(conn, name, tables=("forgotten_sessions",))
    conn.execute("DELETE FROM forgotten_sessions WHERE session_id = ?", (t.session_id,))
    conn.commit()
    if transcript(t):
        return f"{_describe(t)} is off the forgotten list; the next scan indexes it again."
    return f"{_describe(t)} is off the forgotten list, but its transcript is not on disk, so there is nothing to index."


def listing(conn: sqlite3.Connection) -> str:
    found = conn.execute("SELECT session_id, slug, project_dir, forgotten_at, transcript_deleted "
                         "FROM forgotten_sessions ORDER BY forgotten_at DESC").fetchall()
    if not found:
        return "No forgotten sessions."
    lines = [f"{_n(len(found), 'forgotten session')}:"]
    lines += [f"  {r['session_id'][:8]}  {r['forgotten_at'][:10]}  {r['slug'] or ''}  {_project(r['project_dir'])}"
              + ("  (transcript deleted)" if r["transcript_deleted"] else "") for r in found]
    return "\n".join(lines)


def _describe(t: Target, *counts: str) -> str:
    """'title (id8, project, date, N turns, *counts)'."""
    parts = [t.session_id[:8], _project(t.project_dir), (t.started_at or "")[:10], _n(t.turn_count or 0, "turn")]
    return f"{t.slug or t.session_id} ({', '.join(parts + list(counts))})"


def _transcript_line(paths: list[Path]) -> str:
    if not paths:
        return "Transcript: none on disk."
    shown = [f"{p}/ ({_n(sum(f.is_file() for f in p.rglob('*')), 'file')})" if p.is_dir() else str(p) for p in paths]
    return "Transcript: " + ", ".join(shown)


def _project(project_dir: str | None) -> str:
    return (project_dir or "").rstrip("/").split("/")[-1]


def _n(k: int, word: str) -> str:
    return f"{k} {word}{'' if k == 1 else 's'}"
