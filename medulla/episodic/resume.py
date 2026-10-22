"""Pick a session back up, and say when the index is behind the transcripts.

Shared by the MCP tools and the CLI, so `medulla_resume` and `medulla resume` (and search's expand and
staleness line) give the same answer whichever front door is used. Only the hint naming how to
refresh or read on differs, because the two front doors spell it differently.
"""
from __future__ import annotations

import difflib
import re
import sqlite3

from medulla.episodic import scanner

EXPAND_CHARS = 6000   # per expanded chunk; a handoff report fits, a 40k-char tool dump does not

_PART = re.compile(r"^(.*?)(?:Part\d+)?$")
_REPORT = re.compile(r"immediate next action", re.I)
_REPORT_START = re.compile(r"(?im)^#+\s*(?:handoff|1\.\s*overarching goal)")


class NotFound(LookupError):
    """No session matched; the message lists the closest titles."""


def stale_note(conn: sqlite3.Connection, hint: str) -> str:
    """One line when the index is behind the transcripts — a session that ended after the last scan
    is invisible to search, and nothing used to say so. Empty when the index is current."""
    stale = scanner.stale_sessions(conn)
    if not stale:
        return ""
    names = [scanner.title_of(p) or p.stem[:8] for p in stale[-3:]]
    n = len(stale)
    return (f"\n⚠ {n} transcript{'s' if n > 1 else ''} changed since "
            f"{'they were' if n > 1 else 'it was'} indexed ({', '.join(names)}{', …' if n > 3 else ''}): "
            f"results may miss them. {hint}")


def chunk_text(conn: sqlite3.Connection, session_id: str, chunk_index: int) -> str:
    row = conn.execute("SELECT chunk_text FROM session_chunks WHERE session_id = ? AND chunk_index = ?",
                       (session_id, chunk_index)).fetchone()
    return row["chunk_text"] if row else ""


def expanded(conn: sqlite3.Connection, session_id: str, chunk_index: int, read_on: str) -> str:
    """A chunk's text, capped at EXPAND_CHARS, ending with `read_on` (how to fetch the rest) when cut."""
    text = chunk_text(conn, session_id, chunk_index)
    more = len(text) - EXPAND_CHARS
    return text[:EXPAND_CHARS] + (f"\n  … {more:,} more chars: {read_on}" if more > 0 else "")


def _resolve(conn: sqlite3.Connection, name: str):
    rows = conn.execute("""SELECT session_id, slug, project_dir, started_at, ended_at, turn_count
                           FROM sessions WHERE lower(slug) = lower(?) ORDER BY started_at DESC""", (name,)).fetchall()
    if not rows:
        rows = conn.execute("""SELECT session_id, slug, project_dir, started_at, ended_at, turn_count
                               FROM sessions WHERE session_id LIKE ? ORDER BY started_at DESC""", (name + "%",)).fetchall()
    return rows[0] if rows else None


def resume(conn: sqlite3.Connection, name: str, last: int = 2) -> str:
    """Index `name` if its transcript changed, then return its header, its parts (foo, fooPart2, …) by
    date, and its handoff report, or its last `last` chunks when it has none. Raises NotFound."""
    indexed = []
    for p in scanner.stale_sessions(conn, settle_s=0):
        if p.stem.startswith(name) or scanner.title_of(p) == name:
            if scanner.index_session(conn, p) == "indexed":
                indexed.append(p.stem)
    s = _resolve(conn, name)
    if s is None:
        titles = [r[0] for r in conn.execute("SELECT DISTINCT slug FROM sessions WHERE slug IS NOT NULL")]
        close = difflib.get_close_matches(name, titles, n=5, cutoff=0.6)
        raise NotFound(f"Session '{name}' not found by title or id prefix."
                       + (f" Closest titles: {', '.join(close)}" if close else ""))

    sid, slug = s["session_id"], s["slug"] or s["session_id"]
    chunks = conn.execute("SELECT chunk_index, chunk_text FROM session_chunks WHERE session_id = ? "
                          "ORDER BY chunk_index", (sid,)).fetchall()
    lines = [f"Resume: {slug}  ({sid[:8]}, {(s['project_dir'] or '').split('/')[-1]}, "
             f"{(s['started_at'] or '')[:16]} → {(s['ended_at'] or '')[:16]}, {s['turn_count']} turns, "
             f"{len(chunks)} chunks)" + ("  — indexed now" if sid in indexed else "")]

    stem = _PART.match(slug).group(1)
    parts = [r for r in conn.execute("SELECT session_id, slug, started_at, turn_count FROM sessions "
                                     "WHERE slug = ? OR slug LIKE ? ORDER BY started_at", (stem, stem + "Part%"))
             if _PART.match(r["slug"]).group(1) == stem]
    if len(parts) > 1:
        lines += ["", "Parts (oldest first):"]
        for r in parts:
            mark = "▸" if r["session_id"] == sid else " "
            lines.append(f"  {mark} {r['slug']:<34} {(r['started_at'] or '')[:10]}  {r['turn_count']:>4} turns  {r['session_id'][:8]}")

    hits = [i for i, c in enumerate(chunks) if _REPORT.search(c["chunk_text"])]
    if hits:
        k = hits[-1]
        while k > 0 and not _REPORT_START.search(chunks[k]["chunk_text"]) and k > hits[-1] - 2:
            k -= 1
        body = "\n".join(c["chunk_text"] for c in chunks[k:])
        lines += ["", f"Handoff report (chunks {k}–{len(chunks) - 1}):", body[:4 * EXPAND_CHARS]]
    else:
        n = max(1, last or 2)
        lines += ["", f"No handoff report in this session — its last {min(n, len(chunks))} chunk(s):"]
        lines += [c["chunk_text"][:EXPAND_CHARS] for c in chunks[-n:]]
    return "\n".join(lines)
