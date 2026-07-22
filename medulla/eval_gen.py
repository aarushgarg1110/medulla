"""Generate eval-set candidates from the local DB — two modes to compare.

known-item : sample sessions (stratified by project), derive a query from each
             session's own distinctive terms, label it relevant to that session.
             Fully auto-labeled, offline, private — but "easy" (query derived from
             the answer), best for regression.
history    : pull the user's REAL past medulla searches from tool_events; leave
             relevance blank (with candidate suggestions) for hand-labeling. Real
             query distribution, honest accuracy signal, needs human judgment.

Sampling is deterministic (no RNG) so a given DB yields a stable set.
"""
from __future__ import annotations

import collections
import re
import sqlite3

from medulla.episodic.chunker import _STOP_WORDS
from medulla.search import search

_WORD_RE = re.compile(r"\b[a-zA-Z]\w+\b")


# ── known-item ────────────────────────────────────────────────────────────────

def _session_query(conn: sqlite3.Connection, session_id: str, n_terms: int = 5) -> str:
    """Distinctive terms from a session's first chunks → a known-item query."""
    rows = conn.execute(
        "SELECT chunk_text FROM session_chunks WHERE session_id = ? ORDER BY chunk_index LIMIT 5",
        (session_id,),
    ).fetchall()
    text = " ".join(r[0] for r in rows)
    toks = [w.lower() for w in _WORD_RE.findall(text)
            if len(w) > 4 and w.lower() not in _STOP_WORDS]
    return " ".join(w for w, _ in collections.Counter(toks).most_common(n_terms))


def _stratified_sessions(conn: sqlite3.Connection, n: int) -> list[str]:
    """Round-robin across projects (recent-first within each) for coverage."""
    rows = conn.execute("""
        SELECT session_id, COALESCE(project_dir, '') proj, COALESCE(started_at, '') ts
        FROM sessions ORDER BY ts DESC
    """).fetchall()
    by_proj: dict[str, list[str]] = collections.defaultdict(list)
    for r in rows:
        by_proj[r["proj"]].append(r["session_id"])
    picked: list[str] = []
    queues = list(by_proj.values())
    i = 0
    while len(picked) < n and any(queues):
        q = queues[i % len(queues)]
        if q:
            picked.append(q.pop(0))
        i += 1
    return picked


def generate_known_item(conn: sqlite3.Connection, n: int = 20) -> list[dict]:
    cases = []
    for sid in _stratified_sessions(conn, n):
        q = _session_query(conn, sid)
        if q:
            cases.append({"query": q, "relevant": [sid[:8]]})
    return cases


# ── history ─────────────────────────────────────────────────────────────────

def _query_from_search_command(command: str) -> str:
    """A harvested search tool_event's command is '<tool> <query>' — take the query."""
    parts = command.split(None, 1)
    return parts[1].strip() if len(parts) == 2 else ""


def generate_from_history(conn: sqlite3.Connection, n: int = 20) -> list[dict]:
    # Memory searches only — exclude ToolSearch (its tool name also matches %search%)
    # and its 'select:...' tool-loading queries, which aren't real recall queries.
    rows = conn.execute("""
        SELECT command FROM tool_events
        WHERE tool LIKE '%search%' AND tool != 'ToolSearch'
        ORDER BY event_ts DESC
    """).fetchall()
    cases: list[dict] = []
    seen: set[str] = set()
    for (command,) in rows:
        q = _query_from_search_command(command or "")
        # skip tool-loading (select:) and structured-input searches (Slack/mail {json})
        if not q or q in seen or q.startswith("select:") or q.startswith("{"):
            continue
        seen.add(q)
        # dedup candidates (a session can hit as both chunk + command), keep order
        candidates: list[str] = []
        for h in search(conn, q, limit=5):
            base = h.id.split("#")[0][:8]
            if base not in candidates:
                candidates.append(base)
            if len(candidates) >= 3:
                break
        # relevant left blank on purpose — labels are never machine-fabricated
        cases.append({"query": q, "relevant": [], "candidates": candidates})
        if len(cases) >= n:
            break
    return cases


def generate(conn: sqlite3.Connection, mode: str = "known-item", n: int = 20) -> list[dict]:
    if mode == "history":
        return generate_from_history(conn, n)
    return generate_known_item(conn, n)
