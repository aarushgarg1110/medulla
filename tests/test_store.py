"""Tests for medulla.episodic.store — real SQLite DB, no mocks."""
import json
from datetime import datetime

import pytest

from medulla.episodic.parser import ParsedSession, ParsedAgentSession
from medulla.episodic.store import (
    get_agent_scanned_at,
    get_session_scanned_at,
    get_stats,
    list_sessions,
    upsert_agent_session,
    upsert_session,
)


def make_session(session_id="sess-001", project_dir="/proj/a", turn_count=5, messages=None) -> ParsedSession:
    if messages is None:
        messages = ["hello", "world", "foo"]
    return ParsedSession(
        session_id=session_id,
        source="claude",
        project_dir=project_dir,
        git_branch="main",
        slug=session_id,
        model="claude-sonnet-4-6",
        started_at="2026-01-01T10:00:00Z",
        ended_at="2026-01-01T11:00:00Z",
        turn_count=turn_count,
        tool_call_count=2,
        tool_names=["Bash", "Read"],
        files=["/proj/a/foo.py"],
        first_message="hello",
        all_user_text=" ".join(messages),
        messages=messages,
    )


def make_agent(agent_id="agent-001", parent="sess-001") -> ParsedAgentSession:
    return ParsedAgentSession(
        agent_id=agent_id,
        parent_session_id=parent,
        agent_slug="test-agent",
        project_dir="/proj/a",
        cwd="/proj/a/src",
        model="claude-sonnet-4-6",
        turn_count=3,
        tool_call_count=1,
        tool_names=["Bash"],
        first_message="agent task",
        all_user_text="agent task details",
        message_count=6,
        first_seen_at="2026-01-01T10:05:00Z",
        last_updated_at="2026-01-01T10:10:00Z",
    )


# ── upsert_session ─────────────────────────────────────────────────────────────

def test_upsert_session_inserts(db):
    session = make_session()
    upsert_session(db, session)
    row = db.execute("SELECT * FROM sessions WHERE session_id = ?", ("sess-001",)).fetchone()
    assert row is not None
    assert row["project_dir"] == "/proj/a"
    assert row["turn_count"] == 5
    assert row["first_message"] == "hello"


def test_upsert_session_creates_chunks(db):
    messages = [f"msg {i}" for i in range(25)]
    session = make_session(messages=messages)
    upsert_session(db, session)
    chunks = db.execute("SELECT * FROM session_chunks WHERE session_id = ?", ("sess-001",)).fetchall()
    assert len(chunks) >= 2  # 25 messages → at least 2 chunks (default window=20)


def test_upsert_session_updates_on_conflict(db):
    session = make_session(turn_count=5)
    upsert_session(db, session)
    updated = make_session(turn_count=10, messages=["new message"])
    upsert_session(db, updated)
    row = db.execute("SELECT turn_count FROM sessions WHERE session_id = ?", ("sess-001",)).fetchone()
    assert row["turn_count"] == 10


def test_upsert_session_replaces_chunks_on_update(db):
    session = make_session(messages=["a", "b", "c"])
    upsert_session(db, session)
    before = db.execute("SELECT COUNT(*) FROM session_chunks WHERE session_id = ?", ("sess-001",)).fetchone()[0]

    updated = make_session(messages=["x"] * 50)
    upsert_session(db, updated)
    after = db.execute("SELECT COUNT(*) FROM session_chunks WHERE session_id = ?", ("sess-001",)).fetchone()[0]

    assert after > before  # more chunks from more messages


def test_upsert_session_fts_indexed(db):
    session = make_session(messages=["the quick brown fox"])
    upsert_session(db, session)
    rows = db.execute("SELECT * FROM session_chunks_fts WHERE session_chunks_fts MATCH '\"quick\"'").fetchall()
    assert len(rows) >= 1


def test_upsert_session_sets_scanned_at(db):
    session = make_session()
    upsert_session(db, session)
    scanned = get_session_scanned_at(db, "sess-001")
    assert scanned is not None


# ── upsert_agent_session ───────────────────────────────────────────────────────

def test_upsert_agent_session_inserts(db):
    agent = make_agent()
    upsert_agent_session(db, agent)
    row = db.execute("SELECT * FROM agent_sessions WHERE agent_id = ?", ("agent-001",)).fetchone()
    assert row is not None
    assert row["parent_session_id"] == "sess-001"
    assert row["first_message"] == "agent task"


def test_upsert_agent_session_updates_on_conflict(db):
    agent = make_agent()
    upsert_agent_session(db, agent)
    updated = make_agent()
    updated = ParsedAgentSession(
        **{**agent.__dict__, "turn_count": 99}
    )
    upsert_agent_session(db, updated)
    row = db.execute("SELECT turn_count FROM agent_sessions WHERE agent_id = ?", ("agent-001",)).fetchone()
    assert row["turn_count"] == 99


def test_upsert_agent_fts_indexed(db):
    agent = make_agent()
    upsert_agent_session(db, agent)
    rows = db.execute("SELECT * FROM agent_sessions_fts WHERE agent_sessions_fts MATCH '\"agent\"'").fetchall()
    assert len(rows) >= 1


# ── get_session_scanned_at / get_agent_scanned_at ─────────────────────────────

def test_get_session_scanned_at_returns_none_if_missing(db):
    assert get_session_scanned_at(db, "nonexistent") is None


def test_get_session_scanned_at_returns_value(db):
    upsert_session(db, make_session())
    assert get_session_scanned_at(db, "sess-001") is not None


def test_get_agent_scanned_at_returns_none_if_missing(db):
    assert get_agent_scanned_at(db, "nonexistent") is None


def test_get_agent_scanned_at_returns_value(db):
    upsert_agent_session(db, make_agent())
    assert get_agent_scanned_at(db, "agent-001") is not None


# ── list_sessions ─────────────────────────────────────────────────────────────

def test_list_sessions_returns_all(db):
    upsert_session(db, make_session("sess-001", "/proj/a"))
    upsert_session(db, make_session("sess-002", "/proj/b"))
    rows = list_sessions(db)
    assert len(rows) == 2


def test_list_sessions_filter_by_project(db):
    upsert_session(db, make_session("sess-001", "/proj/alpha"))
    upsert_session(db, make_session("sess-002", "/proj/beta"))
    rows = list_sessions(db, project="alpha")
    assert len(rows) == 1
    assert rows[0]["session_id"] == "sess-001"


def test_list_sessions_respects_limit(db):
    for i in range(10):
        upsert_session(db, make_session(f"sess-{i:03d}", f"/proj/{i}"))
    rows = list_sessions(db, limit=3)
    assert len(rows) == 3


def test_list_sessions_empty_db(db):
    rows = list_sessions(db)
    assert rows == []


# ── get_stats ──────────────────────────────────────────────────────────────────

def test_get_stats_empty_db(db):
    s = get_stats(db)
    assert s["sessions"] == 0
    assert s["turns"] == 0
    assert s["chunks"] == 0
    assert s["agent_sessions"] == 0
    assert s["top_tools"] == []


def test_get_stats_counts(db):
    upsert_session(db, make_session("s1", turn_count=10, messages=["a"] * 25))
    upsert_session(db, make_session("s2", turn_count=5, messages=["b"] * 5))
    upsert_agent_session(db, make_agent())
    s = get_stats(db)
    assert s["sessions"] == 2
    assert s["turns"] == 15
    assert s["agent_sessions"] == 1
    assert s["chunks"] >= 2  # s1 has 2 chunks, s2 has 1


def test_get_stats_top_tools(db):
    upsert_session(db, make_session("s1"))  # tool_names = ["Bash", "Read"]
    upsert_session(db, make_session("s2"))  # tool_names = ["Bash", "Read"]
    s = get_stats(db)
    tool_names = [t[0] for t in s["top_tools"]]
    assert "Bash" in tool_names
    assert "Read" in tool_names
