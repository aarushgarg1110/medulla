"""Tests for store functions added in Sprint 2."""
import pytest

from medulla.episodic.store import (
    get_session_detail,
    get_session_tree,
    get_project_context,
    search_events,
    upsert_session,
    upsert_agent_session,
)
from tests.test_store import make_session, make_agent


def _setup(db):
    s = make_session("sess-s2", project_dir="/proj/mlops", messages=["hello logD world"])
    upsert_session(db, s)
    a = make_agent("agent-s2", parent="sess-s2")
    upsert_agent_session(db, a)
    return db


# ── get_session_detail ─────────────────────────────────────────────────────────

def test_get_session_detail_found(db):
    _setup(db)
    detail = get_session_detail(db, "sess-s2")
    assert detail is not None
    assert detail["session"]["session_id"] == "sess-s2"
    assert len(detail["chunks"]) >= 1
    assert len(detail["agents"]) == 1
    assert detail["agents"][0]["agent_id"] == "agent-s2"


def test_get_session_detail_not_found(db):
    assert get_session_detail(db, "missing") is None


def test_get_session_detail_no_agents(db):
    upsert_session(db, make_session("lone", messages=["hi"]))
    detail = get_session_detail(db, "lone")
    assert detail is not None
    assert detail["agents"] == []


# ── get_session_tree ───────────────────────────────────────────────────────────

def test_get_session_tree_found(db):
    _setup(db)
    tree = get_session_tree(db, "sess-s2")
    assert tree is not None
    assert tree["session"]["session_id"] == "sess-s2"
    assert len(tree["agents"]) == 1


def test_get_session_tree_not_found(db):
    assert get_session_tree(db, "missing") is None


def test_get_session_tree_multiple_agents(db):
    upsert_session(db, make_session("parent", messages=["task"]))
    for i in range(3):
        upsert_agent_session(db, make_agent(f"agent-{i}", parent="parent"))
    tree = get_session_tree(db, "parent")
    assert len(tree["agents"]) == 3


# ── get_project_context ────────────────────────────────────────────────────────

def test_get_project_context_found(db):
    _setup(db)
    ctx = get_project_context(db, "mlops")
    assert ctx["project"] == "mlops"
    assert len(ctx["sessions"]) == 1
    assert ctx["sessions"][0]["session_id"] == "sess-s2"


def test_get_project_context_no_match(db):
    ctx = get_project_context(db, "nonexistent")
    assert ctx["sessions"] == []
    assert ctx["events"] == []


def test_get_project_context_respects_session_limit(db):
    for i in range(5):
        upsert_session(db, make_session(f"sess-ctx-{i}", project_dir="/proj/mlops", messages=["hi"]))
    ctx = get_project_context(db, "mlops", session_limit=2)
    assert len(ctx["sessions"]) == 2


# ── search_events ──────────────────────────────────────────────────────────────

def test_search_events_empty(db):
    assert search_events(db, "logD") == []


def test_search_events_finds_match(db):
    db.execute("""
        INSERT INTO tool_events(session_id, project_dir, tool, command, event_ts, event_hash, ingested_at)
        VALUES (?,?,?,?,?,?,datetime('now'))
    """, ("s1", "/proj/a", "Bash", "medulla search logD outliers", "2026-01-01T10:00:00Z", "h1"))
    db.commit()
    results = search_events(db, "logD outliers")
    assert len(results) >= 1


def test_search_events_no_match(db):
    db.execute("""
        INSERT INTO tool_events(session_id, project_dir, tool, command, event_ts, event_hash, ingested_at)
        VALUES (?,?,?,?,?,?,datetime('now'))
    """, ("s1", "/proj/a", "Bash", "echo hello", "2026-01-01T10:00:00Z", "h2"))
    db.commit()
    results = search_events(db, "zzznomatch999")
    assert results == []
