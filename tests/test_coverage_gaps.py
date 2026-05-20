"""Targeted tests for specific uncovered branches."""
import json
import sqlite3
from pathlib import Path

import pytest

from medulla.episodic.parser import parse_session, parse_agent_session, _extract_paths
from medulla.episodic.scanner import scan, _discover_files
from medulla.episodic.store import get_stats, upsert_session
from medulla.search import search, _search_sessions
from tests.conftest import claude_user, make_claude_jsonl
from tests.test_store import make_session


# ── parser — uncovered branches ───────────────────────────────────────────────

def test_parse_session_model_from_assistant_turn(tmp_path):
    """model field is extracted from assistant turn."""
    lines = [
        json.dumps({"sessionId": "s1", "cwd": "/p", "timestamp": "2026-01-01T10:00:00Z",
                    "type": "user", "message": {"role": "user", "content": "hi"}}),
        json.dumps({"sessionId": "s1", "timestamp": "2026-01-01T10:01:00Z",
                    "type": "assistant", "model": "claude-haiku-4-5",
                    "message": {"role": "assistant", "content": []}}),
    ]
    path = tmp_path / "s1.jsonl"
    path.write_text("\n".join(lines))
    result = parse_session(path)
    assert result is not None
    assert result.model == "claude-haiku-4-5"


def test_parse_session_string_content_in_list_item(tmp_path):
    """Content list items can be bare strings, not just dicts."""
    line = json.dumps({
        "sessionId": "s-str",
        "timestamp": "2026-01-01T10:00:00Z",
        "type": "user",
        "message": {"role": "user", "content": ["bare string item", {"type": "text", "text": "dict item"}]},
    })
    path = tmp_path / "s-str.jsonl"
    path.write_text(line)
    result = parse_session(path)
    assert result is not None
    assert "bare string item" in result.all_user_text
    assert "dict item" in result.all_user_text


def test_parse_agent_session_model_from_assistant(tmp_path):
    """Agent parser extracts model from assistant message."""
    subdir = tmp_path / "subagents"
    subdir.mkdir()
    path = subdir / "agent-model-test.jsonl"
    lines = [
        json.dumps({"sessionId": "parent-1", "cwd": "/proj",
                    "timestamp": "2026-01-01T10:00:00Z",
                    "type": "user", "message": {"role": "user", "content": "task"}}),
        json.dumps({"sessionId": "parent-1", "timestamp": "2026-01-01T10:01:00Z",
                    "type": "assistant", "model": "claude-opus-4-7",
                    "message": {"role": "assistant", "content": []}}),
    ]
    path.write_text("\n".join(lines))
    result = parse_agent_session(path)
    assert result is not None
    assert result.model == "claude-opus-4-7"


def test_extract_paths_from_list():
    """_extract_paths handles list input."""
    acc: set[str] = set()
    _extract_paths(["/home/user/file.py", "/tmp/other.sh"], acc)
    assert any("file.py" in p for p in acc)
    assert any("other.sh" in p for p in acc)


def test_extract_paths_ignores_non_string_leaves():
    acc: set[str] = set()
    _extract_paths({"key": 42, "other": None}, acc)
    assert len(acc) == 0


# ── scanner — error branch ─────────────────────────────────────────────────────

def test_scan_handles_parse_error_gracefully(db, tmp_path, monkeypatch):
    """A corrupted file that triggers an exception in upsert should increment errors."""
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    # Valid JSONL that parses OK but we'll make upsert_session raise
    proj = tmp_path / "proj"
    proj.mkdir()
    path = proj / "bad-sess.jsonl"
    path.write_text(make_claude_jsonl([claude_user("hello", session_id="bad-sess")]))

    original_upsert = upsert_session

    def exploding_upsert(conn, session):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr("medulla.episodic.scanner.upsert_session", exploding_upsert)

    counts = scan(db, force=True)
    assert counts["errors"] == 1


def test_discover_files_no_source_includes_both(tmp_path, monkeypatch):
    """source=None discovers both claude and kiro files."""
    claude_dir = tmp_path / "claude"
    kiro_dir = tmp_path / "kiro"
    claude_dir.mkdir()
    kiro_dir.mkdir()

    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", claude_dir)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", kiro_dir)

    (claude_dir / "sess.jsonl").write_text("{}")
    (kiro_dir / "kiro-sess.jsonl").write_text("{}")

    sessions, _ = _discover_files(source=None)
    paths = [str(p) for p in sessions]
    assert any("claude" in p for p in paths)
    assert any("kiro" in p for p in paths)


# ── store — malformed tool_names JSON ─────────────────────────────────────────

def test_get_stats_with_malformed_tool_names(db):
    """Sessions with malformed tool_names JSON should not crash get_stats."""
    db.execute("""
        INSERT INTO sessions(session_id, source, scope, scanned_at, tool_names, turn_count, tool_call_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, ("bad-sess", "claude", "private", "2026-01-01T10:00:00Z", "NOT VALID JSON", 1, 0))
    db.commit()

    s = get_stats(db)
    assert s["sessions"] == 1  # didn't crash, tool just excluded from top_tools


# ── search — session-level FTS fallback & chunk→session upgrade ───────────────

def test_search_upgrades_session_to_chunk_match(db):
    """If FTS finds both a session-level and chunk-level match for same session,
    the chunk match should be kept (more specific)."""
    # Insert short session (1 chunk) so both sessions_fts and session_chunks_fts fire
    s = make_session("sess-upgrade", messages=["logD batch effect analysis"] * 5)
    upsert_session(db, s)

    results = search(db, "logD batch")
    matching = [r for r in results if r.id == "sess-upgrade"]
    assert len(matching) == 1  # deduped to exactly one
    assert matching[0].result_type == "chunk"  # chunk preferred


def test_search_sessions_fts_error_returns_empty(db):
    """_search_sessions handles sqlite OperationalError (e.g. bad FTS state) gracefully."""
    # Drop the FTS table to trigger OperationalError
    db.execute("DROP TABLE IF EXISTS sessions_fts")
    db.commit()
    results = _search_sessions(db, '"hello"', 10)
    assert results == []
