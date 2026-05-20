"""Tests for medulla.db.database — migrations, connection setup."""
import sqlite3
from pathlib import Path

import pytest

from medulla.db.database import connect


def test_connect_creates_db_file(tmp_path):
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    conn.close()
    assert db_path.exists()


def test_connect_creates_parent_dirs(tmp_path):
    db_path = tmp_path / "nested" / "dirs" / "test.db"
    conn = connect(db_path)
    conn.close()
    assert db_path.exists()


def test_connect_runs_migrations(tmp_path):
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "sessions" in tables
    assert "session_chunks" in tables
    assert "agent_sessions" in tables
    assert "tool_events" in tables
    assert "schema_migrations" in tables


def test_connect_creates_fts_tables(tmp_path):
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "sessions_fts" in tables
    assert "session_chunks_fts" in tables
    assert "agent_sessions_fts" in tables
    assert "tool_events_fts" in tables


def test_migrations_applied_only_once(tmp_path):
    db_path = tmp_path / "test.db"
    conn1 = connect(db_path)
    conn1.close()
    conn2 = connect(db_path)
    count = conn2.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
    conn2.close()
    assert count == 1  # only V1 applied, not twice


def test_wal_mode_enabled(tmp_path):
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    conn.close()
    assert mode == "wal"


def test_row_factory_set(tmp_path):
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    row = conn.execute("SELECT 1 AS num").fetchone()
    conn.close()
    assert row["num"] == 1  # dict-style access works


def test_connect_twice_same_db(tmp_path):
    """Two connections to the same DB should both work and see same data."""
    db_path = tmp_path / "shared.db"
    conn1 = connect(db_path)
    conn1.execute("INSERT INTO sessions(session_id, source, scope, scanned_at) VALUES (?,?,?,?)",
                  ("s1", "claude", "private", "2026-01-01"))
    conn1.commit()

    conn2 = connect(db_path)
    row = conn2.execute("SELECT session_id FROM sessions").fetchone()
    assert row["session_id"] == "s1"
    conn1.close()
    conn2.close()
