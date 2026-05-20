"""Shared fixtures for all tests."""
import json
import sqlite3
import textwrap
from pathlib import Path

import pytest

from medulla.db.database import connect


@pytest.fixture
def db(tmp_path):
    """In-memory-style SQLite DB (file in tmp_path so migrations work cleanly)."""
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    yield conn
    conn.close()


def make_claude_jsonl(messages: list[dict]) -> str:
    """Build a minimal Claude Code JSONL string from a list of message dicts."""
    lines = []
    for i, msg in enumerate(messages):
        lines.append(json.dumps(msg))
    return "\n".join(lines)


def claude_user(text: str, session_id: str = "test-session-id", cwd: str = "/home/user/proj", ts: str = "2026-01-01T10:00:00Z") -> dict:
    return {
        "sessionId": session_id,
        "cwd": cwd,
        "gitBranch": "main",
        "timestamp": ts,
        "type": "user",
        "message": {"role": "user", "content": text},
    }


def claude_assistant(tools: list[str] | None = None, ts: str = "2026-01-01T10:01:00Z", session_id: str = "test-session-id") -> dict:
    content = []
    for t in (tools or []):
        content.append({"type": "tool_use", "name": t, "input": {"command": f"run {t}"}})
    return {
        "sessionId": session_id,
        "timestamp": ts,
        "type": "assistant",
        "message": {"role": "assistant", "content": content},
        "model": "claude-sonnet-4-6",
    }
