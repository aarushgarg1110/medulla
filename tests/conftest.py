"""Shared fixtures for all tests."""
import json
import sqlite3
import textwrap
from pathlib import Path

import pytest

from medulla.db.database import connect


class _MockSearchEmbedProvider:
    """Global mock embedding provider — prevents real model load in any test."""
    dimension = 768
    model_name = "mock"
    def embed(self, texts):
        results = []
        for text in texts:
            seed = abs(hash(text)) % 1000
            vec = [(seed + i) / (1000.0 * 10) for i in range(self.dimension)]
            norm = sum(v**2 for v in vec) ** 0.5
            results.append([v / norm for v in vec])
        return results


@pytest.fixture(autouse=True)
def patch_search_provider(monkeypatch):
    """Patch search._get_search_embedding_provider globally so no test loads torch."""
    import medulla.search as search_mod
    monkeypatch.setattr(search_mod, "_get_search_embedding_provider",
                        lambda: _MockSearchEmbedProvider())


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
