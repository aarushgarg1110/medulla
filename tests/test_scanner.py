"""Tests for medulla.episodic.scanner — real files, real SQLite, no mocks."""
import json
import time
from pathlib import Path

import pytest

from medulla.episodic.scanner import _discover_files, scan
from medulla.episodic.store import get_session_scanned_at, list_sessions
from tests.conftest import claude_assistant, claude_user, make_claude_jsonl


class _MockEmbedProvider:
    dimension = 768
    model_name = "mock"
    def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]


@pytest.fixture(autouse=True)
def patch_scanner_embeddings(monkeypatch):
    import medulla.episodic.scanner as scanner_mod
    monkeypatch.setattr(scanner_mod, "_get_embedding_provider", lambda: _MockEmbedProvider())


def _write_session(root: Path, session_id: str, messages: list[str]) -> Path:
    """Write a minimal Claude session JSONL to root/<project>/<session_id>.jsonl."""
    proj = root / "my-project"
    proj.mkdir(exist_ok=True)
    lines = [claude_user(m, session_id=session_id) for m in messages]
    path = proj / f"{session_id}.jsonl"
    path.write_text(make_claude_jsonl(lines))
    return path


def _write_agent_session(root: Path, parent_id: str, agent_id: str) -> Path:
    subdir = root / "my-project" / f"{parent_id}" / "subagents"
    subdir.mkdir(parents=True, exist_ok=True)
    path = subdir / f"agent-{agent_id}.jsonl"
    path.write_text(make_claude_jsonl([
        claude_user("agent task", session_id=parent_id),
    ]))
    return path


# ── basic scan ─────────────────────────────────────────────────────────────────

def test_scan_indexes_new_sessions(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "kiro-missing")

    _write_session(tmp_path, "sess-abc", ["hello", "world"])
    counts = scan(db, force=False)

    assert counts["indexed"] == 1
    assert counts["skipped"] == 0
    assert counts["errors"] == 0
    rows = list_sessions(db)
    assert len(rows) == 1


def test_scan_skips_unchanged_sessions(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    _write_session(tmp_path, "sess-abc", ["hello"])

    # First scan — indexed
    counts1 = scan(db, force=False)
    assert counts1["indexed"] == 1

    # Second scan — skipped (file not modified)
    counts2 = scan(db, force=False)
    assert counts2["indexed"] == 0
    assert counts2["skipped"] == 1


def test_scan_reindexes_modified_session(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    path = _write_session(tmp_path, "sess-abc", ["original content"])
    scan(db, force=False)

    # Modify file (ensure mtime advances)
    time.sleep(0.01)
    path.write_text(path.read_text() + "\n" + json.dumps(claude_user("new content", session_id="sess-abc")))
    import os; os.utime(path, None)  # touch to update mtime

    counts = scan(db, force=False)
    assert counts["indexed"] == 1


def test_scan_force_reindexes_all(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    _write_session(tmp_path, "sess-1", ["hello"])
    _write_session(tmp_path, "sess-2", ["world"])

    scan(db, force=False)
    counts = scan(db, force=True)  # all re-indexed regardless of mtime
    assert counts["indexed"] == 2
    assert counts["skipped"] == 0


def test_scan_indexes_agent_sessions(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    _write_session(tmp_path, "parent-sess", ["parent task"])
    _write_agent_session(tmp_path, "parent-sess", "my-agent")

    counts = scan(db, force=False)
    assert counts["agents_indexed"] == 1


def test_scan_multiple_sessions(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    for i in range(5):
        _write_session(tmp_path, f"sess-{i:03d}", [f"content {i}"])

    counts = scan(db)
    assert counts["indexed"] == 5


def test_scan_source_filter_claude_only(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    _write_session(tmp_path, "sess-1", ["hello"])
    counts = scan(db, source="claude")
    assert counts["indexed"] == 1


def test_scan_empty_directory(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    counts = scan(db)
    assert counts["indexed"] == 0
    assert counts["skipped"] == 0
    assert counts["errors"] == 0


def test_scan_nonexistent_directories(db, tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "no-claude")
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "no-kiro")

    counts = scan(db)
    assert counts["indexed"] == 0


# ── _discover_files ────────────────────────────────────────────────────────────

def test_discover_files_separates_agents(tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    _write_session(tmp_path, "sess-1", ["hello"])
    _write_agent_session(tmp_path, "sess-1", "agt-1")

    sessions, agents = _discover_files(source=None)
    assert len(sessions) == 1
    assert len(agents) == 1


def test_discover_files_kiro_filter(tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "none")
    kiro_dir = tmp_path / "kiro"
    kiro_dir.mkdir()
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", kiro_dir)

    (kiro_dir / "session-1.jsonl").write_text("{}")

    sessions, agents = _discover_files(source="kiro")
    assert len(sessions) == 1

    sessions2, _ = _discover_files(source="claude")
    assert len(sessions2) == 0
