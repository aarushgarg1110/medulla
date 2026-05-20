"""Tests for medulla.cli — Typer CliRunner, real SQLite in tmp_path."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from medulla.cli import app
from tests.conftest import claude_user, make_claude_jsonl

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point config at a tmp_path so tests don't touch ~/.medulla."""
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()

    import medulla.config as cfg_module
    import medulla.db.database as db_module

    # Reset singleton so each test gets a fresh config
    cfg_module._config = None

    monkeypatch.setattr(cfg_module, "_config", cfg_module.Config(medulla_dir=medulla_dir))
    yield


@pytest.fixture
def claude_projects(tmp_path, monkeypatch):
    """Create a fake ~/.claude/projects dir with one session."""
    projects = tmp_path / "claude_projects"
    projects.mkdir()
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", projects)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "kiro_none")

    proj = projects / "my-project"
    proj.mkdir()
    path = proj / "session-abc.jsonl"
    path.write_text(make_claude_jsonl([
        claude_user("tell me about logD outliers", session_id="session-abc-id"),
        claude_user("the CompoundX project had batch effects", session_id="session-abc-id"),
    ]))
    return projects


# ── scan ───────────────────────────────────────────────────────────────────────

def test_scan_command_runs(claude_projects):
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "indexed" in result.output


def test_scan_command_force_flag(claude_projects):
    runner.invoke(app, ["scan"])  # first pass
    result = runner.invoke(app, ["scan", "--force"])
    assert result.exit_code == 0
    assert "1 indexed" in result.output


def test_scan_command_source_flag(claude_projects):
    result = runner.invoke(app, ["scan", "--source", "claude"])
    assert result.exit_code == 0


def test_scan_empty_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "none")
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none2")
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "0 indexed" in result.output


# ── search ─────────────────────────────────────────────────────────────────────

def test_search_command_finds_result(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD outliers"])
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0


def test_search_command_no_results(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "zzznomatch9999"])
    assert result.exit_code == 0
    assert "No results" in result.output


def test_search_command_limit_flag(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD", "--limit", "1"])
    assert result.exit_code == 0


def test_search_command_layer_flag(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD", "--layer", "episodic"])
    assert result.exit_code == 0


# ── list ───────────────────────────────────────────────────────────────────────

def test_list_command_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "none")
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none2")
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No sessions" in result.output


def test_list_command_shows_sessions(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "logD" in result.output or "proj" in result.output


def test_list_command_project_filter(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["list", "--project", "nonexistent-project-xyz"])
    assert result.exit_code == 0
    assert "No sessions" in result.output


def test_list_command_limit_flag(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["list", "--limit", "5"])
    assert result.exit_code == 0


# ── stats ──────────────────────────────────────────────────────────────────────

def test_stats_command_empty(tmp_path, monkeypatch):
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Sessions" in result.output
    assert "0" in result.output


def test_stats_command_after_scan(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "1" in result.output  # 1 session indexed


# ── mcp ────────────────────────────────────────────────────────────────────────

def test_mcp_command_is_registered():
    """mcp command exists and is registered — full stdio test requires integration."""
    result = runner.invoke(app, ["--help"])
    assert "mcp" in result.output
