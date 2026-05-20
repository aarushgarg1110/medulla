"""Sprint 2 CLI tests — session-detail command."""
import pytest
from typer.testing import CliRunner

from medulla.cli import app
from medulla.episodic.store import upsert_session, upsert_agent_session
from tests.conftest import claude_user, make_claude_jsonl
from tests.test_store import make_session, make_agent

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    import medulla.config as cfg
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=tmp_path / ".medulla"))
    yield


@pytest.fixture
def seeded_db(tmp_path, monkeypatch):
    """DB with one session and one agent."""
    import medulla.config as cfg
    conn = __import__("medulla.db.database", fromlist=["connect"]).connect(
        cfg.get_config().db_path
    )
    s = make_session("abcdef12-0000-0000-0000-000000000000", project_dir="/proj/mlops",
                     messages=["what were the logD outliers?",
                               "CHEMBL12345 is suspicious at +6.11 delta."])
    upsert_session(conn, s)
    a = make_agent("agent-xyz-001", parent="abcdef12-0000-0000-0000-000000000000")
    upsert_agent_session(conn, a)
    conn.close()
    return "abcdef12-0000-0000-0000-000000000000"


# ── session-detail ─────────────────────────────────────────────────────────────

def test_session_detail_full_id(seeded_db):
    result = runner.invoke(app, ["session-detail", seeded_db])
    assert result.exit_code == 0
    assert "abcdef12" in result.output
    assert "Chunks" in result.output


def test_session_detail_prefix_lookup(seeded_db):
    """8-char prefix should resolve to the full session."""
    result = runner.invoke(app, ["session-detail", "abcdef12"])
    assert result.exit_code == 0
    assert "abcdef12" in result.output


def test_session_detail_shows_chunks(seeded_db):
    result = runner.invoke(app, ["session-detail", seeded_db])
    assert "Chunk 0" in result.output


def test_session_detail_shows_agents(seeded_db):
    result = runner.invoke(app, ["session-detail", seeded_db])
    assert "Subagents" in result.output
    assert "agent-xy" in result.output  # agent_id[:8]


def test_session_detail_not_found(seeded_db):
    result = runner.invoke(app, ["session-detail", "nonexistent-prefix"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_session_detail_prefix_no_match(seeded_db):
    result = runner.invoke(app, ["session-detail", "zzzzzzzz"])
    assert result.exit_code == 1
