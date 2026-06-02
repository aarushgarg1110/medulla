"""Tests for medulla reset command."""
import pytest
from pathlib import Path
from typer.testing import CliRunner
from medulla.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    import medulla.config as cfg
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=tmp_path / ".medulla"))
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "none")
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none2")
    yield


def _seed_wiki(tmp_path):
    """Create fake wiki content."""
    import medulla.config as cfg
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    for subdir in ["sources", "concepts", "entities", "raw"]:
        (wiki / subdir).mkdir()
        (wiki / subdir / "test.md").write_text("# Test")
    (wiki / "index.md").write_text("# Index")
    (wiki / "log.md").write_text("# Log")
    return wiki


def test_reset_clears_wiki_keeps_raw(tmp_path):
    wiki = _seed_wiki(tmp_path)
    result = runner.invoke(app, ["reset", "--yes"])
    assert result.exit_code == 0
    assert not (wiki / "sources").exists()
    assert not (wiki / "concepts").exists()
    assert not (wiki / "entities").exists()
    assert not (wiki / "index.md").exists()
    assert (wiki / "raw").exists()  # raw/ preserved


def test_reset_all_clears_raw_too(tmp_path):
    wiki = _seed_wiki(tmp_path)
    result = runner.invoke(app, ["reset", "--all", "--yes"])
    assert result.exit_code == 0
    assert not (wiki / "raw").exists()


def test_reset_clears_db_wiki_pages(tmp_path):
    from medulla.db.database import connect
    from medulla.semantic.store import upsert_wiki_page
    import medulla.config as cfg
    conn = connect(cfg.get_config().db_path)
    upsert_wiki_page(conn, "test-page", "concept", "Test", "content", Path("/wiki/test.md"))
    assert conn.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()[0] == 1
    runner.invoke(app, ["reset", "--yes"])
    conn2 = connect(cfg.get_config().db_path)
    assert conn2.execute("SELECT COUNT(*) FROM wiki_pages").fetchone()[0] == 0


def test_reset_all_clears_sessions(tmp_path):
    from medulla.db.database import connect
    from medulla.episodic.store import upsert_session
    from tests.test_store import make_session
    import medulla.config as cfg
    conn = connect(cfg.get_config().db_path)
    upsert_session(conn, make_session("s1", messages=["hi"]))
    assert conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
    runner.invoke(app, ["reset", "--all", "--yes"])
    conn2 = connect(cfg.get_config().db_path)
    assert conn2.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0


def test_reset_confirms_without_yes():
    """Without --yes, should prompt (runner provides 'n' → aborts)."""
    result = runner.invoke(app, ["reset"], input="n\n")
    assert result.exit_code != 0 or "Aborted" in result.output or "Continue" in result.output


def test_reset_missing_dirs_ok(tmp_path):
    """Reset on empty wiki should not crash."""
    result = runner.invoke(app, ["reset", "--yes"])
    assert result.exit_code == 0
    assert "Reset complete" in result.output
