"""Sprint 3 CLI tests — use, status, ingest, wiki commands."""
import json
import pytest
from typer.testing import CliRunner
from medulla.cli import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    import medulla.config as cfg
    cfg._config = None
    test_cfg = cfg.Config(medulla_dir=tmp_path / ".medulla")
    monkeypatch.setattr(cfg, "_config", test_cfg)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / ".medulla" / "config.toml")
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "none_claude")
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none_kiro")
    yield


# ── medulla use ───────────────────────────────────────────────────────────────

def test_use_switches_to_bedrock():
    result = runner.invoke(app, ["use", "bedrock"])
    assert result.exit_code == 0
    assert "bedrock" in result.output


def test_use_switches_to_anthropic():
    result = runner.invoke(app, ["use", "anthropic"])
    assert result.exit_code == 0
    assert "anthropic" in result.output


def test_use_switches_to_ollama():
    result = runner.invoke(app, ["use", "ollama"])
    assert result.exit_code == 0
    assert "ollama" in result.output


def test_use_invalid_provider():
    result = runner.invoke(app, ["use", "gemini"])
    assert result.exit_code == 1
    assert "Unknown" in result.output or "gemini" in result.output


# ── medulla status ────────────────────────────────────────────────────────────

def test_status_shows_provider():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "bedrock" in result.output or "anthropic" in result.output or "ollama" in result.output


def test_status_shows_wiki_section():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Wiki" in result.output or "wiki" in result.output


def test_status_shows_pending_none():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Pending" in result.output or "pending" in result.output


def test_status_shows_pending_count(tmp_path, monkeypatch):
    import medulla.config as cfg
    conn = __import__("medulla.db.database", fromlist=["connect"]).connect(cfg.get_config().db_path)
    from medulla.semantic.store import queue_pending
    queue_pending(conn, "/some/paper.pdf", "pdf", "My Paper")
    conn.close()
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "1" in result.output


# ── medulla stats (now shows wiki) ───────────────────────────────────────────

def test_stats_shows_semantic_section():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Semantic" in result.output or "wiki" in result.output.lower()


# ── medulla ingest — queue when no provider ───────────────────────────────────

def test_ingest_queues_when_provider_unavailable(tmp_path, monkeypatch):
    """When provider raises, source should be queued with a clear message."""
    def raise_env_error():
        raise EnvironmentError("No API key set")

    monkeypatch.setattr("medulla.llm.get_provider", raise_env_error)
    md = tmp_path / "paper.md"
    md.write_text("# Test\nContent.")
    result = runner.invoke(app, ["ingest", str(md)])
    assert result.exit_code == 0
    assert "queued" in result.output.lower() or "process-pending" in result.output or "provider" in result.output.lower()


def test_ingest_with_mock_provider(tmp_path, monkeypatch):
    """Ingest with a working mock provider creates wiki pages."""
    import json
    from tests.test_ingest_pipeline import MockProvider
    monkeypatch.setattr("medulla.llm.get_provider", MockProvider)
    md = tmp_path / "study.md"
    md.write_text("# LogD Study\n\nStudy content about logD.")
    result = runner.invoke(app, ["ingest", str(md)])
    assert result.exit_code == 0
    assert "Ingested" in result.output or "pages" in result.output.lower()


def test_ingest_process_pending_with_mock(tmp_path, monkeypatch):
    """--process-pending processes queued sources."""
    import medulla.config as cfg
    conn = __import__("medulla.db.database", fromlist=["connect"]).connect(cfg.get_config().db_path)
    from medulla.semantic.store import queue_pending
    md = tmp_path / "queued.md"
    md.write_text("# Queued Paper\n\nContent.")
    queue_pending(conn, str(md), "markdown", "Queued Paper")
    conn.close()

    from tests.test_ingest_pipeline import MockProvider
    monkeypatch.setattr("medulla.llm.get_provider", MockProvider)
    result = runner.invoke(app, ["ingest", "--process-pending"])
    assert result.exit_code == 0


def test_ingest_no_source_no_flag():
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 1


def test_ingest_process_pending_none():
    result = runner.invoke(app, ["ingest", "--process-pending"])
    assert result.exit_code == 0
    assert "No pending" in result.output


# ── medulla wiki list ─────────────────────────────────────────────────────────

def test_wiki_list_empty():
    result = runner.invoke(app, ["wiki", "list"])
    assert result.exit_code == 0
    assert "No wiki pages" in result.output or "medulla ingest" in result.output


def test_wiki_list_with_pages(tmp_path, monkeypatch):
    import medulla.config as cfg
    conn = __import__("medulla.db.database", fromlist=["connect"]).connect(cfg.get_config().db_path)
    from medulla.semantic.store import upsert_wiki_page
    upsert_wiki_page(conn, "logd-pred", "concept", "LogD Prediction",
                     "content", tmp_path / "logd.md")
    conn.close()
    result = runner.invoke(app, ["wiki", "list"])
    assert result.exit_code == 0
    assert "logd-pred" in result.output or "LogD" in result.output


def test_wiki_list_type_filter(tmp_path, monkeypatch):
    import medulla.config as cfg
    conn = __import__("medulla.db.database", fromlist=["connect"]).connect(cfg.get_config().db_path)
    from medulla.semantic.store import upsert_wiki_page
    upsert_wiki_page(conn, "src-1", "source", "Source 1", "content", tmp_path / "s.md")
    upsert_wiki_page(conn, "con-1", "concept", "Concept 1", "content", tmp_path / "c.md")
    conn.close()
    result = runner.invoke(app, ["wiki", "list", "--type", "source"])
    assert result.exit_code == 0
    assert "src-1" in result.output
    assert "con-1" not in result.output


# ── medulla wiki lint ─────────────────────────────────────────────────────────

def test_wiki_lint_no_wiki():
    result = runner.invoke(app, ["wiki", "lint"])
    assert result.exit_code == 0
    assert "does not exist" in result.output or "ingest" in result.output


def test_wiki_lint_clean(tmp_path, monkeypatch):
    import medulla.config as cfg
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    (wiki / "concepts").mkdir()
    (wiki / "concepts" / "logd.md").write_text("# LogD\n\nSee [[admet]].\n")
    (wiki / "concepts" / "admet.md").write_text("# ADMET\n\nSee [[logd]].\n")
    result = runner.invoke(app, ["wiki", "lint"])
    assert result.exit_code == 0
    assert "2 pages" in result.output or "pages" in result.output
