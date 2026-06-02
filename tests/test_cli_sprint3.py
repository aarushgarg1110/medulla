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


def test_use_anthropic_warns_no_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    result = runner.invoke(app, ["use", "anthropic"])
    assert result.exit_code == 0
    assert "ANTHROPIC_API_KEY" in result.output


def test_use_anthropic_no_warning_when_key_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    result = runner.invoke(app, ["use", "anthropic"])
    assert result.exit_code == 0
    assert "ANTHROPIC_API_KEY" not in result.output


def test_use_with_model_flag(monkeypatch):
    result = runner.invoke(app, ["use", "anthropic", "--model", "claude-haiku-4-5-20251001"])
    assert result.exit_code == 0
    assert "claude-haiku-4-5-20251001" in result.output


def test_use_switches_to_ollama(monkeypatch):
    import httpx, medulla.cli as cli_mod
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: (_ for _ in ()).throw(Exception("refused")))
    monkeypatch.setattr(cli_mod.subprocess, "run",
        lambda *a, **kw: type("R", (), {"stdout": "", "returncode": 1})())
    result = runner.invoke(app, ["use", "ollama"])
    assert result.exit_code == 0
    assert "ollama" in result.output
    assert "ollama serve" in result.output  # warning shown


def test_use_ollama_server_up_shows_models(monkeypatch):
    import httpx, medulla.cli as cli_mod
    class FakeResp:
        def raise_for_status(self): pass
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: FakeResp())
    monkeypatch.setattr(cli_mod.subprocess, "run", lambda *a, **kw: type("R", (), {
        "stdout": "NAME           ID\nllama3.2:3b    abc123\n", "returncode": 0
    })())
    result = runner.invoke(app, ["use", "ollama"])
    assert result.exit_code == 0
    assert "llama3.2:3b" in result.output


def test_use_ollama_with_model_flag(monkeypatch):
    import httpx
    monkeypatch.setattr(httpx, "get", lambda *a, **kw: (_ for _ in ()).throw(Exception("refused")))
    result = runner.invoke(app, ["use", "ollama", "--model", "mistral:7b"])
    assert result.exit_code == 0
    assert "mistral:7b" in result.output


def test_use_bedrock_with_model_flag():
    result = runner.invoke(app, ["use", "bedrock", "--model", "us.anthropic.claude-haiku-4-5"])
    assert result.exit_code == 0
    assert "us.anthropic.claude-haiku-4-5" in result.output


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


def test_status_shows_raw_section():
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "raw" in result.output.lower() or "intake" in result.output.lower()


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


def test_ingest_with_mock_provider(monkeypatch):
    """Ingest with a working mock provider creates wiki pages and prints success."""
    import medulla.config as cfg
    from tests.test_ingest_pipeline import MockProvider
    monkeypatch.setattr("medulla.llm.get_provider", MockProvider)
    # Put file in raw/ so discover_raw picks it up — avoids sha256 dedup on file path
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    (wiki / "raw").mkdir(exist_ok=True)
    md = wiki / "raw" / "logd-study.md"
    md.write_text("# LogD Study\n\nStudy content about logD.")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    assert "pages" in result.output.lower() or "✓" in result.output


def test_ingest_processes_queued_on_no_args(tmp_path, monkeypatch):
    """medulla ingest with no args processes queued raw/ files."""
    import medulla.config as cfg
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    (wiki / "raw").mkdir()
    md = wiki / "raw" / "queued.md"
    md.write_text("# Queued Paper\n\nContent about logD.")

    from tests.test_ingest_pipeline import MockProvider
    monkeypatch.setattr("medulla.llm.get_provider", MockProvider)
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0


def test_ingest_no_source_processes_raw():
    """medulla ingest with no args is valid — discovers + processes raw/."""
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    # Nothing queued yet — should say so
    assert "Nothing" in result.output or "queued" in result.output or "0" in result.output


def test_ingest_no_source_with_provider_discovers(tmp_path, monkeypatch):
    """With files in raw/, medulla ingest discovers and processes them."""
    import medulla.config as cfg
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    (wiki / "raw").mkdir()
    (wiki / "raw" / "article.md").write_text("# Article\n\nContent about logD.")

    from tests.test_ingest_pipeline import MockProvider
    monkeypatch.setattr("medulla.llm.get_provider", MockProvider)
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0


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


def test_wiki_open_no_wiki(tmp_path, monkeypatch):
    """wiki open fails gracefully when wiki doesn't exist."""
    result = runner.invoke(app, ["wiki", "open"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower() or "ingest" in result.output.lower()


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


def test_ingest_streaming_flag_shows_warning(tmp_path, monkeypatch):
    """--streaming flag shows 4096 token cap warning."""
    from tests.test_ingest_pipeline import MockProvider
    monkeypatch.setattr("medulla.llm.get_provider", MockProvider)
    import medulla.config as cfg
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    (wiki / "raw").mkdir()
    md = wiki / "raw" / "study.md"
    md.write_text("# LogD Study\n\nContent.")
    result = runner.invoke(app, ["ingest", "--streaming"])
    assert result.exit_code == 0
    assert "4096" in result.output or "Streaming" in result.output or "streaming" in result.output.lower()


def test_ingest_default_no_streaming(tmp_path, monkeypatch):
    """Default ingest (no --streaming) does not show streaming warning."""
    from tests.test_ingest_pipeline import MockProvider
    monkeypatch.setattr("medulla.llm.get_provider", MockProvider)
    import medulla.config as cfg
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    (wiki / "raw").mkdir()
    md = wiki / "raw" / "study.md"
    md.write_text("# LogD Study\n\nContent.")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    assert "4096" not in result.output


def test_ingest_broken_wikilinks_shown(tmp_path, monkeypatch):
    """Broken wikilinks in written pages are printed as warnings after ingest."""
    import json
    import medulla.config as cfg

    class BrokenConnectionProvider:
        """Plan includes a connection to a non-existent source — not filtered by
        _filter_wikilinks (which only filters concepts/entities lists), so it
        ends up in the written source page and triggers the wikilink check."""
        @property
        def name(self): return "mock"
        @property
        def model(self): return "mock"
        def generate(self, prompt, system=None, on_token=None):
            if "STAGE: PLAN" in prompt:
                return json.dumps({
                    "source_page": {
                        "title": "Paper", "summary": "S", "key_points": [], "tags": [],
                        "concepts": [], "entities": [],
                        # connection to a source that doesn't exist — passes filter, triggers check
                        "connections": ["[[sources/nonexistent-related-paper]] — related work"],
                        "gaps": [],
                    },
                    "new_concepts": [],
                    "new_entities": [],
                    "update_concepts": [],
                    "update_entities": [],
                })
            return json.dumps({})

    monkeypatch.setattr("medulla.llm.get_provider", BrokenConnectionProvider)
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    (wiki / "raw").mkdir()
    md = wiki / "raw" / "study.md"
    md.write_text("# Study\n\nContent.")
    result = runner.invoke(app, ["ingest"])
    assert result.exit_code == 0
    assert "nonexistent-related-paper" in result.output or "Broken" in result.output
