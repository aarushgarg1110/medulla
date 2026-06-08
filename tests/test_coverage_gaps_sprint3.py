"""Targeted tests for Sprint 3 coverage gaps."""
import pytest
from pathlib import Path
from typer.testing import CliRunner
from medulla.cli import app

runner = CliRunner()


class _MockEmbedProvider:
    dimension = 768
    model_name = "mock"
    def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    import medulla.config as cfg
    import medulla.episodic.scanner as scanner_mod
    import medulla.semantic.ingest as ingest_mod
    cfg._config = None
    test_cfg = cfg.Config(medulla_dir=tmp_path / ".medulla")
    monkeypatch.setattr(cfg, "_config", test_cfg)
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / ".medulla" / "config.toml")
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "none")
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none2")
    monkeypatch.setattr(scanner_mod, "_get_embedding_provider", lambda: _MockEmbedProvider())
    monkeypatch.setattr(ingest_mod, "_get_embedding_provider", lambda: _MockEmbedProvider())
    yield


# ── config.py — TOML parse paths (73-77, 91-92) ──────────────────────────────

def test_config_loads_from_toml(tmp_path, monkeypatch):
    """Exercise the TOML loading path with a real config file."""
    import tomli_w
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir(parents=True)
    (medulla_dir / "config.toml").write_bytes(tomli_w.dumps({
        "llm": {
            "active": "ollama",
            "bedrock": {"model": "b-model", "aws_profile": "test", "aws_region": "eu-west-1"},
            "anthropic": {"model": "a-model"},
            "ollama": {"model": "o-model", "host": "http://localhost:9999"},
        }
    }).encode())
    monkeypatch.setenv("MEDULLA_DIR", str(medulla_dir))
    cfg._config = None
    loaded = cfg.get_config()
    assert loaded.llm.active == "ollama"
    assert loaded.llm.bedrock.model == "b-model"
    assert loaded.llm.anthropic.model == "a-model"
    assert loaded.llm.ollama.model == "o-model"
    assert loaded.llm.ollama.host == "http://localhost:9999"


def test_save_config_writes_toml(tmp_path, monkeypatch):
    import medulla.config as cfg
    cfg._config = None
    monkeypatch.setattr(cfg, "CONFIG_FILE", tmp_path / ".medulla" / "config.toml")
    c = cfg.Config(medulla_dir=tmp_path / ".medulla")
    c.llm.active = "anthropic"
    cfg._config = c
    cfg.save_config(c)
    assert (tmp_path / ".medulla" / "config.toml").exists()


# ── llm.py — check_provider (132-140) ─────────────────────────────────────────

def test_check_provider_success(monkeypatch):
    from medulla.llm import OllamaProvider
    import medulla.llm as llm_mod

    class QuickProvider(OllamaProvider):
        def generate(self, prompt, system=None, on_token=None): return "ok"

    monkeypatch.setattr(llm_mod, "get_provider", lambda: QuickProvider("m", "http://h"))
    from medulla.llm import check_provider
    ok, msg = check_provider()
    assert ok
    assert "reachable" in msg


def test_check_provider_failure(monkeypatch):
    import medulla.llm as llm_mod
    monkeypatch.setattr(llm_mod, "get_provider", lambda: (_ for _ in ()).throw(Exception("conn refused")))
    from medulla.llm import check_provider
    ok, msg = check_provider()
    assert not ok
    assert "conn refused" in msg


# ── cli.py — stats/list/session-detail branches ───────────────────────────────

def test_stats_shows_episodic_and_semantic():
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Episodic" in result.output
    assert "Semantic" in result.output


def test_stats_shows_wiki_type_breakdown_and_tools(tmp_path, monkeypatch):
    """Cover stats wiki type breakdown (line 124) and top tools (127-129)."""
    import medulla.config as cfg
    conn = __import__("medulla.db.database", fromlist=["connect"]).connect(cfg.get_config().db_path)
    # Insert a wiki page so by_type is non-empty
    from medulla.semantic.store import upsert_wiki_page
    upsert_wiki_page(conn, "p1", "concept", "Test", "content", tmp_path / "p.md")
    # Insert a session so top_tools is non-empty
    from tests.test_store import make_session
    from medulla.episodic.store import upsert_session
    upsert_session(conn, make_session("s1", messages=["test"]))
    conn.close()
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "concept" in result.output  # wiki type breakdown
    assert "Bash" in result.output or "Tool" in result.output  # top tools


def test_wiki_lint_shows_broken_and_orphaned(tmp_path, monkeypatch):
    """Cover wiki lint broken links (382-384) and orphaned pages (389-391)."""
    import medulla.config as cfg
    wiki = cfg.get_config().wiki_path
    wiki.mkdir(parents=True)
    (wiki / "concepts").mkdir()
    # Broken link: logd links to nonexistent
    (wiki / "concepts" / "logd.md").write_text("# LogD\n\nSee [[nonexistent-page]].\n")
    # Orphan: nobody links to admet
    (wiki / "concepts" / "admet.md").write_text("# ADMET\n\nNo incoming links.\n")
    result = runner.invoke(app, ["wiki", "lint"])
    assert result.exit_code == 0
    assert "nonexistent" in result.output or "Broken" in result.output
    assert "admet" in result.output or "Orphan" in result.output


def test_session_detail_project_dir_shown(tmp_path, monkeypatch):
    """session-detail with a session that has agents covers the agent display path."""
    import medulla.config as cfg
    conn = __import__("medulla.db.database", fromlist=["connect"]).connect(cfg.get_config().db_path)
    from tests.test_store import make_session, make_agent
    from medulla.episodic.store import upsert_session, upsert_agent_session
    s = make_session("abc12345-0000-0000-0000-000000000000", messages=["logD analysis"])
    upsert_session(conn, s)
    a = make_agent("agent-abc001", parent="abc12345-0000-0000-0000-000000000000")
    upsert_agent_session(conn, a)
    conn.close()
    result = runner.invoke(app, ["session-detail", "abc12345"])
    assert result.exit_code == 0
    assert "Subagents" in result.output


# ── ingest.py — ingest_text edge cases (149, 157-159, 170-171) ───────────────

def test_ingest_text_indexes_to_db(tmp_path):
    """store_wiki_page is the replacement for ingest_text (pure storage)."""
    from tests.test_ingest_pipeline import MockProvider
    from medulla.db.database import connect
    from medulla.semantic.ingest import store_wiki_page
    from medulla.semantic.store import get_wiki_page
    conn = connect(tmp_path / "test.db")
    wiki = tmp_path / "wiki"
    result = store_wiki_page(conn, wiki, "Batch Effect Notes",
                              "---\ntitle: Batch Effect Notes\n---\n\n## Summary\n\nLogD batch effects.")
    assert result["slug"] == "batch-effect-notes"
    page = get_wiki_page(conn, "batch-effect-notes")
    assert page is not None


def test_ingest_pdf_source(tmp_path):
    """Test PDF intake + process via new flow."""
    pytest.importorskip("fitz")
    import fitz
    from tests.test_ingest_pipeline import MockProvider
    from medulla.db.database import connect
    from medulla.semantic.ingest import intake_to_raw, process_pending

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "LogD study on ADMET properties.")
    pdf_path = tmp_path / "study.pdf"
    doc.save(str(pdf_path))
    doc.close()

    conn = connect(tmp_path / "test.db")
    wiki = tmp_path / "wiki"
    intake_to_raw(conn, wiki, str(pdf_path))
    results = process_pending(wiki, conn, MockProvider())
    assert len(results) == 1
    assert results[0]["total_pages"] >= 1


# ── scanner.py — error/skipped-mtime/agent-skipped branches ─────────────────

def test_scan_errors_incremented_on_process_exception(tmp_path, monkeypatch):
    """session processing exception increments errors counter (lines 39-40)."""
    from medulla.episodic.scanner import scan
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")
    # Write a non-empty JSONL that will fail parse as a valid session
    proj = tmp_path / "proj"
    proj.mkdir()
    bad = proj / "bad-session.jsonl"
    bad.write_text("not valid json at all!!!\n")
    # Patch _process_session to raise
    import medulla.episodic.scanner as scanner_mod
    monkeypatch.setattr(scanner_mod, "_process_session", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    from medulla.db.database import connect
    conn = connect(tmp_path / "t.db")
    counts = scan(conn)
    assert counts["errors"] >= 1


def test_scan_agent_skipped_mtime(tmp_path, monkeypatch):
    """Agent skipped_mtime branch (lines 47-48)."""
    from tests.conftest import claude_user, make_claude_jsonl
    from medulla.episodic.scanner import scan
    from medulla.db.database import connect

    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    # Write parent session + agent
    parent_id = "parent-aaa"
    proj = tmp_path / "proj"
    (proj / parent_id / "subagents").mkdir(parents=True)
    (proj / f"{parent_id}.jsonl").write_text(make_claude_jsonl([claude_user("task", session_id=parent_id)]))
    agent_path = proj / parent_id / "subagents" / "agent-agt1.jsonl"
    agent_path.write_text(make_claude_jsonl([claude_user("agent task", session_id=parent_id)]))

    conn = connect(tmp_path / "t.db")
    counts1 = scan(conn)
    assert counts1["agents_indexed"] == 1

    counts2 = scan(conn)
    assert counts2["agents_indexed"] == 0  # skipped (mtime unchanged)


def test_scan_empty_session_increments_empty(tmp_path, monkeypatch):
    """Empty/stub session increments empty counter (lines 37-38)."""
    from medulla.episodic.scanner import scan
    from medulla.db.database import connect

    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")

    proj = tmp_path / "proj"
    proj.mkdir()
    # Write a JSONL with only assistant messages (no user turns → parse_session returns None)
    stub = proj / "stub-sess.jsonl"
    stub.write_text('{"type":"message","role":"assistant","content":[{"type":"text","text":"hi"}]}\n')

    conn = connect(tmp_path / "t.db")
    counts = scan(conn)
    assert counts["empty"] >= 1


def test_scan_agent_none_parse_returns_skipped(tmp_path, monkeypatch):
    """Agent parse returning None → 'skipped' (line 109)."""
    from medulla.episodic.scanner import scan
    from medulla.db.database import connect
    import medulla.episodic.scanner as scanner_mod

    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none")
    monkeypatch.setattr(scanner_mod, "parse_agent_session", lambda _: None)

    from tests.conftest import claude_user, make_claude_jsonl
    parent_id = "par-bbb"
    proj = tmp_path / "proj"
    (proj / parent_id / "subagents").mkdir(parents=True)
    (proj / f"{parent_id}.jsonl").write_text(make_claude_jsonl([claude_user("task", session_id=parent_id)]))
    (proj / parent_id / "subagents" / "agent-x.jsonl").write_text(make_claude_jsonl([claude_user("t", session_id=parent_id)]))

    conn = connect(tmp_path / "t.db")
    counts = scan(conn)
    # agent was discovered but parse returned None → no crash, agents_indexed stays 0
    assert counts["agents_indexed"] == 0


# ── url.py — trafilatura fallback + _extract_title missing (lines 46, 55-57) ─

def test_url_extract_text_falls_back_without_trafilatura(monkeypatch):
    """_extract_text falls back to HTML stripping when trafilatura unavailable (lines 55-57)."""
    import sys
    import medulla.semantic.sources.url as url_mod
    # Remove trafilatura from sys.modules to simulate it being absent
    monkeypatch.setitem(sys.modules, "trafilatura", None)
    html = "<html><body><p>Hello world content here.</p></body></html>"
    text = url_mod._extract_text(html)
    assert "Hello world" in text


def test_url_extract_title_fallback_returns_untitled(monkeypatch):
    """_extract_title returns 'Untitled' when no <title> tag (line 46)."""
    import medulla.semantic.sources.url as url_mod
    title = url_mod._extract_title("<html><body><p>No title here.</p></body></html>")
    assert title == "Untitled"


# ── ingest.py — empty slug in update pathway + parse fallback ─────────────────

def test_update_concepts_skips_empty_slug(db, tmp_path):
    """update_concepts entry with empty slug is silently skipped (lines 332-334)."""
    import json
    from medulla.semantic.ingest import intake_to_raw, process_pending

    class EmptySlugProvider:
        @property
        def name(self): return "mock"
        @property
        def model(self): return "mock"
        def generate(self, prompt, system=None, on_token=None):
            return json.dumps({
                "source_page": {"title": "T", "summary": "S", "key_points": [],
                                "tags": [], "concepts": [], "entities": [], "connections": [], "gaps": []},
                "new_concepts": [],
                "new_entities": [],
                "update_concepts": [{"slug": "", "add_source_note": "note"}],  # empty slug
                "update_entities": [{"slug": "  ", "add_source_note": "note"}],  # whitespace slug
            })

    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# Test\n\nContent.")
    intake_to_raw(db, wiki, str(md))
    results = process_pending(wiki, db, EmptySlugProvider())
    assert len(results) == 1
    assert results[0].get("error") is None


def test_parse_llm_response_returns_concept_entity_fallback():
    """_parse_llm_response fallback includes concept_pages/entity_pages keys (lines 484-485)."""
    from medulla.semantic.ingest import _parse_llm_response
    result = _parse_llm_response("totally invalid!!!")
    assert "concept_pages" in result
    assert "entity_pages" in result


# ── config.py — bad TOML is silently ignored (line 98-99) ─────────────────────

def test_config_bad_toml_silently_ignored(tmp_path, monkeypatch):
    """Malformed TOML triggers except Exception: pass — defaults are kept (lines 98-99)."""
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    (medulla_dir / "config.toml").write_text("this is not valid toml !!!! %%%")
    monkeypatch.setenv("MEDULLA_DIR", str(medulla_dir))
    cfg._config = None
    loaded = cfg.get_config()
    # Defaults intact — bad TOML didn't crash
    assert loaded.llm.active == "bedrock"
    cfg._config = None
