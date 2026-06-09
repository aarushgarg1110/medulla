"""Tests for MCP tool logic — tests the dispatch functions directly, no stdio protocol."""
import pytest

from medulla.episodic.store import upsert_session, upsert_agent_session
from medulla.mcp import (  # noqa: E402
    _tool_search,
    _tool_session_detail,
    _tool_session_tree,
    _tool_project_context,
    _tool_list,
    _tool_stats,
    _tool_events_search,
    _tool_analyze,
    _dispatch,
)
from tests.test_store import make_session, make_agent


class _MockEmbedProvider:
    dimension = 768
    model_name = "mock"
    def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Ensure MCP tool tests never write to real ~/.medulla wiki dir."""
    import medulla.config as cfg
    import medulla.semantic.ingest as ingest_mod
    cfg._config = None
    test_cfg = cfg.Config(medulla_dir=tmp_path / ".medulla")
    monkeypatch.setattr(cfg, "_config", test_cfg)
    monkeypatch.setattr(ingest_mod, "_get_embedding_provider", lambda: _MockEmbedProvider())
    yield
    cfg._config = None


def _setup(db):
    """Insert one session with agents for most tests."""
    s = make_session("sess-mcp-001", project_dir="/proj/mlops", messages=[
        "what were the logD outliers in CompoundX?",
        "CHEMBL12345 has delta logD of +6.11, four sigma above batch mean.",
        "can you look at pKa next?",
        "The pKa prediction for CHEMBL12345 is 9.35 basic.",
    ])
    upsert_session(db, s)
    a = make_agent("agent-mcp-001", parent="sess-mcp-001")
    upsert_agent_session(db, a)
    return db


# ── medulla_search ─────────────────────────────────────────────────────────────

def test_tool_search_finds_results(db):
    _setup(db)
    result = _tool_search(db, {"query": "logD outliers"})
    assert "sess-mcp" in result
    assert "result(s)" in result


def test_tool_search_no_results(db):
    _setup(db)
    result = _tool_search(db, {"query": "zzznomatch99999"})
    assert "No results" in result


def test_tool_search_empty_query(db):
    result = _tool_search(db, {"query": ""})
    assert "Error" in result


def test_tool_search_respects_limit(db):
    for i in range(5):
        upsert_session(db, make_session(f"sess-{i}", messages=["logD compoundx batch outlier"]))
    result = _tool_search(db, {"query": "logD", "limit": 2})
    assert "2 result(s)" in result


def test_tool_search_assistant_content_findable(db):
    """CHEMBL12345 appears in assistant text — must be searchable (Sprint 1.5 fix)."""
    _setup(db)
    result = _tool_search(db, {"query": "CHEMBL12345"})
    assert "sess-mcp" in result


def test_tool_search_chunk_result_includes_chunk_index_hint(db):
    """MCP output for chunk results must include chunk_index so model jumps directly."""
    s = make_session("sess-mcp-ci", messages=["mcp-chunk-index-term content"] * 25)
    upsert_session(db, s)
    result = _tool_search(db, {"query": "mcp-chunk-index-term"})
    assert "chunk_index=" in result
    assert "medulla_session_detail" in result


def test_tool_search_wiki_result_no_chunk_index_hint(db):
    """Wiki results surfaced via medulla_search must not show a chunk_index hint."""
    from medulla.semantic.store import upsert_wiki_page
    import pathlib
    upsert_wiki_page(db, "mcp-ci-wiki", "concept", "MCP CI Wiki",
                     "mcp-no-chunk-hint-term definition here.",
                     pathlib.Path("/wiki/concepts/mcp-ci-wiki.md"))
    result = _tool_search(db, {"query": "mcp-no-chunk-hint-term", "layer": "semantic"})
    assert "mcp-ci-w" in result  # slug truncated to 8 chars in output
    assert "chunk_index=" not in result


# ── medulla_session_detail ─────────────────────────────────────────────────────

def test_tool_session_detail_found(db):
    _setup(db)
    result = _tool_session_detail(db, {"session_id": "sess-mcp-001"})
    assert "sess-mcp-001" in result
    assert "Chunk" in result
    assert "/proj/mlops" in result


def test_tool_session_detail_shows_agents(db):
    _setup(db)
    result = _tool_session_detail(db, {"session_id": "sess-mcp-001"})
    assert "Subagents" in result
    assert "agent-mc" in result  # agent_id[:8] = "agent-mc"


def test_tool_session_detail_prefix_lookup(db):
    """Claude passes 8-char IDs from medulla_list — must resolve to full UUID."""
    _setup(db)
    result = _tool_session_detail(db, {"session_id": "sess-mcp"})  # 8-char prefix
    assert "sess-mcp-001" in result


def test_tool_session_detail_specific_chunk(db):
    """chunk_index param returns that chunk in full with no truncation."""
    _setup(db)
    result = _tool_session_detail(db, {"session_id": "sess-mcp-001", "chunk_index": 0})
    assert "Chunk 0" in result
    assert "Next:" in result or "End of session" in result


def test_tool_session_detail_chunk_out_of_range(db):
    _setup(db)
    result = _tool_session_detail(db, {"session_id": "sess-mcp-001", "chunk_index": 9999})
    assert "not found" in result.lower() or "chunks" in result


def test_tool_session_detail_overview_shows_chunk_hint(db):
    """Overview should tell Claude how to fetch specific chunks."""
    _setup(db)
    result = _tool_session_detail(db, {"session_id": "sess-mcp-001"})
    assert "chunk_index" in result


def test_tool_session_detail_prefix_not_found(db):
    result = _tool_session_detail(db, {"session_id": "zzzznothere"})
    assert "not found" in result.lower()


def test_tool_session_detail_not_found(db):
    result = _tool_session_detail(db, {"session_id": "nonexistent-session-full-uuid-here"})
    assert "not found" in result.lower()


def test_tool_session_detail_empty_id(db):
    result = _tool_session_detail(db, {"session_id": ""})
    assert "Error" in result


# ── medulla_session_tree ───────────────────────────────────────────────────────

def test_tool_session_tree_found(db):
    _setup(db)
    result = _tool_session_tree(db, {"session_id": "sess-mcp-001"})
    assert "sess-mcp-001" in result
    assert "agent-mc" in result  # agent_id[:8] = "agent-mc"


def test_tool_session_tree_prefix_lookup(db):
    """Prefix resolution for session_tree — same bug as session_detail."""
    _setup(db)
    result = _tool_session_tree(db, {"session_id": "sess-mcp"})
    assert "sess-mcp-001" in result


def test_tool_session_tree_prefix_not_found(db):
    result = _tool_session_tree(db, {"session_id": "zzzznothere"})
    assert "not found" in result.lower()


def test_tool_session_tree_no_agents(db):
    upsert_session(db, make_session("lone-sess", messages=["hello"]))
    result = _tool_session_tree(db, {"session_id": "lone-sess"})
    assert "No subagents" in result


def test_tool_session_tree_not_found(db):
    result = _tool_session_tree(db, {"session_id": "gone"})
    assert "not found" in result.lower()


def test_tool_session_tree_empty_id(db):
    result = _tool_session_tree(db, {"session_id": ""})
    assert "Error" in result


# ── medulla_project_context ────────────────────────────────────────────────────

def test_tool_project_context_found(db):
    _setup(db)
    result = _tool_project_context(db, {"project": "mlops"})
    assert "mlops" in result
    assert "sess-mcp" in result


def test_tool_project_context_no_match(db):
    result = _tool_project_context(db, {"project": "nonexistent-proj-xyz"})
    assert "No sessions" in result


def test_tool_project_context_uses_cwd_default(db, monkeypatch):
    _setup(db)
    monkeypatch.setenv("PWD", "/proj/mlops")
    # Should not raise even without explicit project
    result = _tool_project_context(db, {})
    assert isinstance(result, str)


# ── medulla_list ───────────────────────────────────────────────────────────────

def test_tool_list_returns_sessions(db):
    _setup(db)
    result = _tool_list(db, {})
    assert "sess-mcp" in result
    assert "session(s)" in result


def test_tool_list_empty(db):
    result = _tool_list(db, {})
    assert "No sessions" in result


def test_tool_list_respects_limit(db):
    for i in range(5):
        upsert_session(db, make_session(f"sess-list-{i}", messages=["test"]))
    result = _tool_list(db, {"limit": 2})
    assert "2 session(s)" in result


# ── medulla_stats ──────────────────────────────────────────────────────────────

def test_tool_stats_empty_db(db):
    result = _tool_stats(db)
    assert "Sessions:" in result
    assert "0" in result


def test_tool_stats_with_data(db):
    _setup(db)
    result = _tool_stats(db)
    assert "Sessions:" in result
    assert "Chunks:" in result


# ── medulla_events_search ──────────────────────────────────────────────────────

def test_tool_events_search_empty_db(db):
    result = _tool_events_search(db, {"query": "logD"})
    assert "No tool events" in result


def test_tool_events_search_empty_query(db):
    result = _tool_events_search(db, {"query": ""})
    assert "Error" in result


def test_tool_events_search_with_data(db):
    db.execute("""
        INSERT INTO tool_events(session_id, project_dir, tool, command, event_ts, event_hash, ingested_at)
        VALUES (?,?,?,?,?,?,datetime('now'))
    """, ("sess-1", "/proj/a", "Bash", "medulla search logD", "2026-01-01T10:00:00Z", "hash-1"))
    db.commit()
    result = _tool_events_search(db, {"query": "logD"})
    assert "Bash" in result or "logD" in result


# ── medulla_analyze ────────────────────────────────────────────────────────────

def test_tool_analyze_empty_db(db):
    result = _tool_analyze(db, {})
    assert "No tool events" in result
    assert "PostToolUse" in result


def test_tool_analyze_with_events(db):
    db.execute("""
        INSERT INTO tool_events(session_id, project_dir, tool, command, event_ts, event_hash, ingested_at)
        VALUES (?,?,?,?,?,?,datetime('now'))
    """, ("sess-1", "/proj/a", "Bash", "echo hi", "2026-01-01T10:00:00Z", "hash-2"))
    db.commit()
    result = _tool_analyze(db, {})
    assert "tool_events count: 1" in result


# ── wiki stubs ─────────────────────────────────────────────────────────────────

def test_tool_wiki_search_empty(db):
    from medulla.mcp import _HANDLERS
    result = _HANDLERS["medulla_wiki_search"](db, {"query": "logD"})
    assert "No wiki pages" in result or "result" in result


def test_tool_wiki_search_finds_page(db):
    import sqlite3
    from pathlib import Path
    from medulla.mcp import _HANDLERS
    from medulla.semantic.store import upsert_wiki_page
    upsert_wiki_page(db, "logd-pred", "concept", "LogD Prediction",
                     "LogD is a key ADMET property measuring lipophilicity.",
                     Path("/wiki/concepts/logd-pred.md"))
    result = _HANDLERS["medulla_wiki_search"](db, {"query": "lipophilicity"})
    assert "logd-pred" in result or "LogD" in result


def test_tool_wiki_search_missing_query(db):
    from medulla.mcp import _HANDLERS
    result = _HANDLERS["medulla_wiki_search"](db, {})
    assert "Error" in result


def test_tool_wiki_page_found(db):
    from medulla.mcp import _HANDLERS
    from medulla.semantic.store import upsert_wiki_page
    from pathlib import Path
    upsert_wiki_page(db, "logd-pred", "concept", "LogD Prediction",
                     "# LogD\n\nKey ADMET property.", Path("/wiki/concepts/logd.md"))
    result = _HANDLERS["medulla_wiki_page"](db, {"slug": "logd-pred"})
    assert "LogD Prediction" in result
    assert "ADMET" in result


def test_tool_wiki_page_not_found(db):
    from medulla.mcp import _HANDLERS
    result = _HANDLERS["medulla_wiki_page"](db, {"slug": "nonexistent-page"})
    assert "not found" in result.lower()


def test_tool_wiki_page_missing_slug(db):
    from medulla.mcp import _HANDLERS
    result = _HANDLERS["medulla_wiki_page"](db, {})
    assert "Error" in result


def test_tool_ingest_mcp_stores_directly(db, tmp_path, monkeypatch):
    """medulla_ingest MCP tool is pure storage — no LLM call needed."""
    import medulla.config as cfg
    cfg.get_config().medulla_dir.mkdir(parents=True, exist_ok=True)
    from medulla.mcp import _HANDLERS
    content = "---\ntitle: Test Finding\n---\n\n## Summary\n\nLogD batch effects observed."
    result = _HANDLERS["medulla_ingest"](db, {"title": "Test Finding", "content": content})
    assert "Stored" in result or "test-finding" in result


def test_tool_ingest_with_source_path_copies_to_raw(db, tmp_path, monkeypatch):
    """source_path copies local file to wiki/raw/."""
    import medulla.config as cfg
    cfg.get_config().medulla_dir.mkdir(parents=True, exist_ok=True)
    # Create a fake PDF file
    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"fake pdf content")
    content = "---\ntitle: Paper\n---\n\n## Summary\n\nContent."
    from medulla.mcp import _HANDLERS
    result = _HANDLERS["medulla_ingest"](db, {
        "title": "Paper", "content": content,
        "source_path": str(fake_pdf)
    })
    assert "Stored" in result
    assert "raw/" in result or "PDF copied" in result
    # Verify file was copied
    wiki_raw = cfg.get_config().wiki_path / "raw" / "paper.pdf"
    assert wiki_raw.exists()


def test_tool_ingest_mcp_with_source_url(db, tmp_path, monkeypatch):
    """medulla_ingest with source_url creates raw/ backtrace file."""
    import medulla.config as cfg
    cfg.get_config().medulla_dir.mkdir(parents=True, exist_ok=True)
    from medulla.mcp import _HANDLERS
    content = "---\ntitle: Paper\n---\n\n## Summary\n\nContent."
    result = _HANDLERS["medulla_ingest"](db, {
        "title": "Paper", "content": content,
        "source_url": "https://example.com/paper"
    })
    assert "Stored" in result or "paper" in result


def test_tool_ingest_url_with_mock(db, tmp_path, monkeypatch):
    """medulla_ingest_url fetches URL and uses configured LLM."""
    from tests.test_ingest_pipeline import MockProvider
    monkeypatch.setattr("medulla.llm.get_provider", MockProvider)
    import medulla.config as cfg
    cfg.get_config().medulla_dir.mkdir(parents=True, exist_ok=True)

    class MockResponse:
        text = "<html><head><title>LogD Study</title></head><body><p>LogD batch effect analysis.</p></body></html>"
        def raise_for_status(self): pass

    monkeypatch.setattr("httpx.get", lambda url, **kw: MockResponse())
    from medulla.mcp import _HANDLERS
    result = _HANDLERS["medulla_ingest_url"](db, {"url": "https://example.com/logd"})
    assert "Ingested" in result or "raw" in result.lower()


def test_tool_ingest_url_missing_url(db):
    from medulla.mcp import _HANDLERS
    result = _HANDLERS["medulla_ingest_url"](db, {})
    assert "Error" in result


def test_tool_ingest_missing_fields(db):
    from medulla.mcp import _HANDLERS
    result = _HANDLERS["medulla_ingest"](db, {"title": "Only title"})
    assert "Error" in result


def _make_cfg(tmp_path):
    import medulla.config as cfg
    c = cfg.Config(medulla_dir=tmp_path / ".medulla")
    return c


def test_tool_wiki_schema_empty(db, tmp_path, monkeypatch):
    c = _make_cfg(tmp_path)
    c.wiki_path.mkdir(parents=True)
    import medulla.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_config", lambda: c)
    from medulla.mcp import _tool_wiki_schema
    result = _tool_wiki_schema(db, {})
    assert "No existing pages" in result or "first ingest" in result


def test_tool_wiki_schema_with_pages(db, tmp_path, monkeypatch):
    c = _make_cfg(tmp_path)
    c.wiki_path.mkdir(parents=True)
    (c.wiki_path / "concepts").mkdir()
    (c.wiki_path / "concepts" / "multi-head-attention.md").write_text(
        "---\ntitle: Multi-Head Attention\n---\nContent."
    )
    import medulla.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_config", lambda: c)
    from medulla.mcp import _tool_wiki_schema
    result = _tool_wiki_schema(db, {})
    assert "multi-head-attention" in result
    assert "concepts/" in result


def test_tool_list_raw_empty(db, tmp_path, monkeypatch):
    c = _make_cfg(tmp_path)
    import medulla.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_config", lambda: c)
    from medulla.mcp import _tool_list_raw
    result = _tool_list_raw(db, {})
    assert "empty" in result.lower() or "no files" in result.lower()


def test_tool_list_raw_with_files(db, tmp_path, monkeypatch):
    c = _make_cfg(tmp_path)
    raw = c.wiki_path / "raw"
    raw.mkdir(parents=True)
    (raw / "paper.md").write_text("# Paper\nContent.")
    import medulla.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_config", lambda: c)
    from medulla.mcp import _tool_list_raw
    result = _tool_list_raw(db, {})
    assert "paper.md" in result


def test_tool_list_raw_shows_queued_vs_processed(db, tmp_path, monkeypatch):
    from medulla.semantic.store import queue_pending
    c = _make_cfg(tmp_path)
    raw = c.wiki_path / "raw"
    raw.mkdir(parents=True)
    paper = raw / "unprocessed.md"
    paper.write_text("# Unprocessed\nContent.")
    queue_pending(db, str(paper), "md", "Unprocessed")
    import medulla.config as cfg_mod
    monkeypatch.setattr(cfg_mod, "get_config", lambda: c)
    from medulla.mcp import _tool_list_raw
    result = _tool_list_raw(db, {})
    assert "queued" in result.lower() or "⏳" in result


# ── project context with events ───────────────────────────────────────────────

def test_tool_project_context_shows_events(db):
    """Cover the events display branch in _tool_project_context."""
    _setup(db)
    db.execute("""
        INSERT INTO tool_events(session_id, project_dir, tool, command, event_ts, event_hash, ingested_at)
        VALUES (?,?,?,?,?,?,datetime('now'))
    """, ("sess-mcp-001", "/proj/mlops", "Bash", "medulla scan", "2026-01-01T10:00:00Z", "h-ctx"))
    db.commit()
    result = _tool_project_context(db, {"project": "mlops"})
    assert "Bash" in result or "medulla scan" in result


# ── events search with output_preview ─────────────────────────────────────────

def test_tool_events_search_shows_output_preview(db):
    """Cover the output_preview display branch."""
    db.execute("""
        INSERT INTO tool_events(session_id, project_dir, tool, command, output_preview, event_ts, event_hash, ingested_at)
        VALUES (?,?,?,?,?,?,?,datetime('now'))
    """, ("s1", "/proj", "Bash", "medulla search logD", "found 3 results", "2026-01-01T10:00:00Z", "h-prev"))
    db.commit()
    result = _tool_events_search(db, {"query": "logD"})
    assert "found 3 results" in result or "Bash" in result


# ── search result formatting (excerpt display) ─────────────────────────────────

def test_tool_search_formats_excerpt(db):
    """Cover the excerpt/date/proj formatting lines in _tool_search."""
    s = make_session("fmt-sess", project_dir="/proj/mlops",
                     messages=["logD batch analysis result measurement"])
    upsert_session(db, s)
    result = _tool_search(db, {"query": "logD batch"})
    assert "mlops" in result
    assert "fmt-ses" in result  # session_id[:8]


# ── medulla_reindex_edges ─────────────────────────────────────────────────────

def test_tool_reindex_edges_empty_wiki(db):
    """reindex_edges on empty wiki returns 0 pages updated."""
    from medulla.mcp import _tool_reindex_edges
    result = _tool_reindex_edges(db, {})
    assert "0" in result


def test_tool_reindex_edges_registered():
    """medulla_reindex_edges is registered in _HANDLERS."""
    from medulla.mcp import _HANDLERS
    assert "medulla_reindex_edges" in _HANDLERS


# ── dispatch unknown tool ──────────────────────────────────────────────────────

def test_dispatch_unknown_tool(db):
    result = _dispatch("medulla_nonexistent", {})
    assert "Unknown tool" in result
