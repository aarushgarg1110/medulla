"""MCP handler paths that answer with a message instead of a result: failures, empty states and
unknown ids. Each was reachable from a client but never exercised, so a change could turn a
readable message into a traceback over stdio."""
import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

from medulla.episodic.store import upsert_session
from medulla.mcp import (
    _tool_ingest, _tool_ingest_url, _tool_list_raw, _tool_remove,
    _tool_session_detail, _tool_session_tree, _tool_stats,
)
from tests.test_store import make_session


@pytest.fixture
def wiki(tmp_path, monkeypatch):
    w = tmp_path / "wiki"
    w.mkdir()
    monkeypatch.setattr("medulla.config.get_config", lambda: SimpleNamespace(wiki_path=w))
    return w


def test_ingest_lists_broken_wikilinks(db, wiki, monkeypatch):
    monkeypatch.setattr("medulla.semantic.ingest.store_wiki_page",
                        lambda *a, **k: {"slug": "adam", "type": "concept", "broken_wikilinks": ["[[momentum]]"]})
    out = _tool_ingest(db, {"title": "Adam", "content": "uses [[momentum]]"})
    assert "Broken wikilinks" in out and "[[momentum]]" in out


def test_ingest_reports_a_store_failure(db, wiki, monkeypatch):
    def boom(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr("medulla.semantic.ingest.store_wiki_page", boom)
    assert "Store failed: disk full" in _tool_ingest(db, {"title": "T", "content": "C"})


def test_ingest_url_reports_a_failure(db, wiki, monkeypatch):
    def boom():
        raise RuntimeError("no provider configured")
    monkeypatch.setattr("medulla.llm.get_provider", boom)
    assert "Ingest URL failed: no provider configured" in _tool_ingest_url(db, {"url": "https://example.com"})


def test_list_raw_does_not_count_the_url_log_as_a_file(db, wiki):
    (wiki / "raw").mkdir()
    (wiki / "raw" / "url-references.md").write_text("logged urls")
    assert "is empty" in _tool_list_raw(db, {})


def test_remove_needs_a_target_and_passes_plan_errors_through(db, wiki, monkeypatch):
    assert "target is required" in _tool_remove(db, {"target": " "})
    monkeypatch.setattr("medulla.semantic.remove.plan_remove",
                        lambda conn, target, wiki_path=None: {"error": "No page 'concepts/x'"})
    assert _tool_remove(db, {"target": "concepts/x"}) == "No page 'concepts/x'"


def test_detail_and_tree_report_unknown_ids(db):
    full = "00000000-0000-0000-0000-000000000000"
    assert "Session not found" in _tool_session_detail(db, {"session_id": full})
    assert "Session not found" in _tool_session_tree(db, {"session_id": full})
    assert "Session not found" in _tool_session_tree(db, {"session_id": "nope1234"})


def test_detail_overview_names_the_objective(db):
    upsert_session(db, dataclasses.replace(make_session("sess-obj-0000-0000-0000-000000000000"),
                                           slug="inferenceOptimizationPart6"))
    assert "Objective: inferenceOptimizationPart6" in _tool_session_detail(db, {"session_id": "sess-obj"})


def test_stats_names_each_wiki_page_type(db):
    from medulla.semantic.store import upsert_wiki_page
    upsert_wiki_page(db, "adam", "concept", "Adam", "body", Path("/wiki/concepts/adam.md"))
    upsert_wiki_page(db, "acme", "entity", "Acme", "body", Path("/wiki/entities/acme.md"))
    out = _tool_stats(db)
    assert "concepts" in out and "entities" in out
