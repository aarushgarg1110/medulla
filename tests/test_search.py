"""Tests for medulla.search — real FTS5 SQLite, no mocks."""
import pytest

from medulla.episodic.store import upsert_session
from medulla.search import SearchResult, _snippet, _to_fts_query, search
from tests.test_store import make_session


def _insert(db, session_id, messages, project_dir="/proj/x"):
    s = make_session(session_id, project_dir=project_dir, messages=messages)
    upsert_session(db, s)


# ── _to_fts_query ─────────────────────────────────────────────────────────────

def test_to_fts_query_single_word():
    assert _to_fts_query("hello") == '"hello"'


def test_to_fts_query_multiple_words():
    assert _to_fts_query("logD outliers") == '"logD" "outliers"'


def test_to_fts_query_strips_whitespace():
    assert _to_fts_query("  hello  world  ") == '"hello" "world"'


# ── _snippet ──────────────────────────────────────────────────────────────────

def test_snippet_short_text():
    assert _snippet("hello world", 100) == "hello world"


def test_snippet_truncates_at_word_boundary():
    text = "the quick brown fox jumps over"
    result = _snippet(text, 15)
    assert result.endswith("…")
    assert len(result) <= 16


def test_snippet_exact_length():
    text = "a" * 100
    result = _snippet(text, 100)
    assert result == text  # no truncation needed


# ── search ─────────────────────────────────────────────────────────────────────

def test_search_finds_in_chunks(db):
    _insert(db, "sess-logd", ["The logD outlier analysis showed NDI-218229 was suspicious"] * 25)
    results = search(db, "logD outlier")
    assert len(results) > 0
    assert any(r.id == "sess-logd" for r in results)


def test_search_empty_query_returns_empty(db):
    _insert(db, "sess-1", ["hello world"])
    results = search(db, "")
    assert results == []


def test_search_whitespace_query_returns_empty(db):
    _insert(db, "sess-1", ["hello world"])
    results = search(db, "   ")
    assert results == []


def test_search_no_match_returns_empty(db):
    _insert(db, "sess-1", ["hello world"])
    results = search(db, "zzznomatch999")
    assert results == []


def test_search_respects_limit(db):
    for i in range(10):
        _insert(db, f"sess-{i}", [f"hello world logD test session {i}"])
    results = search(db, "logD", limit=3)
    assert len(results) <= 3


def test_search_chunk_preferred_over_session(db):
    """When a session has chunks, chunk results should take priority."""
    messages = [f"unique-search-term content message {i}" for i in range(25)]
    _insert(db, "sess-chunked", messages)
    results = search(db, "unique-search-term")
    types = [r.result_type for r in results if r.id == "sess-chunked"]
    # chunk result should appear; session-level deduped away
    assert "chunk" in types


def test_search_result_has_excerpt(db):
    _insert(db, "sess-1", ["finding about batch effects in Salacia project"])
    results = search(db, "batch effects")
    assert len(results) > 0
    assert results[0].excerpt != ""


def test_search_result_has_date(db):
    _insert(db, "sess-1", ["test content logD"])
    results = search(db, "logD")
    assert len(results) > 0
    assert results[0].date is not None


def test_search_result_has_project_dir(db):
    _insert(db, "sess-1", ["test logD content"], project_dir="/proj/mlops")
    results = search(db, "logD")
    assert len(results) > 0
    assert results[0].project_dir == "/proj/mlops"


def test_search_deduplicates_same_session(db):
    """Same session should not appear twice (once as chunk, once as session)."""
    _insert(db, "sess-dup", ["logD logD logD"] * 25)
    results = search(db, "logD", limit=50)
    ids = [r.id for r in results]
    assert len(ids) == len(set(ids)), "Duplicate session IDs in results"


def test_search_layer_filter_episodic(db):
    _insert(db, "sess-1", ["logD content"])
    results = search(db, "logD", layer="episodic")
    assert all(r.layer == "episodic" for r in results)


def test_search_hyphenated_term(db):
    """Hyphenated compound IDs like NDI-218229 should be findable."""
    _insert(db, "sess-ndi", ["The compound NDI-218229 was the outlier"])
    results = search(db, "NDI-218229")
    # FTS5 treats hyphens as token separators, so NDI and 218229 both match
    assert len(results) > 0


def test_search_across_multiple_sessions(db):
    _insert(db, "sess-a", ["logD batch effect Salacia"])
    _insert(db, "sess-b", ["pKa basic acidic site selection"])
    _insert(db, "sess-c", ["logD Clotho project analysis"])

    results = search(db, "logD")
    ids = {r.id for r in results}
    assert "sess-a" in ids
    assert "sess-c" in ids
    assert "sess-b" not in ids
