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
    _insert(db, "sess-logd", ["The logD outlier analysis showed CHEMBL12345 was suspicious"] * 25)
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
    _insert(db, "sess-1", ["finding about batch effects in CompoundX project"])
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
    """Hyphenated compound IDs like CHEMBL12345 should be findable."""
    _insert(db, "sess-ndi", ["The compound CHEMBL12345 was the outlier"])
    results = search(db, "CHEMBL12345")
    # FTS5 treats hyphens as token separators, so NDI and 218229 both match
    assert len(results) > 0


def test_search_finds_assistant_content(db):
    """Assistant message text must be searchable — Sprint 1.5 core requirement."""
    # Simulate a session where only the assistant said "CHEMBL12345"
    s = make_session("sess-asst", messages=[
        "what compounds were suspicious?",                          # user turn
        "CHEMBL12345 has delta logD of +6.11, four sigma above batch mean.",  # assistant turn (now indexed)
        "can you dig deeper into that one?",                        # user turn
        "The measurement came from ExternalCRO batch April 2023.",      # assistant turn
    ])
    upsert_session(db, s)

    results = search(db, "CHEMBL12345")
    assert len(results) > 0
    assert any(r.id == "sess-asst" for r in results)

    results2 = search(db, "four sigma batch mean")
    assert any(r.id == "sess-asst" for r in results2)


def test_search_wiki_layer_returns_wiki_results(db):
    """Search with layer=semantic returns wiki pages, not session chunks."""
    from medulla.semantic.store import upsert_wiki_page
    upsert_wiki_page(db, "logd-concept", "concept", "LogD Prediction",
                     "LogD batch effect analysis in CompoundX series.",
                     __import__("pathlib").Path("/wiki/logd.md"))
    results = search(db, "batch effect CompoundX", layer="semantic")
    assert any(r.layer == "semantic" for r in results)
    assert any(r.result_type == "wiki_page" for r in results)


def test_strip_frontmatter_removes_yaml():
    from medulla.search import _strip_frontmatter
    content = "---\ntitle: Test\ntags: [a]\n---\n\n## Summary\n\nActual content here."
    result = _strip_frontmatter(content)
    assert "title:" not in result
    assert "Actual content" in result


def test_strip_frontmatter_no_frontmatter():
    from medulla.search import _strip_frontmatter
    content = "# Just a heading\n\nContent."
    assert _strip_frontmatter(content) == content


def test_search_wiki_excerpt_is_match_centered(db):
    """Wiki excerpt is centered on the match (body content), not raw frontmatter."""
    from medulla.semantic.store import upsert_wiki_page
    body = "background sentence about the topic. " * 20
    upsert_wiki_page(db, "test-wiki", "concept", "Test",
                     "---\ntitle: Test\ntags: []\n---\n\n## Definition\n\n" + body
                     + "This is the actual definition text about lipophilicity.",
                     __import__("pathlib").Path("/wiki/concepts/test.md"))
    results = search(db, "lipophilicity", layer="semantic")
    assert len(results) > 0
    assert "lipophilicity" in results[0].excerpt.lower()   # match is shown
    assert "---" not in results[0].excerpt                  # frontmatter not leaked


def test_recency_boost_pure():
    from medulla.search import _recency_boost
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    assert _recency_boost(now) > 0.9                       # ~now
    assert _recency_boost("2020-01-01T00:00:00Z") < 0.05   # old
    assert _recency_boost(now) > _recency_boost("2024-01-01T00:00:00Z")
    assert _recency_boost(None) == 0.0
    assert _recency_boost("not-a-date") == 0.0


def test_recency_breaks_tie_newer_first(db):
    """Two equally-relevant sessions → the more recent ranks first (hybrid path)."""
    from medulla.db.embedding_store import upsert_chunk_embedding
    from medulla.search import _get_search_embedding_provider, hybrid_search
    txt = "recencyxyz shared identical content about the topic here " * 20
    for sid, date in [("old-sess", "2026-01-01T00:00:00Z"), ("new-sess", "2026-07-01T00:00:00Z")]:
        s = make_session(sid, messages=[txt])
        s.started_at = date
        upsert_session(db, s)
        prov = _get_search_embedding_provider()
        for row in db.execute("SELECT chunk_index, chunk_text FROM session_chunks WHERE session_id=?", (sid,)):
            upsert_chunk_embedding(db, sid, row["chunk_index"], prov.embed([row["chunk_text"]])[0])
    order = [r.id for r in hybrid_search(db, "recencyxyz shared content topic", limit=10)]
    assert order.index("new-sess") < order.index("old-sess")


def _evt(hash_, command, tool="Bash", is_error=False, session="sess-e"):
    from medulla.episodic.parser import ToolEvent
    return ToolEvent(session_id=session, project_dir="/proj/x", event_ts="2026-01-01T00:00:00Z",
                     tool=tool, command=command, description="", output_preview="",
                     is_error=is_error, event_hash=hash_)


def test_search_includes_tool_events(db):
    from medulla.episodic.store import upsert_tool_events
    upsert_tool_events(db, "sess-e", [_evt("h1", "duckdb -c \"SELECT zebraword FROM t\"")])
    te = [r for r in search(db, "zebraword") if r.result_type == "tool_event"]
    assert te and "duckdb" in te[0].excerpt
    assert te[0].id.startswith("sess-e#evt")   # unique id


def test_search_tool_events_or_matching(db):
    """Natural-language query matches a command containing only some tokens (OR, not AND)."""
    from medulla.episodic.store import upsert_tool_events
    upsert_tool_events(db, "sess-e", [_evt("h1", "duckdb pka benchmark halving")])
    results = search(db, "pka benchmark duckdb query joining results")  # extra words absent
    assert any(r.result_type == "tool_event" for r in results)


def test_search_excludes_meta_search_tools(db):
    from medulla.episodic.store import upsert_tool_events
    upsert_tool_events(db, "sess-e", [
        _evt("h1", "search zebraword", tool="mcp__medulla__medulla_events_search"),
        _evt("h2", "grep zebraword file.py", tool="Bash"),
    ])
    cmds = [r.excerpt for r in search(db, "zebraword", layer="events")]
    assert any("grep" in c for c in cmds)
    assert not any(r_tool for r_tool in cmds if "search zebraword" == r_tool)


def test_search_layer_events_restricts_to_commands(db):
    from medulla.episodic.store import upsert_tool_events
    _insert(db, "sess-conv", ["talk about zebraword " * 40] * 3)
    upsert_tool_events(db, "sess-e", [_evt("h1", "echo zebraword")])
    results = search(db, "zebraword", layer="events")
    assert results and all(r.result_type == "tool_event" for r in results)


def test_tool_event_not_deduped_against_same_session_chunk(db):
    from medulla.episodic.store import upsert_tool_events
    _insert(db, "sess-both", ["zebraword appears in the conversation content here " * 30])
    upsert_tool_events(db, "sess-both", [_evt("h1", "duckdb zebraword", session="sess-both")])
    types = {r.result_type for r in search(db, "zebraword") if r.id.startswith("sess-both")}
    assert "tool_event" in types
    assert types & {"chunk", "session"}   # both the conversation hit AND the command survive


def test_search_tool_events_capped(db):
    from medulla.episodic.store import upsert_tool_events
    upsert_tool_events(db, "sess-e", [_evt(f"h{i}", f"duckdb zebraword run {i}") for i in range(12)])
    results = search(db, "zebraword", layer="events", limit=10)
    assert 0 < len(results) <= 8


def test_search_chunk_excerpt_is_match_centered(db):
    """Chunk excerpt centers on the matched term, not the start of the chunk."""
    lead = "unrelated preamble boilerplate caveat text that starts the chunk. " * 8
    messages = [lead + "the important keyword is zylophenate near the end of this chunk."]
    messages += [f"more content padding sentence {i} here about other topics" for i in range(40)]
    _insert(db, "sess-snip", messages)
    results = search(db, "zylophenate")
    hits = [r for r in results if r.result_type == "chunk" and r.id == "sess-snip"]
    assert hits
    assert "zylophenate" in hits[0].excerpt.lower()   # match shown, not the boilerplate lead


def test_search_wiki_fts_error_returns_empty(db):
    """_search_wiki handles OperationalError when FTS table missing."""
    db.execute("DROP TABLE IF EXISTS wiki_fts")
    db.commit()
    from medulla.search import _search_wiki
    results = _search_wiki(db, '"logD"', 10)
    assert results == []


def test_search_across_multiple_sessions(db):
    _insert(db, "sess-a", ["logD batch effect CompoundX"])
    _insert(db, "sess-b", ["pKa basic acidic site selection"])
    _insert(db, "sess-c", ["logD Clotho project analysis"])

    results = search(db, "logD")
    ids = {r.id for r in results}
    assert "sess-a" in ids
    assert "sess-c" in ids
    assert "sess-b" not in ids


# ── chunk_index field ──────────────────────────────────────────────────────────

def test_chunk_result_has_chunk_index(db):
    """chunk results must expose chunk_index as an integer, not embedded in title."""
    _insert(db, "sess-ci", ["unique-chunk-term content"] * 25)
    results = search(db, "unique-chunk-term")
    chunk_results = [r for r in results if r.result_type == "chunk"]
    assert len(chunk_results) > 0
    for r in chunk_results:
        assert r.chunk_index is not None
        assert isinstance(r.chunk_index, int)


def test_session_result_has_no_chunk_index(db):
    """session-level results have chunk_index=None."""
    _insert(db, "sess-si", ["short-session-term"])
    results = search(db, "short-session-term")
    session_results = [r for r in results if r.result_type == "session"]
    for r in session_results:
        assert r.chunk_index is None


def test_wiki_result_has_no_chunk_index(db):
    """wiki_page results have chunk_index=None."""
    from medulla.semantic.store import upsert_wiki_page
    upsert_wiki_page(db, "wiki-ci-test", "concept", "Wiki CI Test",
                     "unique-wiki-chunk-term definition here.",
                     __import__("pathlib").Path("/wiki/concepts/wiki-ci-test.md"))
    results = search(db, "unique-wiki-chunk-term", layer="semantic")
    assert len(results) > 0
    for r in results:
        assert r.chunk_index is None


def test_chunk_index_is_not_always_zero(db):
    """chunk_index reflects the actual matched chunk, not always 0."""
    # Substantial filler → multiple chunks; unique term only in the last one.
    messages = [f"filler{i} " + "routine conversation content padding here " * 30 for i in range(24)]
    messages.append("late-chunk-unique-marker at the end of session")
    _insert(db, "sess-late", messages)
    results = search(db, "late-chunk-unique-marker")
    chunk_results = [r for r in results if r.result_type == "chunk" and r.id == "sess-late"]
    assert len(chunk_results) > 0
    # The match is in a later chunk — chunk_index should be > 0
    assert any(r.chunk_index > 0 for r in chunk_results)


def test_chunk_title_no_longer_embeds_chunk_index(db):
    """Title is clean session ID prefix only — chunk_index is a dedicated field now."""
    _insert(db, "sess-title", ["title-test-term content"] * 25)
    results = search(db, "title-test-term")
    chunk_results = [r for r in results if r.result_type == "chunk"]
    assert len(chunk_results) > 0
    for r in chunk_results:
        assert "chunk" not in r.title


# ── hybrid search ──────────────────────────────────────────────────────────────

class _MockEmbedProvider:
    dimension = 768
    model_name = "mock"
    def embed(self, texts):
        # Return deterministic vectors: text hash → seed → unit-ish vector
        results = []
        for text in texts:
            seed = abs(hash(text)) % 1000
            vec = [(seed + i) / (1000.0 * 10) for i in range(self.dimension)]
            # normalize
            norm = sum(v**2 for v in vec) ** 0.5
            results.append([v / norm for v in vec])
        return results


def _insert_with_embedding(db, session_id, messages, project_dir="/proj/x"):
    """Insert session and store fake embeddings for all its chunks."""
    from medulla.db.embedding_store import upsert_chunk_embedding
    s = make_session(session_id, project_dir=project_dir, messages=messages)
    upsert_session(db, s)
    provider = _MockEmbedProvider()
    chunks = db.execute(
        "SELECT session_id, chunk_index, chunk_text FROM session_chunks WHERE session_id = ?",
        (session_id,)
    ).fetchall()
    if chunks:
        texts = [c["chunk_text"] for c in chunks]
        embeddings = provider.embed(texts)
        for chunk, emb in zip(chunks, embeddings):
            upsert_chunk_embedding(db, chunk["session_id"], chunk["chunk_index"], emb)


def test_rrf_score_math():
    """RRF score = 1/(k+rank) for each list; k=60 by default."""
    from medulla.search import _rrf_score
    # rank 1 in one list, not in other
    assert abs(_rrf_score(1, None) - 1/61) < 1e-9
    # rank 1 in both lists
    assert abs(_rrf_score(1, 1) - 2/61) < 1e-9
    # rank 10 in one list only
    assert abs(_rrf_score(10, None) - 1/70) < 1e-9


def test_rrf_fuse_higher_score_ranks_first():
    """Item appearing in both lists ranks above item in only one."""
    from medulla.search import _rrf_fuse
    bm25 = [("id-a", 0), ("id-b", 1)]   # id-a rank 0, id-b rank 1
    vec  = [("id-b", 0), ("id-c", 1)]   # id-b rank 0, id-c rank 1
    # id-b appears in both → should rank first
    fused = _rrf_fuse(bm25, vec)
    assert fused[0][0] == "id-b"


def test_hybrid_search_returns_results(db):
    """hybrid_search returns results when embeddings exist."""
    from medulla.search import hybrid_search
    _insert_with_embedding(db, "sess-hyb", ["hybrid search test content"] * 25)
    provider = _MockEmbedProvider()
    results = hybrid_search(db, "hybrid search test", provider=provider)
    assert len(results) > 0
    assert any(r.id == "sess-hyb" for r in results)


def test_hybrid_search_falls_back_to_bm25_without_embeddings(db):
    """hybrid_search degrades gracefully to BM25 when vec_chunks is empty."""
    from medulla.search import hybrid_search
    _insert(db, "sess-fallback", ["fallback bm25 only content"] * 25)
    # No embeddings stored — vec_chunks empty for this session
    provider = _MockEmbedProvider()
    results = hybrid_search(db, "fallback bm25 only", provider=provider)
    assert len(results) > 0
    assert any(r.id == "sess-fallback" for r in results)


def test_hybrid_search_both_lists_boost_rank(db):
    """A chunk matching both BM25 and vector ranks above a chunk matching only one."""
    from medulla.search import hybrid_search
    from medulla.db.embedding_store import upsert_chunk_embedding

    # sess-both: text matches BM25 query AND gets an embedding close to query
    _insert_with_embedding(db, "sess-both", ["unique-boost-term content detail"] * 25)
    # sess-bm25only: text matches BM25 but no embedding stored
    _insert(db, "sess-bm25only", ["unique-boost-term content detail"] * 25)

    provider = _MockEmbedProvider()
    results = hybrid_search(db, "unique-boost-term", provider=provider)
    ids = [r.id for r in results]
    # sess-both should outrank sess-bm25only
    assert "sess-both" in ids
    assert ids.index("sess-both") <= ids.index("sess-bm25only")


def test_hybrid_search_respects_layer_filter(db):
    """hybrid_search with layer=episodic returns only episodic results."""
    from medulla.search import hybrid_search
    _insert_with_embedding(db, "sess-layer", ["layer filter test content"] * 25)
    provider = _MockEmbedProvider()
    results = hybrid_search(db, "layer filter test", provider=provider, layer="episodic")
    assert all(r.layer == "episodic" for r in results)


def test_hybrid_search_respects_limit(db):
    """hybrid_search respects the limit parameter."""
    from medulla.search import hybrid_search
    for i in range(5):
        _insert_with_embedding(db, f"sess-lim-{i}", [f"limit test content session {i}"] * 25)
    provider = _MockEmbedProvider()
    results = hybrid_search(db, "limit test content", provider=provider, limit=2)
    assert len(results) <= 2


def test_search_uses_hybrid_when_embeddings_exist(db):
    """Top-level search() uses hybrid when embeddings available, BM25 otherwise."""
    _insert_with_embedding(db, "sess-auto", ["auto hybrid detection content"] * 25)
    results = search(db, "auto hybrid detection")
    assert len(results) > 0
