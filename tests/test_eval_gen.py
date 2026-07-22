"""Tests for medulla.eval_gen — known-item + history eval-set generation."""
from medulla.eval_gen import generate_known_item, generate_from_history, generate
from medulla.episodic.store import upsert_session, upsert_tool_events
from medulla.episodic.parser import ToolEvent
from tests.test_store import make_session


def _seed_sessions(db):
    upsert_session(db, make_session("sess-logd", project_dir="/proj/a", messages=[
        "logd lipophilicity chromatographic partition outlier " * 20]))
    upsert_session(db, make_session("sess-pka", project_dir="/proj/b", messages=[
        "protonation ionization acidic triazole titration " * 20]))


def _evt(hash_, command, tool="Bash", session="s1"):
    return ToolEvent(session_id=session, project_dir="/p", event_ts="2026-01-01T00:00:00Z",
                     tool=tool, command=command, description="", output_preview="",
                     is_error=False, event_hash=hash_)


# ── known-item ────────────────────────────────────────────────────────────────

def test_known_item_labels_each_case_to_its_session(db):
    _seed_sessions(db)
    cases = generate_known_item(db, n=10)
    assert cases
    by_rel = {c["relevant"][0]: c for c in cases}
    assert "sess-log" in by_rel   # session_id[:8]
    # the derived query contains a distinctive term from that session
    assert any("lipophilicity" in c["query"] or "logd" in c["query"] for c in cases)


def test_known_item_stratifies_across_projects(db):
    _seed_sessions(db)
    cases = generate_known_item(db, n=10)
    rels = {c["relevant"][0] for c in cases}
    assert {"sess-log", "sess-pka"} <= rels   # both projects represented


def test_generate_default_is_known_item(db):
    _seed_sessions(db)
    cases = generate(db, n=5)  # no mode → known-item → fully labeled
    assert cases and all(c["relevant"] for c in cases)


# ── history ─────────────────────────────────────────────────────────────────

def test_history_extracts_real_queries_blank_relevant(db):
    _seed_sessions(db)
    upsert_tool_events(db, "s1", [
        _evt("h1", "mcp__medulla__medulla_search logd outliers",
             tool="mcp__medulla__medulla_search"),
    ])
    cases = generate_from_history(db, n=10)
    assert cases and cases[0]["query"] == "logd outliers"
    assert cases[0]["relevant"] == []          # never machine-labeled
    assert "candidates" in cases[0]


def test_history_excludes_toolsearch_and_select(db):
    upsert_tool_events(db, "s1", [
        _evt("h1", "ToolSearch select:mcp__medulla__medulla_search", tool="ToolSearch"),
        _evt("h2", "mcp__x__foo_search select:something", tool="mcp__x__foo_search"),
        _evt("h3", "mcp__medulla__medulla_search real user question",
             tool="mcp__medulla__medulla_search"),
    ])
    queries = [c["query"] for c in generate_from_history(db, n=10)]
    assert "real user question" in queries
    assert not any(q.startswith("select:") for q in queries)
    assert not any("ToolSearch" in q for q in queries)


def test_history_excludes_structured_json_queries(db):
    upsert_tool_events(db, "s1", [
        _evt("h1", 'mcp__slack__slack_search {"sender": "jean", "limit": 10}',
             tool="mcp__slack__slack_search"),
        _evt("h2", "mcp__medulla__medulla_search logd outliers question",
             tool="mcp__medulla__medulla_search"),
    ])
    queries = [c["query"] for c in generate_from_history(db, n=10)]
    assert "logd outliers question" in queries
    assert not any(q.startswith("{") for q in queries)


def test_history_respects_n_limit_and_candidate_cap(db):
    # several sessions sharing a term → a query returns >3 hits (cap to 3)
    for i in range(5):
        upsert_session(db, make_session(f"sess-x{i}", project_dir="/p",
                                        messages=[f"sharedterm unique{i} content here " * 20]))
    upsert_tool_events(db, "s1", [
        _evt("h1", "mcp__medulla__medulla_search sharedterm", tool="mcp__medulla__medulla_search"),
        _evt("h2", "mcp__medulla__medulla_search another query here", tool="mcp__medulla__medulla_search"),
    ])
    cases = generate_from_history(db, n=1)   # n limit
    assert len(cases) == 1
    assert len(cases[0]["candidates"]) <= 3   # candidate cap


def test_history_dedups_candidates(db):
    _seed_sessions(db)
    # a session that hits as both a chunk and a command for the same query
    upsert_tool_events(db, "sess-logd", [
        _evt("h0", "mcp__medulla__medulla_search logd lipophilicity",
             tool="mcp__medulla__medulla_search"),
        _evt("h1", "duckdb logd lipophilicity outlier", session="sess-logd"),
    ])
    cases = generate_from_history(db, n=5)
    c = next(c for c in cases if "logd" in c["query"])
    assert len(c["candidates"]) == len(set(c["candidates"]))   # no dup session prefixes
