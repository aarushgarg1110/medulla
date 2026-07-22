"""Tests for medulla.eval — NDCG@k / MRR metrics and the search eval harness."""
import json

import pytest

from medulla.eval import ndcg_at_k, mrr, run_eval, evaluate_case, _relevance_list
from medulla.episodic.store import upsert_session
from tests.test_store import make_session


# ── metric math ───────────────────────────────────────────────────────────────

def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([1, 1, 0, 0], k=5) == 1.0


def test_ndcg_worst_ranking_below_one():
    assert ndcg_at_k([0, 0, 1], k=5) < 1.0


def test_ndcg_no_relevant_is_zero():
    assert ndcg_at_k([0, 0, 0], k=5) == 0.0


def test_ndcg_respects_k_cutoff():
    # relevant item at position 6 is beyond k=5 → NDCG@5 = 0
    assert ndcg_at_k([0, 0, 0, 0, 0, 1], k=5) == 0.0


def test_mrr_first_position():
    assert mrr([1, 0, 0]) == 1.0


def test_mrr_third_position():
    assert mrr([0, 0, 1]) == pytest.approx(1 / 3)


def test_mrr_none_relevant():
    assert mrr([0, 0]) == 0.0


# ── relevance list (dedup to doc level) ────────────────────────────────────────

class _R:
    def __init__(self, id_):
        self.id = id_


def test_relevance_list_dedups_and_matches_tool_event_base():
    results = [_R("sess-a#evt5"), _R("sess-a"), _R("sess-b")]
    rels = _relevance_list(results, {"sess-a"})
    assert rels == [1, 0]   # sess-a counted once (relevant); sess-b once (not)


def test_relevance_list_matches_by_prefix():
    # 8-char prefix (as shown in search output) matches the full UUID
    results = [_R("044d2e97-627c-4ed1-9744-bb50258dbf78")]
    assert _relevance_list(results, {"044d2e97"}) == [1]


# ── harness against a real fixture DB ──────────────────────────────────────────

def _seed(db):
    upsert_session(db, make_session("sess-logd", messages=[
        "logD lipophilicity outlier analysis " * 30]))
    upsert_session(db, make_session("sess-pka", messages=[
        "pka protonation ionization acidic " * 30]))


def test_run_eval_perfect_when_relevant_ranks_first(db):
    _seed(db)
    cases = [
        {"query": "logD lipophilicity outlier", "relevant": ["sess-logd"]},
        {"query": "pka protonation acidic", "relevant": ["sess-pka"]},
    ]
    report = run_eval(db, cases, k=5)
    assert report["n"] == 2
    assert report["ndcg"] == 1.0
    assert report["mrr"] == 1.0
    assert all(r["hits"] == 1 for r in report["per_query"])


def test_evaluate_case_misses_when_not_indexed(db):
    _seed(db)
    case = {"query": "logD lipophilicity outlier", "relevant": ["sess-does-not-exist"]}
    r = evaluate_case(db, case, k=5)
    assert r["ndcg"] == 0.0 and r["mrr"] == 0.0 and r["hits"] == 0


def test_run_eval_empty_cases(db):
    report = run_eval(db, [], k=5)   # no divide-by-zero
    assert report["n"] == 0
    assert report["ndcg"] == 0.0 and report["mrr"] == 0.0
    assert report["per_query"] == []


# ── CLI eval command ───────────────────────────────────────────────────────────

from typer.testing import CliRunner
from medulla.cli import app

_runner = CliRunner()


def test_eval_command_reports_metrics(tmp_path, monkeypatch):
    import medulla.config as cfg
    from medulla.db.database import connect
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    conn = connect()
    _seed(conn)
    conn.close()

    eval_set = tmp_path / "eval.json"
    eval_set.write_text(json.dumps([
        {"query": "logD lipophilicity outlier", "relevant": ["sess-logd"]},
    ]))
    result = _runner.invoke(app, ["eval", str(eval_set)])
    assert result.exit_code == 0
    assert "NDCG@5" in result.output
    assert "MRR" in result.output


def test_eval_command_missing_file_errors(tmp_path, monkeypatch):
    import medulla.config as cfg
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=tmp_path / ".medulla"))
    result = _runner.invoke(app, ["eval", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "No eval set" in result.output
