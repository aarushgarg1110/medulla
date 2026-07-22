"""Offline search-quality evaluation — NDCG@k and MRR over a labeled query set.

A test set is a list of cases:
    [{"query": "hERG ablation cap", "relevant": ["044d2e97", "0ee17c49"], "layer": null}, ...]
`relevant` holds the ids that *should* surface — session_ids, wiki slugs, or the base
session_id of a tool_event hit. Run it against any DB to track whether search changes
(chunking, excerpts, tool_events, ranking) actually improve retrieval instead of eyeballing.
"""
from __future__ import annotations

import math
import sqlite3
from typing import Any

from medulla.search import search


# ── metrics (pure) ───────────────────────────────────────────────────────────

def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg_at_k(rels: list[int], k: int) -> float:
    """Normalized DCG over a binary-relevance list already in ranked order."""
    ideal = _dcg(sorted(rels, reverse=True)[:k])
    return _dcg(rels[:k]) / ideal if ideal > 0 else 0.0


def mrr(rels: list[int]) -> float:
    """Reciprocal rank of the first relevant hit (0 if none)."""
    for i, r in enumerate(rels):
        if r > 0:
            return 1.0 / (i + 1)
    return 0.0


# ── harness ──────────────────────────────────────────────────────────────────

def _base_id(result_id: str) -> str:
    """Doc-level id: strip the tool_event '#evt…' suffix so a command and a chunk of
    the same session both map to the session."""
    return result_id.split("#", 1)[0]


def _relevance_list(results: list, relevant: set[str]) -> list[int]:
    """Binary relevance in ranked order, deduped to one entry per doc.

    A relevant id matches by prefix, so the 8-char session ids shown in search
    output can be used verbatim in the test set (they map to full UUIDs)."""
    rels: list[int] = []
    seen: set[str] = set()
    for r in results:
        base = _base_id(r.id)
        if base in seen:
            continue
        seen.add(base)
        match = any(base == rel or base.startswith(rel) for rel in relevant)
        rels.append(1 if match else 0)
    return rels


def evaluate_case(conn: sqlite3.Connection, case: dict, k: int = 5) -> dict[str, Any]:
    results = search(conn, case["query"], limit=max(k, 10), layer=case.get("layer"))
    rels = _relevance_list(results, set(case.get("relevant", [])))
    return {
        "query": case["query"],
        "ndcg": round(ndcg_at_k(rels, k), 4),
        "mrr": round(mrr(rels), 4),
        "hits": sum(rels),
        "relevant": len(case.get("relevant", [])),
    }


def run_eval(conn: sqlite3.Connection, cases: list[dict], k: int = 5) -> dict[str, Any]:
    """Evaluate every case; return per-query rows + aggregate NDCG@k / MRR."""
    per_query = [evaluate_case(conn, c, k=k) for c in cases]
    n = len(per_query) or 1
    return {
        "k": k,
        "n": len(per_query),
        "ndcg": round(sum(r["ndcg"] for r in per_query) / n, 4),
        "mrr": round(sum(r["mrr"] for r in per_query) / n, 4),
        "per_query": per_query,
    }
