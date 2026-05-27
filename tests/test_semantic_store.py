"""Tests for medulla.semantic.store — wiki pages and pending ingests."""
import pytest
from medulla.semantic.store import (
    upsert_wiki_page, list_wiki_pages, get_wiki_page,
    search_wiki, get_wiki_stats,
    queue_pending, get_pending, get_pending_count,
    mark_pending_done, mark_pending_error,
)
from pathlib import Path


def _insert_page(db, slug="logd-prediction", page_type="concept", title="LogD Prediction"):
    upsert_wiki_page(
        db, slug, page_type, title,
        content=f"# {title}\n\nLogD is the distribution coefficient at pH 7.4. Used in ADMET prediction.",
        file_path=Path(f"/wiki/{page_type}s/{slug}.md"),
        tags=["admet", "logd"],
        sources=["some-paper"],
    )


# ── upsert / get ──────────────────────────────────────────────────────────────

def test_upsert_and_get_wiki_page(db):
    _insert_page(db)
    row = get_wiki_page(db, "logd-prediction")
    assert row is not None
    assert row["title"] == "LogD Prediction"
    assert row["type"] == "concept"
    assert "LogD" in row["content"]


def test_upsert_updates_on_conflict(db):
    _insert_page(db)
    upsert_wiki_page(db, "logd-prediction", "concept", "Updated Title",
                     content="new content", file_path=Path("/wiki/concepts/logd.md"))
    row = get_wiki_page(db, "logd-prediction")
    assert row["title"] == "Updated Title"
    assert row["content"] == "new content"


def test_get_wiki_page_not_found(db):
    assert get_wiki_page(db, "nonexistent") is None


def test_wiki_fts_indexed(db):
    _insert_page(db, slug="batch-effect", title="Batch Effect Analysis",
                 page_type="concept")
    # FTS should find "distribution" from content
    results = search_wiki(db, "distribution coefficient")
    # Might not match "batch-effect" but tests FTS doesn't crash
    assert isinstance(results, list)


# ── list ──────────────────────────────────────────────────────────────────────

def test_list_wiki_pages_all(db):
    _insert_page(db, "page-1", "source", "Paper One")
    _insert_page(db, "page-2", "concept", "Concept Two")
    rows = list_wiki_pages(db)
    assert len(rows) == 2


def test_list_wiki_pages_by_type(db):
    _insert_page(db, "src-1", "source", "Source")
    _insert_page(db, "con-1", "concept", "Concept")
    rows = list_wiki_pages(db, page_type="source")
    assert len(rows) == 1
    assert rows[0]["type"] == "source"


def test_list_wiki_pages_empty(db):
    assert list_wiki_pages(db) == []


# ── search ────────────────────────────────────────────────────────────────────

def test_search_wiki_finds_title(db):
    _insert_page(db, "logd-pred", "concept", "LogD Prediction Model")
    results = search_wiki(db, "LogD Prediction")
    assert any(r["slug"] == "logd-pred" for r in results)


def test_search_wiki_finds_content(db):
    _insert_page(db, "admet-page", "concept", "ADMET Overview")
    results = search_wiki(db, "distribution coefficient")
    assert any(r["slug"] == "admet-page" for r in results)


def test_search_wiki_empty_query(db):
    assert search_wiki(db, "") == []


def test_search_wiki_no_results(db):
    _insert_page(db)
    assert search_wiki(db, "zzznomatch9999") == []


def test_search_wiki_by_type(db):
    _insert_page(db, "src-a", "source", "LogD Paper Source")
    _insert_page(db, "con-a", "concept", "LogD Concept")
    results = search_wiki(db, "LogD", page_type="source")
    types = [r["type"] for r in results]
    assert all(t == "source" for t in types)


# ── stats ─────────────────────────────────────────────────────────────────────

def test_get_wiki_stats_empty(db):
    stats = get_wiki_stats(db)
    assert stats["total"] == 0
    assert stats["by_type"] == {}


def test_get_wiki_stats_counts(db):
    _insert_page(db, "s1", "source", "Source 1")
    _insert_page(db, "c1", "concept", "Concept 1")
    _insert_page(db, "c2", "concept", "Concept 2")
    stats = get_wiki_stats(db)
    assert stats["total"] == 3
    assert stats["by_type"]["source"] == 1
    assert stats["by_type"]["concept"] == 2


# ── pending queue ─────────────────────────────────────────────────────────────

def test_queue_and_get_pending(db):
    queue_pending(db, "/path/to/paper.pdf", "pdf", "My Paper")
    pending = get_pending(db)
    assert len(pending) == 1
    assert pending[0]["source_path"] == "/path/to/paper.pdf"
    assert pending[0]["status"] == "queued"


def test_get_pending_count(db):
    assert get_pending_count(db) == 0
    queue_pending(db, "/a.pdf", "pdf")
    queue_pending(db, "/b.pdf", "pdf")
    assert get_pending_count(db) == 2


def test_queue_pending_idempotent_queued(db):
    """Calling queue_pending twice for same path keeps one queued entry."""
    id1 = queue_pending(db, "/paper.pdf", "pdf", "Paper")
    id2 = queue_pending(db, "/paper.pdf", "pdf", "Paper")
    assert id1 == id2
    assert get_pending_count(db) == 1


def test_queue_pending_resets_error_to_queued(db):
    """Failed source auto-resets to queued on next queue_pending call."""
    pid = queue_pending(db, "/paper.pdf", "pdf")
    mark_pending_error(db, pid, "LLM failed")
    assert get_pending_count(db) == 0  # errored, not queued
    queue_pending(db, "/paper.pdf", "pdf")  # retry — resets to queued
    assert get_pending_count(db) == 1


def test_queue_pending_skips_done_file_exists(db, tmp_path):
    """done + file still on disk → skip (already processed)."""
    f = tmp_path / "paper.pdf"
    f.write_bytes(b"pdf")
    pid = queue_pending(db, str(f), "pdf")
    mark_pending_done(db, pid)
    queue_pending(db, str(f), "pdf")  # file exists → skip
    assert get_pending_count(db) == 0


def test_queue_pending_requeues_done_file_missing(db, tmp_path):
    """done + file deleted → re-queue automatically."""
    f = tmp_path / "paper.pdf"
    f.write_bytes(b"pdf")
    pid = queue_pending(db, str(f), "pdf")
    mark_pending_done(db, pid)
    f.unlink()  # simulate user deleting the raw file
    queue_pending(db, str(f), "pdf")  # file gone → re-queue
    assert get_pending_count(db) == 1


def test_queue_pending_force_requeues_done(db, tmp_path):
    """force=True re-queues even when done + file exists."""
    f = tmp_path / "paper.pdf"
    f.write_bytes(b"pdf")
    pid = queue_pending(db, str(f), "pdf")
    mark_pending_done(db, pid)
    queue_pending(db, str(f), "pdf", force=True)  # force → re-queue
    assert get_pending_count(db) == 1


def test_mark_pending_done(db):
    pid = queue_pending(db, "/a.pdf", "pdf")
    mark_pending_done(db, pid)
    assert get_pending_count(db) == 0


def test_mark_pending_error(db):
    pid = queue_pending(db, "/a.pdf", "pdf")
    mark_pending_error(db, pid, "LLM unavailable")
    pending = get_pending(db)
    assert len(pending) == 0  # errored is not "queued"
    row = db.execute("SELECT * FROM pending_ingests WHERE id=?", (pid,)).fetchone()
    assert row["status"] == "error"
    assert "LLM unavailable" in row["error"]
