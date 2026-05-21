"""Tests for the full ingest pipeline using a MockProvider (no real API calls)."""
import json
import shutil
import pytest
from pathlib import Path

from medulla.semantic.ingest import (
    intake_to_raw, discover_raw, process_pending,
    store_wiki_page, ingest_url_mcp, _parse_llm_response,
)


class MockProvider:
    """Test double implementing LLMProvider — returns realistic structured output."""
    @property
    def name(self): return "mock"
    @property
    def model(self): return "mock-model"

    def generate(self, prompt: str, system: str | None = None, on_token=None) -> str:
        return json.dumps({
            "source_page": {
                "title": "LogD Prediction Study",
                "summary": "A study on logD prediction using chromatographic methods.",
                "key_points": ["LogD measured at pH 7.4", "Batch effects observed"],
                "concepts": ["[[logd-prediction]] — core concept"],
                "entities": ["[[syngene]] — CRO performing measurements"],
                "connections": [],
                "gaps": ["Root cause unclear"],
            },
            "concept_pages": [{
                "slug": "logd-prediction",
                "title": "LogD Prediction",
                "tags": ["admet", "logd"],
                "definition": "LogD is the distribution coefficient at pH 7.4.",
                "how_it_works": "Measured via chromatography.",
                "why_it_matters": "Key ADMET property.",
                "nuances": "Differs from logP by ionization.",
                "evidence": "Salacia series data.",
                "connections": [],
                "open_questions": ["Why batch effect?"],
            }],
            "entity_pages": [{
                "slug": "syngene",
                "title": "Syngene",
                "entity_type": "org",
                "tags": ["cro"],
                "who_what": "Contract research organization.",
                "relevance": "Performed logD measurements.",
                "contributions": ["Chromatographic logD assay"],
                "connections": [],
            }],
        })


@pytest.fixture
def mock_provider():
    return MockProvider()


# ── intake_to_raw ──────────────────────────────────────────────────────────────

def test_intake_markdown_copies_to_raw(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# LogD Study\n\nContent about batch effects.")
    raw_path = intake_to_raw(db, wiki, str(md))
    assert raw_path.exists()
    assert raw_path.parent.name == "raw"
    assert raw_path.name == "paper.md"


def test_intake_pdf_copies_to_raw(db, tmp_path):
    pytest.importorskip("fitz")
    import fitz
    wiki = tmp_path / "wiki"
    doc = fitz.open()
    doc.new_page().insert_text((50, 50), "LogD study content.")
    pdf = tmp_path / "study.pdf"
    doc.save(str(pdf))
    doc.close()
    raw_path = intake_to_raw(db, wiki, str(pdf))
    assert raw_path.exists()
    assert raw_path.suffix == ".pdf"


def test_intake_missing_file_raises(db, tmp_path):
    wiki = tmp_path / "wiki"
    with pytest.raises(ValueError, match="File not found"):
        intake_to_raw(db, wiki, "/nonexistent/paper.pdf")


def test_intake_adds_to_pending(db, tmp_path):
    from medulla.semantic.store import get_pending_count
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# Paper\n\nContent.")
    intake_to_raw(db, wiki, str(md))
    assert get_pending_count(db) == 1


# ── discover_raw ──────────────────────────────────────────────────────────────

def test_discover_raw_finds_new_files(db, tmp_path):
    wiki = tmp_path / "wiki"
    raw = wiki / "raw"
    raw.mkdir(parents=True)
    (raw / "paper1.md").write_text("# Paper 1\nContent.")
    (raw / "paper2.md").write_text("# Paper 2\nContent.")
    new = discover_raw(wiki, db)
    assert len(new) == 2


def test_discover_raw_skips_url_references(db, tmp_path):
    wiki = tmp_path / "wiki"
    raw = wiki / "raw"
    raw.mkdir(parents=True)
    (raw / "url-references.md").write_text("# URL References\n\n## entry")
    (raw / "paper.md").write_text("# Paper\nContent.")
    new = discover_raw(wiki, db)
    assert len(new) == 1
    assert new[0].name == "paper.md"


def test_discover_raw_skips_already_tracked(db, tmp_path):
    from medulla.semantic.store import queue_pending, get_pending_count
    wiki = tmp_path / "wiki"
    raw = wiki / "raw"
    raw.mkdir(parents=True)
    paper = raw / "paper.md"
    paper.write_text("# Paper\nContent.")
    queue_pending(db, str(paper), "md", "Paper")
    new = discover_raw(wiki, db)
    assert len(new) == 0


def test_discover_raw_empty_dir(db, tmp_path):
    wiki = tmp_path / "wiki"
    assert discover_raw(wiki, db) == []


# ── process_pending ────────────────────────────────────────────────────────────

def test_process_pending_creates_wiki_pages(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    md = tmp_path / "logd-study.md"
    md.write_text("# LogD Study\n\nBatch effect analysis content.")
    intake_to_raw(db, wiki, str(md))
    results = process_pending(wiki, db, mock_provider)
    assert len(results) == 1
    assert results[0]["total_pages"] == 3  # source + concept + entity
    assert (wiki / "sources" / "logd-study.md").exists()
    assert (wiki / "concepts" / "logd-prediction.md").exists()
    assert (wiki / "entities" / "syngene.md").exists()


def test_process_pending_marks_done(db, tmp_path, mock_provider):
    from medulla.semantic.store import get_pending_count
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# Paper\n\nContent.")
    intake_to_raw(db, wiki, str(md))
    assert get_pending_count(db) == 1
    process_pending(wiki, db, mock_provider)
    assert get_pending_count(db) == 0


def test_process_pending_handles_error(db, tmp_path):
    from medulla.semantic.store import get_pending_count

    class FailingProvider:
        @property
        def name(self): return "failing"
        @property
        def model(self): return "none"
        def generate(self, *a, **kw): raise RuntimeError("LLM unavailable")

    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# Paper\n\nContent.")
    intake_to_raw(db, wiki, str(md))
    results = process_pending(wiki, db, FailingProvider())
    assert "error" in results[0]
    # Errored items removed from queued
    assert get_pending_count(db) == 0


def test_process_pending_updates_index(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# LogD Study\n\nContent.")
    intake_to_raw(db, wiki, str(md))
    process_pending(wiki, db, mock_provider)
    assert (wiki / "index.md").exists()
    assert "logd-study" in (wiki / "index.md").read_text()


def test_process_pending_empty(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    results = process_pending(wiki, db, mock_provider)
    assert results == []


# ── full flow: intake + discover + process ────────────────────────────────────

def test_full_flow_obsidian_clip_pattern(db, tmp_path, mock_provider):
    """Simulate: Obsidian Clipper drops file → discover → process."""
    wiki = tmp_path / "wiki"
    raw = wiki / "raw"
    raw.mkdir(parents=True)
    # Obsidian Clipper drops file without medulla knowing
    (raw / "article.md").write_text("# Article\n\nContent about logD batch effects.")

    new = discover_raw(wiki, db)
    assert len(new) == 1

    results = process_pending(wiki, db, mock_provider)
    assert results[0]["total_pages"] >= 1


def test_full_flow_multiple_sources(db, tmp_path, mock_provider):
    """Multiple files in raw/ all get processed."""
    wiki = tmp_path / "wiki"
    for i in range(3):
        md = tmp_path / f"paper{i}.md"
        md.write_text(f"# Paper {i}\n\nLogD content {i}.")
        intake_to_raw(db, wiki, str(md))
    results = process_pending(wiki, db, mock_provider)
    assert len(results) == 3
    assert all(r["total_pages"] >= 1 for r in results)


# ── store_wiki_page (MCP pure storage) ────────────────────────────────────────

def test_store_wiki_page_writes_file(db, tmp_path):
    wiki = tmp_path / "wiki"
    content = "---\ntitle: Test\n---\n\n## Summary\n\nLogD batch effects."
    result = store_wiki_page(db, wiki, "Test Page", content)
    assert result["slug"] == "test-page"
    assert Path(result["path"]).exists()


def test_store_wiki_page_with_source_url_appends_log(db, tmp_path):
    wiki = tmp_path / "wiki"
    content = "---\ntitle: Paper\n---\n\n## Summary\n\nContent."
    store_wiki_page(db, wiki, "Paper", content, source_url="https://example.com/paper")
    url_log = wiki / "raw" / "url-references.md"
    assert url_log.exists()
    assert "example.com" in url_log.read_text()


def test_store_wiki_page_indexes_to_db(db, tmp_path):
    from medulla.semantic.store import get_wiki_page
    wiki = tmp_path / "wiki"
    store_wiki_page(db, wiki, "LogD Study", "content", page_type="source")
    page = get_wiki_page(db, "logd-study")
    assert page is not None


# ── ingest_url_mcp ────────────────────────────────────────────────────────────

def test_ingest_url_mcp_creates_raw_and_wiki(db, tmp_path, monkeypatch, mock_provider):
    class MockResponse:
        text = "<html><head><title>LogD Paper</title></head><body><p>Batch effects in logD.</p></body></html>"
        def raise_for_status(self): pass

    monkeypatch.setattr("httpx.get", lambda url, **kw: MockResponse())
    wiki = tmp_path / "wiki"
    result = ingest_url_mcp(db, "https://example.com/paper", wiki, mock_provider)
    assert result.get("total_pages", 0) >= 1
    # raw/ file created
    assert (wiki / "raw").exists()


# ── _parse_llm_response ───────────────────────────────────────────────────────

def test_parse_llm_response_clean_json():
    data = {"source_page": {"title": "Test"}, "concept_pages": [], "entity_pages": []}
    result = _parse_llm_response(json.dumps(data))
    assert result["source_page"]["title"] == "Test"


def test_parse_llm_response_strips_fences():
    data = {"source_page": {"title": "Fenced"}, "concept_pages": [], "entity_pages": []}
    result = _parse_llm_response(f'```json\n{json.dumps(data)}\n```')
    assert result["source_page"]["title"] == "Fenced"


def test_parse_llm_response_bad_json_returns_fallback():
    result = _parse_llm_response("completely invalid!!!")
    assert "source_page" in result
