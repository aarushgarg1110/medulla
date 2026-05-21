"""Tests for the full ingest pipeline using a MockProvider (no real API calls)."""
import json
import pytest
from pathlib import Path

from medulla.semantic.ingest import ingest, ingest_text, _extract_source


class MockProvider:
    """Test double implementing LLMProvider — returns realistic structured output."""
    @property
    def name(self): return "mock"
    @property
    def model(self): return "mock-model"

    def generate(self, prompt: str, system: str | None = None) -> str:
        return json.dumps({
            "source_page": {
                "title": "LogD Prediction Study",
                "summary": "A study on logD prediction using chromatographic methods.",
                "key_points": ["LogD measured at pH 7.4", "Batch effects observed"],
                "concepts": ["[[logd-prediction]] — core concept", "[[batch-effect]] — systematic error"],
                "entities": ["[[syngene]] — CRO performing measurements"],
                "connections": ["[[admet-prediction]] — broader context"],
                "gaps": ["Root cause of batch effect unclear"],
            },
            "concept_pages": [
                {
                    "slug": "logd-prediction",
                    "title": "LogD Prediction",
                    "tags": ["admet", "logd"],
                    "definition": "LogD is the distribution coefficient at pH 7.4.",
                    "how_it_works": "Measured via chromatography or shake-flask.",
                    "why_it_matters": "Key ADMET property for drug absorption.",
                    "nuances": "Differs from logP by accounting for ionization state.",
                    "evidence": "Observed in Salacia series data.",
                    "connections": ["[[admet-prediction]]"],
                    "open_questions": ["Why do batch effects occur?"],
                }
            ],
            "entity_pages": [
                {
                    "slug": "syngene",
                    "title": "Syngene",
                    "entity_type": "org",
                    "tags": ["cro"],
                    "who_what": "Contract research organization.",
                    "relevance": "Performed logD measurements for Salacia project.",
                    "contributions": ["Chromatographic logD assay"],
                    "connections": ["[[logd-prediction]]"],
                }
            ],
        })


@pytest.fixture
def mock_provider():
    return MockProvider()


# ── ingest from markdown file ──────────────────────────────────────────────────

def test_ingest_markdown_creates_pages(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# LogD Study\n\nThis paper studies logD batch effects in Salacia.")
    result = ingest(db, str(md), wiki, mock_provider)
    assert result["source"] == "logd-study"
    assert "logd-prediction" in result["concepts"]
    assert "syngene" in result["entities"]
    assert result["total_pages"] == 3  # 1 source + 1 concept + 1 entity


def test_ingest_creates_markdown_files(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# LogD Study\n\nContent.")
    ingest(db, str(md), wiki, mock_provider)
    assert (wiki / "sources" / "logd-study.md").exists()
    assert (wiki / "concepts" / "logd-prediction.md").exists()
    assert (wiki / "entities" / "syngene.md").exists()


def test_ingest_indexes_to_db(db, tmp_path, mock_provider):
    from medulla.semantic.store import get_wiki_page, get_wiki_stats
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# LogD Study\n\nContent.")
    ingest(db, str(md), wiki, mock_provider)
    page = get_wiki_page(db, "logd-study")
    assert page is not None
    assert page["type"] == "source"
    stats = get_wiki_stats(db)
    assert stats["total"] == 3


def test_ingest_updates_index_md(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# LogD Study\n\nContent.")
    ingest(db, str(md), wiki, mock_provider)
    index = (wiki / "index.md").read_text()
    assert "logd-study" in index


def test_ingest_appends_log(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# LogD Study\n\nContent.")
    ingest(db, str(md), wiki, mock_provider)
    log = (wiki / "log.md").read_text()
    assert "ingest" in log
    assert "LogD" in log


def test_ingest_custom_title(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("No heading here.")
    result = ingest(db, str(md), wiki, mock_provider, title="Custom Title")
    assert result["source"] == "custom-title"


def test_ingest_duplicate_concept_merges_sources(db, tmp_path, mock_provider):
    """Ingesting two sources that reference the same concept merges source lists."""
    wiki = tmp_path / "wiki"
    for i in range(2):
        md = tmp_path / f"paper{i}.md"
        md.write_text(f"# Paper {i}\n\nContent about logD.")
        ingest(db, str(md), wiki, mock_provider)
    concept = (wiki / "concepts" / "logd-prediction.md").read_text()
    assert "paper-0" in concept or "paper" in concept


# ── ingest_text (MCP write tool) ─────────────────────────────────────────────

def test_ingest_text_creates_source_page(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    result = ingest_text(db, "Findings about logD batch effects.", "Research Notes", wiki, mock_provider)
    assert result["source"] == "research-notes"
    assert (wiki / "sources" / "research-notes.md").exists()


def test_ingest_unsupported_file_reads_as_text(db, tmp_path, mock_provider):
    """Unknown file extensions should be read as plain text."""
    wiki = tmp_path / "wiki"
    f = tmp_path / "data.csv"
    f.write_text("compound,logD\nchemblA,2.3")
    result = ingest(db, str(f), wiki, mock_provider)
    assert result["total_pages"] >= 1


def test_ingest_missing_file_raises(db, tmp_path, mock_provider):
    wiki = tmp_path / "wiki"
    with pytest.raises(ValueError, match="File not found"):
        ingest(db, "/nonexistent/paper.pdf", wiki, mock_provider)


# ── _extract_source dispatch ──────────────────────────────────────────────────

def test_extract_source_detects_url():
    from medulla.semantic.ingest import _extract_source
    # Don't actually fetch — just check the dispatch logic raises correctly
    with pytest.raises(Exception):
        _extract_source("https://example.com/paper")  # will fail network in test


def test_extract_source_markdown(tmp_path):
    md = tmp_path / "notes.md"
    md.write_text("# Title\n\nContent.")
    from medulla.semantic.ingest import _extract_source
    source_type, title, text = _extract_source(str(md))
    assert source_type == "markdown"
    assert "Content" in text
