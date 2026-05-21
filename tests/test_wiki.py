"""Tests for medulla.semantic.wiki — templates, file writing, lint."""
import pytest
from pathlib import Path
from medulla.semantic.wiki import (
    slugify, write_source_page, write_concept_page, write_entity_page,
    update_index, append_log, lint_wiki, _fmt_bullets, _fmt_tags, _fmt_list,
)


# ── slugify ───────────────────────────────────────────────────────────────────

def test_slugify_basic():
    assert slugify("LogD Prediction") == "logd-prediction"


def test_slugify_special_chars():
    assert slugify("pKa: Acid/Base") == "pka-acidbase"


def test_slugify_long():
    result = slugify("A" * 100)
    assert len(result) <= 60


def test_slugify_hyphens_collapsed():
    assert "--" not in slugify("hello   world")


# ── page writers ──────────────────────────────────────────────────────────────

def test_write_source_page(tmp_path):
    wiki = tmp_path / "wiki"
    data = {
        "title": "LogD Paper",
        "tags": ["admet", "logd"],
        "summary": "A paper about logD prediction.",
        "key_points": ["Point 1", "Point 2"],
        "concepts": ["[[logd-prediction]]"],
        "entities": ["[[chembl]]"],
        "connections": [],
        "gaps": ["Open question 1"],
    }
    path = write_source_page(wiki, "logd-paper", data, "paper.pdf")
    assert path.exists()
    content = path.read_text()
    assert "LogD Paper" in content
    assert "Point 1" in content
    assert "[[logd-prediction]]" in content
    assert "date_ingested:" in content


def test_write_concept_page(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    data = {
        "title": "LogD Prediction",
        "tags": ["admet"],
        "definition": "LogD is the distribution coefficient at pH 7.4.",
        "how_it_works": "Measured via chromatography.",
        "why_it_matters": "Key ADMET property.",
        "nuances": "Differs from logP by accounting for ionization.",
        "evidence": "Seen in Salacia project data.",
        "connections": ["[[admet-prediction]]"],
        "open_questions": ["Why batch effect?"],
    }
    path = write_concept_page(wiki, "logd-prediction", data, "logd-paper")
    assert path.exists()
    content = path.read_text()
    assert "LogD Prediction" in content
    assert "logd-paper" in content
    assert "[[admet-prediction]]" in content


def test_write_concept_page_merges_sources(tmp_path):
    """Writing a concept page twice merges source lists."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    data = {"title": "LogD", "tags": [], "definition": "d", "how_it_works": "h",
            "why_it_matters": "w", "nuances": "n", "evidence": "e",
            "connections": [], "open_questions": []}
    write_concept_page(wiki, "logd", data, "source-1")
    write_concept_page(wiki, "logd", data, "source-2")
    content = (wiki / "concepts" / "logd.md").read_text()
    assert "source-1" in content
    assert "source-2" in content


def test_write_entity_page(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    data = {
        "title": "ChEMBL",
        "entity_type": "database",
        "tags": ["database"],
        "who_what": "Public bioactivity database.",
        "relevance": "Source of external logD data.",
        "contributions": ["2.2M compounds", "18M bioactivities"],
        "connections": ["[[logd-prediction]]"],
    }
    path = write_entity_page(wiki, "chembl", data, "logd-paper")
    assert path.exists()
    content = path.read_text()
    assert "ChEMBL" in content
    assert "database" in content
    assert "2.2M compounds" in content


# ── index and log ──────────────────────────────────────────────────────────────

def test_update_index_creates_file(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    update_index(wiki, "logd-prediction", "concept", "LogD Prediction", "Distribution coefficient")
    index = (wiki / "index.md").read_text()
    assert "logd-prediction" in index
    assert "Concepts" in index


def test_update_index_no_duplicate(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    update_index(wiki, "logd-prediction", "concept", "LogD", "summary")
    update_index(wiki, "logd-prediction", "concept", "LogD", "summary")
    index = (wiki / "index.md").read_text()
    # The wikilink pattern appears exactly once (not twice = not duplicated)
    assert index.count("[[concepts/logd-prediction|logd-prediction]]") == 1


def test_append_log(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    append_log(wiki, "ingest", "My Paper", "Created 3 pages")
    log = (wiki / "log.md").read_text()
    assert "ingest" in log
    assert "My Paper" in log
    assert "Created 3 pages" in log


def test_append_log_multiple_entries(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    append_log(wiki, "ingest", "Paper A")
    append_log(wiki, "ingest", "Paper B")
    log = (wiki / "log.md").read_text()
    assert "Paper A" in log
    assert "Paper B" in log


# ── lint ──────────────────────────────────────────────────────────────────────

def test_lint_missing_wiki(tmp_path):
    result = lint_wiki(tmp_path / "nonexistent")
    assert "error" in result


def test_lint_clean_wiki(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "concepts" / "logd.md").write_text(
        "---\ntitle: LogD\n---\n\nSee also [[admet]] for context.\n"
    )
    (wiki / "concepts" / "admet.md").write_text(
        "---\ntitle: ADMET\n---\n\nSee [[logd]] for lipophilicity.\n"
    )
    result = lint_wiki(wiki)
    assert result["total_pages"] == 2
    assert result["broken_links"] == []
    assert result["orphaned_pages"] == []


def test_lint_detects_broken_link(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "concepts" / "logd.md").write_text(
        "---\ntitle: LogD\n---\n\nSee [[nonexistent-page]] for details.\n"
    )
    result = lint_wiki(wiki)
    assert len(result["broken_links"]) >= 1
    assert any("nonexistent-page" in link for link in result["broken_links"])


def test_lint_detects_orphan(tmp_path):
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "concepts" / "logd.md").write_text("---\ntitle: LogD\n---\nContent here.\n")
    (wiki / "concepts" / "admet.md").write_text("---\ntitle: ADMET\n---\nNo links here.\n")
    result = lint_wiki(wiki)
    # Both are orphans — no inbound links to either
    assert len(result["orphaned_pages"]) == 2


# ── helpers ───────────────────────────────────────────────────────────────────

def test_fmt_bullets_empty():
    assert "(none identified)" in _fmt_bullets([])


def test_fmt_bullets_items():
    result = _fmt_bullets(["a", "b"])
    assert "- a" in result
    assert "- b" in result


def test_fmt_tags_empty():
    assert _fmt_tags([]) == "[]"


def test_fmt_tags_items():
    result = _fmt_tags(["admet", "logd"])
    assert "admet" in result
    assert "logd" in result


def test_fmt_list_empty():
    assert _fmt_list([]) == "[]"


def test_fmt_list_items():
    result = _fmt_list(["source-1", "source-2"])
    assert '"source-1"' in result
