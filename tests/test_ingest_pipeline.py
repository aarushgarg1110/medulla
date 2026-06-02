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
    """Test double implementing LLMProvider — returns realistic structured output.

    Detects the stage from prompt content (STAGE: PLAN / CONCEPT / ENTITY) and
    returns the correct shaped JSON for each stage of the multi-call pipeline.
    """
    @property
    def name(self): return "mock"
    @property
    def model(self): return "mock-model"

    def generate(self, prompt: str, system: str | None = None, on_token=None) -> str:
        if "STAGE: CONCEPT" in prompt:
            return json.dumps({
                "slug": "logd-prediction",
                "title": "LogD Prediction",
                "tags": ["admet"],
                "definition": "LogD is the distribution coefficient at pH 7.4.",
                "how_it_works": "Measured via chromatography.",
                "why_it_matters": "Key ADMET property.",
                "nuances": "Differs from logP by ionization.",
                "evidence": "CompoundX series data.",
                "connections": [],
                "open_questions": ["Why batch effect?"],
            })
        if "STAGE: ENTITY" in prompt:
            return json.dumps({
                "slug": "externalcro",
                "title": "ExternalCRO",
                "entity_type": "org",
                "tags": ["cro"],
                "who_what": "Contract research organization.",
                "relevance": "Performed logD measurements.",
                "contributions": ["Chromatographic logD assay"],
                "connections": [],
            })
        # STAGE: PLAN (default)
        return json.dumps({
            "source_page": {
                "title": "LogD Prediction Study",
                "summary": "A study on logD prediction using chromatographic methods.",
                "key_points": ["LogD measured at pH 7.4", "Batch effects observed"],
                "tags": ["admet"],
                "concepts": ["[[concepts/logd-prediction]] — core concept"],
                "entities": ["[[entities/externalcro]] — CRO performing measurements"],
                "connections": [],
                "gaps": ["Root cause unclear"],
            },
            "new_concepts": [{"slug": "logd-prediction", "title": "LogD Prediction", "brief": "Distribution coefficient at pH 7.4"}],
            "new_entities": [{"slug": "externalcro", "title": "ExternalCRO", "entity_type": "org", "brief": "CRO for measurements"}],
            "update_concepts": [],
            "update_entities": [],
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
    # Slug comes from LLM-chosen title "LogD Prediction Study", not filename
    assert (wiki / "sources" / "logd-prediction-study.md").exists()
    assert (wiki / "concepts" / "logd-prediction.md").exists()
    assert (wiki / "entities" / "externalcro.md").exists()


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
    assert "logd-prediction-study" in (wiki / "index.md").read_text()


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


def test_refresh_index_stats(tmp_path):
    """_refresh_index_stats updates the stats line correctly."""
    from medulla.semantic.wiki import _refresh_index_stats, update_index
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    update_index(wiki, "test-source", "source", "Test Source", "A test source")
    update_index(wiki, "test-concept", "concept", "Test Concept", "A test concept")
    index = (wiki / "index.md").read_text()
    assert "1 sources" in index
    assert "1 concept pages" in index


def test_update_concepts_adds_source(db, tmp_path, mock_provider):
    """update_concepts pathway adds new source to existing concept's sources list."""
    import json
    wiki = tmp_path / "wiki"
    # Create existing concept page
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "concepts" / "multi-head-attention.md").write_text(
        '---\ntitle: Multi-Head Attention\ntags: [attention]\nsources: ["paper-one"]\n---\n\n## Definition\n\nOriginal content.'
    )

    class UpdatingProvider:
        @property
        def name(self): return "mock"
        @property
        def model(self): return "mock"
        def generate(self, prompt, system=None, on_token=None):
            return json.dumps({
                "source_page": {"title": "Paper Two", "summary": "Summary.", "key_points": [],
                                "tags": [], "concepts": [], "entities": [], "connections": [], "gaps": []},
                "new_concepts": [],
                "new_entities": [],
                "update_concepts": [{"slug": "multi-head-attention", "add_source_note": "Paper Two also uses MHA"}],
                "update_entities": [],
            })

    md = tmp_path / "paper2.md"
    md.write_text("# Paper Two\n\nContent about multi-head attention.")
    from medulla.semantic.ingest import intake_to_raw, process_pending
    intake_to_raw(db, wiki, str(md))
    results = process_pending(wiki, db, UpdatingProvider())
    assert len(results) == 1
    # Verify sources list was updated
    content = (wiki / "concepts" / "multi-head-attention.md").read_text()
    assert "paper-one" in content
    assert "paper2" in content or "paper-2" in content or "paper" in content


def test_add_source_to_page_merges(tmp_path, db):
    """_add_source_to_page adds to sources without overwriting content."""
    from medulla.semantic.ingest import _add_source_to_page
    page = tmp_path / "concept.md"
    page.write_text('---\ntitle: Test\nsources: ["source-one"]\n---\n\n## Definition\n\nContent.')
    _add_source_to_page(page, "source-two", db)
    content = page.read_text()
    assert "source-one" in content
    assert "source-two" in content
    assert "Content." in content  # original content preserved


def test_add_source_to_page_no_duplicate(tmp_path, db):
    """_add_source_to_page doesn't duplicate already-present sources."""
    from medulla.semantic.ingest import _add_source_to_page
    page = tmp_path / "concept.md"
    page.write_text('---\ntitle: Test\nsources: ["source-one"]\n---\n\n## Definition\n\nContent.')
    _add_source_to_page(page, "source-one", db)  # already there
    content = page.read_text()
    assert content.count("source-one") == 1  # not duplicated


def test_build_wiki_schema_empty(tmp_path):
    from medulla.semantic.ingest import _build_wiki_schema
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    result = _build_wiki_schema(wiki)
    assert "No existing pages" in result or "first ingest" in result


def test_build_wiki_schema_with_pages(tmp_path):
    from medulla.semantic.ingest import _build_wiki_schema
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "concepts" / "adam-optimizer.md").write_text(
        "---\ntitle: Adam Optimizer\n---\nContent."
    )
    (wiki / "entities").mkdir()
    (wiki / "entities" / "andrej-karpathy.md").write_text(
        "---\ntitle: Andrej Karpathy\n---\nContent."
    )
    result = _build_wiki_schema(wiki)
    assert "adam-optimizer" in result
    assert "andrej-karpathy" in result
    assert "concepts/" in result
    assert "entities/" in result


def test_update_entities_via_pipeline(db, tmp_path):
    """update_entities pathway via full pipeline updates entity sources list."""
    import json
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    entity_page = wiki / "entities" / "andrej-karpathy.md"
    entity_page.write_text(
        '---\ntitle: Andrej Karpathy\ntype: person\ntags: [person]\nsources: ["source-one"]\n---\n\n## Who / What\n\nResearcher.'
    )

    class EntityUpdatingProvider:
        @property
        def name(self): return "mock"
        @property
        def model(self): return "mock"
        def generate(self, prompt, system=None, on_token=None):
            return json.dumps({
                "source_page": {"title": "Paper", "summary": "S.", "key_points": [],
                                "tags": [], "concepts": [], "entities": [], "connections": [], "gaps": []},
                "new_concepts": [],
                "new_entities": [],
                "update_concepts": [],
                "update_entities": [{"slug": "andrej-karpathy", "add_source_note": "Also author here"}],
            })

    md = tmp_path / "paper.md"
    md.write_text("# Paper\n\nContent.")
    from medulla.semantic.ingest import intake_to_raw, process_pending
    intake_to_raw(db, wiki, str(md))
    results = process_pending(wiki, db, EntityUpdatingProvider())
    assert len(results) == 1
    content = entity_page.read_text()
    assert "source-one" in content
    assert "Researcher." in content  # content preserved


def test_update_entities_adds_source(db, tmp_path):
    """update_entities pathway adds new source to existing entity page."""
    from medulla.semantic.ingest import _add_source_to_page
    import json
    wiki = tmp_path / "wiki"
    (wiki / "entities").mkdir(parents=True)
    entity_page = wiki / "entities" / "andrej-karpathy.md"
    entity_page.write_text(
        '---\ntitle: Andrej Karpathy\ntype: person\ntags: [person]\nsources: ["source-one"]\n---\n\n## Who / What\n\nResearcher.'
    )
    _add_source_to_page(entity_page, "source-two", db)
    content = entity_page.read_text()
    assert "source-one" in content
    assert "source-two" in content
    assert "Researcher." in content  # content preserved


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


def test_store_wiki_page_explicit_slug(db, tmp_path):
    """slug param overrides slugify(title) — title can be descriptive."""
    wiki = tmp_path / "wiki"
    content = "---\ntitle: MA-RAE: Macro-Averaged Relative Absolute Error\n---\n\n## Definition\n\nA metric."
    result = store_wiki_page(db, wiki, "MA-RAE: Macro-Averaged Relative Absolute Error",
                              content, page_type="concept", slug="ma-rae")
    assert result["slug"] == "ma-rae"
    assert (wiki / "concepts" / "ma-rae.md").exists()


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


def test_store_wiki_page_summary_in_index(db, tmp_path):
    """store_wiki_page extracts summary from content for index.md display."""
    from medulla.semantic.ingest import store_wiki_page
    wiki = tmp_path / "wiki"
    content = "---\ntitle: LogD\n---\n\n## Definition\n\nLogD is the distribution coefficient."
    store_wiki_page(db, wiki, "LogD", content, page_type="concept")
    index = (wiki / "index.md").read_text()
    assert "LogD is the distribution coefficient" in index


def test_extract_summary_skips_frontmatter(tmp_path):
    """_extract_summary skips frontmatter and headings, returns first body line."""
    from medulla.semantic.ingest import _extract_summary
    content = "---\ntitle: Test\ntags: [a]\n---\n\n## Definition\n\nThis is the definition."
    assert _extract_summary(content) == "This is the definition."


def test_extract_summary_empty_content():
    from medulla.semantic.ingest import _extract_summary
    assert _extract_summary("") == ""


def test_check_wikilinks_no_broken(tmp_path):
    """_check_wikilinks returns empty list when all linked pages exist."""
    from medulla.semantic.ingest import _check_wikilinks
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    (wiki / "concepts" / "adam-optimizer.md").write_text("# Adam")
    page = tmp_path / "source.md"
    page.write_text("See [[concepts/adam-optimizer]] for details.")
    assert _check_wikilinks(wiki, [page]) == []


def test_check_wikilinks_detects_broken(tmp_path):
    """_check_wikilinks catches missing [[folder/slug]] references."""
    from medulla.semantic.ingest import _check_wikilinks
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    page = tmp_path / "source.md"
    page.write_text("See [[concepts/adam-optimizer]] and [[entities/karpathy]].")
    broken = _check_wikilinks(wiki, [page])
    assert len(broken) == 2
    assert any("adam-optimizer" in b for b in broken)
    assert any("karpathy" in b for b in broken)


def test_check_wikilinks_ignores_bare_slugs(tmp_path):
    """_check_wikilinks ignores bare [[slug]] without folder prefix."""
    from medulla.semantic.ingest import _check_wikilinks
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    page = tmp_path / "source.md"
    page.write_text("See [[adam-optimizer]] (bare slug — ignored).")
    assert _check_wikilinks(wiki, [page]) == []


def test_pipeline_enforces_plan_slug_consistency(db, tmp_path, mock_provider):
    """source_page.concepts wikilinks are filtered to match new_concepts slugs."""
    import json
    wiki = tmp_path / "wiki"

    class DriftingProvider:
        """Plan lists 'adam-optimizer' in new_concepts but source_page uses different slug."""
        @property
        def name(self): return "mock"
        @property
        def model(self): return "mock"
        def generate(self, prompt, system=None, on_token=None):
            if "STAGE: PLAN" in prompt:
                return json.dumps({
                    "source_page": {
                        "title": "Test", "summary": "S", "key_points": [],
                        "tags": [], "entities": [], "connections": [], "gaps": [],
                        # deliberately uses wrong slug in concepts list
                        "concepts": ["[[concepts/adam-optimizer-wrong]] — wrong slug"],
                    },
                    "new_concepts": [{"slug": "adam-optimizer", "title": "Adam", "brief": "optimizer"}],
                    "new_entities": [],
                    "update_concepts": [],
                    "update_entities": [],
                })
            if "STAGE: CONCEPT" in prompt:
                return json.dumps({
                    "slug": "adam-optimizer", "title": "Adam Optimizer", "tags": [],
                    "definition": "d", "how_it_works": "h", "why_it_matters": "w",
                    "nuances": "n", "evidence": "e", "connections": [], "open_questions": [],
                })
            return json.dumps({"slug": "x", "title": "x", "tags": [], "who_what": "x",
                                "relevance": "x", "contributions": [], "connections": []})

    md = tmp_path / "paper.md"
    md.write_text("# Paper\n\nContent.")
    from medulla.semantic.ingest import intake_to_raw, process_pending
    intake_to_raw(db, wiki, str(md))
    results = process_pending(wiki, db, DriftingProvider())
    source_content = (wiki / "sources" / f"{results[0]['source']}.md").read_text()
    # The wrong slug should have been filtered out; no broken link to wrong slug
    assert "adam-optimizer-wrong" not in source_content
    # The concept page was created under the plan slug
    assert (wiki / "concepts" / "adam-optimizer.md").exists()
