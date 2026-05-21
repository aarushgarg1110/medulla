"""Tests for semantic source extractors — PDF, URL, markdown."""
import pytest
from pathlib import Path


# ── PDF extraction ────────────────────────────────────────────────────────────

def test_pdf_extract_returns_text(tmp_path):
    """Create a minimal PDF and verify text extraction."""
    pytest.importorskip("fitz")
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "LogD prediction is important for ADMET modeling.")
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    from medulla.semantic.sources.pdf import extract
    text = extract(pdf_path)
    assert "LogD" in text
    assert len(text) > 10


def test_pdf_extract_respects_max_chars(tmp_path):
    pytest.importorskip("fitz")
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "X" * 1000)
    pdf_path = tmp_path / "big.pdf"
    doc.save(str(pdf_path))
    doc.close()
    from medulla.semantic.sources.pdf import extract
    text = extract(pdf_path, max_chars=100)
    assert len(text) <= 100


# ── URL extraction ────────────────────────────────────────────────────────────

def test_url_extract_title():
    from medulla.semantic.sources.url import _extract_title
    html = "<html><head><title>LogD Prediction Paper</title></head><body>content</body></html>"
    assert _extract_title(html) == "LogD Prediction Paper"


def test_url_extract_title_fallback_h1():
    from medulla.semantic.sources.url import _extract_title
    html = "<html><body><h1>Article Title</h1><p>content</p></body></html>"
    assert "Article Title" in _extract_title(html)


def test_url_html_to_text_strips_tags():
    from medulla.semantic.sources.url import _html_to_text
    html = "<p>Hello <b>world</b></p><script>evil()</script>"
    text = _html_to_text(html)
    assert "Hello" in text
    assert "world" in text
    assert "<" not in text
    assert "evil" not in text


def test_url_html_to_text_collapses_whitespace():
    from medulla.semantic.sources.url import _html_to_text
    html = "<p>too   many    spaces</p>"
    text = _html_to_text(html)
    assert "  " not in text


# ── Markdown extraction ───────────────────────────────────────────────────────

def test_markdown_extract_title_from_h1(tmp_path):
    md = tmp_path / "paper.md"
    md.write_text("# My Paper Title\n\nContent here.")
    from medulla.semantic.sources.markdown import extract
    title, text = extract(md)
    assert title == "My Paper Title"
    assert "Content here" in text


def test_markdown_extract_title_fallback_to_stem(tmp_path):
    md = tmp_path / "my-paper-notes.md"
    md.write_text("No heading, just content.")
    from medulla.semantic.sources.markdown import extract
    title, text = extract(md)
    assert "My Paper Notes" in title


def test_markdown_extract_respects_max_chars(tmp_path):
    md = tmp_path / "big.md"
    md.write_text("# Title\n\n" + "A" * 1000)
    from medulla.semantic.sources.markdown import extract
    _, text = extract(md, max_chars=100)
    assert len(text) <= 100


# ── Ingest _parse_llm_response ────────────────────────────────────────────────

def test_parse_llm_response_clean_json():
    from medulla.semantic.ingest import _parse_llm_response
    import json
    data = {"source_page": {"title": "Test"}, "concept_pages": [], "entity_pages": []}
    result = _parse_llm_response(json.dumps(data))
    assert result["source_page"]["title"] == "Test"


def test_parse_llm_response_strips_fences():
    from medulla.semantic.ingest import _parse_llm_response
    response = '```json\n{"source_page": {"title": "Fenced"}, "concept_pages": [], "entity_pages": []}\n```'
    result = _parse_llm_response(response)
    assert result["source_page"]["title"] == "Fenced"


def test_parse_llm_response_extracts_embedded_json():
    from medulla.semantic.ingest import _parse_llm_response
    response = 'Here is the result: {"source_page": {"title": "Embedded"}, "concept_pages": [], "entity_pages": []} Done.'
    result = _parse_llm_response(response)
    assert result["source_page"]["title"] == "Embedded"


def test_parse_llm_response_bad_json_returns_fallback():
    from medulla.semantic.ingest import _parse_llm_response
    result = _parse_llm_response("not valid json at all!!!")
    assert "source_page" in result


def test_extract_source_unsupported_file(tmp_path):
    from medulla.semantic.ingest import _extract_source
    import pytest
    bad = tmp_path / "test.xyz"
    bad.write_text("content")
    # Should try as plain text and succeed
    source_type, title, text = _extract_source(str(bad))
    assert "content" in text


def test_extract_source_missing_file():
    from medulla.semantic.ingest import _extract_source
    with pytest.raises(ValueError, match="File not found"):
        _extract_source("/nonexistent/path/file.pdf")


def test_url_extract_mocked(monkeypatch):
    """Test full url.extract() with mocked httpx."""
    class MockResponse:
        text = "<html><head><title>Test Article</title></head><body><p>LogD content here.</p></body></html>"
        def raise_for_status(self): pass

    monkeypatch.setattr("httpx.get", lambda url, **kw: MockResponse())
    from medulla.semantic.sources.url import extract
    title, text = extract("https://example.com/article")
    assert title == "Test Article"
    assert "LogD content" in text
