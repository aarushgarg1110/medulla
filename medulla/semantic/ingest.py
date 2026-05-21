"""Ingest orchestrator: source → LLM → wiki pages → DB + markdown files."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


def ingest(
    conn: sqlite3.Connection,
    source: str,              # file path or URL
    wiki_path: Path,
    provider,                 # LLMProvider instance
    title: str | None = None,
    scope: str = "personal",
) -> dict:
    """
    Full ingest pipeline. Returns summary dict with pages created.
    Raises: ValueError on unsupported source type.
    """
    from medulla.semantic.wiki import (
        INGEST_SYSTEM_PROMPT, INGEST_PROMPT_TEMPLATE,
        slugify, write_source_page, write_concept_page,
        write_entity_page, update_index, append_log, write_raw_source,
    )
    from medulla.semantic.store import upsert_wiki_page

    # 1. Extract text based on source type
    source_type, raw_title, text = _extract_source(source)

    final_title = title or raw_title or Path(source).stem

    # 1b. Write raw/ file for URLs (backtrace from wiki → raw → original source)
    wiki_path.mkdir(parents=True, exist_ok=True)
    if source_type == "url":
        write_raw_source(
            wiki_path, slugify(final_title), text,
            url=source, title=final_title, source_type=source_type,
        )

    # 2. Call LLM to generate structured wiki content
    from datetime import date
    prompt = INGEST_PROMPT_TEMPLATE.format(
        title=final_title,
        source_type=source_type,
        today=date.today().isoformat(),
        text=text[:40_000],  # stay well within context window
    )
    response = provider.generate(prompt, system=INGEST_SYSTEM_PROMPT)
    data = _parse_llm_response(response)

    # 3. Write wiki pages to disk + index to DB
    source_slug = slugify(final_title)
    source_data = data.get("source_page", {})
    source_data["title"] = source_data.get("title") or final_title
    source_data["tags"] = source_data.get("tags", [])

    wiki_path.mkdir(parents=True, exist_ok=True)

    source_path = write_source_page(wiki_path, source_slug, source_data, source, scope)
    upsert_wiki_page(
        conn, source_slug, "source", source_data["title"],
        source_path.read_text(), source_path,
        tags=source_data.get("tags", []),
        scope=scope,
    )
    update_index(wiki_path, source_slug, "source", source_data["title"],
                 (source_data.get("summary") or "")[:80])

    concept_slugs = []
    for cp in data.get("concept_pages", []):
        slug = cp.get("slug") or slugify(cp.get("title", "unknown"))
        path = write_concept_page(wiki_path, slug, cp, source_slug, scope)
        upsert_wiki_page(
            conn, slug, "concept", cp.get("title", slug),
            path.read_text(), path,
            tags=cp.get("tags", []),
            sources=[source_slug],
            scope=scope,
        )
        update_index(wiki_path, slug, "concept", cp.get("title", slug), cp.get("definition", "")[:80])
        concept_slugs.append(slug)

    entity_slugs = []
    for ep in data.get("entity_pages", []):
        slug = ep.get("slug") or slugify(ep.get("title", "unknown"))
        path = write_entity_page(wiki_path, slug, ep, source_slug, scope)
        upsert_wiki_page(
            conn, slug, "entity", ep.get("title", slug),
            path.read_text(), path,
            tags=ep.get("tags", []),
            sources=[source_slug],
            scope=scope,
        )
        update_index(wiki_path, slug, "entity", ep.get("title", slug), ep.get("who_what", "")[:80])
        entity_slugs.append(slug)

    # 4. Append to log
    details = f"Created: {source_slug}\nConcepts: {', '.join(concept_slugs)}\nEntities: {', '.join(entity_slugs)}"
    append_log(wiki_path, "ingest", final_title, details)

    return {
        "source": source_slug,
        "concepts": concept_slugs,
        "entities": entity_slugs,
        "total_pages": 1 + len(concept_slugs) + len(entity_slugs),
    }


def ingest_url(
    conn: sqlite3.Connection,
    url: str,
    wiki_path: Path,
    provider,
    title: str | None = None,
    scope: str = "personal",
) -> dict:
    """Fetch a URL, write raw/ file, call LLM, store wiki pages.

    Used by both CLI (`medulla ingest https://...`) and the
    `medulla_ingest_url` MCP tool (for clients without WebFetch).
    """
    from medulla.semantic.sources.url import extract
    from medulla.semantic.wiki import write_raw_source, slugify
    fetch_title, text = extract(url)
    final_title = title or fetch_title
    slug = slugify(final_title)
    wiki_path.mkdir(parents=True, exist_ok=True)
    write_raw_source(wiki_path, slug, text, url=url, title=final_title, source_type="url")
    return ingest(conn, url, wiki_path, provider, title=final_title, scope=scope)


def store_wiki_page(
    conn: sqlite3.Connection,
    wiki_path: Path,
    title: str,
    content: str,
    page_type: str = "source",
    tags: list[str] | None = None,
    source_url: str | None = None,
    scope: str = "personal",
) -> dict:
    """Store Claude-synthesized content directly — no LLM call.

    Used by `medulla_ingest` MCP tool when Claude has already done the synthesis.
    If source_url provided, also writes a raw/ reference file for backtrace.
    """
    from medulla.semantic.wiki import (
        slugify, write_raw_source, update_index, append_log,
    )
    from medulla.semantic.store import upsert_wiki_page
    from datetime import date

    slug = slugify(title)
    wiki_path.mkdir(parents=True, exist_ok=True)

    # Write the wiki page markdown
    page_dir = wiki_path / f"{page_type}s"
    page_dir.mkdir(exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    page_path.write_text(content)

    # Write raw/ URL reference for backtrace (when Claude used WebFetch)
    if source_url:
        write_raw_source(
            wiki_path, slug,
            f"[Source fetched by LLM — see wiki page for synthesized content]\n\nURL: {source_url}",
            url=source_url, title=title, source_type="url",
        )

    upsert_wiki_page(conn, slug, page_type, title, content, page_path,
                     tags=tags or [], scope=scope)
    update_index(wiki_path, slug, page_type, title, "")
    append_log(wiki_path, "ingest", title, f"Stored via medulla_ingest MCP tool")

    return {"slug": slug, "type": page_type, "path": str(page_path)}


def ingest_text(
    conn: sqlite3.Connection,
    text: str,
    title: str,
    wiki_path: Path,
    provider,
    scope: str = "personal",
) -> dict:
    """Ingest raw text directly (used by medulla_ingest MCP tool)."""
    from medulla.semantic.wiki import (
        INGEST_SYSTEM_PROMPT, INGEST_PROMPT_TEMPLATE,
        slugify, write_source_page, update_index, append_log,
    )
    from medulla.semantic.store import upsert_wiki_page
    from datetime import date

    prompt = INGEST_PROMPT_TEMPLATE.format(
        title=title, source_type="text",
        today=date.today().isoformat(),
        text=text[:40_000],
    )
    response = provider.generate(prompt, system=INGEST_SYSTEM_PROMPT)
    data = _parse_llm_response(response)

    source_slug = slugify(title)
    source_data = data.get("source_page", {})
    source_data["title"] = source_data.get("title") or title
    wiki_path.mkdir(parents=True, exist_ok=True)
    source_path = write_source_page(wiki_path, source_slug, source_data, "agent-ingested", scope)
    upsert_wiki_page(
        conn, source_slug, "source", source_data["title"],
        source_path.read_text(), source_path,
        tags=source_data.get("tags", []),
        scope=scope,
    )
    update_index(wiki_path, source_slug, "source", source_data["title"], "")
    append_log(wiki_path, "ingest", title, f"Ingested via medulla_ingest MCP tool")
    return {"source": source_slug, "concepts": [], "entities": [], "total_pages": 1}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_source(source: str) -> tuple[str, str, str]:
    """Returns (source_type, title, text)."""
    if source.startswith("http://") or source.startswith("https://"):
        from medulla.semantic.sources.url import extract
        title, text = extract(source)
        return "url", title, text

    path = Path(source)
    if not path.exists():
        raise ValueError(f"File not found: {source}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from medulla.semantic.sources.pdf import extract
        text = extract(path)
        return "pdf", path.stem.replace("-", " ").title(), text

    if suffix in (".md", ".txt", ".markdown"):
        from medulla.semantic.sources.markdown import extract
        title, text = extract(path)
        return "markdown", title, text

    # Try as plain text for anything else
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
        return "text", path.stem, text
    except OSError as e:
        raise ValueError(f"Cannot read file: {e}")


def _parse_llm_response(response: str) -> dict:
    """Parse JSON from LLM response, handling fenced code blocks."""
    text = response.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to extract JSON object from response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
        return {"source_page": {"title": "Parse error", "summary": response[:500]}, "concept_pages": [], "entity_pages": []}
