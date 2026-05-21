"""Ingest orchestrator — raw/ is always the intake point.

Flow:
  intake_to_raw()      → copy/fetch source to wiki/raw/, track in pending_ingests
  process_pending()    → run all queued raw/ files through LLM → wiki pages
  discover_raw()       → find raw/ files not yet tracked (e.g. Obsidian Clipper drops)

CLI:
  medulla ingest                → discover_raw() + process_pending()
  medulla ingest paper.pdf      → intake_to_raw() + process_immediately()
  medulla ingest https://url    → intake_to_raw() + process_immediately()

MCP (pure storage — Claude already synthesized):
  store_wiki_page()             → write Claude's content to disk + DB, no LLM call
  medulla_ingest_url tool       → intake_to_raw() + process (for clients without WebFetch)
"""
from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path


# ── Raw/ intake ───────────────────────────────────────────────────────────────

def intake_to_raw(
    conn: sqlite3.Connection,
    wiki_path: Path,
    source: str,
    title: str | None = None,
) -> Path:
    """Copy/fetch source into wiki/raw/ and register in pending_ingests.

    Returns the raw/ path. Does NOT process — call process_pending() after.
    """
    from medulla.semantic.store import queue_pending
    from medulla.semantic.wiki import slugify, write_raw_source

    raw_dir = wiki_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if source.startswith("http://") or source.startswith("https://"):
        from medulla.semantic.sources.url import extract
        fetch_title, text = extract(source)
        final_title = title or fetch_title
        slug = slugify(final_title)
        raw_path = write_raw_source(
            wiki_path, slug, text,
            url=source, title=final_title, source_type="url",
        )
        queue_pending(conn, str(raw_path), "url", final_title)
        return raw_path

    src = Path(source)
    if not src.exists():
        raise ValueError(f"File not found: {source}")

    raw_path = raw_dir / src.name
    shutil.copy2(src, raw_path)
    source_type = src.suffix.lstrip(".").lower() or "text"
    queue_pending(conn, str(raw_path), source_type, title or src.stem)
    return raw_path


def discover_raw(wiki_path: Path, conn: sqlite3.Connection) -> list[Path]:
    """Find raw/ files not yet tracked in pending_ingests.

    Handles files dropped in raw/ by Obsidian Clipper or manually.
    Adds newly discovered files as queued in pending_ingests.
    Returns list of newly discovered paths.
    """
    from medulla.semantic.store import queue_pending

    raw_dir = wiki_path / "raw"
    if not raw_dir.exists():
        return []

    skip = {"url-references.md"}
    tracked = {
        row[0] for row in
        conn.execute("SELECT source_path FROM pending_ingests").fetchall()
    }

    new_files: list[Path] = []
    for f in sorted(raw_dir.iterdir()):
        if f.name in skip or not f.is_file():
            continue
        if str(f) not in tracked:
            source_type = f.suffix.lstrip(".").lower() or "text"
            queue_pending(conn, str(f), source_type, f.stem)
            new_files.append(f)

    return new_files


def process_pending(
    wiki_path: Path,
    conn: sqlite3.Connection,
    provider,
    scope: str = "personal",
) -> list[dict]:
    """Process all queued raw/ files through the LLM into wiki pages.

    Returns list of result dicts per file.
    """
    from medulla.semantic.store import get_pending, mark_pending_done, mark_pending_error

    results = []
    for row in get_pending(conn):
        try:
            result = _process_raw_file(
                conn, Path(row["source_path"]), wiki_path, provider, scope=scope
            )
            mark_pending_done(conn, row["id"])
            result["source_path"] = row["source_path"]
            results.append(result)
        except Exception as e:
            mark_pending_error(conn, row["id"], str(e))
            results.append({"source_path": row["source_path"], "error": str(e)})

    return results


def _process_raw_file(
    conn: sqlite3.Connection,
    raw_path: Path,
    wiki_path: Path,
    provider,
    scope: str = "personal",
) -> dict:
    """Run one raw/ file through the LLM and store wiki pages."""
    suffix = raw_path.suffix.lower()

    if suffix == ".pdf":
        from medulla.semantic.sources.pdf import extract
        text = extract(raw_path)
        title = raw_path.stem.replace("-", " ").title()
        source_ref = str(raw_path)
    elif suffix in (".md", ".txt", ".markdown"):
        # URL-fetched files and markdown clips both land here
        from medulla.semantic.sources.markdown import extract
        title, text = extract(raw_path)
        # Check raw/ frontmatter for original URL
        content = raw_path.read_text(errors="replace")
        source_ref = _extract_frontmatter_url(content) or str(raw_path)
    else:
        text = raw_path.read_text(encoding="utf-8", errors="replace")
        title = raw_path.stem
        source_ref = str(raw_path)

    return _run_llm_pipeline(conn, text, title, wiki_path, provider, source_ref, scope)


def _extract_frontmatter_url(content: str) -> str | None:
    """Extract url: field from markdown frontmatter."""
    import re
    m = re.search(r"^url:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


# ── Core LLM pipeline ─────────────────────────────────────────────────────────

def _run_llm_pipeline(
    conn: sqlite3.Connection,
    text: str,
    title: str,
    wiki_path: Path,
    provider,
    source_ref: str,
    scope: str = "personal",
) -> dict:
    """Call LLM, parse response, write all wiki pages to disk + DB."""
    from medulla.semantic.wiki import (
        INGEST_SYSTEM_PROMPT, INGEST_PROMPT_TEMPLATE,
        slugify, write_source_page, write_concept_page,
        write_entity_page, update_index, append_log,
    )
    from medulla.semantic.store import upsert_wiki_page
    from datetime import date

    source_type = "url" if source_ref.startswith("http") else "file"
    prompt = INGEST_PROMPT_TEMPLATE.format(
        title=title,
        source_type=source_type,
        today=date.today().isoformat(),
        text=text[:40_000],
    )
    response = provider.generate(prompt, system=INGEST_SYSTEM_PROMPT)
    data = _parse_llm_response(response)

    source_slug = slugify(title)
    wiki_path.mkdir(parents=True, exist_ok=True)

    source_data = data.get("source_page", {})
    source_data["title"] = source_data.get("title") or title
    source_data["tags"] = source_data.get("tags", [])

    source_path = write_source_page(wiki_path, source_slug, source_data, source_ref, scope)
    upsert_wiki_page(
        conn, source_slug, "source", source_data["title"],
        source_path.read_text(), source_path,
        tags=source_data.get("tags", []), scope=scope,
    )
    update_index(wiki_path, source_slug, "source", source_data["title"],
                 (source_data.get("summary") or "")[:80])

    concept_slugs = []
    for cp in data.get("concept_pages", []):
        slug = cp.get("slug") or slugify(cp.get("title", "unknown"))
        path = write_concept_page(wiki_path, slug, cp, source_slug, scope)
        upsert_wiki_page(conn, slug, "concept", cp.get("title", slug),
                         path.read_text(), path, tags=cp.get("tags", []),
                         sources=[source_slug], scope=scope)
        update_index(wiki_path, slug, "concept", cp.get("title", slug),
                     cp.get("definition", "")[:80])
        concept_slugs.append(slug)

    entity_slugs = []
    for ep in data.get("entity_pages", []):
        slug = ep.get("slug") or slugify(ep.get("title", "unknown"))
        path = write_entity_page(wiki_path, slug, ep, source_slug, scope)
        upsert_wiki_page(conn, slug, "entity", ep.get("title", slug),
                         path.read_text(), path, tags=ep.get("tags", []),
                         sources=[source_slug], scope=scope)
        update_index(wiki_path, slug, "entity", ep.get("title", slug),
                     ep.get("who_what", "")[:80])
        entity_slugs.append(slug)

    append_log(wiki_path, "ingest", title,
               f"Source: {source_slug}\nConcepts: {', '.join(concept_slugs)}\nEntities: {', '.join(entity_slugs)}")

    return {
        "source": source_slug,
        "concepts": concept_slugs,
        "entities": entity_slugs,
        "total_pages": 1 + len(concept_slugs) + len(entity_slugs),
    }


# ── MCP pure-storage path (Claude already synthesized) ────────────────────────

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

    Claude IS the LLM when using MCP. This is pure storage.
    If source_url: appends to url-references.md for backtrace.
    """
    from medulla.semantic.wiki import slugify, append_url_reference, update_index, append_log
    from medulla.semantic.store import upsert_wiki_page

    slug = slugify(title)
    wiki_path.mkdir(parents=True, exist_ok=True)

    page_dir = wiki_path / f"{page_type}s"
    page_dir.mkdir(exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    page_path.write_text(content)

    if source_url:
        append_url_reference(wiki_path, slug, source_url, title=title)

    upsert_wiki_page(conn, slug, page_type, title, content, page_path,
                     tags=tags or [], scope=scope)
    update_index(wiki_path, slug, page_type, title, "")
    append_log(wiki_path, "ingest", title, "Stored via medulla_ingest MCP tool")

    return {"slug": slug, "type": page_type, "path": str(page_path)}


# ── Legacy helpers (kept for MCP medulla_ingest_url) ──────────────────────────

def ingest_url_mcp(
    conn: sqlite3.Connection,
    url: str,
    wiki_path: Path,
    provider,
    title: str | None = None,
    scope: str = "personal",
) -> dict:
    """Fetch URL + write raw/ + process. For MCP clients without WebFetch."""
    raw_path = intake_to_raw(conn, wiki_path, url, title)
    results = process_pending(wiki_path, conn, provider, scope)
    return results[0] if results else {"source": "", "total_pages": 0}


def _parse_llm_response(response: str) -> dict:
    text = response.strip()
    if text.startswith("```"):
        text = "\n".join(text.split("\n")[1:])
    if text.endswith("```"):
        text = "\n".join(text.split("\n")[:-1])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {"source_page": {"title": "Parse error", "summary": response[:500]},
                "concept_pages": [], "entity_pages": []}
