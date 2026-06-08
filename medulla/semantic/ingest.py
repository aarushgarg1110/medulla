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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ── Raw/ intake ───────────────────────────────────────────────────────────────

def intake_to_raw(
    conn: sqlite3.Connection,
    wiki_path: Path,
    source: str,
    title: str | None = None,
    force: bool = False,
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
        from medulla.semantic.wiki import append_url_reference
        # Dedup key is the URL itself — deterministic regardless of LLM title choice
        fetch_title, text = extract(source)
        final_title = title or fetch_title
        slug = slugify(final_title)
        append_url_reference(wiki_path, slug, source, title=final_title)
        import tempfile
        tmp = Path(tempfile.mktemp(suffix=".md", prefix=f"{slug}_"))
        tmp.write_text(f"---\nurl: {source}\ntitle: {final_title}\nsource_type: url\n---\n\n{text}")
        queue_pending(conn, source, "url", final_title, force=force, processing_path=str(tmp))
        return tmp

    src = Path(source)
    if not src.exists():
        raise ValueError(f"File not found: {source}")

    raw_path = raw_dir / src.name
    shutil.copy2(src, raw_path)
    source_type = src.suffix.lstrip(".").lower() or "text"

    # For binary files use SHA-256 of content as dedup key — catches same PDF under different filenames
    import hashlib
    content_hash = hashlib.sha256(src.read_bytes()).hexdigest()
    dedup_key = f"sha256:{content_hash}"
    queue_pending(conn, dedup_key, source_type, title or src.stem, force=force, processing_path=str(raw_path))
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
    on_token=None,
) -> list[dict]:
    """Process all queued raw/ files through the LLM into wiki pages.

    Returns list of result dicts per file.
    """
    from medulla.semantic.store import get_pending, mark_pending_done, mark_pending_error

    results = []
    for row in get_pending(conn):
        # processing_path is the actual file; source_path is the dedup key (URL or sha256)
        processing_path = row["processing_path"] or row["source_path"]
        raw_path = Path(processing_path)
        if not raw_path.exists():  # pragma: no cover
            mark_pending_error(conn, row["id"], "raw file no longer exists")
            continue
        try:
            result = _process_raw_file(
                conn, raw_path, wiki_path, provider, scope=scope, on_token=on_token
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
    on_token=None,
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
    else:  # pragma: no cover
        text = raw_path.read_text(encoding="utf-8", errors="replace")
        title = raw_path.stem
        source_ref = str(raw_path)

    return _run_llm_pipeline(conn, text, title, wiki_path, provider, source_ref, scope, on_token=on_token)


def _extract_frontmatter_url(content: str) -> str | None:
    """Extract url: field from markdown frontmatter."""
    import re
    m = re.search(r"^url:\s*(.+)$", content, re.MULTILINE)
    return m.group(1).strip() if m else None


def _check_wikilinks(wiki_path: Path, pages: list[Path]) -> list[str]:
    """Return broken [[folder/slug]] wikilinks found in the given pages.

    Only checks links that use the folder-prefixed format (concepts/X, entities/X,
    sources/X). Bare [[slug]] links are ignored.
    """
    import re
    pattern = re.compile(r"\[\[(concepts|entities|sources)/([^\]|]+)\]\]")
    broken = []
    for page in pages:
        if not page.exists():
            continue
        content = page.read_text(errors="replace")
        for m in pattern.finditer(content):
            folder, slug = m.group(1), m.group(2).strip()
            target = wiki_path / folder / f"{slug}.md"
            if not target.exists():
                broken.append(f"[[{folder}/{slug}]] (referenced in {page.name})")
    return broken


# ── Core LLM pipeline ─────────────────────────────────────────────────────────

def _extract_summary(content: str) -> str:
    """Extract first substantive sentence from wiki page content for index display."""
    in_frontmatter = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if not stripped or stripped.startswith("#"):
            continue
        return stripped[:80]
    return ""


def _clean_slug(slug: str) -> str:
    """Strip folder prefixes and reject placeholder slugs from LLM template copying."""
    slug = slug.split("/")[-1] if "/" in slug else slug
    # Reject slugs that are literally the prompt placeholder examples
    _PLACEHOLDERS = {"concept-slug", "entity-slug", "actual-concept-name", "actual-entity-name",
                     "existing-concept-slug", "existing-entity-slug", "new-concept-slug"}
    return "" if slug in _PLACEHOLDERS else slug


def _build_wiki_schema(wiki_path: Path) -> str:
    """Build a compact schema of all existing wiki pages for LLM injection.

    This is the key to accurate wikilinks — the LLM can only link to slugs
    that exist (or that it's about to create). Without this, it invents slugs
    that don't match existing pages and the graph fragments.
    """
    import re
    _DIRS = {"concepts": "concept", "entities": "entity", "sources": "source"}
    lines = []
    for dir_name, page_type in _DIRS.items():
        d = wiki_path / dir_name
        if not d.exists():
            continue
        for md in sorted(d.glob("*.md")):
            slug = md.stem
            content = md.read_text(errors="replace")
            # Extract title from frontmatter
            m = re.search(r"^title:\s*(.+)$", content, re.MULTILINE)
            title_str = m.group(1).strip() if m else slug
            lines.append(f"[[{dir_name}/{slug}]] ({page_type}) — {title_str}")
    if not lines:
        return "No existing pages yet. This is the first ingest — create concepts and entities freely."
    return "\n".join(lines)


def _run_llm_pipeline(
    conn: sqlite3.Connection,
    text: str,
    title: str,
    wiki_path: Path,
    provider,
    source_ref: str,
    scope: str = "personal",
    on_token=None,
) -> dict:
    """Multi-call ingest pipeline: plan → source page → per-concept → per-entity.

    Each call targets <2000 output tokens, fitting within Bedrock's streaming cap.
    CLI sees checkpoint lines via print(); on_token still streams tokens within each call.
    """
    from medulla.semantic.wiki import (
        PLAN_SYSTEM_PROMPT, PLAN_PROMPT_TEMPLATE,
        CONCEPT_PROMPT_TEMPLATE, ENTITY_PROMPT_TEMPLATE,
        TAG_VOCABULARY,
        slugify, write_source_page, write_concept_page,
        write_entity_page, update_index, append_log,
    )
    from medulla.semantic.store import upsert_wiki_page
    from datetime import date

    source_type = "url" if source_ref.startswith("http") else "file"
    schema = _build_wiki_schema(wiki_path)
    tag_vocab = ", ".join(TAG_VOCABULARY)
    today = date.today().isoformat()
    truncated_text = text[:40_000]
    wiki_path.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: planning call ────────────────────────────────────────────────
    plan_prompt = PLAN_PROMPT_TEMPLATE.format(
        title=title,
        source_type=source_type,
        today=today,
        text=truncated_text,
        wiki_schema=schema,
        tag_vocabulary=tag_vocab,
    )
    plan_response = provider.generate(plan_prompt, system=PLAN_SYSTEM_PROMPT, on_token=on_token)
    plan = _parse_llm_response(plan_response)

    source_data = plan.get("source_page", {})
    # Slug from LLM title — better than programmatic extraction for messy files
    # (e.g. Perplexity exports have "# You" as H1; LLM chooses a real title)
    source_data["title"] = source_data.get("title") or title
    source_slug = slugify(source_data["title"])
    source_data.setdefault("tags", [])

    new_concepts = plan.get("new_concepts", [])

    # Filter author-only person entities — route them to source_page.authors instead.
    # A person entity is an author if their slug matches a slugified author name from
    # source_page.authors. This enforces the prompt rule in code so a single LLM lapse
    # doesn't silently create a useless person page.
    author_slugs = {
        slugify(name)
        for name in source_data.get("authors", [])
        if name
    }
    new_entities_raw = plan.get("new_entities", [])
    new_entities = [
        ne for ne in new_entities_raw
        if not (ne.get("entity_type") == "person"
                and _clean_slug(ne.get("slug", "")) in author_slugs)
    ]
    skipped_authors = [
        ne["slug"] for ne in new_entities_raw if ne not in new_entities
    ]
    if skipped_authors:
        print(f"  ↷ Skipped author entities (added to source authors): {', '.join(skipped_authors)}")

    n_concepts = len(new_concepts)
    n_entities = len(new_entities)
    n_updates = len(plan.get("update_concepts", [])) + len(plan.get("update_entities", []))

    # Enforce consistency: rebuild source_page concept/entity wikilinks from the
    # plan slugs so they can never drift from what will actually be created.
    planned_concept_slugs = {_clean_slug(nc.get("slug", "")) for nc in new_concepts}
    planned_concept_slugs |= {_clean_slug(uc.get("slug", "")) for uc in plan.get("update_concepts", [])}
    planned_entity_slugs = {_clean_slug(ne.get("slug", "")) for ne in new_entities}
    planned_entity_slugs |= {_clean_slug(ue.get("slug", "")) for ue in plan.get("update_entities", [])}

    def _filter_wikilinks(entries: list, folder: str, valid_slugs: set) -> list:
        """Keep only entries whose [[folder/slug]] matches the plan."""
        import re
        out = []
        for entry in entries:
            m = re.search(r"\[\[" + folder + r"/([^\]|]+)\]\]", entry)
            if m and _clean_slug(m.group(1)) in valid_slugs:
                out.append(entry)
        return out

    source_data["concepts"] = _filter_wikilinks(
        source_data.get("concepts", []), "concepts", planned_concept_slugs)
    source_data["entities"] = _filter_wikilinks(
        source_data.get("entities", []), "entities", planned_entity_slugs)

    print(f"  ✓ Source page ready — plan: {n_concepts} concepts, {n_entities} entities"
          + (f", {n_updates} updates" if n_updates else ""))

    # Write source page
    source_path = write_source_page(wiki_path, source_slug, source_data, source_ref, scope)
    upsert_wiki_page(
        conn, source_slug, "source", source_data["title"],
        source_path.read_text(), source_path,
        tags=source_data.get("tags", []), scope=scope,
    )
    update_index(wiki_path, source_slug, "source", source_data["title"],
                 (source_data.get("summary") or "")[:80])

    # Rebuild schema after adding source page so concepts can wikilink it
    schema = _build_wiki_schema(wiki_path)

    # Session slugs: all concepts + entities being created this ingest — valid wikilink targets
    # even before their pages are written. Prevents drift (autograd vs autograd-engine).
    session_concept_slugs = [_clean_slug(nc.get("slug", "")) for nc in new_concepts if nc.get("slug")]
    session_entity_slugs = [_clean_slug(ne.get("slug", "")) for ne in new_entities if ne.get("slug")]
    session_schema = "\n".join(
        [f"[[concepts/{s}]]" for s in session_concept_slugs if s] +
        [f"[[entities/{s}]]" for s in session_entity_slugs if s]
    ) or "None — this is the only page being created this session."

    # ── Stage 2: per-concept calls (parallel LLM, sequential writes) ─────────
    # Workers do only the LLM call + parse (thread-safe).
    # Main thread handles disk + DB writes as futures complete via as_completed
    # so checkpoints print as each call finishes, not in a batch at the end.
    concept_slugs = []

    def _llm_concept(nc: dict, index: int) -> tuple[str, dict, int]:
        slug = _clean_slug(nc.get("slug") or slugify(nc.get("title", "unknown")))
        prompt = CONCEPT_PROMPT_TEMPLATE.format(
            title=nc.get("title", slug),
            concept_slug=slug,
            brief=nc.get("brief", ""),
            source_title=title,
            wiki_schema=schema,
            session_schema=session_schema,
            tag_vocabulary=tag_vocab,
            text=truncated_text[:8_000],
        )
        response = provider.generate(prompt, system=PLAN_SYSTEM_PROMPT, on_token=on_token)
        cp = _parse_llm_response(response)
        cp["title"] = cp.get("title") or nc.get("title", slug)
        return slug, cp, index

    with ThreadPoolExecutor(max_workers=min(len(new_concepts), 8) or 1) as executor:
        futures = {executor.submit(_llm_concept, nc, i): nc
                   for i, nc in enumerate(new_concepts, 1)}
        for future in as_completed(futures):
            slug, cp, idx = future.result()
            path = write_concept_page(wiki_path, slug, cp, source_slug, scope)
            upsert_wiki_page(conn, slug, "concept", cp.get("title", slug),
                             path.read_text(), path, tags=cp.get("tags", []),
                             sources=[source_slug], scope=scope)
            update_index(wiki_path, slug, "concept", cp.get("title", slug),
                         cp.get("definition", "")[:80])
            concept_slugs.append(slug)
            print(f"    ✓ {slug} ({idx}/{n_concepts})")

    if concept_slugs:
        schema = _build_wiki_schema(wiki_path)

    # ── Stage 3: per-entity calls (parallel, after all concepts finish) ───────
    entity_slugs = []

    def _llm_entity(ne: dict, index: int) -> tuple[str, dict, int]:
        slug = _clean_slug(ne.get("slug") or slugify(ne.get("title", "unknown")))
        prompt = ENTITY_PROMPT_TEMPLATE.format(
            title=ne.get("title", slug),
            entity_slug=slug,
            entity_type=ne.get("entity_type", "tool"),
            brief=ne.get("brief", ""),
            source_title=title,
            wiki_schema=schema,
            session_schema=session_schema,
            tag_vocabulary=tag_vocab,
            text=truncated_text[:8_000],
        )
        response = provider.generate(prompt, system=PLAN_SYSTEM_PROMPT, on_token=on_token)
        ep = _parse_llm_response(response)
        ep["title"] = ep.get("title") or ne.get("title", slug)
        return slug, ep, index

    with ThreadPoolExecutor(max_workers=min(len(new_entities), 8) or 1) as executor:
        futures = {executor.submit(_llm_entity, ne, i): ne
                   for i, ne in enumerate(new_entities, 1)}
        for future in as_completed(futures):
            slug, ep, idx = future.result()
            path = write_entity_page(wiki_path, slug, ep, source_slug, scope)
            upsert_wiki_page(conn, slug, "entity", ep.get("title", slug),
                             path.read_text(), path, tags=ep.get("tags", []),
                             sources=[source_slug], scope=scope)
            update_index(wiki_path, slug, "entity", ep.get("title", slug),
                         ep.get("who_what", "")[:80])
            entity_slugs.append(slug)
            print(f"    ✓ {slug} ({idx}/{n_entities})")

    # ── Stage 4: update existing pages (no LLM call needed) ──────────────────
    updated_concepts = []
    for uc in plan.get("update_concepts", []):
        slug = _clean_slug(uc.get("slug", "").strip())
        if not slug:
            continue
        existing_path = wiki_path / "concepts" / f"{slug}.md"
        if existing_path.exists():
            _add_source_to_page(existing_path, source_slug, conn)
            updated_concepts.append(slug)

    updated_entities = []
    for ue in plan.get("update_entities", []):
        slug = _clean_slug(ue.get("slug", "").strip())
        if not slug:
            continue
        existing_path = wiki_path / "entities" / f"{slug}.md"
        if existing_path.exists():
            _add_source_to_page(existing_path, source_slug, conn)
            updated_entities.append(slug)

    if updated_concepts or updated_entities:
        print(f"    ↻ Updated existing: {', '.join(updated_concepts + updated_entities)}")

    updated_note = ""
    if updated_concepts or updated_entities:
        updated_note = f"\nUpdated: {', '.join(updated_concepts + updated_entities)}"

    append_log(wiki_path, "ingest", title,
               f"Source: {source_slug}\nConcepts: {', '.join(concept_slugs)}"
               f"\nEntities: {', '.join(entity_slugs)}{updated_note}")

    # ── Final wikilink validation ─────────────────────────────────────────────
    pages_written = (
        [wiki_path / "sources" / f"{source_slug}.md"]
        + [wiki_path / "concepts" / f"{s}.md" for s in concept_slugs]
        + [wiki_path / "entities" / f"{s}.md" for s in entity_slugs]
    )
    broken = _check_wikilinks(wiki_path, pages_written)
    if broken:
        print(f"\n  ⚠ Broken wikilinks in pages written this session:")
        for link in broken:
            print(f"    {link}")

    return {
        "source": source_slug,
        "concepts": concept_slugs,
        "entities": entity_slugs,
        "updated": updated_concepts + updated_entities,
        "total_pages": 1 + len(concept_slugs) + len(entity_slugs),
        "broken_wikilinks": broken,
    }


def _add_source_to_page(page_path: Path, source_slug: str, conn) -> None:
    """Add source_slug to the sources: list in a concept/entity page's frontmatter."""
    import re as _re
    content = page_path.read_text(errors="replace")
    # Parse existing sources list from frontmatter
    sources_match = _re.search(r"^sources:\s*\[([^\]]*)\]", content, _re.MULTILINE)
    if sources_match:
        existing = [s.strip().strip('"').strip("'")
                    for s in sources_match.group(1).split(",") if s.strip()]
        if source_slug not in existing:
            existing.append(source_slug)
            new_sources = "[" + ", ".join(f'"{s}"' for s in existing) + "]"
            content = content[:sources_match.start()] + f"sources: {new_sources}" + content[sources_match.end():]
            page_path.write_text(content)
            # Update DB content
            from medulla.semantic.store import upsert_wiki_page
            slug = page_path.stem
            page_type = page_path.parent.name.rstrip("s")  # concepts→concept, entities→entity
            if page_type == "entitie":
                page_type = "entity"
            title_match = _re.search(r"^title:\s*(.+)$", content, _re.MULTILINE)
            title = title_match.group(1).strip() if title_match else slug
            upsert_wiki_page(conn, slug, page_type, title, content, page_path,
                             tags=[], sources=existing)


# ── MCP pure-storage path (Claude already synthesized) ────────────────────────

def store_wiki_page(
    conn: sqlite3.Connection,
    wiki_path: Path,
    title: str,
    content: str,
    page_type: str = "source",
    tags: list[str] | None = None,
    source_url: str | None = None,
    source_path: str | None = None,
    scope: str = "personal",
    slug: str | None = None,
) -> dict:
    """Store Claude-synthesized content directly — no LLM call.

    Claude IS the LLM when using MCP. This is pure storage.
    - slug: explicit slug override — use this to decouple the wikilink slug
      from the title. If omitted, slugify(title) is used.
    - source_url: WebFetch URL → appended to url-references.md log
    - source_path: local file path → file copied to wiki/raw/ for backtrace
    Both can be provided (e.g. PDF downloaded from a URL).
    """
    import shutil
    from medulla.semantic.wiki import (
        slugify, append_url_reference, update_index, append_log, write_raw_source,
    )
    from medulla.semantic.store import upsert_wiki_page

    slug = slug or slugify(title)
    wiki_path.mkdir(parents=True, exist_ok=True)

    _DIR = {"source": "sources", "concept": "concepts", "entity": "entities"}
    page_dir = wiki_path / _DIR.get(page_type, f"{page_type}s")
    page_dir.mkdir(exist_ok=True)
    page_path = page_dir / f"{slug}.md"
    page_path.write_text(content)

    # Local file → copy to raw/ for immutable archive + backtrace
    if source_path:
        src = Path(source_path)
        if src.exists():
            raw_dir = wiki_path / "raw"
            raw_dir.mkdir(exist_ok=True)
            raw_dest = raw_dir / src.name
            if not raw_dest.exists():
                shutil.copy2(src, raw_dest)

    # WebFetch URL → append to url-references.md ONLY when no local file was copied.
    # If source_path was provided, the file is in raw/ and its frontmatter has the URL.
    if source_url and not source_path:
        append_url_reference(wiki_path, slug, source_url, title=title)

    upsert_wiki_page(conn, slug, page_type, title, content, page_path,
                     tags=tags or [], scope=scope)
    update_index(wiki_path, slug, page_type, title, _extract_summary(content))
    append_log(wiki_path, "ingest", title, "Stored via medulla_ingest MCP tool")

    broken = _check_wikilinks(wiki_path, [page_path])
    return {"slug": slug, "type": page_type, "path": str(page_path), "broken_wikilinks": broken}


# ── Legacy helpers (kept for MCP medulla_ingest_url) ──────────────────────────

def ingest_url_mcp(
    conn: sqlite3.Connection,
    url: str,
    wiki_path: Path,
    provider,
    title: str | None = None,
    scope: str = "personal",
) -> dict:
    """Fetch URL + log to url-references.md + process. For MCP clients without WebFetch.

    URLs are logged in url-references.md only — raw/ reserved for binary files.
    """
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
