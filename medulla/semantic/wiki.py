"""Wiki page templates and index/log maintenance.

Page format inherited from Nimbus-Brain / Karpathy LLM Wiki pattern.
Three page types: source, concept, entity.
"""
from __future__ import annotations

import re
from datetime import date
from pathlib import Path


# ── Page templates ─────────────────────────────────────────────────────────────

SOURCE_TEMPLATE = """\
---
title: {title}
source: {source}
date_ingested: {date_ingested}
tags: {tags}
scope: {scope}
ingested_by: medulla
---

## Summary

{summary}

## Key Points

{key_points}

## Concepts Introduced or Updated

{concepts}

## Entities Mentioned

{entities}

## Connections

{connections}

## Gaps / Open Questions

{gaps}
"""

CONCEPT_TEMPLATE = """\
---
title: {title}
tags: {tags}
sources: {sources}
scope: {scope}
ingested_by: medulla
---

## Definition

{definition}

## How It Works

{how_it_works}

## Why It Matters

{why_it_matters}

## Nuances & Caveats

{nuances}

## Evidence & Examples

{evidence}

## Connections

{connections}

## Open Questions

{open_questions}
"""

ENTITY_TEMPLATE = """\
---
title: {title}
type: {entity_type}
tags: {tags}
sources: {sources}
scope: {scope}
ingested_by: medulla
---

## Who / What

{who_what}

## Relevance

{relevance}

## Key Contributions / Features

{contributions}

## Connections

{connections}
"""

INGEST_SYSTEM_PROMPT = """\
You are maintaining a personal knowledge wiki following the Karpathy LLM Wiki pattern.
Your job is to read a source document and produce structured wiki pages in JSON format.

Rules:
- Use [[slug]] wikilinks for cross-references (slugs are lowercase-hyphenated)
- Be concrete and specific — no vague generalities
- Extract actual facts, findings, and connections from the source
- Identify 2-5 significant concepts and 1-3 significant entities worth their own pages
- For existing concepts/entities, note what this source adds or contradicts

Output valid JSON only — no markdown fences, no explanation outside the JSON.
"""

INGEST_PROMPT_TEMPLATE = """\
Source title: {title}
Source type: {source_type}
Date: {today}

Source text:
---
{text}
---

Produce a JSON object with this exact structure:
{{
  "source_page": {{
    "title": "...",
    "summary": "2-4 paragraph synthesis",
    "key_points": ["bullet 1", "bullet 2", ...],
    "concepts": ["[[concept-slug]] — one-line note", ...],
    "entities": ["[[entity-slug]] — role description", ...],
    "connections": ["[[related-page]] — how connected", ...],
    "gaps": ["open question 1", ...]
  }},
  "concept_pages": [
    {{
      "slug": "concept-slug",
      "title": "Concept Name",
      "tags": ["tag1", "tag2"],
      "definition": "1-3 sentence plain-language definition",
      "how_it_works": "concrete mechanism",
      "why_it_matters": "practical significance",
      "nuances": "edge cases, caveats",
      "evidence": "examples from this source",
      "connections": ["[[related]]", ...],
      "open_questions": ["question 1", ...]
    }}
  ],
  "entity_pages": [
    {{
      "slug": "entity-slug",
      "title": "Entity Name",
      "entity_type": "person|org|tool|project|database",
      "tags": ["tag1"],
      "who_what": "1-2 sentence description",
      "relevance": "why it matters in this wiki",
      "contributions": ["feature 1", "feature 2"],
      "connections": ["[[related]]"]
    }}
  ]
}}
"""


# ── Slug generation ────────────────────────────────────────────────────────────

def slugify(title: str) -> str:
    s = title.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")[:60]


# ── File writers ───────────────────────────────────────────────────────────────

def write_source_page(wiki_path: Path, slug: str, data: dict, source_ref: str, scope: str = "personal") -> Path:
    wiki_path.mkdir(parents=True, exist_ok=True)
    sources_dir = wiki_path / "sources"
    sources_dir.mkdir(exist_ok=True)
    tags = _fmt_tags(data.get("tags", []))
    content = SOURCE_TEMPLATE.format(
        title=data["title"],
        source=source_ref,
        date_ingested=date.today().isoformat(),
        tags=tags,
        scope=scope,
        summary=data.get("summary", ""),
        key_points=_fmt_bullets(data.get("key_points", [])),
        concepts=_fmt_bullets(data.get("concepts", [])),
        entities=_fmt_bullets(data.get("entities", [])),
        connections=_fmt_bullets(data.get("connections", [])),
        gaps=_fmt_bullets(data.get("gaps", [])),
    )
    path = sources_dir / f"{slug}.md"
    path.write_text(content)
    return path


def write_concept_page(wiki_path: Path, slug: str, data: dict, source_slug: str, scope: str = "personal") -> Path:
    concepts_dir = wiki_path / "concepts"
    concepts_dir.mkdir(exist_ok=True)
    path = concepts_dir / f"{slug}.md"
    # Merge sources if page already exists
    existing_sources: list[str] = []
    if path.exists():
        existing = path.read_text()
        m = re.search(r"^sources:\s*\[(.+?)\]", existing, re.MULTILINE)
        if m:
            existing_sources = [s.strip().strip('"').strip("'") for s in m.group(1).split(",")]
    sources = list(dict.fromkeys(existing_sources + [source_slug]))
    tags = _fmt_tags(data.get("tags", []))
    content = CONCEPT_TEMPLATE.format(
        title=data["title"],
        tags=tags,
        sources=_fmt_list(sources),
        scope=scope,
        definition=data.get("definition", ""),
        how_it_works=data.get("how_it_works", ""),
        why_it_matters=data.get("why_it_matters", ""),
        nuances=data.get("nuances", ""),
        evidence=data.get("evidence", ""),
        connections=_fmt_bullets(data.get("connections", [])),
        open_questions=_fmt_bullets(data.get("open_questions", [])),
    )
    path.write_text(content)
    return path


def write_entity_page(wiki_path: Path, slug: str, data: dict, source_slug: str, scope: str = "personal") -> Path:
    entities_dir = wiki_path / "entities"
    entities_dir.mkdir(exist_ok=True)
    path = entities_dir / f"{slug}.md"
    existing_sources: list[str] = []
    if path.exists():
        existing = path.read_text()
        m = re.search(r"^sources:\s*\[(.+?)\]", existing, re.MULTILINE)
        if m:
            existing_sources = [s.strip().strip('"').strip("'") for s in m.group(1).split(",")]
    sources = list(dict.fromkeys(existing_sources + [source_slug]))
    tags = _fmt_tags(data.get("tags", []))
    content = ENTITY_TEMPLATE.format(
        title=data["title"],
        entity_type=data.get("entity_type", "tool"),
        tags=tags,
        sources=_fmt_list(sources),
        scope=scope,
        who_what=data.get("who_what", ""),
        relevance=data.get("relevance", ""),
        contributions=_fmt_bullets(data.get("contributions", [])),
        connections=_fmt_bullets(data.get("connections", [])),
    )
    path.write_text(content)
    return path


def update_index(wiki_path: Path, slug: str, page_type: str, title: str, summary_line: str) -> None:
    index_path = wiki_path / "index.md"
    if not index_path.exists():
        index_path.write_text("# Wiki Index\n\nContent catalog. Updated on every ingest.\n\n")
    content = index_path.read_text()
    # Use a unique anchor pattern to detect existing entries
    entry_anchor = f"[[{page_type}s/{slug}|{slug}]]"
    if entry_anchor in content:
        return  # already indexed, skip
    entry = f"| {entry_anchor} | {summary_line[:80]} |\n"
    section = f"## {page_type.capitalize()}s"
    if section not in content:
        content += f"\n{section}\n\n| Page | Summary |\n|---|---|\n"
    content = content.replace(
        f"{section}\n\n| Page | Summary |\n|---|---|\n",
        f"{section}\n\n| Page | Summary |\n|---|---|\n{entry}",
        1,
    )
    index_path.write_text(content)


def append_log(wiki_path: Path, operation: str, title: str, details: str = "") -> None:
    log_path = wiki_path / "log.md"
    if not log_path.exists():
        log_path.write_text("# Wiki Log\n\nAppend-only record. Format: `## [YYYY-MM-DD] <op> | <title>`\n\n")
    entry = f"## [{date.today().isoformat()}] {operation} | {title}\n"
    if details:
        entry += f"\n{details}\n"
    entry += "\n"
    log_path.write_text(log_path.read_text() + entry)


# ── Lint ───────────────────────────────────────────────────────────────────────

def lint_wiki(wiki_path: Path) -> dict:
    """Structural lint — no LLM needed. Returns report dict."""
    if not wiki_path.exists():
        return {"error": "Wiki directory does not exist. Run `medulla ingest` first."}

    all_slugs: set[str] = set()
    all_files: list[Path] = []
    for md in wiki_path.rglob("*.md"):
        if md.name in ("index.md", "log.md"):
            continue
        all_files.append(md)
        all_slugs.add(md.stem)

    broken_links: list[str] = []
    inbound: dict[str, int] = {s: 0 for s in all_slugs}
    wikilink_re = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

    for md in all_files:
        text = md.read_text()
        for m in wikilink_re.finditer(text):
            target = m.group(1).split("/")[-1]  # strip path prefix
            if target in all_slugs:
                inbound[target] = inbound.get(target, 0) + 1
            else:
                broken_links.append(f"{md.stem} → [[{m.group(1)}]]")

    orphans = [slug for slug, count in inbound.items() if count == 0]

    return {
        "total_pages": len(all_files),
        "broken_links": broken_links,
        "orphaned_pages": orphans,
        "ok": len(broken_links) == 0 and len(orphans) == 0,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_bullets(items: list[str]) -> str:
    if not items:
        return "- (none identified)"
    return "\n".join(f"- {item}" for item in items)


def _fmt_tags(tags: list[str]) -> str:
    if not tags:
        return "[]"
    return "[" + ", ".join(tags) + "]"


def _fmt_list(items: list[str]) -> str:
    if not items:
        return "[]"
    return "[" + ", ".join(f'"{i}"' for i in items) + "]"
