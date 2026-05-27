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
---

## Who / What

{who_what}

## Relevance to This Wiki

{relevance}

## Key Contributions / Features

{contributions}

## Connections

{connections}
"""

# Tag vocabulary — reuse these before inventing new ones (same philosophy as Nimbus-Brain)
TAG_VOCABULARY = [
    "deep-learning", "machine-learning", "nlp", "transformer", "attention",
    "neural-networks", "optimization", "gradient-descent", "training",
    "architecture", "language-models", "gpt", "bert", "encoder-decoder",
    "self-attention", "multi-head-attention", "positional-encoding",
    "autograd", "backpropagation", "automatic-differentiation",
    "adaptive-learning-rate", "adam", "rmsprop", "adagrad",
    "software-2-0", "programming-paradigm", "dataset-as-source-code",
    "knowledge-management", "pkm", "llm-wiki", "obsidian",
    "drug-discovery", "admet", "cheminformatics", "molecular-property-prediction",
    "research", "paper", "education", "tools", "person", "org", "database",
    "ai", "generative-models", "sequence-modeling", "machine-translation",
    "benchmarks", "evaluation", "robustness", "security",
]

PLAN_SYSTEM_PROMPT = """\
You are maintaining a personal knowledge wiki using the Karpathy LLM Wiki pattern.
Your job is to read source documents and produce structured wiki pages in JSON format.

Rules:
- Use [[folder/slug]] wikilinks: [[concepts/slug]], [[entities/slug]], [[sources/slug]]
- Be concrete: extract actual facts, findings, numbers, and connections from the source
- Tags: reuse the provided tag vocabulary before inventing new tags
- Wikilinks: only link to slugs in the provided wiki schema or pages you're creating in this response
- Output valid JSON only — no markdown fences, no explanation outside the JSON
"""

PLAN_PROMPT_TEMPLATE = """\
STAGE: PLAN
Source title: {title}
Source type: {source_type}
Date: {today}

EXISTING WIKI PAGES — use ONLY these exact slugs for [[wikilinks]]:
{wiki_schema}

TAG VOCABULARY — reuse before inventing new tags:
{tag_vocabulary}

Source text:
---
{text}
---

Produce a JSON object with:
1. source_page — complete source wiki page (title, summary, key_points, tags, concept/entity/connection/gap lists)
2. new_concepts — NEW concept page briefs (slugs NOT in the existing wiki schema above)
3. new_entities — NEW entity page briefs (slugs NOT in the existing wiki schema above)
4. update_concepts — existing concept slugs (from wiki schema) that this source adds to
5. update_entities — existing entity slugs (from wiki schema) that this source adds to

Wikilinks in source_page: use [[concepts/slug]], [[entities/slug]], [[sources/slug]] — only slugs from the schema or from new_concepts/new_entities you list here.

{{
  "source_page": {{
    "title": "...",
    "summary": "2-4 paragraph synthesis of this source",
    "key_points": ["bullet 1", "bullet 2"],
    "tags": ["tag1", "tag2"],
    "concepts": ["[[concepts/slug]] — one-line note on what this source adds"],
    "entities": ["[[entities/slug]] — role in this source"],
    "connections": ["[[sources/related-slug]] — how connected"],
    "gaps": ["open question 1"]
  }},
  "new_concepts": [
    {{"slug": "concept-slug", "title": "Concept Name", "brief": "one sentence on what to capture"}}
  ],
  "new_entities": [
    {{"slug": "entity-slug", "title": "Entity Name", "entity_type": "person|org|tool|project|database", "brief": "one sentence"}}
  ],
  "update_concepts": [
    {{"slug": "existing-concept-slug", "add_source_note": "what this source contributes to this concept"}}
  ],
  "update_entities": [
    {{"slug": "existing-entity-slug", "add_source_note": "this entity's role in the current source"}}
  ]
}}
"""

CONCEPT_PROMPT_TEMPLATE = """\
STAGE: CONCEPT
Write a single concept wiki page.

Concept: {title} (slug: {concept_slug})
Brief: {brief}
From source: {source_title}

EXISTING WIKI PAGES for [[wikilinks]]:
{wiki_schema}

TAG VOCABULARY:
{tag_vocabulary}

Source excerpt (for evidence and examples):
---
{text}
---

Produce a JSON object for this one concept page. Fill all fields substantively.
Connections: link to related slugs from the wiki schema only.

{{
  "slug": "{concept_slug}",
  "title": "...",
  "tags": ["tag1"],
  "definition": "1-3 sentence plain-language definition",
  "how_it_works": "concrete mechanism, can be multiple sentences",
  "why_it_matters": "practical significance",
  "nuances": "edge cases, caveats, common misconceptions",
  "evidence": "specific examples and findings from this source",
  "connections": ["[[concepts/related-slug]] — how connected"],
  "open_questions": ["question 1"]
}}
"""

ENTITY_PROMPT_TEMPLATE = """\
STAGE: ENTITY
Write a single entity wiki page.

Entity: {title} (slug: {entity_slug}, type: {entity_type})
Brief: {brief}
From source: {source_title}

EXISTING WIKI PAGES for [[wikilinks]]:
{wiki_schema}

TAG VOCABULARY:
{tag_vocabulary}

Source excerpt:
---
{text}
---

Produce a JSON object for this one entity page.

{{
  "slug": "{entity_slug}",
  "title": "...",
  "entity_type": "{entity_type}",
  "tags": ["tag1"],
  "who_what": "1-2 sentence description",
  "relevance": "why this entity matters in this wiki's context",
  "contributions": ["key contribution or feature 1"],
  "connections": ["[[concepts/related-slug]] — connection type"]
}}
"""


# ── Raw source storage ────────────────────────────────────────────────────────

RAW_TEMPLATE = """\
---
url: {url}
fetched_at: {fetched_at}
title: {title}
source_type: {source_type}
---

# {title}

{content}
"""


def write_raw_source(
    wiki_path: Path,
    slug: str,
    content: str,
    url: str = "",
    title: str = "",
    source_type: str = "url",
) -> Path:
    """Write raw extracted content to wiki/raw/<slug>.md for backtrace.

    Use when medulla itself fetched the content (CLI url, medulla_ingest_url).
    Has actual extracted text so the raw file is genuinely useful.
    """
    raw_dir = wiki_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{slug}.md"
    path.write_text(RAW_TEMPLATE.format(
        url=url,
        fetched_at=date.today().isoformat(),
        title=title or slug,
        source_type=source_type,
        content=content[:20_000],
    ))
    return path


def append_url_reference(
    wiki_path: Path,
    slug: str,
    url: str,
    title: str = "",
) -> Path:
    """Append a URL reference to wiki/raw/url-references.md.

    Use when Claude fetched the URL via WebFetch and passed source_url to
    medulla_ingest. We don't have the raw text, just the URL. Rather than
    20 near-empty files, one running log keeps it clean.
    """
    raw_dir = wiki_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = raw_dir / "url-references.md"
    if not log_path.exists():
        log_path.write_text(
            "# URL References\n\n"
            "Sources ingested via WebFetch in Claude Code / Kiro sessions.\n"
            "Format: `## [YYYY-MM-DD] slug`\n\n"
        )
    entry = (
        f"## [{date.today().isoformat()}] {slug}\n"
        f"URL: {url}\n"
        f"Title: {title or slug}\n"
        f"Wiki: [[sources/{slug}]]\n\n"
    )
    log_path.write_text(log_path.read_text() + entry)
    return log_path


# ── Slug generation ────────────────────────────────────────────────────────────

def slugify(title: str) -> str:
    s = title.lower()
    # Preserve version numbers: "2.0" → "2-0", "GPT-2" → "gpt-2"
    s = re.sub(r"(\d)\.(\d)", r"\1-\2", s)
    # Remove non-alphanumeric except spaces and hyphens
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
        who_what=data.get("who_what", ""),
        relevance=data.get("relevance", ""),
        contributions=_fmt_bullets(data.get("contributions", [])),
        connections=_fmt_bullets(data.get("connections", [])),
    )
    path.write_text(content)
    return path


def update_index(wiki_path: Path, slug: str, page_type: str, title: str, summary_line: str) -> None:
    """Update index.md with Nimbus-Brain format: [[folder/slug|slug]] in tables + stats header."""
    index_path = wiki_path / "index.md"
    if not index_path.exists():
        index_path.write_text(
            "# Wiki Index\n\n"
            "Content catalog for the Medulla wiki. Updated on every ingest.\n\n"
            f"**Stats:** 0 sources · 0 concept pages · 0 entity pages · "
            f"Last updated: {date.today().isoformat()}\n\n---\n\n"
        )
    content = index_path.read_text()

    _DIR = {"source": "sources", "concept": "concepts", "entity": "entities"}
    dir_name = _DIR.get(page_type, f"{page_type}s")
    # No pipe alias inside table cells — Obsidian treats | as column separator
    entry_anchor = f"[[{dir_name}/{slug}]]"

    if entry_anchor in content:
        _refresh_index_stats(index_path)
        return  # already indexed

    entry = f"| {entry_anchor} | {summary_line[:80]} |\n"
    _SECTION = {"source": "## Sources", "concept": "## Concepts", "entity": "## Entities"}
    section = _SECTION.get(page_type, f"## {page_type.capitalize()}s")

    if section not in content:
        content += f"\n{section}\n\n| Page | Summary |\n|---|---|\n"

    content = content.replace(
        f"{section}\n\n| Page | Summary |\n|---|---|\n",
        f"{section}\n\n| Page | Summary |\n|---|---|\n{entry}",
        1,
    )
    index_path.write_text(content)
    _refresh_index_stats(index_path)


def _refresh_index_stats(index_path: Path) -> None:
    """Recompute the stats line in index.md from actual link counts."""
    content = index_path.read_text()
    sources = content.count("[[sources/")
    concepts = content.count("[[concepts/")
    entities = content.count("[[entities/")
    stats = (f"**Stats:** {sources} sources · {concepts} concept pages · "
             f"{entities} entity pages · Last updated: {date.today().isoformat()}")
    updated = re.sub(
        r"\*\*Stats:\*\*.*?Last updated: \d{4}-\d{2}-\d{2}", stats, content
    )
    if updated != content:
        index_path.write_text(updated)


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

    # System files excluded from lint — not real wiki pages
    _SYSTEM_FILES = {"index.md", "log.md", "url-references.md"}
    all_slugs: set[str] = set()
    all_files: list[Path] = []
    for md in wiki_path.rglob("*.md"):
        if md.name in _SYSTEM_FILES:
            continue
        if md.parent.name == "raw" or "raw" in md.parts:
            continue  # raw/ files are source archives, not wiki pages
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

    # Source pages are expected leaf nodes — exclude from orphan reporting
    orphans = [slug for slug, count in inbound.items()
               if count == 0 and slug not in {f.stem for f in all_files
                                              if f.parent.name == "sources"}]

    return {
        "total_pages": len(all_files),
        "suggested_pages": broken_links,   # forward refs to pages not yet created
        "orphaned_pages": orphans,
        "ok": len(orphans) == 0,           # broken_links are normal; only orphans matter
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
