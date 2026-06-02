# Medulla Wiki Schema

Authoritative format spec for all wiki pages. Referenced by:
- `medulla/semantic/wiki.py` — CLI ingest templates
- `medulla/mcp.py` — `medulla_ingest` tool description
- `~/.medulla/wiki/CLAUDE.md` — if user creates one for interactive Obsidian sessions

---

## Directory structure

```
~/.medulla/wiki/
├── CLAUDE.md          # optional — for interactive Claude sessions in this dir
├── index.md           # content catalog, auto-maintained by medulla
├── log.md             # append-only operation record, auto-maintained
├── raw/               # immutable source documents (URLs, PDFs, clips)
│   └── <slug>.md      # raw extracted text with frontmatter (url, fetched_at)
├── sources/           # LLM-synthesized source summaries
│   └── <slug>.md
├── concepts/          # topic/idea pages
│   └── <slug>.md
└── entities/          # people, orgs, tools, databases, projects
    └── <slug>.md
```

**Slugs:** lowercase, hyphen-separated, max 60 chars. `"LogD Prediction"` → `logd-prediction`.

**Wikilinks:** `[[slug]]` or `[[slug|Display Text]]`. Always link to slug without path prefix.

---

## raw/ page format

Raw pages are immutable. Medulla writes them; no one edits them.

```markdown
---
url: https://example.com/article
fetched_at: YYYY-MM-DD
title: Article Title
source_type: url | pdf | markdown
---

# Article Title

[raw extracted text]
```

---

## Source page (`sources/<slug>.md`)

```markdown
---
title: Full Source Title
source: https://url-or-filename
date_ingested: YYYY-MM-DD
tags: [tag1, tag2]
scope: personal
ingested_by: medulla | human
---

## Summary

2–4 paragraphs synthesizing the source.

## Key Points

- Specific claim or finding
- Another key point

## Concepts Introduced or Updated

- [[concept-slug]] — one-line note on what this source adds

## Entities Mentioned

- [[entity-slug]] — role in this source

## Connections

- [[related-page]] — how it connects

## Gaps / Open Questions

- What remains unclear or worth investigating
```

---

## Concept page (`concepts/<slug>.md`)

```markdown
---
title: Concept Name
tags: [tag1, tag2]
sources: ["source-slug-1", "source-slug-2"]
scope: personal
ingested_by: medulla | human
---

## Definition

1–3 sentence plain-language definition.

## How It Works

Concrete mechanism. Be specific.

## Why It Matters

Practical significance.

## Nuances & Caveats

Edge cases, common misunderstandings, tensions.

## Evidence & Examples

Specific examples, data points, quotes from sources.

## Connections

- [[related-concept]] — how connected
- [[entity]] — role

## Open Questions

- Unresolved questions this concept raises
```

---

## Entity page (`entities/<slug>.md`)

```markdown
---
title: Entity Name
type: person | org | tool | project | database
tags: [tag1]
sources: ["source-slug-1"]
scope: personal
ingested_by: medulla | human
---

## Who / What

1–2 sentence description.

## Relevance

Why this entity matters in this wiki.

## Key Contributions / Features

- Feature or contribution 1
- Feature or contribution 2

## Connections

- [[related-entity]] — relationship
- [[related-concept]] — connection
```

---

## Tags

Reuse existing tags before creating new ones. Common tags:
`admet`, `machine-learning`, `drug-discovery`, `mlops`, `research`, `paper`, `tool`, `database`, `person`, `org`

---

## index.md

Auto-maintained. Format:
```markdown
# Wiki Index
**Stats:** N sources · N concepts · N entities · Last updated: YYYY-MM-DD

## Sources
| Page | Summary |
|---|---|
| [[sources/slug\|slug]] | One-line summary |

## Concepts
...

## Entities
...

## Tag Index
| Tag | Pages |
|---|---|
| `admet` | page1, page2 |
```

## log.md

Append-only. Format: `## [YYYY-MM-DD] <operation> | <title>`
Grep-friendly: `grep "^## \[" log.md | tail -5`
