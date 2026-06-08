# Feat: Concept/Entity Discrimination — Skip Well-Known & Author-Only Pages

**Branch:** `feat/concept-entity-discrimination`  
**Issue:** #15  
**Date:** 2026-06-08  
**Tests:** 450 passing · 96% statement coverage

---

## What Was Wrong

The plan prompt had no selectivity guidance. The LLM created wiki pages for anything
mentioned in a source — including authors (Kingma, Ba), their universities, well-known
benchmark datasets (MNIST, CIFAR-10, IMDB), and the conference venue (ICLR). These pages
added zero recall value; all their content is general knowledge independent of the source.

Adam Optimizer PDF ingest before this fix: 10 concepts + 7 entities (incl. Kingma, Ba,
University of Amsterdam, University of Toronto, OpenAI, MNIST, CIFAR-10, IMDB).
After: 8 concepts + 1 entity. Authors routed to `authors:` frontmatter on the source page.

---

## What Changed

### Prompt — `PLAN_PROMPT_TEMPLATE` in `wiki.py`

Two new selectivity rules added before the JSON schema:

**Concept selectivity:** Only add to `new_concepts` if this source contributes novel,
source-specific content — findings, a concrete implementation, empirical results, or a
framing that wouldn't exist without this source. Ask: would this wiki page contain
anything a reader couldn't get from general knowledge alone?

**Entity selectivity:** Same source-specificity test. For people: default to omit — list
them in `source_page.authors` instead. Only create a person entity page if the person is
the primary subject of the source. For tools/databases/orgs: only if the source adds
specific detail you'd actually reference.

### Schema — `source_page.authors` field

`PLAN_PROMPT_TEMPLATE` JSON schema now includes `"authors": ["Name 1", "Name 2"]` on
`source_page`. `SOURCE_TEMPLATE` gains an `authors:` frontmatter line. `write_source_page`
passes `authors` from `source_data`.

### Code guard — author entity filter in `ingest.py`

After the plan stage, person entities whose slug matches a slugified author name from
`source_page.authors` are filtered from `new_entities` before stage 3. A `↷ Skipped`
line reports what was filtered. This enforces the prompt rule in code so a single LLM
lapse doesn't silently create a useless person page.

---

## Known Rough Edge

`iclr-2015` (conference venue) survived as an entity in the Adam paper ingest — the LLM
judged it worth keeping. Venue entities have zero source-specific detail and ideally
would be filtered. This is prompt-only selectivity working as expected: mostly right,
occasionally lenient. No code fix planned for venues specifically.

---

## Tests Added (`tests/test_ingest_pipeline.py`)

`AuthorEntityProvider` — plan with two author persons + one tool entity:

- `test_author_entities_not_written_as_pages` — person slugs matching authors not written to disk
- `test_tool_entities_still_written` — non-person entities unaffected
- `test_authors_written_to_source_frontmatter` — author names appear in source page `authors:`
- `test_author_entities_not_in_db` — author slugs absent from `wiki_pages` DB
- `test_plan_prompt_contains_information_gain_rule` — prompt contains selectivity language
- `test_plan_prompt_contains_author_routing_rule` — prompt mentions `authors` field
- `test_plan_prompt_schema_includes_authors_field` — JSON schema has `"authors"` key
- `test_source_template_includes_authors_frontmatter` — SOURCE_TEMPLATE has `authors:` line
