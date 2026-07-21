"""Wiki page removal — plan (preview) and execute (delete + cleanup).

Targets:
  sources/slug   — remove source page, clean sources: on concepts/entities
  concepts/slug  — remove concept page, clean related: on all pages
  entities/slug  — remove entity page, clean related: on all pages
  raw/filename   — remove raw archive file + its linked source page

After every execute_remove, reindex_wiki_edges() is called automatically.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any


def plan_remove(
    conn: sqlite3.Connection,
    target: str,
    wiki_path: Path | None = None,
) -> dict[str, Any]:
    """Return a preview of what will be removed/changed without touching anything.

    target format: "sources/slug", "concepts/slug", "entities/slug", or "raw/filename"
    Returns a plan dict describing all affected pages.
    """
    parts = target.split("/", 1)
    if len(parts) != 2:
        return {"error": f"Invalid target format: {target!r}. Use folder/slug or raw/filename."}

    folder, name = parts[0], parts[1]

    if folder == "raw":
        return _plan_raw(conn, name, wiki_path)
    elif folder in ("sources", "concepts", "entities"):
        return _plan_wiki_page(conn, folder, name)
    else:
        return {"error": f"Unknown target folder: {folder!r}. Use sources/, concepts/, entities/, or raw/."}


def _plan_wiki_page(conn: sqlite3.Connection, folder: str, slug: str) -> dict[str, Any]:
    row = conn.execute(
        "SELECT slug, type, file_path FROM wiki_pages WHERE slug = ?", (slug,)
    ).fetchone()
    if row is None:
        return {"error": f"Page not found: {folder}/{slug}"}

    page_type = row["type"]
    plan: dict[str, Any] = {
        "target_slug": slug,
        "target_type": page_type,
        "target_file": row["file_path"],
        "affected_sources_update": [],   # slugs that will lose this from their sources:
        "would_orphan": [],              # slugs whose sources: becomes empty (cascade candidates)
        "related_cleanup": [],           # slugs that have this in their related:
    }

    if page_type == "source":
        # Find all pages that list this source in sources:
        rows = conn.execute(
            "SELECT slug, sources FROM wiki_pages WHERE type != 'source'"
        ).fetchall()
        for r in rows:
            srcs = _parse_json_list(r["sources"])
            if slug in srcs:
                plan["affected_sources_update"].append(r["slug"])
                if len(srcs) == 1:
                    plan["would_orphan"].append(r["slug"])

    # Find all pages with this slug in their related: frontmatter
    all_pages = conn.execute("SELECT slug, file_path FROM wiki_pages").fetchall()
    for r in all_pages:
        if r["slug"] == slug or not r["file_path"]:
            continue
        p = Path(r["file_path"])
        if p.exists():
            content = p.read_text(errors="replace")
            if _slug_in_related(slug, content):
                plan["related_cleanup"].append(r["slug"])

    return plan


def _plan_raw(conn: sqlite3.Connection, filename: str, wiki_path: Path | None) -> dict[str, Any]:
    plan: dict[str, Any] = {
        "target_type": "raw",
        "target_filename": filename,
        "linked_source_slug": None,
        "linked_source_file": None,
    }
    # Look up by raw_path column
    rows = conn.execute(
        "SELECT slug, file_path, raw_path FROM wiki_pages WHERE type = 'source'"
    ).fetchall()
    for r in rows:
        if r["raw_path"] and Path(r["raw_path"]).name == filename:
            plan["linked_source_slug"] = r["slug"]
            plan["linked_source_file"] = r["file_path"]
            break

    # Fallback: resolve via wiki_path/raw/filename if raw_path column not set
    if plan["linked_source_slug"] is None and wiki_path:
        raw_file = wiki_path / "raw" / filename
        for r in rows:
            if r["file_path"]:
                src_content = Path(r["file_path"]).read_text(errors="replace") if Path(r["file_path"]).exists() else ""
                if str(raw_file) in src_content or filename in src_content:
                    plan["linked_source_slug"] = r["slug"]
                    plan["linked_source_file"] = r["file_path"]
                    break

    return plan


def execute_remove(
    conn: sqlite3.Connection,
    target: str,
    wiki_path: Path | None = None,
    cascade: bool = False,
) -> dict[str, Any]:
    """Execute the removal described by plan_remove(target).

    Always calls reindex_wiki_edges() at the end.
    Returns summary of what was removed.
    """
    plan = plan_remove(conn, target, wiki_path=wiki_path)
    if "error" in plan:
        return plan

    removed: list[str] = []
    cleaned: list[str] = []

    if plan["target_type"] == "raw":
        raw_file = _resolve_raw_file(target, wiki_path)
        if raw_file and raw_file.exists():
            raw_file.unlink()
            removed.append(str(raw_file))
        # Always remove the linked source page too. Build the source's own plan
        # so sources: frontmatter cleanup and cascade run — the raw plan doesn't
        # carry affected_sources_update / would_orphan.
        if plan["linked_source_slug"]:
            src_plan = _plan_wiki_page(conn, "sources", plan["linked_source_slug"])
            r = _remove_wiki_page(conn, plan["linked_source_slug"], src_plan, cascade)
            removed.extend(r["removed"])
            cleaned.extend(r["cleaned"])
    else:
        r = _remove_wiki_page(conn, plan["target_slug"], plan, cascade)
        removed.extend(r["removed"])
        cleaned.extend(r["cleaned"])

    # Reindex edges after any removal
    try:
        from medulla.semantic.wiki import reindex_wiki_edges
        reindex_wiki_edges(conn, top_k=5)
    except Exception:
        pass

    return {"removed": removed, "cleaned": cleaned}


def _remove_wiki_page(
    conn: sqlite3.Connection,
    slug: str,
    plan: dict[str, Any],
    cascade: bool,
) -> dict[str, Any]:
    removed: list[str] = []
    cleaned: list[str] = []

    row = conn.execute(
        "SELECT file_path, type FROM wiki_pages WHERE slug = ?", (slug,)
    ).fetchone()
    if row is None:
        return {"removed": [], "cleaned": []}

    page_type = row["type"]

    # 1. Delete file
    if row["file_path"]:
        p = Path(row["file_path"])
        if p.exists():
            p.unlink()
            removed.append(str(p))

    # 2. Delete DB row + vec_wiki embedding
    conn.execute("DELETE FROM wiki_pages WHERE slug = ?", (slug,))
    conn.execute("DELETE FROM vec_wiki WHERE slug = ?", (slug,))
    conn.commit()

    # 3. Clean related: references across all pages
    all_pages = conn.execute("SELECT slug, file_path FROM wiki_pages").fetchall()
    for r in all_pages:
        if not r["file_path"]:
            continue
        p = Path(r["file_path"])
        if p.exists():
            content = p.read_text(errors="replace")
            if _slug_in_related(slug, content):
                new_content = _remove_from_related(slug, content)
                p.write_text(new_content)
                cleaned.append(r["slug"])

    # 4. If removing a source: clean sources: frontmatter on concepts/entities
    if page_type == "source":
        for affected_slug in plan.get("affected_sources_update", []):
            r = conn.execute(
                "SELECT file_path, sources FROM wiki_pages WHERE slug = ?", (affected_slug,)
            ).fetchone()
            if not r:
                continue
            srcs = _parse_json_list(r["sources"])
            srcs = [s for s in srcs if s != slug]
            conn.execute(
                "UPDATE wiki_pages SET sources = ? WHERE slug = ?",
                (json.dumps(srcs), affected_slug)
            )
            # Update frontmatter in file
            if r["file_path"]:
                fp = Path(r["file_path"])
                if fp.exists():
                    _remove_from_sources_frontmatter(fp, slug)
                    cleaned.append(affected_slug)
        conn.commit()

        # 5. Cascade: delete orphaned concepts/entities
        if cascade:
            for orphan_slug in plan.get("would_orphan", []):
                orphan_row = conn.execute(
                    "SELECT slug, type FROM wiki_pages WHERE slug = ?", (orphan_slug,)
                ).fetchone()
                if orphan_row:
                    orphan_plan = _plan_wiki_page(conn, orphan_row["type"] + "s", orphan_slug)
                    r2 = _remove_wiki_page(conn, orphan_slug, orphan_plan, cascade=False)
                    removed.extend(r2["removed"])
                    cleaned.extend(r2["cleaned"])

    return {"removed": removed, "cleaned": cleaned}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        return json.loads(value)
    except Exception:
        return []


def _item_slug(item: str) -> str:
    """Normalize a related: list item to its bare slug.

    Handles [[concepts/foo]], "concepts/foo", concepts/foo, and alias forms
    like [[concepts/foo|Foo]] — returns the final path segment ('foo').
    """
    s = item.strip().strip('"').strip("'")
    s = s.removeprefix("[[").removesuffix("]]")
    s = s.split("|", 1)[0]          # drop |alias
    s = s.rsplit("/", 1)[-1]        # drop folder prefix
    return s.strip()


def _slug_in_related(slug: str, content: str) -> bool:
    m = re.search(r"^related:\s*\[(.+?)\]", content, re.MULTILINE)
    if not m:
        return False
    return any(_item_slug(i) == slug for i in m.group(1).split(","))


def _remove_from_related(slug: str, content: str) -> str:
    def _filter_related(m: re.Match) -> str:
        items = [i.strip() for i in m.group(1).split(",")]
        items = [i for i in items if _item_slug(i) != slug]
        new_list = "[" + ", ".join(items) + "]" if items else "[]"
        return f"related: {new_list}"
    return re.sub(r"^related:\s*\[(.+?)\]", _filter_related, content, flags=re.MULTILINE)


def _remove_from_sources_frontmatter(path: Path, source_slug: str) -> None:
    content = path.read_text(errors="replace")
    m = re.search(r'^sources:\s*\[(.+?)\]', content, re.MULTILINE)
    if not m:
        return
    items = [i.strip().strip('"').strip("'") for i in m.group(1).split(",")]
    items = [i for i in items if i != source_slug]
    new_list = "[" + ", ".join(f'"{i}"' for i in items) + "]" if items else "[]"
    content = re.sub(r'^sources:\s*\[.+?\]', f"sources: {new_list}", content, flags=re.MULTILINE)
    path.write_text(content)


def _resolve_raw_file(target: str, wiki_path: Path | None) -> Path | None:
    _, filename = target.split("/", 1)
    if wiki_path:
        return wiki_path / "raw" / filename
    from medulla.config import get_config
    return get_config().wiki_path / "raw" / filename
