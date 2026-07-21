"""Tests for medulla remove — wiki page and raw file deletion."""
import json
import pytest
from pathlib import Path

from medulla.db.database import connect
from medulla.semantic.store import upsert_wiki_page
from medulla.episodic.store import upsert_session
from tests.test_store import make_session


# ── V5 migration — raw_path column ────────────────────────────────────────────

def test_wiki_pages_has_raw_path_column(db):
    """V5 migration adds raw_path column to wiki_pages."""
    cols = {r[1] for r in db.execute("PRAGMA table_info(wiki_pages)").fetchall()}
    assert "raw_path" in cols


def test_upsert_wiki_page_stores_raw_path(db, tmp_path):
    """upsert_wiki_page stores raw_path for source pages."""
    raw = tmp_path / "paper.pdf"
    raw.write_bytes(b"fake pdf")
    upsert_wiki_page(db, "test-source", "source", "Test Source",
                     "content", Path("/wiki/sources/test-source.md"),
                     raw_path=raw)
    row = db.execute("SELECT raw_path FROM wiki_pages WHERE slug = ?", ("test-source",)).fetchone()
    assert row["raw_path"] == str(raw)


def test_upsert_wiki_page_raw_path_defaults_none(db, tmp_path):
    """Concept/entity pages have raw_path=None."""
    upsert_wiki_page(db, "test-concept", "concept", "Test",
                     "content", Path("/wiki/concepts/test-concept.md"))
    row = db.execute("SELECT raw_path FROM wiki_pages WHERE slug = ?", ("test-concept",)).fetchone()
    assert row["raw_path"] is None


# ── remove module imports ──────────────────────────────────────────────────────

def test_remove_module_importable():
    from medulla.semantic import remove as remove_mod
    assert hasattr(remove_mod, "plan_remove")
    assert hasattr(remove_mod, "execute_remove")


# ── plan_remove — dry-run preview ─────────────────────────────────────────────

def _setup_wiki(db, tmp_path):
    """Insert a source + two concepts (one with two sources, one orphan)."""
    wiki = tmp_path / "wiki"
    wiki.mkdir()
    (wiki / "sources").mkdir()
    (wiki / "concepts").mkdir()
    (wiki / "raw").mkdir()
    raw_file = wiki / "raw" / "paper.pdf"
    raw_file.write_bytes(b"fake")

    src_path = wiki / "sources" / "my-source.md"
    src_path.write_text("---\ntitle: My Source\nsources: []\nrelated: []\n---\n")
    upsert_wiki_page(db, "my-source", "source", "My Source", src_path.read_text(),
                     src_path, raw_path=raw_file)

    con_a = wiki / "concepts" / "concept-a.md"
    con_a.write_text('---\ntitle: A\nsources: ["my-source"]\nrelated: []\n---\n')
    upsert_wiki_page(db, "concept-a", "concept", "A", con_a.read_text(), con_a,
                     sources=["my-source"])

    con_b = wiki / "concepts" / "concept-b.md"
    con_b.write_text('---\ntitle: B\nsources: ["my-source", "other-source"]\nrelated: ["[[concepts/concept-a]]"]\n---\n')
    upsert_wiki_page(db, "concept-b", "concept", "B", con_b.read_text(), con_b,
                     sources=["my-source", "other-source"])

    return wiki, raw_file


def test_plan_remove_source_shows_affected_pages(db, tmp_path):
    """plan_remove for a source shows which pages lose it from sources:."""
    from medulla.semantic.remove import plan_remove
    wiki, _ = _setup_wiki(db, tmp_path)
    plan = plan_remove(db, "sources/my-source")
    assert plan["target_slug"] == "my-source"
    assert plan["target_type"] == "source"
    assert "concept-a" in plan["affected_sources_update"]   # loses my-source
    assert "concept-b" in plan["affected_sources_update"]   # loses my-source (still has other)
    assert "concept-a" in plan["would_orphan"]              # only source was my-source
    assert "concept-b" not in plan["would_orphan"]          # still has other-source


def test_plan_remove_concept_shows_related_cleanup(db, tmp_path):
    """plan_remove for a concept shows pages that have it in related:."""
    from medulla.semantic.remove import plan_remove
    wiki, _ = _setup_wiki(db, tmp_path)
    plan = plan_remove(db, "concepts/concept-a")
    assert plan["target_slug"] == "concept-a"
    assert "concept-b" in plan["related_cleanup"]   # concept-b has [[concepts/concept-a]] in related:


def test_plan_remove_raw_finds_source_slug(db, tmp_path):
    """plan_remove for raw/file.pdf resolves to the linked source slug."""
    from medulla.semantic.remove import plan_remove
    wiki, raw_file = _setup_wiki(db, tmp_path)
    plan = plan_remove(db, f"raw/{raw_file.name}", wiki_path=wiki)
    assert plan["target_type"] == "raw"
    assert plan["linked_source_slug"] == "my-source"


def test_plan_remove_raw_no_linked_source(db, tmp_path):
    """plan_remove for raw file with no linked source page shows None."""
    from medulla.semantic.remove import plan_remove
    wiki, _ = _setup_wiki(db, tmp_path)
    # Create an orphan raw file with no source page
    orphan = wiki / "raw" / "orphan.pdf"
    orphan.write_bytes(b"orphan")
    plan = plan_remove(db, "raw/orphan.pdf", wiki_path=wiki)
    assert plan["linked_source_slug"] is None


# ── execute_remove ─────────────────────────────────────────────────────────────

def test_execute_remove_concept_deletes_file_and_db(db, tmp_path):
    """execute_remove removes concept page from disk and DB."""
    from medulla.semantic.remove import execute_remove
    wiki, _ = _setup_wiki(db, tmp_path)
    execute_remove(db, "concepts/concept-a", wiki_path=wiki, cascade=False)
    assert not (wiki / "concepts" / "concept-a.md").exists()
    row = db.execute("SELECT slug FROM wiki_pages WHERE slug = ?", ("concept-a",)).fetchone()
    assert row is None


def test_execute_remove_concept_cleans_related_refs(db, tmp_path):
    """execute_remove removes slug from related: in other pages."""
    from medulla.semantic.remove import execute_remove
    wiki, _ = _setup_wiki(db, tmp_path)
    execute_remove(db, "concepts/concept-a", wiki_path=wiki, cascade=False)
    content = (wiki / "concepts" / "concept-b.md").read_text()
    assert "concept-a" not in content


def test_execute_remove_source_cleans_sources_frontmatter(db, tmp_path):
    """execute_remove removes source from sources: frontmatter on concepts."""
    from medulla.semantic.remove import execute_remove
    wiki, _ = _setup_wiki(db, tmp_path)
    execute_remove(db, "sources/my-source", wiki_path=wiki, cascade=False)
    content = (wiki / "concepts" / "concept-b.md").read_text()
    assert "my-source" not in content
    assert "other-source" in content   # other source preserved


def test_execute_remove_source_cascade_deletes_orphans(db, tmp_path):
    """execute_remove with cascade=True deletes concepts with empty sources:."""
    from medulla.semantic.remove import execute_remove
    wiki, _ = _setup_wiki(db, tmp_path)
    execute_remove(db, "sources/my-source", wiki_path=wiki, cascade=True)
    # concept-a had only my-source → deleted
    assert not (wiki / "concepts" / "concept-a.md").exists()
    # concept-b had other-source too → survives
    assert (wiki / "concepts" / "concept-b.md").exists()


def test_execute_remove_source_no_cascade_leaves_orphans(db, tmp_path):
    """execute_remove without cascade leaves orphaned concepts intact."""
    from medulla.semantic.remove import execute_remove
    wiki, _ = _setup_wiki(db, tmp_path)
    execute_remove(db, "sources/my-source", wiki_path=wiki, cascade=False)
    assert (wiki / "concepts" / "concept-a.md").exists()


def test_execute_remove_raw_removes_file_and_source(db, tmp_path):
    """execute_remove for raw file removes raw file + linked source page."""
    from medulla.semantic.remove import execute_remove
    wiki, raw_file = _setup_wiki(db, tmp_path)
    execute_remove(db, f"raw/{raw_file.name}", wiki_path=wiki, cascade=False)
    assert not raw_file.exists()
    assert not (wiki / "sources" / "my-source.md").exists()
    row = db.execute("SELECT slug FROM wiki_pages WHERE slug = ?", ("my-source",)).fetchone()
    assert row is None


def test_execute_remove_raw_cleans_source_frontmatter(db, tmp_path):
    """Removing raw/file.pdf cleans sources: on concepts, like removing the source directly."""
    from medulla.semantic.remove import execute_remove
    wiki, raw_file = _setup_wiki(db, tmp_path)
    execute_remove(db, f"raw/{raw_file.name}", wiki_path=wiki, cascade=False)
    # concept-b referenced my-source (the source linked to this raw file) + other-source
    content = (wiki / "concepts" / "concept-b.md").read_text()
    assert "my-source" not in content
    assert "other-source" in content


def test_execute_remove_raw_cascade_deletes_orphans(db, tmp_path):
    """Removing raw/file.pdf with cascade deletes concepts orphaned by the source removal."""
    from medulla.semantic.remove import execute_remove
    wiki, raw_file = _setup_wiki(db, tmp_path)
    execute_remove(db, f"raw/{raw_file.name}", wiki_path=wiki, cascade=True)
    assert not (wiki / "concepts" / "concept-a.md").exists()   # only source was my-source
    assert (wiki / "concepts" / "concept-b.md").exists()       # still has other-source


def test_execute_remove_related_slug_no_substring_collision(db, tmp_path):
    """Removing concepts/adam must not touch related: entries for adam-optimizer."""
    from medulla.semantic.remove import execute_remove
    wiki = tmp_path / "wiki"
    (wiki / "concepts").mkdir(parents=True)

    adam = wiki / "concepts" / "adam.md"
    adam.write_text("---\ntitle: Adam\nsources: []\nrelated: []\n---\n")
    upsert_wiki_page(db, "adam", "concept", "Adam", adam.read_text(), adam)

    optimizer = wiki / "concepts" / "adam-optimizer.md"
    optimizer.write_text("---\ntitle: Adam Optimizer\nsources: []\nrelated: []\n---\n")
    upsert_wiki_page(db, "adam-optimizer", "concept", "Adam Optimizer", optimizer.read_text(), optimizer)

    # A page whose related: links ONLY adam-optimizer, not adam
    ref = wiki / "concepts" / "ref.md"
    ref.write_text('---\ntitle: Ref\nsources: []\nrelated: ["[[concepts/adam-optimizer]]"]\n---\n')
    upsert_wiki_page(db, "ref", "concept", "Ref", ref.read_text(), ref)

    execute_remove(db, "concepts/adam", wiki_path=wiki, cascade=False)
    content = (wiki / "concepts" / "ref.md").read_text()
    assert "adam-optimizer" in content   # collision-adjacent slug survives


def test_store_wiki_page_sets_raw_path_to_archive_copy(db, tmp_path, monkeypatch):
    """MCP store path records raw_path pointing at the wiki/raw/ copy, not the original."""
    import medulla.semantic.ingest as ing
    monkeypatch.setattr(ing, "_embed_new_wiki_pages", lambda conn, result: None)

    original = tmp_path / "downloads" / "paper.pdf"
    original.parent.mkdir()
    original.write_bytes(b"fake pdf")
    wiki = tmp_path / "wiki"

    ing.store_wiki_page(
        db, wiki, "Paper", "---\ntitle: Paper\n---\n## Summary\n",
        page_type="source", source_path=str(original),
    )
    row = db.execute("SELECT raw_path FROM wiki_pages WHERE slug = ?", ("paper",)).fetchone()
    assert row["raw_path"] == str(wiki / "raw" / "paper.pdf")


def test_execute_remove_clears_vec_wiki_entry(db, tmp_path):
    """execute_remove deletes the vec_wiki embedding row."""
    from medulla.semantic.remove import execute_remove
    from medulla.db.embedding_store import upsert_wiki_embedding
    wiki, _ = _setup_wiki(db, tmp_path)
    upsert_wiki_embedding(db, "concept-a", [0.1] * 768)
    execute_remove(db, "concepts/concept-a", wiki_path=wiki, cascade=False)
    row = db.execute("SELECT slug FROM vec_wiki WHERE slug = ?", ("concept-a",)).fetchone()
    assert row is None


# ── CLI medulla remove ─────────────────────────────────────────────────────────

from typer.testing import CliRunner
from medulla.cli import app

_runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_remove_config(tmp_path, monkeypatch):
    import medulla.config as cfg
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=tmp_path / ".medulla"))
    yield
    cfg._config = None


def test_remove_command_registered():
    result = _runner.invoke(app, ["--help"])
    assert "remove" in result.output


def test_remove_command_unknown_slug_errors():
    result = _runner.invoke(app, ["remove", "concepts/nonexistent", "--yes"])
    assert result.exit_code != 0 or "not found" in result.output.lower()


def _setup_cli_wiki(medulla_dir):
    """Build a source + two concepts under an isolated medulla_dir for CLI tests."""
    import medulla.config as cfg
    wiki = medulla_dir / "wiki"
    for sub in ("sources", "concepts", "raw"):
        (wiki / sub).mkdir(parents=True, exist_ok=True)
    raw_file = wiki / "raw" / "paper.pdf"
    raw_file.write_bytes(b"fake")

    conn = connect(cfg.get_config().db_path)
    src = wiki / "sources" / "my-source.md"
    src.write_text("---\ntitle: My Source\nsources: []\nrelated: []\n---\n")
    upsert_wiki_page(conn, "my-source", "source", "My Source", src.read_text(),
                     src, raw_path=raw_file)
    ca = wiki / "concepts" / "concept-a.md"
    ca.write_text('---\ntitle: A\nsources: ["my-source"]\nrelated: []\n---\n')
    upsert_wiki_page(conn, "concept-a", "concept", "A", ca.read_text(), ca, sources=["my-source"])
    cb = wiki / "concepts" / "concept-b.md"
    cb.write_text('---\ntitle: B\nsources: ["my-source", "other"]\nrelated: ["[[concepts/concept-a]]"]\n---\n')
    upsert_wiki_page(conn, "concept-b", "concept", "B", cb.read_text(), cb, sources=["my-source", "other"])
    conn.close()
    return wiki, raw_file


def test_remove_command_source_shows_all_sections(tmp_path, monkeypatch):
    """Source removal preview shows affected sources, orphan warning, and related cleanup."""
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    _setup_cli_wiki(medulla_dir)
    result = _runner.invoke(app, ["remove", "sources/my-source", "--yes"])
    assert result.exit_code == 0
    assert "Removes from" in result.output          # affected_sources_update
    assert "orphaned" in result.output.lower()       # would_orphan (no cascade)


def test_remove_command_concept_shows_related_cleanup(tmp_path, monkeypatch):
    """Concept removal preview shows the related: links it will clean."""
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    _setup_cli_wiki(medulla_dir)
    result = _runner.invoke(app, ["remove", "concepts/concept-a", "--yes"])
    assert result.exit_code == 0
    assert "Cleans" in result.output                 # concept-b has [[concepts/concept-a]]


def test_remove_command_cascade_lists_orphans(tmp_path, monkeypatch):
    """--cascade preview lists the orphaned pages that will also be removed."""
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    wiki, _ = _setup_cli_wiki(medulla_dir)
    result = _runner.invoke(app, ["remove", "sources/my-source", "--yes", "--cascade"])
    assert result.exit_code == 0
    assert "concepts/concept-a" in result.output
    assert not (wiki / "concepts" / "concept-a.md").exists()


def test_remove_command_raw_shows_linked_source(tmp_path, monkeypatch):
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    _setup_cli_wiki(medulla_dir)
    result = _runner.invoke(app, ["remove", "raw/paper.pdf", "--yes"])
    assert result.exit_code == 0
    assert "Also removes source page" in result.output


def test_remove_command_raw_no_linked_source(tmp_path, monkeypatch):
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    wiki, _ = _setup_cli_wiki(medulla_dir)
    (wiki / "raw" / "orphan.pdf").write_bytes(b"x")
    result = _runner.invoke(app, ["remove", "raw/orphan.pdf", "--yes"])
    assert result.exit_code == 0
    assert "No linked source page found" in result.output


def test_remove_command_abort_on_no(tmp_path, monkeypatch):
    """Answering 'n' at the prompt aborts without deleting."""
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    wiki, _ = _setup_cli_wiki(medulla_dir)
    result = _runner.invoke(app, ["remove", "concepts/concept-a"], input="n\n")
    assert "Aborted" in result.output
    assert (wiki / "concepts" / "concept-a.md").exists()


def test_remove_command_invalid_target_format(tmp_path, monkeypatch):
    import medulla.config as cfg
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    result = _runner.invoke(app, ["remove", "noslash", "--yes"])
    assert result.exit_code == 1


# ── remove.py unit branches ─────────────────────────────────────────────────────

def test_plan_remove_invalid_format(db):
    from medulla.semantic.remove import plan_remove
    assert "error" in plan_remove(db, "noslash")


def test_plan_remove_unknown_folder(db):
    from medulla.semantic.remove import plan_remove
    assert "error" in plan_remove(db, "bogus/thing")


def test_execute_remove_passes_through_error(db):
    from medulla.semantic.remove import execute_remove
    result = execute_remove(db, "concepts/does-not-exist")
    assert "error" in result


def test_plan_raw_fallback_content_scan(db, tmp_path):
    """When raw_path column is unset, _plan_raw falls back to scanning source content."""
    from medulla.semantic.remove import plan_remove
    wiki = tmp_path / "wiki"
    (wiki / "sources").mkdir(parents=True)
    (wiki / "raw").mkdir()
    raw_file = wiki / "raw" / "legacy.pdf"
    raw_file.write_bytes(b"x")
    src = wiki / "sources" / "legacy-src.md"
    src.write_text(f"---\ntitle: Legacy\n---\nArchived at {raw_file}\n")
    # raw_path deliberately NOT set → forces fallback branch
    upsert_wiki_page(db, "legacy-src", "source", "Legacy", src.read_text(), src)
    plan = plan_remove(db, "raw/legacy.pdf", wiki_path=wiki)
    assert plan["linked_source_slug"] == "legacy-src"


def test_remove_command_yes_flag_skips_prompt(db, tmp_path, monkeypatch):
    """--yes flag skips interactive Y/N and executes directly."""
    import medulla.config as cfg
    from medulla.semantic.store import upsert_wiki_page

    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    wiki = medulla_dir / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    page = wiki / "concepts" / "test-remove-concept.md"
    page.write_text("---\ntitle: Test\nsources: []\nrelated: []\n---\n")

    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    conn = connect(cfg.get_config().db_path)
    upsert_wiki_page(conn, "test-remove-concept", "concept", "Test", page.read_text(), page)
    conn.close()

    result = _runner.invoke(app, ["remove", "concepts/test-remove-concept", "--yes"])
    assert result.exit_code == 0
    assert not page.exists()


# ── MCP medulla_remove tool ────────────────────────────────────────────────────

def test_mcp_remove_tool_registered():
    """medulla_remove is in the MCP tool registry."""
    from medulla.mcp import _HANDLERS
    assert "medulla_remove" in _HANDLERS


def test_mcp_remove_concept(db, tmp_path, monkeypatch):
    """medulla_remove removes a concept page via MCP."""
    import medulla.config as cfg
    from medulla.mcp import _HANDLERS

    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()
    wiki = medulla_dir / "wiki"
    (wiki / "concepts").mkdir(parents=True)
    page = wiki / "concepts" / "mcp-remove-test.md"
    page.write_text("---\ntitle: Test\nsources: []\nrelated: []\n---\n")

    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=medulla_dir))
    conn = connect(cfg.get_config().db_path)
    upsert_wiki_page(conn, "mcp-remove-test", "concept", "Test", page.read_text(), page)

    result = _HANDLERS["medulla_remove"](conn, {"target": "concepts/mcp-remove-test"})
    assert "removed" in result.lower() or "deleted" in result.lower()
    assert not page.exists()
    conn.close()
