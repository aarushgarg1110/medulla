"""Tests for sqlite-vec integration, embedding store, and medulla embed CLI."""
import json
import sqlite3
from pathlib import Path

import pytest

from medulla.db.database import connect


# ── MockEmbeddingProvider ─────────────────────────────────────────────────────

class MockEmbeddingProvider:
    """Returns deterministic fake 768-dim embeddings without downloading any model."""
    dimension = 768
    model_name = "mock-embeddings"

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Each text gets a unique vector based on its hash so similarity tests work
        results = []
        for text in texts:
            seed = hash(text) % 1000
            vec = [(seed + i) / 1000.0 for i in range(self.dimension)]
            results.append(vec)
        return results


# ── sqlite-vec loads ──────────────────────────────────────────────────────────

def test_sqlite_vec_loads_without_error(tmp_path):
    """sqlite-vec extension must load cleanly on this platform."""
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    # If sqlite-vec loaded, vec_version() is available
    row = conn.execute("SELECT vec_version()").fetchone()
    assert row[0] is not None
    conn.close()


def test_vec_chunks_table_exists(tmp_path):
    """V4 migration must create vec_chunks virtual table."""
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    conn.close()
    assert "vec_chunks" in tables


def test_vec_wiki_table_exists(tmp_path):
    """V4 migration must create vec_wiki virtual table."""
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    conn.close()
    assert "vec_wiki" in tables


# ── chunk embedding store ─────────────────────────────────────────────────────

def test_upsert_and_get_chunk_embedding(db):
    from medulla.db.embedding_store import upsert_chunk_embedding, get_chunk_embedding
    provider = MockEmbeddingProvider()
    embedding = provider.embed(["test chunk text"])[0]
    upsert_chunk_embedding(db, "sess-abc", 0, embedding)
    result = get_chunk_embedding(db, "sess-abc", 0)
    assert result is not None
    assert len(result) == 768
    assert abs(result[0] - embedding[0]) < 1e-5


def test_upsert_chunk_embedding_updates_not_duplicates(db):
    from medulla.db.embedding_store import upsert_chunk_embedding, get_chunk_embedding
    provider = MockEmbeddingProvider()
    emb1 = provider.embed(["first version"])[0]
    emb2 = provider.embed(["second version"])[0]
    upsert_chunk_embedding(db, "sess-abc", 0, emb1)
    upsert_chunk_embedding(db, "sess-abc", 0, emb2)
    result = get_chunk_embedding(db, "sess-abc", 0)
    # Should have updated, not duplicated
    assert abs(result[0] - emb2[0]) < 1e-5
    count = db.execute("SELECT COUNT(*) FROM vec_chunks").fetchone()[0]
    assert count == 1


def test_get_chunk_embedding_missing_returns_none(db):
    from medulla.db.embedding_store import get_chunk_embedding
    assert get_chunk_embedding(db, "nonexistent", 0) is None


# ── wiki embedding store ──────────────────────────────────────────────────────

def test_upsert_and_get_wiki_embedding(db):
    from medulla.db.embedding_store import upsert_wiki_embedding, get_wiki_embedding
    provider = MockEmbeddingProvider()
    embedding = provider.embed(["adam optimizer concept"])[0]
    upsert_wiki_embedding(db, "adam-optimizer", embedding)
    result = get_wiki_embedding(db, "adam-optimizer")
    assert result is not None
    assert len(result) == 768
    assert abs(result[0] - embedding[0]) < 1e-5


def test_upsert_wiki_embedding_updates_not_duplicates(db):
    from medulla.db.embedding_store import upsert_wiki_embedding
    provider = MockEmbeddingProvider()
    emb1 = provider.embed(["v1"])[0]
    emb2 = provider.embed(["v2"])[0]
    upsert_wiki_embedding(db, "adam-optimizer", emb1)
    upsert_wiki_embedding(db, "adam-optimizer", emb2)
    count = db.execute("SELECT COUNT(*) FROM vec_wiki").fetchone()[0]
    assert count == 1


def test_get_wiki_embedding_missing_returns_none(db):
    from medulla.db.embedding_store import get_wiki_embedding
    assert get_wiki_embedding(db, "nonexistent-slug") is None


# ── missing-embeddings queries ────────────────────────────────────────────────

def test_get_chunks_without_embeddings(db):
    from medulla.db.embedding_store import get_chunks_without_embeddings, upsert_chunk_embedding
    from medulla.episodic.store import upsert_session
    from tests.test_store import make_session

    # Index a session — produces chunks
    s = make_session("sess-embed-test", messages=["content about logD"] * 30)
    upsert_session(db, s)

    missing = get_chunks_without_embeddings(db)
    assert len(missing) > 0
    assert all("session_id" in row.keys() for row in missing)

    # Embed one chunk — it should no longer be missing
    provider = MockEmbeddingProvider()
    first = missing[0]
    emb = provider.embed([first["chunk_text"]])[0]
    upsert_chunk_embedding(db, first["session_id"], first["chunk_index"], emb)

    still_missing = get_chunks_without_embeddings(db)
    assert len(still_missing) == len(missing) - 1


def test_get_wiki_pages_without_embeddings(db):
    from medulla.db.embedding_store import get_wiki_pages_without_embeddings, upsert_wiki_embedding
    from medulla.semantic.store import upsert_wiki_page

    upsert_wiki_page(db, "concept-a", "concept", "Concept A",
                     "## Definition\n\nSome definition.",
                     Path("/wiki/concepts/concept-a.md"))
    upsert_wiki_page(db, "concept-b", "concept", "Concept B",
                     "## Definition\n\nAnother definition.",
                     Path("/wiki/concepts/concept-b.md"))

    missing = get_wiki_pages_without_embeddings(db)
    assert len(missing) == 2

    # Embed one — it should no longer appear
    provider = MockEmbeddingProvider()
    emb = provider.embed([missing[0]["content"]])[0]
    upsert_wiki_embedding(db, missing[0]["slug"], emb)

    still_missing = get_wiki_pages_without_embeddings(db)
    assert len(still_missing) == 1


# ── cosine similarity ordering ────────────────────────────────────────────────

def test_cosine_similarity_returns_closest_first(db):
    """vec_chunks cosine similarity must rank semantically-close embeddings higher."""
    from medulla.db.embedding_store import upsert_chunk_embedding, find_similar_chunks
    from medulla.episodic.store import upsert_session
    from tests.test_store import make_session

    # Need real session_chunks rows for the JOIN to work
    upsert_session(db, make_session("sess-close", messages=["close content"] * 5))
    upsert_session(db, make_session("sess-far", messages=["far content"] * 5))

    dim = 768
    close = [1.0] + [0.0] * (dim - 1)
    far   = [0.0] + [1.0] + [0.0] * (dim - 2)
    query = [0.9] + [0.1] + [0.0] * (dim - 2)

    upsert_chunk_embedding(db, "sess-close", 0, close)
    upsert_chunk_embedding(db, "sess-far",   0, far)

    results = find_similar_chunks(db, query, top_k=2)
    assert len(results) == 2
    assert results[0]["session_id"] == "sess-close"
    assert results[1]["session_id"] == "sess-far"


def test_find_similar_wiki_pages(db):
    """vec_wiki cosine similarity returns pages in order of relevance."""
    from medulla.db.embedding_store import upsert_wiki_embedding, find_similar_wiki_pages
    from medulla.semantic.store import upsert_wiki_page

    # Need real wiki_pages rows for the JOIN
    upsert_wiki_page(db, "slug-close", "concept", "Close Concept",
                     "Close content.", Path("/wiki/concepts/slug-close.md"))
    upsert_wiki_page(db, "slug-far", "concept", "Far Concept",
                     "Far content.", Path("/wiki/concepts/slug-far.md"))

    dim = 768
    close = [1.0] + [0.0] * (dim - 1)
    far   = [0.0] + [1.0] + [0.0] * (dim - 2)
    query = [0.9] + [0.1] + [0.0] * (dim - 2)

    upsert_wiki_embedding(db, "slug-close", close)
    upsert_wiki_embedding(db, "slug-far",   far)

    results = find_similar_wiki_pages(db, query, top_k=2)
    assert results[0]["slug"] == "slug-close"
    assert results[1]["slug"] == "slug-far"


# ── EmbeddingProvider protocol + SentenceTransformersProvider ────────────────

def test_sentence_transformers_provider_init():
    """SentenceTransformersProvider initialises without downloading the model."""
    from medulla.embeddings import SentenceTransformersProvider
    p = SentenceTransformersProvider()
    assert p.model_name == "intfloat/e5-base-v2"
    assert p.dimension == 768
    assert p._model is None  # lazy — not loaded yet


def test_sentence_transformers_provider_custom_model():
    from medulla.embeddings import SentenceTransformersProvider
    p = SentenceTransformersProvider(model="sentence-transformers/all-MiniLM-L6-v2")
    assert p.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_get_embedding_provider_returns_provider():
    """get_embedding_provider returns an EmbeddingProvider instance."""
    from medulla.embeddings import get_embedding_provider, EmbeddingProvider, SentenceTransformersProvider
    p = get_embedding_provider()
    assert isinstance(p, SentenceTransformersProvider)
    assert isinstance(p, EmbeddingProvider)


# ── embed CLI command ─────────────────────────────────────────────────────────

from typer.testing import CliRunner
from medulla.cli import app

_runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_embed_config(tmp_path, monkeypatch):
    import medulla.config as cfg
    cfg._config = None
    monkeypatch.setattr(cfg, "_config", cfg.Config(medulla_dir=tmp_path / ".medulla"))
    yield
    cfg._config = None


def test_embed_command_registered():
    result = _runner.invoke(app, ["--help"])
    assert "embed" in result.output


def test_embed_command_empty_db_exits_zero(monkeypatch):
    monkeypatch.setattr("medulla.cli._get_embedding_provider",
                        lambda: MockEmbeddingProvider())
    result = _runner.invoke(app, ["embed"])
    assert result.exit_code == 0


def test_embed_command_backfills_chunks(monkeypatch):
    from medulla.db.database import connect as db_connect
    from medulla.episodic.store import upsert_session
    from medulla.db.embedding_store import get_chunks_without_embeddings
    from tests.test_store import make_session
    import medulla.config as cfg

    monkeypatch.setattr("medulla.cli._get_embedding_provider",
                        lambda: MockEmbeddingProvider())

    conn = db_connect(cfg.get_config().db_path)
    s = make_session("sess-embed-cli", messages=["content about logD analysis"] * 30)
    upsert_session(conn, s)
    conn.close()

    assert len(get_chunks_without_embeddings(db_connect(cfg.get_config().db_path))) > 0
    result = _runner.invoke(app, ["embed"])
    assert result.exit_code == 0
    assert len(get_chunks_without_embeddings(db_connect(cfg.get_config().db_path))) == 0


def test_embed_command_backfills_wiki_pages(monkeypatch):
    from medulla.db.database import connect as db_connect
    from medulla.semantic.store import upsert_wiki_page
    from medulla.db.embedding_store import get_wiki_pages_without_embeddings
    import medulla.config as cfg

    monkeypatch.setattr("medulla.cli._get_embedding_provider",
                        lambda: MockEmbeddingProvider())

    conn = db_connect(cfg.get_config().db_path)
    upsert_wiki_page(conn, "test-concept", "concept", "Test Concept",
                     "## Definition\n\nSomething.", Path("/wiki/concepts/test-concept.md"))
    conn.close()

    result = _runner.invoke(app, ["embed"])
    assert result.exit_code == 0
    assert len(get_wiki_pages_without_embeddings(db_connect(cfg.get_config().db_path))) == 0


def test_embed_command_shows_counts(monkeypatch):
    from medulla.db.database import connect as db_connect
    from medulla.episodic.store import upsert_session
    from tests.test_store import make_session
    import medulla.config as cfg

    monkeypatch.setattr("medulla.cli._get_embedding_provider",
                        lambda: MockEmbeddingProvider())

    conn = db_connect(cfg.get_config().db_path)
    upsert_session(conn, make_session("sess-counts", messages=["data"] * 30))
    conn.close()

    result = _runner.invoke(app, ["embed"])
    assert result.exit_code == 0
    assert "chunk" in result.output.lower()


def test_embed_force_reembeds_all(monkeypatch):
    from medulla.db.database import connect as db_connect
    from medulla.episodic.store import upsert_session
    from medulla.db.embedding_store import upsert_chunk_embedding, get_chunks_without_embeddings
    from tests.test_store import make_session
    import medulla.config as cfg

    monkeypatch.setattr("medulla.cli._get_embedding_provider",
                        lambda: MockEmbeddingProvider())

    conn = db_connect(cfg.get_config().db_path)
    s = make_session("sess-force", messages=["content"] * 30)
    upsert_session(conn, s)
    # Pre-embed one chunk
    chunks = get_chunks_without_embeddings(conn)
    upsert_chunk_embedding(conn, chunks[0]["session_id"], chunks[0]["chunk_index"],
                           [0.1] * 768)
    conn.close()

    # --force should re-embed everything including already-embedded
    result = _runner.invoke(app, ["embed", "--force"])
    assert result.exit_code == 0


# ── auto-embed on scan ────────────────────────────────────────────────────────

def test_scan_auto_embeds_new_chunks(monkeypatch, tmp_path):
    """After scan, new session chunks have embeddings in vec_chunks."""
    import medulla.config as cfg
    from medulla.db.database import connect as db_connect
    from medulla.episodic.scanner import scan
    from medulla.db.embedding_store import get_chunks_without_embeddings
    from tests.conftest import claude_user, make_claude_jsonl

    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "kiro-none")
    monkeypatch.setattr("medulla.episodic.scanner._get_embedding_provider",
                        lambda: MockEmbeddingProvider())

    proj = tmp_path / "my-project"
    proj.mkdir()
    messages = [claude_user(f"message {i}", session_id="sess-scan-embed")
                for i in range(30)]
    (proj / "sess-scan-embed.jsonl").write_text(make_claude_jsonl(messages))

    conn = db_connect(cfg.get_config().db_path)
    scan(conn, force=False)
    missing = get_chunks_without_embeddings(conn)
    conn.close()
    assert len(missing) == 0


# ── auto-embed on ingest ──────────────────────────────────────────────────────

def test_ingest_auto_embeds_wiki_pages(monkeypatch, tmp_path):
    """After process_pending, new wiki pages have embeddings in vec_wiki."""
    import medulla.config as cfg
    from medulla.db.database import connect as db_connect
    from medulla.semantic.ingest import intake_to_raw, process_pending
    from medulla.db.embedding_store import get_wiki_pages_without_embeddings
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_ingest_pipeline import MockProvider

    monkeypatch.setattr("medulla.semantic.ingest._get_embedding_provider",
                        lambda: MockEmbeddingProvider())

    conn = db_connect(cfg.get_config().db_path)
    wiki = tmp_path / "wiki"
    md = tmp_path / "paper.md"
    md.write_text("# LogD Study\n\nContent about batch effects.")

    intake_to_raw(conn, wiki, str(md))
    process_pending(wiki, conn, MockProvider())

    missing = get_wiki_pages_without_embeddings(conn)
    conn.close()
    assert len(missing) == 0
