-- V2: Semantic layer — wiki pages, pending ingest queue

CREATE TABLE IF NOT EXISTS wiki_pages (
    id          INTEGER PRIMARY KEY,
    slug        TEXT UNIQUE NOT NULL,
    type        TEXT NOT NULL,  -- source | concept | entity
    title       TEXT NOT NULL,
    tags        TEXT,           -- JSON array
    sources     TEXT,           -- JSON array of source slugs
    content     TEXT NOT NULL,  -- full markdown content
    file_path   TEXT,           -- absolute path to .md file
    scope       TEXT NOT NULL DEFAULT 'personal',  -- personal | org
    ingested_by TEXT NOT NULL DEFAULT 'medulla',   -- medulla | human
    session_id  TEXT,           -- if created from a session
    ingested_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS pending_ingests (
    id          INTEGER PRIMARY KEY,
    source_path TEXT NOT NULL,  -- original file path or URL
    source_type TEXT NOT NULL,  -- pdf | url | markdown | text
    title       TEXT,
    status      TEXT NOT NULL DEFAULT 'queued',  -- queued | processing | done | error
    error       TEXT,
    queued_at   TEXT NOT NULL DEFAULT (datetime('now')),
    processed_at TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS wiki_fts USING fts5(
    slug        UNINDEXED,
    type        UNINDEXED,
    title,
    content,
    content='wiki_pages',
    content_rowid='id'
);

CREATE TRIGGER IF NOT EXISTS wiki_ai AFTER INSERT ON wiki_pages BEGIN
    INSERT INTO wiki_fts(rowid, slug, type, title, content)
    VALUES (new.id, new.slug, new.type, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS wiki_au AFTER UPDATE ON wiki_pages BEGIN
    INSERT INTO wiki_fts(wiki_fts, rowid, slug, type, title, content)
    VALUES ('delete', old.id, old.slug, old.type, old.title, old.content);
    INSERT INTO wiki_fts(rowid, slug, type, title, content)
    VALUES (new.id, new.slug, new.type, new.title, new.content);
END;
CREATE TRIGGER IF NOT EXISTS wiki_ad AFTER DELETE ON wiki_pages BEGIN
    INSERT INTO wiki_fts(wiki_fts, rowid, slug, type, title, content)
    VALUES ('delete', old.id, old.slug, old.type, old.title, old.content);
END;

CREATE INDEX IF NOT EXISTS idx_wiki_type ON wiki_pages(type);
CREATE INDEX IF NOT EXISTS idx_wiki_slug ON wiki_pages(slug);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_ingests(status);
