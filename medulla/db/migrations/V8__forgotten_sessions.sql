-- V8: sessions `medulla forget` removed from the index (#65). Their transcripts can stay on
-- disk, so every scan skips these ids instead of indexing them again; --undo deletes the row.
CREATE TABLE IF NOT EXISTS forgotten_sessions (
    session_id          TEXT PRIMARY KEY,
    source              TEXT,
    slug                TEXT,
    project_dir         TEXT,
    started_at          TEXT,
    turn_count          INTEGER,
    forgotten_at        TEXT NOT NULL,
    transcript_deleted  INTEGER NOT NULL DEFAULT 0
);
