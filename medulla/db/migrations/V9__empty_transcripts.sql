-- V9: transcripts that parsed as no conversation (no reply, only commands), keyed by file name
-- like sessions' scanned_at lookup, with the file's mtime then. Until the file changes, a scan
-- does not parse it again and the stale note does not list it.
CREATE TABLE IF NOT EXISTS empty_transcripts (
    session_id  TEXT PRIMARY KEY,
    mtime       TEXT NOT NULL
);
