-- V1: Episodic layer — sessions, chunks, agent sessions, tool events

CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT UNIQUE NOT NULL,
    source      TEXT NOT NULL DEFAULT 'claude',  -- claude | kiro | codex | gemini
    project_dir TEXT,
    git_branch  TEXT,
    slug        TEXT,
    model       TEXT,
    started_at  TEXT,
    ended_at    TEXT,
    turn_count  INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    tool_names  TEXT,   -- JSON array
    files_json  TEXT,   -- JSON array
    first_message TEXT, -- up to 500 chars
    all_user_text TEXT, -- FULL text, no cap
    scope       TEXT NOT NULL DEFAULT 'private',  -- private | public
    scanned_at  TEXT NOT NULL DEFAULT (datetime('now')),
    source_instance TEXT NOT NULL DEFAULT 'local'
);

CREATE TABLE IF NOT EXISTS session_chunks (
    id          INTEGER PRIMARY KEY,
    session_id  TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text  TEXT NOT NULL,
    turn_start  INTEGER NOT NULL,
    turn_end    INTEGER NOT NULL,
    UNIQUE(session_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS agent_sessions (
    id               INTEGER PRIMARY KEY,
    agent_id         TEXT UNIQUE NOT NULL,
    parent_session_id TEXT,
    agent_slug       TEXT,
    project_dir      TEXT,
    cwd              TEXT,
    model            TEXT,
    turn_count       INTEGER DEFAULT 0,
    tool_call_count  INTEGER DEFAULT 0,
    tool_names       TEXT,   -- JSON array
    first_message    TEXT,   -- up to 500 chars
    all_user_text    TEXT,   -- FULL text, no cap
    first_seen_at    TEXT,
    last_updated_at  TEXT,
    message_count    INTEGER DEFAULT 0,
    scanned_at       TEXT NOT NULL DEFAULT (datetime('now')),
    source_instance  TEXT NOT NULL DEFAULT 'local'
);

CREATE TABLE IF NOT EXISTS tool_events (
    id            INTEGER PRIMARY KEY,
    event_ts      TEXT,
    session_id    TEXT,
    project_dir   TEXT,
    tool          TEXT,
    command       TEXT,   -- up to 500 chars
    manifest_key  TEXT,
    output_preview TEXT,  -- up to 200 chars of output
    manifest_version TEXT,
    event_hash    TEXT UNIQUE,  -- SHA-256 dedup
    ingested_at   TEXT NOT NULL DEFAULT (datetime('now')),
    source_instance TEXT NOT NULL DEFAULT 'local'
);

-- FTS5 indexes
CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
    session_id  UNINDEXED,
    first_message,
    all_user_text,
    content='sessions',
    content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS session_chunks_fts USING fts5(
    session_id  UNINDEXED,
    chunk_index UNINDEXED,
    chunk_text,
    content='session_chunks',
    content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS agent_sessions_fts USING fts5(
    agent_id    UNINDEXED,
    agent_slug,
    first_message,
    all_user_text,
    content='agent_sessions',
    content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS tool_events_fts USING fts5(
    command,
    project_dir,
    content='tool_events',
    content_rowid='id'
);

-- Sync triggers: sessions
CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
    INSERT INTO sessions_fts(rowid, session_id, first_message, all_user_text)
    VALUES (new.id, new.session_id, new.first_message, new.all_user_text);
END;
CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, session_id, first_message, all_user_text)
    VALUES ('delete', old.id, old.session_id, old.first_message, old.all_user_text);
    INSERT INTO sessions_fts(rowid, session_id, first_message, all_user_text)
    VALUES (new.id, new.session_id, new.first_message, new.all_user_text);
END;
CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
    INSERT INTO sessions_fts(sessions_fts, rowid, session_id, first_message, all_user_text)
    VALUES ('delete', old.id, old.session_id, old.first_message, old.all_user_text);
END;

-- Sync triggers: session_chunks
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON session_chunks BEGIN
    INSERT INTO session_chunks_fts(rowid, session_id, chunk_index, chunk_text)
    VALUES (new.id, new.session_id, new.chunk_index, new.chunk_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON session_chunks BEGIN
    INSERT INTO session_chunks_fts(session_chunks_fts, rowid, session_id, chunk_index, chunk_text)
    VALUES ('delete', old.id, old.session_id, old.chunk_index, old.chunk_text);
    INSERT INTO session_chunks_fts(rowid, session_id, chunk_index, chunk_text)
    VALUES (new.id, new.session_id, new.chunk_index, new.chunk_text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON session_chunks BEGIN
    INSERT INTO session_chunks_fts(session_chunks_fts, rowid, session_id, chunk_index, chunk_text)
    VALUES ('delete', old.id, old.session_id, old.chunk_index, old.chunk_text);
END;

-- Sync triggers: agent_sessions
CREATE TRIGGER IF NOT EXISTS agent_sessions_ai AFTER INSERT ON agent_sessions BEGIN
    INSERT INTO agent_sessions_fts(rowid, agent_id, agent_slug, first_message, all_user_text)
    VALUES (new.id, new.agent_id, new.agent_slug, new.first_message, new.all_user_text);
END;
CREATE TRIGGER IF NOT EXISTS agent_sessions_au AFTER UPDATE ON agent_sessions BEGIN
    INSERT INTO agent_sessions_fts(agent_sessions_fts, rowid, agent_id, agent_slug, first_message, all_user_text)
    VALUES ('delete', old.id, old.agent_id, old.agent_slug, old.first_message, old.all_user_text);
    INSERT INTO agent_sessions_fts(rowid, agent_id, agent_slug, first_message, all_user_text)
    VALUES (new.id, new.agent_id, new.agent_slug, new.first_message, new.all_user_text);
END;
CREATE TRIGGER IF NOT EXISTS agent_sessions_ad AFTER DELETE ON agent_sessions BEGIN
    INSERT INTO agent_sessions_fts(agent_sessions_fts, rowid, agent_id, agent_slug, first_message, all_user_text)
    VALUES ('delete', old.id, old.agent_id, old.agent_slug, old.first_message, old.all_user_text);
END;

-- Sync triggers: tool_events
CREATE TRIGGER IF NOT EXISTS tool_events_ai AFTER INSERT ON tool_events BEGIN
    INSERT INTO tool_events_fts(rowid, command, project_dir)
    VALUES (new.id, new.command, new.project_dir);
END;
CREATE TRIGGER IF NOT EXISTS tool_events_ad AFTER DELETE ON tool_events BEGIN
    INSERT INTO tool_events_fts(tool_events_fts, rowid, command, project_dir)
    VALUES ('delete', old.id, old.command, old.project_dir);
END;

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project_dir);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_source ON sessions(source_instance);
CREATE INDEX IF NOT EXISTS idx_chunks_session ON session_chunks(session_id, chunk_index);
CREATE INDEX IF NOT EXISTS idx_agent_sessions_parent ON agent_sessions(parent_session_id);
CREATE INDEX IF NOT EXISTS idx_tool_events_session ON tool_events(session_id);
CREATE INDEX IF NOT EXISTS idx_tool_events_ts ON tool_events(event_ts DESC);
CREATE INDEX IF NOT EXISTS idx_tool_events_hash ON tool_events(event_hash);
