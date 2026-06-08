-- V4: Embedding layer — vector storage for session chunks and wiki pages.
-- Uses regular BLOB columns with vec_distance_cosine() for similarity search.
-- Exact cosine search is fast enough at medulla's scale (thousands of rows, not millions).

CREATE TABLE IF NOT EXISTS vec_chunks (
    session_id  TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding   BLOB NOT NULL,
    PRIMARY KEY (session_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS vec_wiki (
    slug        TEXT NOT NULL PRIMARY KEY,
    embedding   BLOB NOT NULL
);
