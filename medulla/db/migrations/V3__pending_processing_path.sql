-- Add processing_path column to pending_ingests.
-- source_path is now the dedup key (URL string or sha256:hash for binaries).
-- processing_path is the actual file path used during LLM processing.
-- For existing rows, processing_path falls back to source_path.
ALTER TABLE pending_ingests ADD COLUMN processing_path TEXT;
