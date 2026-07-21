-- V6: Add is_error to tool_events so backfilled command history records
-- whether each tool call failed. Enables error/correction recall.
ALTER TABLE tool_events ADD COLUMN is_error INTEGER NOT NULL DEFAULT 0;
