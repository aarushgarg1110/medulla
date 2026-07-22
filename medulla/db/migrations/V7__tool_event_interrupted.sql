-- V7: distinguish user-aborted/interrupted tool calls from genuine failures.
-- An abort ("tool use was rejected", [Request interrupted]) is a choice, not a
-- fixable error, so it must not count as is_error.
ALTER TABLE tool_events ADD COLUMN interrupted INTEGER NOT NULL DEFAULT 0;
