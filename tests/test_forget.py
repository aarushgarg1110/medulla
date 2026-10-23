"""`medulla forget` removes a session from the index, CLI and MCP (#65), and the scanner uses the same delete
when a transcript re-parses as empty (the ahtest2Part2 stub that survived the 2026-10-23 rescan).

The real index on 2026-10-23: 215 sessions, 23 with one turn or fewer, ~10 test/demo sessions (ahtest*, ahProbe,
MedullaDemo, medullaMCPTest, TestMedullaForDan), 6 whose transcript is gone from disk. Removing any of them took
hand-written SQL. Transcripts are ground truth and only ever read; the wiki layer is authored and is never
touched by a forget.

Planned behaviours (TDD step 2 — stubs until implemented):
"""
import pytest

PLANNED = pytest.mark.skip(reason="planned for v0.2.1, not implemented yet")


@PLANNED
def test_forget_removes_every_row_of_the_session():
    """sessions, session_chunks, vec_chunks (no trigger: explicit delete), tool_events, and agent_sessions whose
    parent is the session; the FTS tables follow through their triggers. Counts before/after per table."""


@PLANNED
def test_forget_keeps_wiki_pages_that_cite_the_session():
    """wiki_pages.session_id is authored content: kept, not cascaded."""


@PLANNED
def test_forget_by_title_or_id_prefix_and_unknown_is_a_clear_error():
    """Resolves like resume (title, then id prefix); an unknown name lists the closest titles and deletes nothing."""


@PLANNED
def test_forget_missing_lists_orphans_and_deletes_only_with_yes():
    """`--missing`: sessions whose transcript is gone from disk. Lists them; deletes only with --yes, never on
    scan (Claude Code deletes transcripts after 30 days by default, and medulla exists to keep that history)."""


@PLANNED
def test_a_transcript_that_now_parses_empty_loses_its_old_rows_on_rescan():
    """scanner: parse_session -> None for a file that HAS rows -> the same delete, result 'removed_empty'."""


@PLANNED
def test_mcp_medulla_forget_matches_the_cli():
    """Same resolution and the same report text from both front doors (like resume)."""
