"""`medulla clean` lists the junk in the index and forgets it only when asked (v0.2.1, built on `forget`).

Planned behaviours (TDD step 2 — stubs until implemented):
"""
import pytest

PLANNED = pytest.mark.skip(reason="planned for v0.2.1, not implemented yet")


@PLANNED
def test_clean_lists_orphans_and_forgets_them_only_with_yes():
    """Sessions whose transcript is gone from disk (6 in the real index on 2026-10-23). Lists them; forgets only
    with --yes, never on scan (Claude Code deletes transcripts after 30 days by default, and medulla exists to
    keep that history)."""
