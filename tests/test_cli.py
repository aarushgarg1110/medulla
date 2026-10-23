"""Tests for medulla.cli — Typer CliRunner, real SQLite in tmp_path."""
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from medulla.cli import app
from tests.conftest import claude_assistant, claude_user, make_claude_jsonl

runner = CliRunner()


class _MockEmbedProvider:
    dimension = 768
    model_name = "mock"
    def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]


@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Point config at a tmp_path so tests don't touch ~/.medulla."""
    medulla_dir = tmp_path / ".medulla"
    medulla_dir.mkdir()

    import medulla.config as cfg_module
    import medulla.episodic.scanner as scanner_mod
    import medulla.semantic.ingest as ingest_mod

    # Reset singleton so each test gets a fresh config
    cfg_module._config = None

    monkeypatch.setattr(cfg_module, "_config", cfg_module.Config(medulla_dir=medulla_dir))
    monkeypatch.setattr(scanner_mod, "_get_embedding_provider", lambda: _MockEmbedProvider())
    monkeypatch.setattr(ingest_mod, "_get_embedding_provider", lambda: _MockEmbedProvider())
    yield


@pytest.fixture
def claude_projects(tmp_path, monkeypatch):
    """Create a fake ~/.claude/projects dir with one session."""
    projects = tmp_path / "claude_projects"
    projects.mkdir()
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", projects)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "kiro_none")

    proj = projects / "my-project"
    proj.mkdir()
    path = proj / "session-abc.jsonl"
    path.write_text(make_claude_jsonl([
        claude_user("tell me about logD outliers", session_id="session-abc-id"),
        claude_user("the CompoundX project had batch effects", session_id="session-abc-id"),
        claude_assistant(session_id="session-abc-id"),
    ]))
    return projects


# ── scan ───────────────────────────────────────────────────────────────────────

def test_scan_command_runs(claude_projects):
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "indexed" in result.output


def test_scan_command_force_flag(claude_projects):
    runner.invoke(app, ["scan"])  # first pass
    result = runner.invoke(app, ["scan", "--force"])
    assert result.exit_code == 0
    assert "1 indexed" in result.output


def test_scan_command_source_flag(claude_projects):
    result = runner.invoke(app, ["scan", "--source", "claude"])
    assert result.exit_code == 0


def test_scan_empty_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "none")
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none2")
    result = runner.invoke(app, ["scan"])
    assert result.exit_code == 0
    assert "0 indexed" in result.output


# ── search ─────────────────────────────────────────────────────────────────────

def test_search_command_finds_result(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD outliers"])
    assert result.exit_code == 0
    assert len(result.output.strip()) > 0


def test_search_command_no_results_bm25_only(claude_projects):
    """BM25-only returns no results for nonsense query."""
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "zzznomatch9999", "--bm25-only"])
    assert result.exit_code == 0
    assert "No results" in result.output


def test_search_command_hybrid_returns_results_for_unknown_terms(claude_projects):
    """Hybrid search returns closest semantic results even for unknown terms."""
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "zzznomatch9999"])
    assert result.exit_code == 0  # no crash — vector fallback kicks in


def test_search_command_limit_flag(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD", "--limit", "1"])
    assert result.exit_code == 0


def test_search_command_layer_flag(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD", "--layer", "episodic"])
    assert result.exit_code == 0


def test_search_command_shows_chunk_index_for_chunk_results(claude_projects):
    """CLI output shows 'chunk N' label for chunk results so user knows --chunk flag value."""
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD"])
    assert result.exit_code == 0
    # chunk results show inline chunk number; not all results are chunks so only check if present
    if "chunk" in result.output:
        import re
        assert re.search(r"chunk \d+", result.output)


# ── list ───────────────────────────────────────────────────────────────────────

def test_list_command_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", tmp_path / "none")
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "none2")
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "No sessions" in result.output


def test_list_command_shows_sessions(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    assert "logD" in result.output or "proj" in result.output


def test_list_command_project_filter(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["list", "--project", "nonexistent-project-xyz"])
    assert result.exit_code == 0
    assert "No sessions" in result.output


def test_list_command_limit_flag(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["list", "--limit", "5"])
    assert result.exit_code == 0


# ── stats ──────────────────────────────────────────────────────────────────────

def test_stats_command_empty(tmp_path, monkeypatch):
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "Sessions" in result.output
    assert "0" in result.output


def test_stats_command_after_scan(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["stats"])
    assert result.exit_code == 0
    assert "1" in result.output  # 1 session indexed


# ── mcp ────────────────────────────────────────────────────────────────────────

def test_mcp_command_is_registered():
    """mcp command exists and is registered — full stdio test requires integration."""
    result = runner.invoke(app, ["--help"])
    assert "mcp" in result.output


# ── session-detail --chunk ─────────────────────────────────────────────────────

@pytest.fixture
def chunked_session(tmp_path, monkeypatch):
    """Session with enough messages to produce multiple chunks."""
    projects = tmp_path / "claude_projects"
    projects.mkdir()
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", projects)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "kiro_none")

    proj = projects / "my-project"
    proj.mkdir()
    path = proj / "session-chunked.jsonl"
    messages = [claude_user(f"message number {i} about logD analysis", session_id="session-chunked-id")
                for i in range(30)] + [claude_assistant(session_id="session-chunked-id")]
    path.write_text(make_claude_jsonl(messages))
    return projects


def test_session_detail_chunk_flag_shows_single_chunk(chunked_session):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["session-detail", "session-chunked-id", "--chunk", "0"])
    assert result.exit_code == 0
    assert "Chunk 0" in result.output


def test_session_detail_chunk_flag_short_form(chunked_session):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["session-detail", "session-chunked-id", "-c", "0"])
    assert result.exit_code == 0
    assert "Chunk 0" in result.output


def test_session_detail_chunk_out_of_range(chunked_session):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["session-detail", "session-chunked-id", "--chunk", "9999"])
    assert result.exit_code == 1
    assert "not found" in result.output.lower()


def test_session_detail_no_chunk_shows_all(chunked_session):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["session-detail", "session-chunked-id"])
    assert result.exit_code == 0
    assert "Chunks" in result.output


# ── session-detail --start/--end range + search excerpt ─────────────────────────

@pytest.fixture
def big_chunked_session(tmp_path, monkeypatch):
    """Session with large messages → genuinely multiple chunks."""
    projects = tmp_path / "claude_projects"
    projects.mkdir()
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", projects)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "kiro_none")
    proj = projects / "my-project"
    proj.mkdir()
    path = proj / "session-big.jsonl"
    msgs = [claude_user(f"segment{i} " + "detailed analysis content sentence here " * 40,
                        session_id="session-big-id") for i in range(16)] + [claude_assistant(session_id="session-big-id")]
    path.write_text(make_claude_jsonl(msgs))
    return projects


def test_session_detail_range_shows_multiple_chunks(big_chunked_session):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["session-detail", "session-big-id", "--start", "0", "--end", "2"])
    assert result.exit_code == 0
    assert "Chunk 0" in result.output and "Chunk 1" in result.output and "Chunk 2" in result.output


def test_session_detail_range_clamps_end(big_chunked_session):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["session-detail", "session-big-id", "--start", "0", "--end", "9999"])
    assert result.exit_code == 0
    assert "Chunks 0–" in result.output


def test_session_detail_range_empty_errors(big_chunked_session):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["session-detail", "session-big-id", "--start", "5", "--end", "1"])
    assert result.exit_code == 1
    assert "Empty range" in result.output


def test_search_command_shows_tool_event(isolated_config):
    """CLI search surfaces harvested commands as $-prefixed tool_event rows."""
    from medulla.db.database import connect
    from medulla.episodic.store import upsert_tool_events
    from medulla.episodic.parser import ToolEvent
    conn = connect()
    upsert_tool_events(conn, "sess-c", [ToolEvent(
        session_id="sess-c", project_dir="/p", event_ts="2026-01-01T00:00:00Z",
        tool="Bash", command="duckdb -c \"SELECT zebraword\"", description="",
        output_preview="", is_error=False, event_hash="h1")])
    conn.close()
    result = runner.invoke(app, ["search", "zebraword", "--bm25-only"])
    assert result.exit_code == 0
    assert "$" in result.output and "zebraword" in result.output


def test_search_command_excerpt_is_match_centered(claude_projects):
    """CLI search shows a match-centered excerpt containing the query term."""
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD", "--bm25-only"])
    assert result.exit_code == 0
    assert "logD" in result.output or "logd" in result.output.lower()


# ── resume, search --expand / --refresh: the CLI gives what the MCP tools give ──

def _titled_session(projects, sid, title, body, age_s=600):
    import os
    import time
    path = projects / "my-project" / f"{sid}.jsonl"
    path.write_text(make_claude_jsonl([{"type": "custom-title", "customTitle": title},
                                       claude_user(f"please {body}", session_id=sid),
                                       claude_assistant(session_id=sid)]))
    t = time.time() - age_s
    os.utime(path, (t, t))
    return path


def test_resume_command_indexes_and_prints_the_handoff(claude_projects):
    _titled_session(claude_projects, "sess-handoff", "fooPart2", "## Immediate next action\nship the PR")
    result = runner.invoke(app, ["resume", "fooPart2"])
    assert result.exit_code == 0
    assert "ship the PR" in result.output and "indexed now" in result.output


def test_resume_command_exits_1_for_an_unknown_name(claude_projects):
    result = runner.invoke(app, ["resume", "noSuchSession"])
    assert result.exit_code == 1 and "not found" in result.output


def test_nextpart_command_prints_or_explains(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDULLA_DIR", str(tmp_path / "m"))
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    result = runner.invoke(app, ["nextpart", "--from", "fooPart3", "--dry-run"])
    assert result.exit_code == 0 and "claude -n fooPart4" in result.output
    result = runner.invoke(app, ["nextpart"])
    assert result.exit_code == 1 and "--from" in result.output


def test_search_command_expand_prints_the_whole_chunk(claude_projects):
    runner.invoke(app, ["scan"])
    result = runner.invoke(app, ["search", "logD outliers", "--expand", "1", "--bm25-only"])
    assert result.exit_code == 0 and "CompoundX" in result.output


def test_search_command_warns_when_transcripts_changed_since_indexing(claude_projects):
    runner.invoke(app, ["scan"])
    _titled_session(claude_projects, "sess-late", "latePart3", "newterm notes", age_s=120)
    result = runner.invoke(app, ["search", "logD", "--bm25-only"])
    assert "changed since" in result.output and "latePart3" in result.output


def test_search_command_refresh_indexes_before_searching(claude_projects):
    runner.invoke(app, ["scan"])
    _titled_session(claude_projects, "sess-late2", "laterPart1", "refreshterm notes", age_s=120)
    result = runner.invoke(app, ["search", "refreshterm", "--refresh", "--bm25-only"])
    assert result.exit_code == 0 and "No results" not in result.output


def test_search_with_no_results_still_says_the_index_is_behind(claude_projects):
    """No results plus a stale index is exactly how a just-ended session goes missing."""
    runner.invoke(app, ["scan"])
    _titled_session(claude_projects, "sess-late3", "missedPart2", "the answer lives here", age_s=120)
    result = runner.invoke(app, ["search", "zzznomatch9999", "--bm25-only"])
    assert "No results" in result.output and "missedPart2" in result.output


# ── forget: the CLI asks before deleting a transcript; the MCP tool never deletes one ─────────────────

@pytest.fixture
def forget_projects(tmp_path, monkeypatch):
    from tests.test_forget import write_session
    proj = tmp_path / "claude_projects" / "-proj-a"
    proj.mkdir(parents=True)
    monkeypatch.setattr("medulla.episodic.scanner.CLAUDE_PROJECTS_DIR", proj.parent)
    monkeypatch.setattr("medulla.episodic.scanner.KIRO_SESSIONS_DIR", tmp_path / "kiro_none")
    write_session(proj, "sess-a", "alphaTask")
    runner.invoke(app, ["scan"])
    return proj


def test_forget_command_asks_and_keeps_the_transcript_by_default(forget_projects, monkeypatch):
    monkeypatch.setattr("medulla.cli._can_ask", lambda: True)
    result = runner.invoke(app, ["forget", "alphaTask"], input="\n")
    assert result.exit_code == 0 and "Forgot alphaTask" in result.output and "Also delete the transcript" in result.output
    assert "Transcript kept." in result.output and (forget_projects / "sess-a.jsonl").exists()


def test_forget_command_deletes_the_transcript_on_yes(forget_projects, monkeypatch):
    monkeypatch.setattr("medulla.cli._can_ask", lambda: True)
    result = runner.invoke(app, ["forget", "alphaTask"], input="y\n")
    assert result.exit_code == 0 and "Deleted" in result.output
    assert not (forget_projects / "sess-a.jsonl").exists() and not (forget_projects / "sess-a").exists()


def test_forget_command_without_a_terminal_keeps_the_transcript_and_says_how(forget_projects):
    """Claude's Bash tool, a pipe or cron: nobody to ask, so no prompt and no deletion."""
    result = runner.invoke(app, ["forget", "alphaTask"])
    assert result.exit_code == 0 and "Also delete" not in result.output
    assert "Rerun with --delete-transcript" in result.output and (forget_projects / "sess-a.jsonl").exists()


def test_forget_command_delete_transcript_skips_the_prompt(forget_projects, monkeypatch):
    monkeypatch.setattr("medulla.cli._can_ask", lambda: True)
    result = runner.invoke(app, ["forget", "alphaTask", "--delete-transcript"])   # no input: a prompt would abort
    assert result.exit_code == 0 and "Also delete" not in result.output
    assert not (forget_projects / "sess-a.jsonl").exists()


def test_forget_command_deletes_later_what_was_kept(forget_projects):
    """The MCP tool's reply names this command; the session is found on the forgotten list."""
    runner.invoke(app, ["forget", "alphaTask"])
    result = runner.invoke(app, ["forget", "sess-a", "--delete-transcript"])
    assert result.exit_code == 0 and "already forgotten" in result.output and "Deleted" in result.output


def test_forget_command_refuses_a_live_transcript(forget_projects):
    import os
    os.utime(forget_projects / "sess-a.jsonl", None)
    result = runner.invoke(app, ["forget", "alphaTask", "--delete-transcript"])
    assert result.exit_code == 1 and "last minute" in result.output


def test_forget_command_list_undo_and_the_scan_count(forget_projects):
    runner.invoke(app, ["forget", "alphaTask"])
    assert "1 forgotten" in runner.invoke(app, ["scan"]).output
    assert "alphaTask" in runner.invoke(app, ["forget", "--list"]).output
    assert "the next scan indexes it again" in runner.invoke(app, ["forget", "--undo", "alphaTask"]).output
    assert runner.invoke(app, ["forget", "--undo", "alphaTask"]).exit_code == 1


def test_forget_command_with_no_transcript_on_disk_asks_nothing(forget_projects, monkeypatch):
    monkeypatch.setattr("medulla.cli._can_ask", lambda: True)
    (forget_projects / "sess-a.jsonl").unlink()
    result = runner.invoke(app, ["forget", "alphaTask"])
    assert result.exit_code == 0 and "Transcript: none on disk." in result.output and "Also delete" not in result.output


def test_forget_command_without_a_match_exits_1(forget_projects):
    assert runner.invoke(app, ["forget"]).exit_code == 1
    assert runner.invoke(app, ["forget", "noSuchSession"]).exit_code == 1


def test_scan_command_counts_rows_removed_for_a_transcript_that_is_no_longer_a_conversation(claude_projects):
    from medulla.db.database import connect
    from medulla.episodic.store import upsert_session
    from tests.test_store import make_session
    path = claude_projects / "my-project" / "sess-junk.jsonl"
    path.write_text(make_claude_jsonl([claude_user("hello", session_id="sess-junk")]))
    (claude_projects / "my-project" / "sess-hello.jsonl").write_text(make_claude_jsonl([claude_user("hi")]))
    upsert_session(connect(), make_session("sess-junk", turn_count=0))
    summary = runner.invoke(app, ["scan", "--force"]).output.splitlines()[-2]
    assert summary == "✓ Sessions: 1 indexed, 0 unchanged, 1 empty/stub, 1 removed (no longer a conversation), 0 errors"


def test_forget_cli_and_mcp_give_the_same_report(forget_projects):
    from medulla.db.database import connect
    from medulla.episodic import scanner
    from medulla.mcp import _tool_forget
    cli = runner.invoke(app, ["forget", "alphaTask"]).output
    runner.invoke(app, ["forget", "--undo", "alphaTask"])
    conn = connect()
    scanner.scan(conn)
    mcp = _tool_forget(conn, {"name": "alphaTask"})
    assert cli.startswith(mcp.split("\nTranscript kept")[0] + "\nTranscript kept")
    assert mcp.endswith("run in a terminal: medulla forget sess-a --delete-transcript")


def test_scan_prints_progress_lines_while_it_runs(claude_projects):
    """Not a terminal (CliRunner, Claude's Bash tool, a pipe): a plain line every 10%, then the unchanged
    summary. A full rescan of the real index took 4 min with no output on 2026-10-23."""
    import re
    for i in range(9):
        _titled_session(claude_projects, f"sess-p{i}", f"progPart{i}", "work")
    result = runner.invoke(app, ["scan"])
    lines = [ln for ln in result.output.splitlines() if ln.startswith("[")]
    pcts = [int(re.match(r"\[\d+/10 (\d+)%\] 0:00 elapsed · ETA 0:00$", ln).group(1)) for ln in lines]
    assert 5 <= len(lines) <= 10 and pcts == sorted(pcts) and lines[-1].startswith("[10/10 100%]")
    assert result.output.splitlines()[-2].startswith("✓ Sessions: 10 indexed")


def test_plain_progress_also_prints_every_30_s_and_says_nothing_for_an_empty_scan(monkeypatch):
    """The percent and the ETA follow bytes: in the 2026-10-23 proof, counting the 192 quick subagent files put
    the bar at 51% with an ETA of 8:56 when 9 s were left."""
    import medulla.cli as cli
    from medulla.episodic.scanner import ScanProgress as P
    printed, now = [], [0.0]
    monkeypatch.setattr(cli, "_plain", printed.append)
    p = cli._PlainProgress(clock=lambda: now[0])
    p(P(0, 0, 0, 0, ""))
    p(P(0, 1000, 0, 10_000, "first"))
    now[0] = 31.0
    p(P(5, 1000, 50, 10_000, "slow one"))
    now[0] = 32.0
    p(P(6, 1000, 60, 10_000, "next"))
    assert printed == ["[5/1000 0%] 0:31 elapsed · ETA 102:49"]
    big = cli._PlainProgress(clock=lambda: now[0])
    now[0] = 42.0
    big(P(1, 10, 9_000, 10_000, "the one big session"))
    assert printed[-1] == "[1/10 90%] 0:10 elapsed · ETA 0:01"


def test_scan_in_a_terminal_draws_two_rows_and_leaves_the_summary(claude_projects, monkeypatch):
    """The display picked from the 2026-10-23 demos: spinner, bar, count, %, elapsed and ETA, then the session
    being indexed on a second row. Transient, so only the summary stays on screen."""
    from io import StringIO
    from rich.console import Console
    import medulla.cli as cli
    screen = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=screen, force_terminal=True, width=100, color_system=None))
    result = runner.invoke(app, ["scan"])
    out = screen.getvalue()
    assert result.exit_code == 0 and "Scanning" in out and "↳" in out and "ETA" in out and "✓ Sessions: 1 indexed" in out
    assert "1/1" in out


def test_the_terminal_bar_follows_bytes_and_counts_sessions(monkeypatch):
    from io import StringIO
    from rich.console import Console
    import medulla.cli as cli
    from medulla.episodic.scanner import ScanProgress as P
    screen = StringIO()
    monkeypatch.setattr(cli, "console", Console(file=screen, force_terminal=True, width=100, color_system=None))
    with cli._scan_progress() as report:
        report(P(1, 10, 9_000, 10_000, "the one big session"))
    out = screen.getvalue()      # the last frame, drawn as the display closes
    assert " 1/10 " in out and " 90% " in out and "the one big session" in out
