"""Antigravity / Gemini episodic source registry, parser, tool events, scanner, and resume tests."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import pytest

from medulla.episodic import forget as fg
from medulla.episodic import parser as ps
from medulla.episodic import resume as rs
from medulla.episodic import scanner as sc
from medulla.episodic.store import list_sessions, search_events


def _make_agy_app(tmp_path: Path, rows: list[tuple] | None = None) -> Path:
    """Create a fake ~/.gemini/antigravity-cli directory with optional conversation_summaries.db."""
    app = tmp_path / "antigravity-cli"
    (app / "brain").mkdir(parents=True, exist_ok=True)
    if rows is not None:
        db = app / "conversation_summaries.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE conversation_summaries ("
            "conversation_id TEXT PRIMARY KEY, title TEXT, workspace_uris TEXT, "
            "parent_conversation_id TEXT DEFAULT '', nesting_depth INTEGER DEFAULT 0)"
        )
        conn.executemany("INSERT INTO conversation_summaries VALUES (?,?,?,?,?)", rows)
        conn.commit()
        conn.close()
    return app


def _write_gemini(app: Path, cid: str, steps: list[dict], *, age_s: float = 120) -> Path:
    p = app / "brain" / cid / ".system_generated" / "logs" / "transcript.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(s) for s in steps))
    # Also drop a sibling transcript_full.jsonl to prove discovery skips it
    (p.parent / "transcript_full.jsonl").write_text("{}\n")
    t = time.time() - age_s
    os.utime(p, (t, t))
    return p


def _user(idx: int, text: str, ts: str = "2026-11-19T20:00:00Z", wrapped: bool = True) -> dict:
    body = (
        f"<USER_REQUEST>\n{text}\n</USER_REQUEST>\n"
        f"<ADDITIONAL_METADATA>\nThe current local time is: {ts}\n</ADDITIONAL_METADATA>\n"
        f"<USER_SETTINGS_CHANGE>\nModel changed.\n</USER_SETTINGS_CHANGE>"
    ) if wrapped else f"{text}\n<ADDITIONAL_METADATA>\ntime\n</ADDITIONAL_METADATA>"
    return {"step_index": idx, "source": "USER_EXPLICIT", "type": "USER_INPUT",
            "status": "DONE", "created_at": ts, "content": body}


def _planner(idx: int, content: str = "", tool_calls: list[dict] | None = None,
             thinking: str = "secret inner thought", model: str = "gemini-4-neon-high",
             ts: str = "2026-11-19T20:01:00Z") -> dict:
    d: dict = {"step_index": idx, "source": "MODEL", "type": "PLANNER_RESPONSE",
               "status": "DONE", "created_at": ts, "model": model, "thinking": thinking}
    if content:
        d["content"] = content
    if tool_calls:
        d["tool_calls"] = tool_calls
    return d


def _generic(idx: int, content: str = "", status: str = "DONE", error: str | None = None,
             ts: str = "2026-11-19T20:01:05Z") -> dict:
    d: dict = {"step_index": idx, "source": "MODEL", "type": "GENERIC",
               "status": status, "created_at": ts, "content": content}
    if error is not None:
        d["error"] = error
    return d


def _sysmsg(idx: int, content: str, ts: str = "2026-11-19T20:02:00Z") -> dict:
    return {"step_index": idx, "source": "SYSTEM", "type": "SYSTEM_MESSAGE",
            "status": "DONE", "created_at": ts, "content": content}


def test_parse_gemini_extracts_clean_user_requests_assistant_replies_and_db_metadata(tmp_path):
    app = _make_agy_app(tmp_path, [
        ("conv-1111", "introToNeon", '["file:///Users/agarg/my%20repo"]', "", 0),
    ])
    steps = [
        _user(0, "compare neon high vs opus 5.5"),
        _planner(1, tool_calls=[
            {"name": "read_file", "args": {"Path": json.dumps("/Users/agarg/code/app.py")}},
            {"name": "call_mcp_tool", "args": {"ServerName": json.dumps("aarush-ops"),
                                               "ToolName": json.dumps("capabilities"), "Arguments": "{}"}},
            {"name": "call_mcp_tool", "args": {"ToolName": "standalone_tool"}},
        ]),
        _generic(2, "file contents"),
        _generic(3, "mcp output"),
        _generic(4, "standalone output"),
        _sysmsg(5, "Reminder: work efficiently."),
        _planner(6, content="Neon High is faster end-to-end."),
        _user(7, "unwrapped follow-up", wrapped=False, ts="2026-11-19T20:05:00Z"),
        _planner(8, content="Confirmed.", ts="2026-11-19T20:06:00Z"),
    ]
    p = _write_gemini(app, "conv-1111", steps)
    s = ps.parse_session(p)

    assert s is not None
    assert s.source == "gemini"
    assert s.session_id == "conv-1111"
    assert s.slug == "introToNeon"
    assert s.project_dir == "/Users/agarg/my repo"
    assert s.model == "gemini-4-neon-high"
    assert s.started_at == "2026-11-19T20:00:00Z" and s.ended_at == "2026-11-19T20:06:00Z"
    assert s.turn_count == 3 and s.tool_call_count == 3
    assert s.tool_names == ["mcp:aarush-ops/capabilities", "read_file", "standalone_tool"]
    assert "/Users/agarg/code/app.py" in s.files
    assert s.first_message == "compare neon high vs opus 5.5"
    assert "ADDITIONAL_METADATA" not in s.all_user_text
    assert "USER_SETTINGS_CHANGE" not in s.all_user_text
    assert "secret inner thought" not in s.all_user_text
    assert "Reminder: work efficiently" not in s.all_user_text
    assert "Neon High is faster end-to-end." in s.all_user_text
    assert "unwrapped follow-up" in s.all_user_text

    # Missing/corrupt DB and standalone (non-brain) path fail open cleanly
    (app / "conversation_summaries.db").unlink()
    assert ps.gemini_summary_meta(p, "conv-1111") == {}
    (app / "conversation_summaries.db").write_text("not sqlite")
    assert ps.gemini_summary_meta(p, "conv-1111") == {}
    standalone = tmp_path / "standalone.jsonl"
    standalone.write_text(
        "\n"
        + json.dumps({"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "content": ""}) + "\n"
        + json.dumps(_user(1, "hi")) + "\nnot json\n"
        + json.dumps({"step_index": 2, "source": "MODEL", "type": "PLANNER_RESPONSE",
                      "content": "yo", "tool_calls": ["bad-tc", {"name": ""}]})
    )
    s2 = ps.parse_session(standalone)
    assert s2 is not None and s2.source == "gemini" and s2.session_id == "standalone" and s2.project_dir is None


def test_parse_gemini_skips_stubs_and_parses_subagent_sessions(tmp_path):
    app = _make_agy_app(tmp_path, [
        ("sub-1", "research helper", '["file:///Users/agarg"]', "parent-1", 1),
        ("bad-uris", "", "not-json-uris", "", 0),
        ("empty-uris", "untitled", "[]", "", 0),
    ])
    # Stub: user only, no PLANNER_RESPONSE -> None
    stub = _write_gemini(app, "stub-1", [_user(0, "hello")])
    assert ps.parse_session(stub) is None
    assert ps.parse_agent_session(stub) is None

    # Empty user text / missing reply -> None
    empty_u = _write_gemini(app, "stub-2", [_user(0, ""), _planner(1, content="hi")])
    assert ps.parse_session(empty_u) is None

    # Bad or empty workspace_uris fails open to project_dir=None
    assert ps.gemini_summary_meta(
        app / "brain" / "bad-uris" / ".system_generated" / "logs" / "transcript.jsonl", "bad-uris"
    )["project_dir"] is None
    assert ps.gemini_summary_meta(
        app / "brain" / "empty-uris" / ".system_generated" / "logs" / "transcript.jsonl", "empty-uris"
    )["project_dir"] is None

    # Nested subagent parsed via parse_agent_session
    sub = _write_gemini(app, "sub-1", [_user(0, "find references"), _planner(1, content="found 3")])
    ag = ps.parse_agent_session(sub)
    assert ag is not None
    assert ag.agent_id == "sub-1" and ag.parent_session_id == "parent-1"
    assert ag.agent_slug == "research helper" and ag.project_dir == "/Users/agarg"
    assert ag.first_message == "find references" and "found 3" in ag.all_user_text


def test_extract_gemini_tool_events_handles_unquoting_errors_bg_tasks_and_canceled_batches(tmp_path):
    app = _make_agy_app(tmp_path, [("conv-2", "tools", '["file:///Users/agarg"]', "", 0)])
    steps = [
        _user(0, "run tests"),
        # 1. Successful run_command + non-zero exit (status=DONE!) + trivial cd (skipped) + non-string raw arg
        _planner(1, tool_calls=[
            {"name": "run_command", "args": {"CommandLine": json.dumps("uv run pytest"),
                                             "toolSummary": json.dumps("Run pytest")}},
            {"name": "run_command", "args": {"CommandLine": json.dumps("uv run ruff check")}},
            {"name": "run_command", "args": {"CommandLine": json.dumps("cd /tmp")}},
            {"name": "read_file", "args": {"Path": "unquoted-plain-string", "StartLine": 1}},
        ]),
        _generic(2, "Command exited with code 0.\n10 passed"),
        _generic(3, "Command exited with code 1.\nE501 line too long"),
        _generic(4, "Command exited with code 0."),
        _generic(5, "file body"),
        # 2. Background task failing later via SYSTEM_MESSAGE + canceled background task
        _planner(6, tool_calls=[
            {"name": "run_command", "args": {"CommandLine": json.dumps("uv run pytest -q")}},
            {"name": "run_command", "args": {"CommandLine": json.dumps("uv run long-job")}},
        ]),
        _generic(7, "Tool is running as a background task with task id: conv-2/task-7", status="RUNNING"),
        _generic(8, "Tool is running as a background task with task id: conv-2/task-8", status="RUNNING"),
        _sysmsg(9, 'Task id "conv-2/task-7" finished with result:\nCommand exited with code 1.'),
        _sysmsg(10, 'Task id "conv-2/task-8" was canceled with result:\nTool execution was canceled'),
        _sysmsg(11, "Unrelated system reminder"),
        # 3. Parallel batch where call #1 is denied by user and calls #2-#3 have NO GENERIC steps
        _planner(12, tool_calls=[
            {"name": "run_command", "args": {"CommandLine": json.dumps("git commit -m test")}},
            {"name": "write_file", "args": {"Path": json.dumps("/tmp/a.txt")}},
        ]),
        _generic(13, "Encountered error", status="ERROR", error="permission check failed: user denied permission"),
        _user(14, "wait review first"),
        # 4. Real harness ERROR (file does not exist)
        _planner(15, tool_calls=[
            {"name": "read_file", "args": {"Path": json.dumps("/tmp/missing.py")}},
        ]),
        _generic(16, "", status="ERROR", error="file does not exist"),
        # 5. Edge cases: non-dict args & non-dict tc & empty tool_calls
        _planner(17, content="done"),
        {"step_index": 18, "type": "PLANNER_RESPONSE", "tool_calls": ["bad-tc", {"name": "noop", "args": "not-dict"}]},
        _generic(19, "ok"),
        _generic(20, "ok"),
    ]
    p = _write_gemini(app, "conv-2", steps)
    events = ps.extract_tool_events(p)

    by_cmd = {e.command: e for e in events}
    assert "cd /tmp" not in by_cmd                                    # trivial skipped
    assert by_cmd["uv run pytest"].is_error is False and by_cmd["uv run pytest"].description == "Run pytest"
    assert by_cmd["uv run ruff check"].is_error is True               # DONE + exit code 1
    assert by_cmd["read_file unquoted-plain-string"].is_error is False
    assert by_cmd["uv run pytest -q"].is_error is True and by_cmd["uv run pytest -q"].interrupted is False
    assert by_cmd["uv run long-job"].is_error is False and by_cmd["uv run long-job"].interrupted is True
    assert by_cmd["git commit -m test"].is_error is False and by_cmd["git commit -m test"].interrupted is True
    assert "write_file /tmp/a.txt" not in by_cmd                      # unpaired after user denial, not mis-paired with USER_INPUT
    assert by_cmd["read_file /tmp/missing.py"].is_error is True and by_cmd["read_file /tmp/missing.py"].interrupted is False
    assert all(e.session_id == "conv-2" and e.project_dir == "/Users/agarg" for e in events)


def test_scanner_discovers_gemini_by_uuid_routes_subagents_and_respects_mtime_and_forget(db, tmp_path, monkeypatch):
    app = _make_agy_app(tmp_path, [
        ("main-1", "introToNeon", '["file:///Users/agarg"]', "", 0),
        ("sub-1", "helper", '["file:///Users/agarg"]', "main-1", 1),
    ])
    _write_gemini(app, "main-1", [
        _user(0, "check neon benchmarks"),
        _planner(1, content="Neon scores 65.2% on Terminal-Bench.",
                 tool_calls=[{"name": "run_command", "args": {"CommandLine": json.dumps("uv run pytest")}}]),
        _generic(2, "Command exited with code 0."),
    ])
    _write_gemini(app, "sub-1", [_user(0, "subagent work"), _planner(1, content="subagent answer")])
    monkeypatch.setattr(sc, "GEMINI_BRAIN_DIR", app / "brain")

    # First scan indexes 1 session + 1 subagent (and skips transcript_full.jsonl)
    c1 = sc.scan(db, source="gemini")
    assert c1["indexed"] == 1 and c1["agents_indexed"] == 1 and c1["errors"] == 0
    rows = list_sessions(db)
    assert len(rows) == 1 and rows[0]["session_id"] == "main-1" and rows[0]["slug"] == "introToNeon"
    assert rows[0]["source"] == "gemini"
    assert len(search_events(db, "pytest")) == 1

    # Second scan skips unchanged by mtime (using UUID, not path.stem == "transcript"!)
    c2 = sc.scan(db, source="gemini")
    assert c2["indexed"] == 0 and c2["skipped"] == 1 and c2["agents_skipped"] == 1

    # Forget removes main-1 from index and skips both main-1 and its subagent on rescan
    t = fg.find(db, "introToNeon")
    assert fg.transcript(t) == []                                     # delete_transcript stays Claude-only
    fg.forget(db, t)
    c3 = sc.scan(db, force=True, source="gemini")
    assert c3["forgotten"] == 1 and c3["indexed"] == 0 and c3["agents_indexed"] == 0


def test_stale_sessions_and_resume_index_live_gemini_session_by_title_and_id_prefix(db, tmp_path, monkeypatch):
    app = _make_agy_app(tmp_path, [
        ("709bf40f-1111", "introToNeon", '["file:///Users/agarg/medulla"]', "", 0),
    ])
    report = (
        "# Handoff\n\n"
        "**1. Overarching goal.** Add Antigravity support to medulla.\n\n"
        "**4. Immediate next action.** Run coverage gate and open PR."
    )
    _write_gemini(app, "709bf40f-1111", [
        _user(0, "build agy support"),
        _planner(1, content=report),
    ], age_s=120)
    monkeypatch.setattr(sc, "GEMINI_BRAIN_DIR", app / "brain")

    # Stale note sees the unindexed Gemini session by title
    note = rs.stale_note(db, "Run medulla scan.")
    assert "introToNeon" in note and "1 transcript changed" in note

    # Resume by title indexes on the spot and returns the handoff report
    out = rs.resume(db, "introToNeon")
    assert "Resume: introToNeon  (709bf40f, medulla" in out
    assert "indexed now" in out
    assert "Run coverage gate and open PR." in out
    assert rs.stale_note(db, "Run medulla scan.") == ""

    # Resume by UUID prefix works too
    assert "Resume: introToNeon" in rs.resume(db, "709bf40f")


def test_source_registry_dispatches_claude_kiro_gemini_and_rejects_unknown(tmp_path, monkeypatch):
    assert set(sc.SOURCES) == {"claude", "kiro", "gemini"}
    assert sc.get_source("gemini").name == "gemini"
    with pytest.raises(ValueError, match="unknown source"):
        sc.get_source("bogus")
    with pytest.raises(ValueError, match="unknown source"):
        sc._discover_files("bogus")

    # Corrupt summaries DB in _gemini_subagent_ids fails open to empty set
    app = _make_agy_app(tmp_path)
    (app / "conversation_summaries.db").write_text("corrupt")
    monkeypatch.setattr(sc, "GEMINI_BRAIN_DIR", app / "brain")
    assert sc._gemini_subagent_ids() == set()
    # Missing brain dir returns empty discovery
    monkeypatch.setattr(sc, "GEMINI_BRAIN_DIR", tmp_path / "missing-brain")
    assert sc._discover_gemini() == ([], [])
    # _has_user_message routes via source_for
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"type": "user"}) + "\n")
    assert sc._has_user_message(p) is True
