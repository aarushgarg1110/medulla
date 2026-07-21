"""Tests for tool_events backfill — extraction, storage, scan integration."""
import json
from pathlib import Path

import pytest

from medulla.episodic.parser import extract_tool_events, ToolEvent
from medulla.episodic.store import upsert_tool_events, search_events


def _rec(**kw) -> str:
    return json.dumps(kw)


def _write_session(path: Path, tool_use, result, *, cwd="/proj/a", is_error=False,
                   tool_use_result=None):
    lines = [
        _rec(type="user", sessionId="s1", cwd=cwd, timestamp="2026-01-01T10:00:00Z",
             message={"role": "user", "content": "do the thing"}),
        _rec(type="assistant", sessionId="s1", cwd=cwd, timestamp="2026-01-01T10:00:01Z",
             message={"role": "assistant", "content": [tool_use]}),
        _rec(type="user", sessionId="s1", cwd=cwd, timestamp="2026-01-01T10:00:02Z",
             toolUseResult=tool_use_result or {"stdout": "col1\n1\n2", "stderr": ""},
             message={"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": tool_use["id"],
                  "is_error": is_error, "content": result}]}),
    ]
    path.write_text("\n".join(lines))


def _bash(id_, command, description=""):
    return {"type": "tool_use", "id": id_, "name": "Bash",
            "input": {"command": command, "description": description}}


# ── V6 migration ────────────────────────────────────────────────────────────

def test_tool_events_has_is_error_column(db):
    cols = {r[1] for r in db.execute("PRAGMA table_info(tool_events)").fetchall()}
    assert "is_error" in cols


# ── extract_tool_events ──────────────────────────────────────────────────────

def test_extract_basic_bash_command(tmp_path):
    p = tmp_path / "s1.jsonl"
    _write_session(p, _bash("tu1", "duckdb -c \"SELECT * FROM 'x.csv'\"", "run sql"),
                   [{"type": "text", "text": "col1\n1\n2"}])
    events = extract_tool_events(p)
    assert len(events) == 1
    e = events[0]
    assert e.tool == "Bash"
    assert "duckdb" in e.command and "SELECT" in e.command
    assert e.description == "run sql"
    assert e.event_ts == "2026-01-01T10:00:01Z"   # timestamp of the tool_use
    assert e.project_dir == "/proj/a"
    assert e.is_error is False
    assert e.output_preview.startswith("col1")


def test_extract_captures_is_error(tmp_path):
    p = tmp_path / "s1.jsonl"
    _write_session(p, _bash("tu1", "python broken.py"),
                   [{"type": "text", "text": "Traceback..."}],
                   is_error=True, tool_use_result={"stdout": "", "stderr": "Traceback: boom"})
    events = extract_tool_events(p)
    assert events[0].is_error is True
    assert "boom" in events[0].output_preview


def test_extract_skips_trivial_cd_and_assignment(tmp_path):
    p = tmp_path / "s1.jsonl"
    _write_session(p, _bash("tu1", "cd /Users/agarg/code/mlops"),
                   [{"type": "text", "text": "ok"}])
    assert extract_tool_events(p) == []
    _write_session(p, _bash("tu2", "CSV=/tmp/x.csv"), [{"type": "text", "text": "ok"}])
    assert extract_tool_events(p) == []


def test_extract_keeps_multiline_even_if_starts_with_assignment(tmp_path):
    p = tmp_path / "s1.jsonl"
    cmd = "PW=$(cat .pw)\nduckdb -c \"SELECT count(*) FROM t\""
    _write_session(p, _bash("tu1", cmd), [{"type": "text", "text": "42"}])
    events = extract_tool_events(p)
    assert len(events) == 1 and "duckdb" in events[0].command


def test_extract_scrubs_secret_in_command(tmp_path):
    p = tmp_path / "s1.jsonl"
    _write_session(p, _bash("tu1", "psql 'postgresql://u:hunter2@h/db' -c 'SELECT 1'"),
                   [{"type": "text", "text": "1"}])
    e = extract_tool_events(p)[0]
    assert "hunter2" not in e.command


def test_extract_non_bash_tool_derives_command(tmp_path):
    p = tmp_path / "s1.jsonl"
    tu = {"type": "tool_use", "id": "tu1", "name": "Read",
          "input": {"file_path": "/proj/a/model.py"}}
    _write_session(p, tu, [{"type": "text", "text": "..."}])
    e = extract_tool_events(p)[0]
    assert e.tool == "Read" and "/proj/a/model.py" in e.command


def test_extract_non_dict_input(tmp_path):
    p = tmp_path / "s1.jsonl"
    tu = {"type": "tool_use", "id": "tu1", "name": "Weird", "input": "just-a-string"}
    _write_session(p, tu, [{"type": "text", "text": "ok"}])
    e = extract_tool_events(p)[0]
    assert e.tool == "Weird" and "just-a-string" in e.command


def test_extract_unknown_input_keys_falls_back_to_json(tmp_path):
    p = tmp_path / "s1.jsonl"
    tu = {"type": "tool_use", "id": "tu1", "name": "Custom", "input": {"foo": "bar123"}}
    _write_session(p, tu, [{"type": "text", "text": "ok"}])
    e = extract_tool_events(p)[0]
    assert "bar123" in e.command   # json.dumps fallback


def test_extract_output_from_result_content_when_no_tooluseresult(tmp_path):
    p = tmp_path / "s1.jsonl"
    # a record with a tool_result block but NO top-level toolUseResult
    lines = [
        _rec(type="assistant", sessionId="s1", cwd="/proj/a", timestamp="2026-01-01T10:00:01Z",
             message={"role": "assistant", "content": [_bash("tu1", "python run.py")]}),
        _rec(type="user", sessionId="s1", cwd="/proj/a", timestamp="2026-01-01T10:00:02Z",
             message={"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "tu1", "is_error": False,
                  "content": [{"type": "text", "text": "output-from-content-block"}]}]}),
    ]
    p.write_text("\n".join(lines))
    e = extract_tool_events(p)[0]
    assert "output-from-content-block" in e.output_preview


def test_extract_ignores_subagent_files(tmp_path):
    d = tmp_path / "subagents"
    d.mkdir()
    p = d / "agent-abc.jsonl"
    _write_session(p, _bash("tu1", "duckdb SELECT"), [{"type": "text", "text": "x"}])
    assert extract_tool_events(p) == []


def test_extract_missing_file_returns_empty(tmp_path):
    assert extract_tool_events(tmp_path / "nope.jsonl") == []


# ── upsert_tool_events + search ───────────────────────────────────────────────

def _mk_event(hash_, command="duckdb -c \"SELECT 1\"", is_error=False):
    return ToolEvent(session_id="s1", project_dir="/proj/a", event_ts="2026-01-01T10:00:01Z",
                     tool="Bash", command=command, description="", output_preview="1",
                     is_error=is_error, event_hash=hash_)


def test_upsert_and_search(db):
    upsert_tool_events(db, "s1", [_mk_event("h1")])
    rows = search_events(db, "duckdb")
    assert len(rows) == 1
    assert rows[0]["command"].startswith("duckdb")
    assert rows[0]["is_error"] == 0


def test_upsert_dedups_by_hash(db):
    upsert_tool_events(db, "s1", [_mk_event("h1"), _mk_event("h1", command="duckdb dup")])
    assert db.execute("SELECT COUNT(*) FROM tool_events WHERE session_id='s1'").fetchone()[0] == 1


def test_upsert_replaces_previous_scan(db):
    upsert_tool_events(db, "s1", [_mk_event("h1")])
    upsert_tool_events(db, "s1", [_mk_event("h2", command="python train.py")])
    rows = db.execute("SELECT command FROM tool_events WHERE session_id='s1'").fetchall()
    assert len(rows) == 1 and rows[0]["command"] == "python train.py"
    # FTS kept in sync — old command no longer matches
    assert search_events(db, "duckdb") == []


def test_search_events_flags_error(db):
    upsert_tool_events(db, "s1", [_mk_event("h1", command="python boom.py", is_error=True)])
    assert search_events(db, "boom")[0]["is_error"] == 1


# ── scanner integration ───────────────────────────────────────────────────────

def test_scan_populates_tool_events(db, tmp_path, monkeypatch):
    import medulla.episodic.scanner as scanner
    monkeypatch.setattr(scanner, "_embed_session_chunks", lambda conn, sid: None)
    p = tmp_path / "s1.jsonl"
    _write_session(p, _bash("tu1", "duckdb -c \"SELECT foo FROM bar\"", "q"),
                   [{"type": "text", "text": "ok"}])
    assert scanner._process_session(db, p, force=True) == "indexed"
    rows = search_events(db, "duckdb")
    assert len(rows) == 1 and "bar" in rows[0]["command"]
