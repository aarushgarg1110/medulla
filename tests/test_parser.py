"""Tests for medulla.episodic.parser — real JSONL files, no mocks."""
import json
from pathlib import Path

import pytest

from medulla.episodic.parser import (
    MAX_FIRST_MSG,
    ParsedSession,
    _extract_assistant_text,
    _extract_paths,
    _extract_user_text,
    _read_json,
    is_subagent_file,
    parse_agent_session,
    parse_session,
)
from tests.conftest import claude_assistant, claude_user, make_claude_jsonl


# ── is_subagent_file ──────────────────────────────────────────────────────────

def test_is_subagent_file_detects_subagent(tmp_path):
    p = tmp_path / "proj" / "session-abc" / "subagents" / "agent-xyz.jsonl"
    p.parent.mkdir(parents=True)
    p.touch()
    assert is_subagent_file(p) is True


def test_is_subagent_file_normal_session(tmp_path):
    p = tmp_path / "proj" / "session-abc.jsonl"
    p.parent.mkdir(parents=True)
    p.touch()
    assert is_subagent_file(p) is False


# ── parse_session — happy paths ───────────────────────────────────────────────

def test_parse_basic_session(tmp_path):
    content = make_claude_jsonl([
        claude_user("hello world"),
        claude_assistant(["Bash"]),
        claude_user("next question"),
    ])
    path = tmp_path / "abc123.jsonl"
    path.write_text(content)

    result = parse_session(path)

    assert result is not None
    assert result.session_id == "test-session-id"
    assert result.source == "claude"
    assert result.project_dir == "/home/user/proj"
    assert result.git_branch == "main"
    assert result.model == "claude-sonnet-4-6"
    assert result.turn_count == 1
    assert result.tool_call_count == 1
    assert "Bash" in result.tool_names
    assert result.first_message == "hello world"
    assert "hello world" in result.all_user_text
    assert "next question" in result.all_user_text
    assert len(result.messages) >= 2  # user + assistant messages interleaved


def test_parse_session_uses_stem_as_fallback_id(tmp_path):
    content = json.dumps({"type": "user", "message": {"role": "user", "content": "hi"}})
    path = tmp_path / "fallback-id.jsonl"
    path.write_text(content)
    result = parse_session(path)
    assert result is not None
    assert result.session_id == "fallback-id"


def test_parse_session_first_message_capped(tmp_path):
    long_msg = "x" * (MAX_FIRST_MSG + 100)
    content = make_claude_jsonl([claude_user(long_msg)])
    path = tmp_path / "sess.jsonl"
    path.write_text(content)
    result = parse_session(path)
    assert result is not None
    assert len(result.first_message) == MAX_FIRST_MSG


def test_parse_session_all_user_text_has_no_cap(tmp_path):
    # 50 messages of 200 chars each = 10KB total, well over the old 8KB cap
    messages = [claude_user("A" * 200, ts=f"2026-01-01T{i:02d}:00:00Z") for i in range(50)]
    content = make_claude_jsonl(messages)
    path = tmp_path / "big.jsonl"
    path.write_text(content)
    result = parse_session(path)
    assert result is not None
    assert len(result.all_user_text) > 8000, "all_user_text must not be capped at 8KB"
    assert len(result.messages) >= 50


def test_parse_session_extracts_file_paths(tmp_path):
    tool_msg = {
        "type": "assistant",
        "timestamp": "2026-01-01T10:01:00Z",
        "message": {
            "role": "assistant",
            "content": [{
                "type": "tool_use",
                "name": "Read",
                "input": {"file_path": "/Users/agarg/code/medulla/foo.py"},
            }],
        },
    }
    content = make_claude_jsonl([claude_user("read it"), tool_msg])
    path = tmp_path / "sess.jsonl"
    path.write_text(content)
    result = parse_session(path)
    assert result is not None
    assert any("foo.py" in f for f in result.files)


def test_parse_session_multiple_tools(tmp_path):
    content = make_claude_jsonl([
        claude_user("do stuff"),
        claude_assistant(["Bash", "Read", "Write"]),
        claude_user("more"),
        claude_assistant(["Bash", "Edit"]),
    ])
    path = tmp_path / "sess.jsonl"
    path.write_text(content)
    result = parse_session(path)
    assert result is not None
    assert result.tool_call_count == 5
    assert set(result.tool_names) == {"Bash", "Read", "Write", "Edit"}


def test_parse_session_timestamps(tmp_path):
    content = make_claude_jsonl([
        claude_user("first", ts="2026-01-01T09:00:00Z"),
        claude_user("last", ts="2026-01-01T11:00:00Z"),
    ])
    path = tmp_path / "sess.jsonl"
    path.write_text(content)
    result = parse_session(path)
    assert result is not None
    assert result.started_at == "2026-01-01T09:00:00Z"
    assert result.ended_at == "2026-01-01T11:00:00Z"


# ── parse_session — edge/error cases ──────────────────────────────────────────

def test_parse_session_returns_none_for_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("")
    assert parse_session(path) is None


def test_parse_session_returns_none_for_no_user_messages(tmp_path):
    content = make_claude_jsonl([claude_assistant(["Bash"])])
    path = tmp_path / "no_user.jsonl"
    path.write_text(content)
    assert parse_session(path) is None


def test_parse_session_returns_none_for_subagent_file(tmp_path):
    subdir = tmp_path / "proj" / "sess" / "subagents"
    subdir.mkdir(parents=True)
    path = subdir / "agent-xyz.jsonl"
    path.write_text(make_claude_jsonl([claude_user("hi")]))
    assert parse_session(path) is None


def test_parse_session_returns_none_for_unreadable_file(tmp_path):
    path = tmp_path / "missing.jsonl"
    # Don't create the file — it won't exist
    result = parse_session(path)
    assert result is None


def test_parse_session_skips_bad_json_lines(tmp_path):
    content = "not json\n" + json.dumps(claude_user("valid message")) + "\nmore bad json{"
    path = tmp_path / "mixed.jsonl"
    path.write_text(content)
    result = parse_session(path)
    assert result is not None
    assert result.first_message == "valid message"


def test_parse_session_human_role_alias(tmp_path):
    """'human' role should be treated same as 'user'."""
    line = json.dumps({
        "sessionId": "sess-1",
        "timestamp": "2026-01-01T10:00:00Z",
        "type": "human",
        "message": {"role": "human", "content": "hello from human"},
    })
    path = tmp_path / "human.jsonl"
    path.write_text(line)
    result = parse_session(path)
    assert result is not None
    assert result.first_message == "hello from human"


def test_parse_session_indexes_assistant_text(tmp_path):
    """Assistant text blocks must appear in all_user_text and messages — the core Sprint 1.5 fix."""
    lines = [
        json.dumps(claude_user("what compounds were suspicious in Salacia?")),
        json.dumps({
            "sessionId": "test-session-id",
            "timestamp": "2026-01-01T10:01:00Z",
            "type": "assistant",
            "model": "claude-sonnet-4-6",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "NDI-218229 has delta logD of +6.11, four sigma above batch mean. Almost certainly a measurement error from the Syngene batch."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "echo hi"}},
                ],
            },
        }),
    ]
    path = tmp_path / "salacia.jsonl"
    path.write_text("\n".join(lines))

    result = parse_session(path)
    assert result is not None
    assert "NDI-218229" in result.all_user_text
    assert "four sigma" in result.all_user_text
    assert "Bash" not in result.all_user_text  # tool_use excluded
    assert any("NDI-218229" in m for m in result.messages)


def test_parse_session_assistant_only_no_result(tmp_path):
    """Session with only assistant messages (no user turns) should still return None."""
    line = json.dumps({
        "sessionId": "asst-only",
        "timestamp": "2026-01-01T10:00:00Z",
        "type": "assistant",
        "message": {"role": "assistant", "content": [{"type": "text", "text": "I am Claude."}]},
    })
    path = tmp_path / "asst-only.jsonl"
    path.write_text(line)
    # No user messages means we can't establish first_message or intent
    assert parse_session(path) is None


def test_parse_session_content_as_list(tmp_path):
    """User message content can be a list of {type, text} blocks."""
    line = json.dumps({
        "sessionId": "sess-list",
        "timestamp": "2026-01-01T10:00:00Z",
        "type": "user",
        "message": {
            "role": "user",
            "content": [
                {"type": "text", "text": "block one"},
                {"type": "tool_result", "content": "should be ignored"},
                {"type": "text", "text": "block two"},
            ],
        },
    })
    path = tmp_path / "list_content.jsonl"
    path.write_text(line)
    result = parse_session(path)
    assert result is not None
    assert "block one" in result.all_user_text
    assert "block two" in result.all_user_text
    assert "should be ignored" not in result.all_user_text


# ── parse_agent_session ───────────────────────────────────────────────────────

def test_parse_agent_session_basic(tmp_path):
    subdir = tmp_path / "proj" / "sess" / "subagents"
    subdir.mkdir(parents=True)
    path = subdir / "agent-myagent.jsonl"
    content = make_claude_jsonl([
        {
            "sessionId": "parent-session-id",
            "cwd": "/home/user/proj/src",
            "timestamp": "2026-01-01T10:00:00Z",
            "type": "user",
            "message": {"role": "user", "content": "agent task"},
        },
        claude_assistant(["Bash"], session_id="parent-session-id"),
    ])
    path.write_text(content)

    result = parse_agent_session(path)
    assert result is not None
    assert result.agent_id == "myagent"
    assert result.parent_session_id == "parent-session-id"
    assert result.first_message == "agent task"
    assert result.turn_count == 1


def test_parse_agent_session_returns_none_for_non_subagent(tmp_path):
    path = tmp_path / "regular.jsonl"
    path.write_text(make_claude_jsonl([claude_user("hi")]))
    assert parse_agent_session(path) is None


def test_parse_agent_session_returns_none_for_empty(tmp_path):
    subdir = tmp_path / "subagents"
    subdir.mkdir()
    path = subdir / "agent-empty.jsonl"
    path.write_text("")
    assert parse_agent_session(path) is None


def test_parse_agent_session_returns_none_no_user_messages(tmp_path):
    subdir = tmp_path / "subagents"
    subdir.mkdir()
    path = subdir / "agent-nouser.jsonl"
    path.write_text(make_claude_jsonl([claude_assistant(["Bash"])]))
    assert parse_agent_session(path) is None


# ── _extract_user_text ────────────────────────────────────────────────────────

def test_extract_user_text_string_content():
    assert _extract_user_text({"content": "hello"}) == "hello"


def test_extract_user_text_list_content():
    msg = {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}
    assert _extract_user_text(msg) == "a b"


def test_extract_user_text_empty():
    assert _extract_user_text({}) == ""


def test_extract_user_text_skips_tool_result():
    msg = {"content": [
        {"type": "text", "text": "keep"},
        {"type": "tool_result", "content": "drop"},
    ]}
    result = _extract_user_text(msg)
    assert "keep" in result
    assert "drop" not in result


def test_extract_user_text_non_string_content():
    assert _extract_user_text({"content": 42}) == ""


# ── _extract_assistant_text ───────────────────────────────────────────────────

def test_extract_assistant_text_plain_string():
    assert _extract_assistant_text({"content": "The answer is 42."}) == "The answer is 42."


def test_extract_assistant_text_extracts_text_blocks():
    msg = {"content": [
        {"type": "text", "text": "Here is my analysis:"},
        {"type": "tool_use", "name": "Bash", "input": {}},
        {"type": "text", "text": "The result confirms NDI-218229 is an outlier."},
    ]}
    result = _extract_assistant_text(msg)
    assert "Here is my analysis" in result
    assert "NDI-218229 is an outlier" in result
    assert "Bash" not in result  # tool_use excluded


def test_extract_assistant_text_skips_tool_use():
    msg = {"content": [
        {"type": "tool_use", "name": "Read", "input": {"path": "/foo"}},
    ]}
    assert _extract_assistant_text(msg) == ""


def test_extract_assistant_text_empty():
    assert _extract_assistant_text({}) == ""


def test_extract_assistant_text_bare_string_items():
    msg = {"content": ["bare string analysis"]}
    result = _extract_assistant_text(msg)
    assert "bare string analysis" in result


# ── _read_json ─────────────────────────────────────────────────────────────────

def test_read_json_valid():
    assert _read_json('{"key": "val"}') == {"key": "val"}


def test_read_json_invalid():
    assert _read_json("not json") is None


def test_read_json_non_dict():
    assert _read_json("[1,2,3]") is None


# ── _extract_paths ────────────────────────────────────────────────────────────

def test_extract_paths_from_string():
    acc: set[str] = set()
    _extract_paths("/Users/agarg/code/foo.py", acc)
    assert any("foo.py" in p for p in acc)


def test_extract_paths_from_dict():
    acc: set[str] = set()
    _extract_paths({"file_path": "/tmp/bar.py"}, acc)
    assert any("bar.py" in p for p in acc)


def test_extract_paths_from_nested():
    acc: set[str] = set()
    _extract_paths({"a": {"b": ["/home/user/script.sh"]}}, acc)
    assert any("script.sh" in p for p in acc)


def test_extract_paths_ignores_proc():
    acc: set[str] = set()
    _extract_paths("/proc/12345/status", acc)
    assert not any("/proc" in p for p in acc)


def test_extract_paths_ignores_short_paths():
    acc: set[str] = set()
    _extract_paths("/ab", acc)
    assert len(acc) == 0
