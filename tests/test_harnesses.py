"""Harness registry: Claude Code and Antigravity (gemini) share one interface."""
import json
import sqlite3
from pathlib import Path

import pytest

from medulla import harnesses
from medulla.harnesses import HookOutcome


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDULLA_DIR", str(tmp_path / "medulla"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    monkeypatch.setenv("ANTIGRAVITY_CONFIG_DIR", str(tmp_path / "agy"))
    for k in ("CLAUDE_CODE_SESSION_ID", "ANTIGRAVITY_CONVERSATION_ID", "ANTIGRAVITY_APP_DATA_DIR"):
        monkeypatch.delenv(k, raising=False)
    return tmp_path


def test_registry_lists_claude_and_gemini_and_rejects_unknown(isolated):
    assert set(harnesses.HARNESSES) == {"claude", "gemini"}
    assert harnesses.get("claude").settings_path() == isolated / "claude" / "settings.json"
    assert harnesses.get("gemini").settings_path() == isolated / "agy" / "settings.json"
    assert harnesses.get("claude").default_threshold_pct == 93.0
    assert harnesses.get("gemini").default_threshold_pct == 45.0
    with pytest.raises(ValueError, match="unknown harness"):
        harnesses.get("nonexistent")


def test_detect_event_and_active_harness(isolated, monkeypatch):
    assert harnesses.detect_event({"hook_event_name": "Stop"}).name == "claude"
    assert harnesses.detect_event({"hookEventName": "Stop"}).name == "gemini"
    assert harnesses.detect_event({}) is None

    assert harnesses.active_harness({}).name == "claude"
    assert harnesses.active_harness({"ANTIGRAVITY_CONVERSATION_ID": "c1"}).name == "gemini"
    assert harnesses.active_harness({"CLAUDE_CODE_SESSION_ID": "s1", "ANTIGRAVITY_CONVERSATION_ID": "c1"}).name == "claude"


def test_gemini_title_falls_back_from_ctx_json_to_summaries_db(isolated):
    g = harnesses.get("gemini")
    assert g.title_of("c1", None) is None

    # Fallback to conversation_summaries.db when ctx.json is absent
    db = isolated / "agy" / "conversation_summaries.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE conversation_summaries (conversation_id TEXT PRIMARY KEY, title TEXT)")
    conn.execute("INSERT INTO conversation_summaries VALUES ('c1', 'introToNeon'), ('c2', '')")
    conn.commit()
    conn.close()
    assert g.title_of("c1", None) == "introToNeon"
    assert g.title_of("c2", None) is None
    assert g.title_of("missing", None) is None

    # Corrupt DB fails open to None
    db.write_text("not sqlite")
    assert g.title_of("c1", None) is None

    # ctx.json wins over DB
    sdir = isolated / "medulla" / "state"
    sdir.mkdir(parents=True)
    (sdir / "c1.ctx.json").write_text(json.dumps({"name": "fromStatusline"}))
    assert g.title_of("c1", None) == "fromStatusline"
    (sdir / "c3.ctx.json").write_text(json.dumps({"name": None}))
    assert g.title_of("c3", None) is None


def test_claude_and_gemini_hook_normalization_and_output_formatting(isolated):
    c, g = harnesses.get("claude"), harnesses.get("gemini")

    # Unknown events return None
    assert c.normalize_event({"hook_event_name": "SessionStart"}) is None
    assert g.normalize_event({"hookEventName": "SessionStart"}) is None

    # Claude formatting
    assert c.format_output("tool", HookOutcome(system_message="x")) is None
    assert c.format_output("prompt", HookOutcome()) is None
    assert c.format_output("stop", HookOutcome()) is None
    assert c.format_output("stop", HookOutcome(decision="continue", reason="go")) == {"decision": "block", "reason": "go"}
    assert c.format_output("stop", HookOutcome(system_message="done")) == {"systemMessage": "done"}

    # Gemini formatting
    assert g.format_output("tool", None) == {}
    assert g.format_output("prompt", None) == {}
    assert g.format_output("prompt", HookOutcome()) == {}
    assert g.format_output("prompt", HookOutcome(additional_context="[context] 10%")) == {
        "injectSteps": [{"systemMessage": {"systemMessage": "[context] 10%"}}]
    }
    assert g.format_output("stop", HookOutcome(decision="continue", reason="nudge")) == {
        "decision": "continue", "reason": "nudge"
    }
    assert g.format_output("stop", HookOutcome(system_message="opened")) == {}

    # Entry usage & role dispatch across harnesses
    assert harnesses.entry_usage({"type": "PLANNER_RESPONSE", "input_tokens": 100, "cache_read_tokens": 50}) == 150
    assert harnesses.entry_usage({"type": "PLANNER_RESPONSE"}) is None
    assert harnesses.entry_usage({"type": "GENERIC"}) is None
    assert harnesses.entry_role({"type": "USER_INPUT"}) == "user"
    assert harnesses.entry_role({"type": "PLANNER_RESPONSE"}) == "assistant"
    assert harnesses.entry_role({"type": "GENERIC"}) is None


def test_last_gemini_reply_reads_tail_and_handles_missing_or_corrupt(isolated, monkeypatch):
    g = harnesses.get("gemini")
    assert g.normalize_event({"hookEventName": "Stop", "conversationId": "c1"}).last_message == ""
    assert g.normalize_event({"hookEventName": "Stop", "conversationId": "c1",
                              "transcriptPath": str(isolated / "missing.jsonl")}).last_message == ""

    p = isolated / "transcript.jsonl"
    p.write_text("\n".join([
        "x" * 64,
        "not json",
        json.dumps({"type": "PLANNER_RESPONSE", "content": "first"}),
        json.dumps({"type": "PLANNER_RESPONSE", "tool_calls": []}),
        json.dumps({"type": "PLANNER_RESPONSE", "content": "final answer"}),
    ]))
    monkeypatch.setattr(harnesses, "_TAIL_BYTES", len(p.read_bytes()) - 10)
    ev = g.normalize_event({"hookEventName": "Stop", "conversationId": "c1", "transcriptPath": str(p)})
    assert ev.last_message == "final answer"
