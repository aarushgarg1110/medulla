"""autohandoff: when it asks for the handoff, when it opens the next part, how it sizes the window,
and that it does nothing at all until the user turns it on."""
import io
import json
import os
import subprocess
from pathlib import Path

import pytest

from medulla import autohandoff as ah

ROOT = Path(__file__).resolve().parents[1]
REPORT = "## 4. Immediate next action\nShip the PR."


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDULLA_DIR", str(tmp_path / "medulla"))
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    for k in ("MEDULLA_AUTOHANDOFF", "MEDULLA_AUTOHANDOFF_AT", "MEDULLA_CONTEXT_WINDOW",
              "CLAUDE_PLUGIN_OPTION_AUTOHANDOFF", "CLAUDE_PLUGIN_OPTION_AUTOHANDOFF_AT",
              "CLAUDE_PLUGIN_OPTION_CONTEXT_WINDOW"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(ah, "_sleep", lambda s: None)
    return tmp_path


def transcript(home, tokens, title="fooPart3", last="assistant", sidechain_tokens=None):
    rows = [{"type": "custom-title", "customTitle": title}] if title else []
    rows.append({"type": "user", "message": {"role": "user", "content": "go"}})
    if sidechain_tokens:
        rows.append({"type": "assistant", "isSidechain": True,
                     "message": {"usage": {"input_tokens": sidechain_tokens}}})
    if last == "assistant":
        rows.append({"type": "assistant", "message": {"usage": {
            "input_tokens": 10, "cache_creation_input_tokens": 0, "cache_read_input_tokens": tokens - 10}}})
    p = home / "t.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    return str(p)


def stop(home, tokens, **kw):
    return {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": transcript(home, tokens), **kw}


# ── sizing the window ─────────────────────────────────────────────────────────

def test_next_names():
    assert ah.next_names("fooPart3") == ("fooPart4", "fooPart3")
    assert ah.next_names("foo") == ("fooPart2", "foo")


def test_context_tokens_are_the_last_main_chain_request(home):
    assert ah.context_tokens(transcript(home, 120_000, sidechain_tokens=900_000)) == 120_000


def test_the_statusline_record_wins(home):
    (home / "medulla" / "state").mkdir(parents=True)
    (home / "medulla" / "state" / "s1.ctx.json").write_text(json.dumps({"size": 500_000}))
    assert ah.window_size("s1") == 500_000


def test_then_a_configured_size_then_the_settings_model(home, monkeypatch):
    assert ah.window_size("s1") == 200_000
    (home / "claude").mkdir()
    (home / "claude" / "settings.json").write_text(json.dumps({"model": "opus[1m]"}))
    assert ah.window_size("s1") == 1_000_000
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_CONTEXT_WINDOW", "400000")
    assert ah.window_size("s1") == 400_000


def test_more_than_200k_tokens_can_only_be_the_1m_window(home):
    assert ah.window_size("s1", tokens=250_000) == 1_000_000
    assert ah.window_size("s1", tokens=150_000) == 200_000


def test_context_line_says_the_handoff_is_automatic_near_the_threshold(home):
    assert ah.context_line("s1", transcript(home, 100_000)) == "[context] 50.0% of 200,000 tokens (100,000)"
    assert "autohandoff takes over at 93%" in ah.context_line("s1", transcript(home, 180_000))
    assert ah.context_line("s1", None) is None


# ── the Stop decision ─────────────────────────────────────────────────────────

def test_below_the_threshold_nothing_happens(home):
    assert ah.on_stop(stop(home, 100_000), spawn=None) is None


def test_at_the_threshold_it_asks_for_the_handoff_once(home):
    out = ah.on_stop(stop(home, 190_000), spawn=None)
    assert out["decision"] == "block" and "95%" in out["reason"]
    assert ah.on_stop(stop(home, 190_000, last_assistant_message="still working"), spawn=None) is None


def test_the_report_opens_the_next_part(home):
    ah._mark("s1", "", "hook")
    calls = []

    def spawn(prev, nxt, event):
        calls.append((prev, nxt))
        return True, "opened"
    out = ah.on_stop(stop(home, 190_000, last_assistant_message=REPORT), spawn)
    assert calls == [("fooPart3", "fooPart4")] and "fooPart4: opened" in out["systemMessage"]
    assert ah.on_stop(stop(home, 195_000), spawn) is None            # this session already handed off


def test_an_unnamed_session_is_told_instead_of_guessed(home):
    ah._mark("s1", "", "user")
    ev = {"hook_event_name": "Stop", "session_id": "s1", "transcript_path": transcript(home, 10, title=None),
          "last_assistant_message": REPORT}
    assert "no /rename title" in ah.on_stop(ev, spawn=None)["systemMessage"]


def test_a_failed_open_is_reported(home):
    ah._mark("s1", "", "user")
    out = ah.on_stop(stop(home, 10, last_assistant_message=REPORT), lambda p, n, e: (False, "no terminal"))
    assert "could not open fooPart4: no terminal" in out["systemMessage"]


def test_subagents_and_a_continued_stop_are_left_alone(home):
    assert ah.on_stop(stop(home, 199_000, agent_id="a1"), spawn=None) is None
    assert ah.on_stop(stop(home, 199_000, stop_hook_active=True), spawn=None) is None


def test_waiting_for_the_reply_gives_up_after_a_moment(home, monkeypatch):
    t = iter([0.0, 0.5, 2.0, 0.0])          # deadline, still waiting, past it; then the second call's deadline
    monkeypatch.setattr(ah, "_clock", lambda: next(t))
    assert ah.wait_for_reply(transcript(home, 10, last="user")) is False
    assert ah.wait_for_reply(transcript(home, 10)) is True


# ── how a handoff gets marked ─────────────────────────────────────────────────

def test_a_typed_handoff_marks_one_and_nonext_clears_it(home):
    ev = {"hook_event_name": "UserPromptSubmit", "session_id": "s1", "transcript_path": transcript(home, 50_000)}
    out = ah.on_prompt({**ev, "prompt": "/medulla:handoff focus on the tests"})
    assert (home / "medulla" / "state" / "s1.handoff").read_text() == "user"
    assert out["hookSpecificOutput"]["additionalContext"].startswith("[context] 25.0%")
    ah.on_prompt({**ev, "prompt": "/handoff nonext"})
    assert not (home / "medulla" / "state" / "s1.handoff").exists()


def test_a_handoff_claude_starts_through_the_skill_tool_is_marked(home):
    ah.on_skill({"tool_name": "Skill", "session_id": "s1", "tool_input": {"skill": "medulla:handoff"}})
    assert (home / "medulla" / "state" / "s1.handoff").read_text() == "skill"
    ah.on_skill({"tool_name": "Skill", "session_id": "s2", "tool_input": {"skill": "medulla:pickoff"}})
    assert not (home / "medulla" / "state" / "s2.handoff").exists()


def test_where_no_terminal_can_be_driven_the_command_is_printed(home, monkeypatch):
    monkeypatch.setenv("MEDULLA_TERMINAL", "print")
    ok, detail = ah.terminal_spawn("fooPart3", "fooPart4", {"cwd": "/work"})
    assert ok and detail == f"run in a new terminal: cd /work && {ah.launch_line('fooPart3', 'fooPart4')}"
    assert ah.launch_line("fooPart3", "fooPart4").startswith("claude -n fooPart4 '/medulla:pickoff fooPart3 ")


# ── off until turned on ───────────────────────────────────────────────────────

def _main(event):
    out = io.StringIO()
    ah.main(stdin=io.StringIO(json.dumps(event)), stdout=out)
    return out.getvalue()


def test_off_by_default_even_at_the_threshold(home):
    assert _main(stop(home, 199_000)) == ""


def test_turned_on_it_routes_each_event(home, monkeypatch):
    monkeypatch.setenv("CLAUDE_PLUGIN_OPTION_AUTOHANDOFF", "true")
    assert json.loads(_main(stop(home, 199_000)))["decision"] == "block"
    ev = {"hook_event_name": "UserPromptSubmit", "session_id": "s9", "prompt": "hi",
          "transcript_path": transcript(home, 10)}
    assert "[context]" in _main(ev)
    assert _main({"hook_event_name": "PostToolUse", "session_id": "s9", "tool_name": "Skill",
                  "tool_input": {"skill": "handoff"}}) == ""
    assert (home / "medulla" / "state" / "s9.handoff").exists()


def test_a_broken_event_never_blocks(home, monkeypatch):
    monkeypatch.setenv("MEDULLA_AUTOHANDOFF", "1")
    out = io.StringIO()
    ah.main(stdin=io.StringIO("not json"), stdout=out)
    assert out.getvalue() == ""


@pytest.mark.parametrize("value,runs", [(None, False), ("false", False), ("true", True), ("ON", True)])
def test_the_shell_gate_matches_the_option(home, value, runs):
    env = {k: v for k, v in os.environ.items() if k != "CLAUDE_PLUGIN_OPTION_AUTOHANDOFF"}
    env["CLAUDE_PLUGIN_ROOT"] = str(ROOT)
    if value is not None:
        env["CLAUDE_PLUGIN_OPTION_AUTOHANDOFF"] = value
    r = subprocess.run([str(ROOT / "hooks" / "autohandoff.sh")], input=json.dumps(stop(home, 199_000)),
                       capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0 and (('"block"' in r.stdout) == runs)


def test_a_long_or_messy_transcript_is_read_from_its_tail(home, monkeypatch):
    """Real transcripts run to hundreds of MB: only the tail is read, and broken lines are skipped."""
    p = Path(transcript(home, 120_000))
    p.write_text("x" * 64 + "\n" + "not json\n" + json.dumps({"type": "assistant", "message": {"role": "assistant"}})
                 + "\n" + p.read_text())
    monkeypatch.setattr(ah, "_TAIL_BYTES", len(p.read_bytes()) - 10)
    assert ah.context_tokens(str(p)) == 120_000
    assert _main({"hook_event_name": "SessionStart", "session_id": "s1"}) == ""
