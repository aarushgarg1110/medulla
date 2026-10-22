"""Which terminal a session is in, and how each adapter opens the next part beside it. A fake runner
stands in for tmux, osascript and ps, so this runs anywhere; the live checks are a manual script."""
from types import SimpleNamespace

import pytest

from medulla import terminals as t


class Runner:
    """Records each command; answers by program, with osascript answered by which script it got."""

    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    def __call__(self, cmd, input=None, **kw):
        self.calls.append((cmd, input))
        key = cmd[0]
        if key == "osascript":
            key = next((k for k in self.answers if k.startswith("osa:") and k[4:] in (input or "")), "osa")
        out = self.answers.get(key, "")
        if callable(out):
            out = out(cmd)
        rc, text = (out if isinstance(out, tuple) else (0, out))
        return SimpleNamespace(returncode=rc, stdout=text, stderr="boom" if rc else "")


@pytest.fixture(autouse=True)
def quick(monkeypatch):
    monkeypatch.setattr(t, "_sleep", lambda s: None)
    monkeypatch.setattr(t, "_display_asleep", lambda: False)


@pytest.mark.parametrize("env,platform,kind", [
    ({"TMUX": "/tmp/tmux-1/default,1,0", "TERM_PROGRAM": "ghostty"}, "darwin", "tmux"),
    ({"TMUX": "/tmp/tmux-1/default,1,0"}, "linux", "tmux"),
    ({"TERM_PROGRAM": "ghostty"}, "darwin", "ghostty"),
    ({"TERM_PROGRAM": "iTerm.app"}, "darwin", "iterm"),
    ({"TERM_PROGRAM": "Apple_Terminal"}, "darwin", "terminal"),
    ({"TERM_PROGRAM": "vscode"}, "darwin", "print"),
    ({"TERM_PROGRAM": "ghostty"}, "linux", "print"),
    ({"TERM_PROGRAM": "ghostty", "MEDULLA_TERMINAL": "print"}, "darwin", "print"),
    ({"CLAUDE_PLUGIN_OPTION_TERMINAL": "tmux"}, "darwin", "tmux"),
])
def test_detect(monkeypatch, env, platform, kind):
    monkeypatch.setattr(t.sys, "platform", platform)
    assert t.detect(env) == kind


def test_pane_command_runs_the_line_in_a_login_shell_and_keeps_one():
    cmd = t.pane_command("claude -n x", {"SHELL": "/bin/zsh"}, cwd="/w d")
    assert cmd == "/bin/zsh -lic 'cd '\"'\"'/w d'\"'\"' && claude -n x; exec /bin/zsh -l'"


def test_tmux_splits_this_sessions_own_pane():
    r = Runner()
    assert t.tmux_open("echo hi", "/w", {"TMUX_PANE": "%3", "SHELL": "/bin/sh"}, r) == "split the tmux pane"
    assert r.calls[0][0][:6] == ["tmux", "split-window", "-h", "-c", "/w", "-t"] and r.calls[0][0][6] == "%3"
    with pytest.raises(RuntimeError):
        t.tmux_open("echo hi", "/w", {}, Runner({"tmux": (1, "")}))


LISTING = "w1\tt1\tp1\t✳ fooPart3\t/w\nw1\tt2\tp2\tother\t/x\n"


def test_ghostty_finds_the_pane_by_title_and_splits_it_right_when_alone():
    r = Runner({"osa:repeat with w in windows": LISTING, "osa:new surface configuration": "p9"})
    assert t.ghostty_open("echo hi", "/w", {"SHELL": "/bin/sh"}, r, title="fooPart3") == "split right of Ghostty pane p1"
    assert r.calls[-1][0][2:5] == ["p1", "right", "/w"]


def test_find_pane_breaks_a_title_tie_by_directory_and_never_guesses():
    panes = [dict(window="w", tab="t", id="a", title="✳ foo", cwd="/one"),
             dict(window="w", tab="t", id="b", title="foo", cwd="/two")]
    assert t.find_pane(panes, "foo", "/two")["id"] == "b" and t.split_direction(panes, panes[0]) == "down"
    with pytest.raises(LookupError):
        t.find_pane(panes, "foo")
    with pytest.raises(LookupError):
        t.find_pane(panes, "bar")


def test_a_sleeping_display_is_woken_or_reported(monkeypatch):
    states = iter([True, True, False])
    monkeypatch.setattr(t, "_display_asleep", lambda: next(states))
    r = Runner()
    assert t.wake_display(r) is True and r.calls[0][0][0] == "caffeinate"
    monkeypatch.setattr(t, "_display_asleep", lambda: True)
    clock = iter([0.0, 10.0])
    monkeypatch.setattr(t, "_clock", lambda: next(clock))
    with pytest.raises(RuntimeError):
        t.wake_display(Runner())


def test_iterm_splits_the_session_on_this_processes_terminal():
    ps = iter(["123 ??", "45 ttys004"])
    r = Runner({"ps": lambda cmd: next(ps), "osa:sessions of t": "split"})
    assert t.iterm_open("echo hi", "/w", {"SHELL": "/bin/sh"}, r) == "split the iTerm2 session on /dev/ttys004"
    assert r.calls[-1][0][2] == "/dev/ttys004"
    with pytest.raises(LookupError):
        t.iterm_open("x", "/w", {}, Runner({"ps": "1 ttys004", "osa:sessions of t": "none"}))
    with pytest.raises(RuntimeError):
        t.iterm_open("x", "/w", {}, Runner({"ps": ""}))


def test_terminal_app_opens_a_window_in_the_right_directory():
    r = Runner()
    assert t.terminal_open("echo hi", "/w d", {}, r) == "opened a new Terminal window"
    assert r.calls[0][0][2] == "cd '/w d' && echo hi"


def test_osascript_failures_carry_the_apps_message():
    with pytest.raises(RuntimeError, match="boom"):
        t._osascript("x", runner=Runner({"osa": (1, "")}))


def test_claude_started_matches_the_part_not_its_wrapper_shells():
    out = "/bin/zsh -lic claude -n fooPart4 '/medulla:pickoff fooPart3'\nclaude -n fooPart4 /medulla:pickoff\n"
    assert t.claude_started("fooPart4", Runner({"ps": out}))() is True
    assert t.claude_started("fooPart4", Runner({"ps": out.splitlines()[0]}))() is False


def test_open_beside_reports_running_failed_or_printed(monkeypatch):
    ok, detail = t.open_beside("echo hi", "/w", "foo", lambda: True, env={"MEDULLA_TERMINAL": "print"})
    assert ok and detail == "run in a new terminal: cd /w && echo hi"
    ok, detail = t.open_beside("echo hi", "/w", "foo", lambda: True, env={"MEDULLA_TERMINAL": "tmux"},
                               runner=Runner())
    assert ok and "split the tmux pane; running in" in detail
    ok, detail = t.open_beside("echo hi", "/w", "foo", lambda: True, env={"MEDULLA_TERMINAL": "tmux"},
                               runner=Runner({"tmux": (1, "")}))
    assert not ok and "Run it yourself: cd /w && echo hi" in detail
    clock = iter([0.0, 11.0])
    monkeypatch.setattr(t, "_clock", lambda: next(clock))
    ok, detail = t.open_beside("echo hi", "/w", "foo", lambda: False, env={"MEDULLA_TERMINAL": "tmux"},
                               runner=Runner())
    assert not ok and "did not start within 10 s" in detail
