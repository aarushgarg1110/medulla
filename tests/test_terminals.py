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


def test_tmux_splits_this_sessions_own_pane_and_can_close_the_new_one():
    r = Runner({"tmux": "%7\n"})
    detail, close = t.tmux_open("echo hi", "/w", {"TMUX_PANE": "%3", "SHELL": "/bin/sh"}, r)
    assert detail == "split the tmux pane"
    assert r.calls[0][0][:9] == ["tmux", "split-window", "-h", "-P", "-F", "#{pane_id}", "-c", "/w", "-t"]
    assert r.calls[0][0][9] == "%3"
    close()
    assert r.calls[-1][0] == ["tmux", "kill-pane", "-t", "%7"]
    with pytest.raises(RuntimeError):
        t.tmux_open("echo hi", "/w", {}, Runner({"tmux": (1, "")}))


LISTING = "w1\tt1\tp1\t✳ fooPart3\t/w\nw1\tt2\tp2\tother\t/x\n"


def test_ghostty_finds_the_pane_by_title_and_splits_it_right_when_alone():
    r = Runner({"osa:repeat with w in windows": LISTING, "osa:new surface configuration": "p9"})
    detail, close = t.ghostty_open("echo hi", "/w", {"SHELL": "/bin/sh"}, r, title="fooPart3")
    assert detail == "split right of Ghostty pane p1" and r.calls[-1][0][2:5] == ["p1", "right", "/w"]
    close()
    assert r.calls[-1][0][2] == "p9" and "close" in r.calls[-1][1]          # the new pane, not this one


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
    assert t.iterm_open("echo hi", "/w", {"SHELL": "/bin/sh"}, r) == ("split the iTerm2 session on /dev/ttys004", None)
    assert r.calls[-1][0][2] == "/dev/ttys004"
    with pytest.raises(LookupError):
        t.iterm_open("x", "/w", {}, Runner({"ps": "1 ttys004", "osa:sessions of t": "none"}))
    with pytest.raises(RuntimeError):
        t.iterm_open("x", "/w", {}, Runner({"ps": ""}))


def test_terminal_app_opens_a_window_in_the_right_directory():
    r = Runner()
    assert t.terminal_open("echo hi", "/w d", {}, r) == ("opened a new Terminal window", None)
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


def test_a_part_that_never_starts_has_its_pane_closed_where_it_can_be(monkeypatch):
    """A pane whose claude never runs is dead weight (Ghostty's "terminal failed to initialize"); close it
    where the adapter knows exactly which pane it made, and say so either way."""
    monkeypatch.setattr(t, "_clock", iter([0.0, 11.0, 0.0, 11.0, 0.0, 11.0]).__next__)
    r = Runner({"tmux": "%7"})
    ok, detail = t.open_beside("x", "/w", "f", lambda: False, env={"MEDULLA_TERMINAL": "tmux"}, runner=r)
    assert not ok and "did not start within 10 s; closed it. Run it yourself: cd /w && x" in detail
    assert r.calls[-1][0] == ["tmux", "kill-pane", "-t", "%7"]
    r = Runner({"tmux": lambda cmd: (1, "") if cmd[1] == "kill-pane" else (0, "%7")})
    ok, detail = t.open_beside("x", "/w", "f", lambda: False, env={"MEDULLA_TERMINAL": "tmux"}, runner=r)
    assert not ok and "; could not close it: " in detail
    ok, detail = t.open_beside("x", "/w", "f", lambda: False, env={"MEDULLA_TERMINAL": "terminal"}, runner=Runner())
    assert not ok and "did not start within 10 s; it is still open. Run it yourself" in detail


def test_a_sleeping_display_is_woken_for_the_macos_apps_but_not_for_tmux(monkeypatch):
    woken = []
    monkeypatch.setattr(t, "wake_display", lambda runner: woken.append(1) or True)
    ok, detail = t.open_beside("x", "/w", "f", lambda: True, env={"MEDULLA_TERMINAL": "terminal"}, runner=Runner())
    assert ok and detail.startswith("opened a new Terminal window, display woken")
    t.open_beside("x", "/w", "f", lambda: True, env={"MEDULLA_TERMINAL": "tmux"}, runner=Runner())
    assert woken == [1]


def test_a_display_that_cannot_wake_says_why_and_gives_the_command(monkeypatch):
    def lid_closed(runner):
        raise RuntimeError("the display is asleep and did not wake within 5 s")
    monkeypatch.setattr(t, "wake_display", lid_closed)
    ok, detail = t.open_beside("x", "/w", "f", lambda: True, env={"MEDULLA_TERMINAL": "ghostty"}, runner=Runner())
    assert not ok and "did not wake" in detail and "Run it yourself: cd /w && x" in detail
