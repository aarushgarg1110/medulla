"""Open the next part of a session beside it, in whatever terminal it runs in.

detect() picks the adapter from the environment a hook inherits from its terminal: tmux when $TMUX is
set (any OS), then Ghostty, iTerm2 or Terminal.app on macOS by $TERM_PROGRAM, and otherwise none:
the command is printed for the user to run. MEDULLA_TERMINAL (or the plugin's `terminal` option)
overrides it. The macOS apps are driven through their own AppleScript dictionaries: Apple Events, no
keystrokes, so no Accessibility permission. Values reach a script as argv, never pasted into its
text, so a title or path cannot break it.

open_beside() reports success only once the caller's started() check comes true (by default, a
`claude -n <next>` process), so "opened" means running, not just requested. An opener returns its
message and a closer for the pane it made (None where it has no exact handle), so a pane whose part
never starts is closed rather than left dead on screen.
"""
from __future__ import annotations

import ctypes
import os
import re
import shlex
import subprocess
import sys
import time
from typing import Callable

_GLYPH = re.compile(r"^\W+")     # Claude Code prefixes a pane title with a status glyph: "✳ name"
_START_WAIT_S, _START_POLL_S = 10.0, 0.25
_WAKE_WAIT_S, _WAKE_POLL_S = 5.0, 0.1
_sleep, _clock = time.sleep, time.monotonic


def detect(env=None) -> str:
    env = os.environ if env is None else env
    forced = (env.get("MEDULLA_TERMINAL") or env.get("CLAUDE_PLUGIN_OPTION_TERMINAL") or "auto").lower()
    if forced != "auto":
        return forced
    if env.get("TMUX"):
        return "tmux"
    if sys.platform == "darwin":
        return {"ghostty": "ghostty", "iTerm.app": "iterm",
                "Apple_Terminal": "terminal"}.get(env.get("TERM_PROGRAM", ""), "print")
    return "print"


def pane_command(line: str, env, cwd: str | None = None) -> str:
    """`line` in an interactive login shell (the same PATH and rc files as a pane the user opens), then
    a shell to keep when it exits."""
    shell = env.get("SHELL") or "/bin/zsh"
    body = (f"cd {shlex.quote(cwd)} && " if cwd else "") + f"{line}; exec {shell} -l"
    return f"{shell} -lic {shlex.quote(body)}"


def _osascript(script: str, *args: str, runner=subprocess.run) -> str:
    r = runner(["osascript", "-", *args], input=script, capture_output=True, text=True, timeout=15)
    if r.returncode:
        raise RuntimeError((r.stderr or r.stdout).strip() or f"osascript exit {r.returncode}")
    return r.stdout.strip("\n")


def claude_started(nxt: str, runner=subprocess.run) -> Callable[[], bool]:
    """A check for a process whose arguments start `claude -n <nxt>`: the part itself, not the shells
    that carry the same line further along theirs."""
    own = f"claude -n {nxt}"

    def check() -> bool:
        try:
            out = runner(["ps", "-axww", "-o", "args="], capture_output=True, text=True, timeout=5).stdout
        except (OSError, subprocess.TimeoutExpired):
            return False
        return any(a == own or a.startswith(own + " ") for a in map(str.lstrip, out.splitlines()))
    return check


def _wait(started: Callable[[], bool]) -> float | None:
    start = _clock()
    while not started():
        if _clock() - start >= _START_WAIT_S:
            return None
        _sleep(_START_POLL_S)
    return _clock() - start


# ── tmux ──────────────────────────────────────────────────────────────────────

def tmux_open(line, cwd, env, runner, **_) -> tuple[str, Callable[[], None]]:
    cmd = ["tmux", "split-window", "-h", "-P", "-F", "#{pane_id}", "-c", cwd]
    if env.get("TMUX_PANE"):
        cmd += ["-t", env["TMUX_PANE"]]                # this session's own pane, not the focused one
    r = runner(cmd + [pane_command(line, env)], capture_output=True, text=True, timeout=15)
    if r.returncode:
        raise RuntimeError(r.stderr.strip() or "tmux split-window failed")
    new = r.stdout.strip()

    def close() -> None:
        k = runner(["tmux", "kill-pane", "-t", new], capture_output=True, text=True, timeout=15)
        if k.returncode:
            raise RuntimeError(k.stderr.strip() or "tmux kill-pane failed")
    return "split the tmux pane", close


# ── Ghostty: find this session's pane by its title, then split it ──────────────

_GHOSTTY_LIST = """
tell application "Ghostty"
	set sep to character id 9
	set out to ""
	repeat with w in windows
		repeat with t in tabs of w
			repeat with s in terminals of t
				set out to out & (id of w) & sep & (id of t) & sep & (id of s) & sep & ¬
					(name of s) & sep & (working directory of s) & linefeed
			end repeat
		end repeat
	end repeat
	return out
end tell
"""

_GHOSTTY_SPLIT = """
on run argv
	tell application "Ghostty"
		set cfg to new surface configuration
		set initial working directory of cfg to item 3 of argv
		set command of cfg to item 4 of argv
		set src to first terminal whose id is (item 1 of argv)
		if (item 2 of argv) is "down" then
			set newPane to split src direction down with configuration cfg
		else
			set newPane to split src direction right with configuration cfg
		end if
		return id of newPane
	end tell
end run
"""


_GHOSTTY_CLOSE = """
on run argv
	tell application "Ghostty" to close (first terminal whose id is (item 1 of argv))
end run
"""


def ghostty_panes(runner=subprocess.run) -> list[dict]:
    keys = ("window", "tab", "id", "title", "cwd")
    rows = (line.split("\t") for line in _osascript(_GHOSTTY_LIST, runner=runner).splitlines())
    return [dict(zip(keys, r)) for r in rows if len(r) == len(keys)]


def _same_dir(a: str, b: str) -> bool:
    """Ghostty reports the path the shell went through (a symlink), Claude the resolved one."""
    return bool(a and b) and os.path.realpath(a) == os.path.realpath(b)


def find_pane(panes: list[dict], title: str, cwd: str | None = None) -> dict:
    """The one pane showing `title`; a tie is broken by working directory, never guessed."""
    hits = [p for p in panes if _GLYPH.sub("", p["title"]).strip() == title]
    if len(hits) > 1 and cwd:
        hits = [p for p in hits if _same_dir(p["cwd"], cwd)] or hits
    if len(hits) != 1:
        raise LookupError(f"{len(hits)} Ghostty panes are titled {title!r}")
    return hits[0]


def split_direction(panes: list[dict], pane: dict) -> str:
    """Right while the pane is alone in its tab, down once the tab is already split."""
    alone = sum(p["window"] == pane["window"] and p["tab"] == pane["tab"] for p in panes) == 1
    return "right" if alone else "down"


_CORE_GRAPHICS = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"


def _display_asleep() -> bool:
    try:
        cg = ctypes.CDLL(_CORE_GRAPHICS)
    except OSError:
        return False
    cg.CGMainDisplayID.restype = ctypes.c_uint32
    cg.CGDisplayIsAsleep.argtypes = [ctypes.c_uint32]
    return bool(cg.CGDisplayIsAsleep(cg.CGMainDisplayID()))


def wake_display(runner=subprocess.run) -> bool:
    """Wake a sleeping display as a keypress would. Ghostty 1.3.1 cannot create a surface without an
    active display (it fails as error.OutOfMemory and the pane's shell never runs; fixed upstream in
    ghostty#13639, unreleased). A locked screen is fine; a closed lid cannot wake, and fails here."""
    if not _display_asleep():
        return False
    runner(["caffeinate", "-u", "-t", "1"], capture_output=True, timeout=5)
    deadline = _clock() + _WAKE_WAIT_S
    while _display_asleep():
        if _clock() >= deadline:
            raise RuntimeError(f"the display is asleep and did not wake within {_WAKE_WAIT_S:g} s")
        _sleep(_WAKE_POLL_S)
    return True


def ghostty_open(line, cwd, env, runner, title=None, **_) -> tuple[str, Callable[[], None]]:
    panes = ghostty_panes(runner)
    src = find_pane(panes, title or "", cwd)
    direction = split_direction(panes, src)
    new = _osascript(_GHOSTTY_SPLIT, src["id"], direction, src["cwd"] or cwd, pane_command(line, env),
                     runner=runner)

    def close() -> None:
        _osascript(_GHOSTTY_CLOSE, new, runner=runner)
    return f"split {direction} of Ghostty pane {src['id'][:8]}", close


# ── iTerm2: the session on this process's terminal device ─────────────────────

_ITERM_SPLIT = """
on run argv
	tell application "iTerm"
		repeat with w in windows
			repeat with t in tabs of w
				repeat with s in sessions of t
					if tty of s is (item 1 of argv) then
						tell s to split vertically with default profile command (item 2 of argv)
						return "split"
					end if
				end repeat
			end repeat
		end repeat
	end tell
	return "none"
end run
"""


def _tty(runner=subprocess.run, pid: int | None = None) -> str | None:
    """The terminal device of the nearest ancestor that has one: a hook has none, its claude does."""
    pid = pid or os.getpid()
    for _ in range(8):
        out = runner(["ps", "-o", "ppid=,tty=", "-p", str(pid)], capture_output=True, text=True,
                     timeout=5).stdout.split()
        if len(out) < 2:
            return None
        if out[1] not in ("??", "?", "-"):
            return out[1] if out[1].startswith("/dev/") else f"/dev/{out[1]}"
        pid = int(out[0])
    return None


def iterm_open(line, cwd, env, runner, **_) -> tuple[str, None]:
    tty = _tty(runner)
    if not tty:
        raise RuntimeError("this session has no terminal device")
    if _osascript(_ITERM_SPLIT, tty, pane_command(line, env, cwd), runner=runner) != "split":
        raise LookupError(f"no iTerm2 session is on {tty}")
    return f"split the iTerm2 session on {tty}", None


# ── Terminal.app: no splits, so a new window ──────────────────────────────────

_TERMINAL_NEW = """
on run argv
	tell application "Terminal" to do script (item 1 of argv)
end run
"""


def terminal_open(line, cwd, env, runner, **_) -> tuple[str, None]:
    _osascript(_TERMINAL_NEW, f"cd {shlex.quote(cwd)} && {line}", runner=runner)
    return "opened a new Terminal window", None


OPENERS = {"tmux": tmux_open, "ghostty": ghostty_open, "iterm": iterm_open, "terminal": terminal_open}


def _close(close: Callable[[], None] | None) -> str:
    """Close the pane of a part that never started, where the opener knows exactly which pane it made."""
    if close is None:
        return "; it is still open"
    try:
        close()
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        return f"; could not close it: {e}"
    return "; closed it"


def open_beside(line: str, cwd: str, title: str | None, started: Callable[[], bool],
                env=None, runner=subprocess.run) -> tuple[bool, str]:
    """Open `line` in `cwd` next to the session titled `title`; (ok, what happened)."""
    env = os.environ if env is None else env
    kind = detect(env)
    run_it = f"cd {shlex.quote(cwd)} && {line}"
    if kind not in OPENERS:
        return True, f"run in a new terminal: {run_it}"
    try:
        woke = kind != "tmux" and wake_display(runner)     # a macOS app may need an active display
        detail, close = OPENERS[kind](line=line, cwd=cwd, env=env, runner=runner, title=title)
        detail += ", display woken" if woke else ""
    except (RuntimeError, LookupError, OSError, subprocess.TimeoutExpired) as e:
        return False, f"{kind}: {e}. Run it yourself: {run_it}"
    took = _wait(started)
    if took is None:
        return False, f"{detail}, but it did not start within {_START_WAIT_S:g} s{_close(close)}. Run it yourself: {run_it}"
    return True, f"{detail}; running in {took:.1f} s"
