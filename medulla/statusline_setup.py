"""Put medulla's statusline in front of a user without taking theirs away.

plan() reads the user's Claude Code settings and says what it would do; apply() does it, backing up
settings.json once; undo() reverses it. Three cases:
    install   no statusline yet: medulla's two lines
    wrap      the user has their own: medulla records the context size, then runs their command on the
              same input and prints their output unchanged. Their command is kept inside ours, so undo
              restores it exactly
    present   already medulla's: nothing to do

Settings point at a launcher in medulla's own folder, which never moves. statusline.target names the
Python and script it runs, and refresh_target() rewrites that file when the install moves, which a
plugin update does. settings.json is written through, not replaced, so a symlinked one stays a link.
"""
from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path

from medulla import statusline

LAUNCHER = """#!/bin/sh
# medulla's statusline launcher. It never moves: statusline.target names the Python and the script to
# run, and medulla rewrites that file whenever its install moves.
dir="$(dirname "$0")/.."
{ IFS= read -r py && IFS= read -r script; } < "$dir/statusline.target" || exit 0
exec "$py" -I "$script" "$@"
"""


def settings_path() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "settings.json"


def launcher_path() -> Path:
    return Path(statusline.medulla_dir()) / "bin" / "statusline"


def _target_path() -> Path:
    return Path(statusline.medulla_dir()) / "statusline.target"


def write_target(python: str | None = None) -> bool:
    """Point the launcher at this install's Python and statusline.py. True when that changed it."""
    text = f"{python or sys.executable}\n{Path(statusline.__file__).resolve()}\n"
    t = _target_path()
    if t.exists() and t.read_text() == text:
        return False
    t.parent.mkdir(parents=True, exist_ok=True)
    t.write_text(text)
    return True


def refresh_target() -> bool:
    """At MCP server start: only for someone who installed the launcher, and only if the install moved."""
    return launcher_path().exists() and write_target()


def _ours(cmd: str | None) -> bool:
    return bool(cmd) and str(launcher_path()) in cmd


def _wrapped(cmd: str) -> str | None:
    """The user's own command inside one of our `--wrap` statuslines."""
    parts = shlex.split(cmd)
    return parts[parts.index("--wrap") + 1] if "--wrap" in parts[:-1] else None


def _shows_context(cmd: str) -> bool:
    """Whether the user's statusline script already displays the context window (a courtesy note only)."""
    for tok in shlex.split(cmd):
        p = Path(os.path.expanduser(tok))
        if p.is_file():
            text = p.read_text(errors="replace")
            if "context_window" in text or "used_percentage" in text:
                return True
    return False


def _load(sp: Path) -> dict:
    return json.loads(sp.read_text()) if sp.exists() else {}


def plan(sp: Path | None = None) -> dict:
    sp = sp or settings_path()
    cur = (_load(sp).get("statusLine") or {}).get("command")
    if _ours(cur):
        return {"mode": "present", "current": cur}
    if not cur:
        return {"mode": "install", "current": None}
    return {"mode": "wrap", "current": cur, "shows_context": _shows_context(cur)}


def install_launcher() -> Path:
    """The launcher and its target, without touching settings: what a `--tap` user needs."""
    lp = launcher_path()
    lp.parent.mkdir(parents=True, exist_ok=True)
    lp.write_text(LAUNCHER)
    lp.chmod(0o755)
    write_target()
    return lp


def tap_line() -> str:
    """The line a user adds to their own statusline script, naming this install's launcher."""
    return f"printf '%s' \"$input\" | {shlex.quote(str(launcher_path()))} --tap"


def apply(sp: Path | None = None) -> dict:
    """Do what plan() says. The first apply copies settings.json to settings.json.medulla-backup."""
    sp = sp or settings_path()
    p = plan(sp)
    if p["mode"] == "present":
        return p
    settings = _load(sp)
    backup = sp.with_name(sp.name + ".medulla-backup")
    if sp.exists() and not backup.exists():
        backup.write_text(sp.read_text())
    lp = shlex.quote(str(install_launcher()))
    cmd = lp if p["mode"] == "install" else f"{lp} --wrap {shlex.quote(p['current'])}"
    settings["statusLine"] = {**(settings.get("statusLine") or {}), "type": "command", "command": cmd}
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(settings, indent=2) + "\n")
    return {**p, "command": cmd}


def undo(sp: Path | None = None) -> str:
    sp = sp or settings_path()
    settings = _load(sp)
    cur = (settings.get("statusLine") or {}).get("command")
    if not _ours(cur):
        return "The statusline is not medulla's; nothing to undo."
    orig = _wrapped(cur)
    if orig:
        settings["statusLine"]["command"] = orig
    else:
        settings.pop("statusLine")
    sp.write_text(json.dumps(settings, indent=2) + "\n")
    return f"Restored your statusline: {orig}" if orig else "Removed medulla's statusline."


def describe(p: dict) -> str:
    """One paragraph for a person deciding whether to apply the plan."""
    if p["mode"] == "present":
        return "medulla's statusline is already set up. Nothing to do."
    if p["mode"] == "install":
        return ("You have no statusline. medulla would add one: [model] | session name, then context "
                "and 5-hour usage bars, with the context's token count.")
    note = (" It already shows the context window, so wrapping only records the window size for "
            "autohandoff.") if p.get("shows_context") else ""
    return (f"You have your own statusline ({p['current']}). medulla would wrap it: record the context "
            f"window size, then run yours on the same input and print its output unchanged.{note} "
            "`medulla statusline undo` restores it exactly.")
