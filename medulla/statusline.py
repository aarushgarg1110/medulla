"""Claude Code statusline: shows the context window filling up, and records its size where autohandoff
reads it. Claude Code reports the window size only to the statusline, never to hooks.

It runs on every refresh, so it imports nothing beyond the standard library (medulla.config alone
costs 13 ms). Modes:
    statusline.py               record, then print medulla's two lines
    statusline.py --tap         record only, print nothing: for calling from your own statusline script
    statusline.py --wrap CMD    record, then run your CMD on the same input and print its output unchanged
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

_BAR = 8


def medulla_dir() -> str:
    """Same rule as medulla.config._default_medulla_dir, without its imports; a test keeps them equal."""
    return os.environ.get("MEDULLA_DIR") or os.path.join(os.path.expanduser("~"), ".medulla")


def state_dir() -> str:
    return os.path.join(medulla_dir(), "state")


def record(d: dict, sdir: str | None = None) -> str | None:
    """Write this session's window size, fill and name. Atomic, so a reader never sees half a file."""
    sid = d.get("session_id")
    if not sid:
        return None
    cw = d.get("context_window") or {}
    sdir = sdir or state_dir()
    os.makedirs(sdir, exist_ok=True)
    out = os.path.join(sdir, f"{sid}.ctx.json")
    with open(out + ".tmp", "w") as fh:
        json.dump({"size": cw.get("context_window_size"), "pct": cw.get("used_percentage"),
                   "name": d.get("session_name"), "model": (d.get("model") or {}).get("id")}, fh)
    os.replace(out + ".tmp", out)
    return out


def _bar(pct: int, color: int) -> str:
    n = min(_BAR, max(0, pct * _BAR // 100))
    return f"\033[{color}m" + "█" * n + "░" * (_BAR - n) + "\033[0m"


def _tokens(n: float) -> str:
    return f"{n / 1e6:.3g}M" if n >= 1e6 else f"{round(n / 1e3)}k"


def render(d: dict) -> str:
    """[Model] | session name, then context and 5-hour usage bars, the context with its token count."""
    cw = d.get("context_window") or {}
    used = cw.get("used_percentage") or 0
    size = cw.get("context_window_size")
    five = (((d.get("rate_limits") or {}).get("five_hour") or {}).get("used_percentage")) or 0
    model = (d.get("model") or {}).get("display_name") or "Claude"
    name = d.get("session_name")
    tokens = f" ({_tokens(size * used / 100)}/{_tokens(size)})" if size else ""
    line1 = f"\033[36m[{model}]" + (f" | {name}" if name else "") + "\033[0m"
    line2 = (f"Context {_bar(round(used), 32)} {round(used)}%{tokens} | "
             f"Usage {_bar(round(five), 34)} {round(five)}%")
    return f"{line1}\n{line2}"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    raw = sys.stdin.read()
    try:
        d = json.loads(raw or "{}")
    except ValueError:
        d = {}
    try:
        record(d)
    except OSError:
        pass   # a read-only or full home directory must not blank the statusline
    if argv[:1] == ["--tap"]:
        return 0
    if argv[:1] == ["--wrap"] and len(argv) > 1:
        r = subprocess.run(argv[1], shell=True, input=raw, capture_output=True, text=True, timeout=10)
        sys.stdout.write(r.stdout)
        return r.returncode
    sys.stdout.write(render(d))
    return 0


if __name__ == "__main__":  # pragma: no cover  (run by the launcher, in a subprocess)
    sys.exit(main())
