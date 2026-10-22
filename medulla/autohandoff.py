"""Hand a long session to its next part without the user doing it: Stop, UserPromptSubmit and
PostToolUse hooks, off unless the user turns them on.

At the threshold (93% of the context window by default), a Stop hook keeps the turn going with one
instruction: run the handoff skill. When that report ends a turn (it has the skill's "Immediate next
action" section), the next part is opened as `claude -n fooPart<N+1> '/medulla:pickoff fooPart<N> …'`.
A handoff the user types, or one Claude starts through the Skill tool, opens the next part the same
way at any context level; `nonext` in its arguments opts out.

Context % is the last main-chain request's input-side tokens (what the next request re-sends) over the
window size. Claude Code reports the size only to the statusline, so it comes from, in order: the
statusline's record (medulla statusline), a configured size, a `[1m]` model in Claude settings, and
finally the transcript itself, since more than 200k tokens can only fit the 1M window.

Fails open: any error means no output, because a missed handoff is recoverable and a blocked stop is not.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

from medulla import statusline

_HANDOFF_CMD = re.compile(r"\s*/(?:medulla:)?handoff(?:\s+(.*))?\s*$", re.S)
_REPORT = re.compile(r"immediate next action", re.I)
_PART = re.compile(r"(.*?Part)(\d+)$", re.I)
_TAIL_BYTES = 4 << 20
_SMALL_WINDOW, _LARGE_WINDOW = 200_000, 1_000_000
PLAN_FROM = 0.9      # from this fraction of the threshold, the context line says the handoff is automatic

NUDGE = ("Context is at {pct:.0f}% of the window (autohandoff threshold {at:g}%). Run the medulla "
         "handoff skill now, without asking the user first: they turned autohandoff on. Its report "
         "must be your final message; the next part opens when this turn ends.")


def _option(name: str, default: str = "") -> str:
    """A plugin option (CLAUDE_PLUGIN_OPTION_*, set in /config) or its MEDULLA_* environment override."""
    return os.environ.get(f"MEDULLA_{name}") or os.environ.get(f"CLAUDE_PLUGIN_OPTION_{name}") or default


def enabled() -> bool:
    return _option("AUTOHANDOFF", "false").strip().lower() in ("1", "true", "yes", "on")


def threshold() -> float:
    return float(_option("AUTOHANDOFF_AT", "93"))


def state_dir() -> Path:
    d = Path(statusline.state_dir())
    d.mkdir(parents=True, exist_ok=True)
    return d


def next_names(title: str) -> tuple[str, str]:
    """(next part, previous part): fooPart3 -> fooPart4; a name without PartN becomes Part2."""
    m = _PART.fullmatch(title)
    return (f"{m.group(1)}{int(m.group(2)) + 1}" if m else f"{title}Part2"), title


def _entries(path: str, tail: bool):
    with open(path, "rb") as fh:
        if tail:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - _TAIL_BYTES))
            if size > _TAIL_BYTES:
                fh.readline()
        for line in fh:
            try:
                yield json.loads(line)
            except ValueError:
                continue


def _last_usage(entries) -> int | None:
    tokens = None
    for e in entries:
        msg = e.get("message")
        if e.get("type") != "assistant" or e.get("isSidechain") or not isinstance(msg, dict):
            continue
        u = msg.get("usage")
        if u:
            tokens = sum(u.get(k) or 0 for k in
                         ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))
    return tokens


def context_tokens(transcript_path: str) -> int | None:
    """Input-side tokens of the last main-chain request: what the next request re-sends."""
    tokens = _last_usage(_entries(transcript_path, tail=True))
    return tokens if tokens is not None else _last_usage(_entries(transcript_path, tail=False))


_REPLY_WAIT_S, _REPLY_POLL_S = 1.5, 0.05
_sleep, _clock = time.sleep, time.monotonic


def _reply_written(entries) -> bool:
    last = None
    for e in entries:
        if e.get("type") in ("user", "assistant") and not e.get("isSidechain"):
            last = e
    return last is not None and last["type"] == "assistant"


def wait_for_reply(transcript_path: str) -> bool:
    """Stop can fire before the turn's reply reaches the transcript (77 ms early on a first turn), so
    wait briefly for it; otherwise the count is the previous turn's."""
    deadline = _clock() + _REPLY_WAIT_S
    while not _reply_written(_entries(transcript_path, tail=True)):
        if _clock() >= deadline:
            return False
        _sleep(_REPLY_POLL_S)
    return True


def _settings_model() -> str:
    try:
        sp = Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "settings.json"
        return str(json.loads(sp.read_text()).get("model") or "")
    except (OSError, ValueError, AttributeError):
        return ""


def window_size(sid: str, tokens: int | None = None) -> int:
    """The statusline's record if there is one; else a configured size; else 1M for a `[1m]` model in
    Claude settings; else 200k. Never below 1M once the context holds more than 200k tokens."""
    size = None
    try:
        size = int(json.loads((state_dir() / f"{sid}.ctx.json").read_text())["size"])
    except (OSError, ValueError, KeyError, TypeError):
        configured = int(float(_option("CONTEXT_WINDOW", "0") or 0))
        size = configured or (_LARGE_WINDOW if "[1m]" in _settings_model() else _SMALL_WINDOW)
    return max(size, _LARGE_WINDOW) if tokens and tokens > _SMALL_WINDOW else size


def context_pct(sid: str, transcript_path: str) -> float | None:
    tokens = context_tokens(transcript_path)
    return None if tokens is None else 100.0 * tokens / window_size(sid, tokens)


def _flag(sid: str) -> Path:
    return state_dir() / f"{sid}.handoff"


def _spawned(sid: str) -> Path:
    return state_dir() / f"{sid}.spawned"


def _log(**rec) -> None:
    with open(state_dir() / "autohandoff.log.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")


def _mark(sid: str, args: str, who: str) -> None:
    if "nonext" in args.split():
        _flag(sid).unlink(missing_ok=True)
    else:
        _flag(sid).write_text(who)


def context_line(sid: str, transcript_path: str | None) -> str | None:
    """`[context] 31.2% of 1,000,000 tokens (312,345)` and, near the threshold, that the handoff is the
    hook's job, so Claude keeps working instead of wrapping up early."""
    tokens = context_tokens(transcript_path) if transcript_path else None
    if tokens is None:
        return None
    size = window_size(sid, tokens)
    pct, at = 100.0 * tokens / size, threshold()
    line = f"[context] {pct:.1f}% of {size:,} tokens ({tokens:,})"
    if pct >= PLAN_FROM * at:
        line += f" · autohandoff takes over at {at:g}%; keep working"
    return line


def launch_line(prev: str, nxt: str) -> str:
    """The command that opens the next part, told to restate and wait rather than start working."""
    pickoff = f"/medulla:pickoff {prev} opened automatically: restate, then wait for my reply before doing any work"
    return f"claude -n {shlex.quote(nxt)} {shlex.quote(pickoff)}"


def terminal_spawn(prev: str, nxt: str, event: dict) -> tuple[bool, str]:
    """Open the next part beside this session in its terminal (tmux, Ghostty, iTerm2, Terminal.app),
    or print the command where medulla cannot drive the terminal."""
    from medulla import terminals
    return terminals.open_beside(launch_line(prev, nxt), event.get("cwd") or os.getcwd(), title=prev,
                                 started=terminals.claude_started(nxt))


def on_prompt(event: dict) -> dict | None:
    """A typed /handoff marks a handoff pending; each prompt carries the context line."""
    m = _HANDOFF_CMD.fullmatch(event.get("prompt") or "")
    if m:
        _mark(event["session_id"], m.group(1) or "", "user")
    line = context_line(event["session_id"], event.get("transcript_path"))
    return line and {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit", "additionalContext": line}}


def on_skill(event: dict) -> None:
    """A handoff Claude started itself ("wrap up", "I'm out of context") through the Skill tool."""
    inp = event.get("tool_input") or {}
    if event.get("tool_name") == "Skill" and inp.get("skill") in ("handoff", "medulla:handoff"):
        _mark(event["session_id"], inp.get("args") or "", "skill")


def on_stop(event: dict, spawn) -> dict | None:
    """The Stop decision. `spawn(prev, nxt, event) -> (ok, detail)` opens the next part."""
    sid, transcript = event["session_id"], event["transcript_path"]
    if _spawned(sid).exists() or event.get("agent_id"):
        return None
    if _flag(sid).exists():
        if not _REPORT.search(event.get("last_assistant_message") or ""):
            return None
        _flag(sid).unlink()
        from medulla.episodic.scanner import title_of
        title = title_of(Path(transcript))
        if not title:
            _log(sid=sid, action="skip", why="unnamed")
            return {"systemMessage": "autohandoff: this session has no /rename title, so no next part "
                                     "was opened. Rename it, then open the next part yourself."}
        nxt, prev = next_names(title)
        ok, detail = spawn(prev, nxt, event)
        _log(sid=sid, action="spawn", ok=ok, prev=prev, next=nxt, detail=detail)
        if not ok:
            return {"systemMessage": f"autohandoff: could not open {nxt}: {detail}"}
        _spawned(sid).write_text(nxt)
        return {"systemMessage": f"autohandoff: {nxt}: {detail}"}
    if event.get("stop_hook_active"):
        return None
    wait_for_reply(transcript)
    pct = context_pct(sid, transcript)
    if pct is None or pct < threshold():
        return None
    _flag(sid).write_text("hook")
    _log(sid=sid, action="request", pct=round(pct, 1))
    return {"decision": "block", "reason": NUDGE.format(pct=pct, at=threshold())}


def main(stdin=sys.stdin, stdout=sys.stdout, spawn=terminal_spawn) -> None:
    try:
        if not enabled():
            return
        event = json.load(stdin)
        name, out = event.get("hook_event_name"), None
        if name == "UserPromptSubmit":
            out = on_prompt(event)
        elif name == "PostToolUse":
            on_skill(event)
        elif name == "Stop":
            out = on_stop(event, spawn)
        if out:
            stdout.write(json.dumps(out))
    except Exception:
        return


if __name__ == "__main__":  # pragma: no cover  (run by the plugin's hooks, in a subprocess)
    main()
