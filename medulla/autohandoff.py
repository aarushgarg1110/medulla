"""Hand a long session to its next part without the user doing it: prompt, tool and Stop hooks across
any registered agent harness (Claude Code, Antigravity CLI), off unless the user turns them on.

At the harness's threshold (93% of the context window for Claude, 45% for Antigravity by default),
a Stop hook keeps the turn going with one instruction: run the handoff skill. When that report ends
a turn (it has the skill's "Immediate next action" section), the next part is opened beside this
session through the active harness's launch command and terminal adapters. A handoff the user types,
or one the agent starts through the handoff skill, opens the next part the same way at any context
level; `nonext` in its arguments opts out.

Context % is the last main-chain request's input-side tokens (what the next request re-sends) over the
window size. Window size comes from, in order: the statusline's record (`medulla statusline`), a
configured size, a `[1m]` model in Claude settings, and finally the transcript itself, since more than
200k tokens can only fit the 1M window.

Fails open: any error means no output, because a missed handoff is recoverable and a blocked stop is not.
An open is recorded before it is attempted, so one the hook does not live to finish (the Mac slept and
killed the hook on wake) is reported on the session's next prompt instead of vanishing.
`medulla nextpart` opens the next part by hand, through the same harness and terminal adapters.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
import time
import tomllib
from pathlib import Path

from medulla import harnesses, statusline
from medulla.harnesses import Harness, HookEvent, HookOutcome

_HANDOFF_CMD = re.compile(r"\s*/(?:medulla:)?handoff(?:\s+(.*))?\s*$", re.S)
_REPORT = re.compile(r"immediate next action", re.I)
_PART = re.compile(r"(.*?Part)(\d+)$", re.I)
_TAIL_BYTES = 4 << 20
_SMALL_WINDOW, _LARGE_WINDOW = 200_000, 1_000_000
PLAN_FROM = 0.9      # from this fraction of the threshold, the context line says the handoff is automatic

NUDGE = ("Context is at {pct:.0f}% of the window (autohandoff threshold {at:g}%, {src}). Run the medulla "
         "handoff skill now, without asking the user first: they turned autohandoff on. Its report "
         "must be your final message; the next part opens when this turn ends.")


def _config_section() -> dict:
    """Optional `[autohandoff]` table in `<MEDULLA_DIR>/config.toml`."""
    p = Path(statusline.medulla_dir()) / "config.toml"
    if not p.exists():
        return {}
    try:
        sec = tomllib.loads(p.read_text()).get("autohandoff")
        return sec if isinstance(sec, dict) else {}
    except (OSError, ValueError):
        return {}


def _option(name: str, default: str = "", config_key: str | None = None) -> str:
    """Precedence: MEDULLA_* env -> CLAUDE_PLUGIN_OPTION_* env -> config.toml [autohandoff] -> default."""
    if os.environ.get(f"MEDULLA_{name}"):
        return os.environ[f"MEDULLA_{name}"]
    if os.environ.get(f"CLAUDE_PLUGIN_OPTION_{name}"):
        return os.environ[f"CLAUDE_PLUGIN_OPTION_{name}"]
    if config_key is not None:
        val = _config_section().get(config_key)
        if val is not None:
            return "true" if val is True else ("false" if val is False else str(val))
    return default


def enabled() -> bool:
    return _option("AUTOHANDOFF", "false", config_key="enabled").strip().lower() in ("1", "true", "yes", "on")


def threshold(harness: str = "claude") -> float:
    return float(_option("AUTOHANDOFF_AT", str(harnesses.get(harness).default_threshold_pct), config_key="at"))


def threshold_source() -> str:
    """Where threshold() came from, in _option's order, so the next part can tell an odd one is deliberate."""
    if os.environ.get("MEDULLA_AUTOHANDOFF_AT"):
        return "from MEDULLA_AUTOHANDOFF_AT"
    if os.environ.get("CLAUDE_PLUGIN_OPTION_AUTOHANDOFF_AT"):
        return "set in /config"
    return "from config.toml" if _config_section().get("at") is not None else "the default"


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
        val = harnesses.entry_usage(e)
        if val is not None:
            tokens = val
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
        role = harnesses.entry_role(e)
        if role is not None:
            last = role
    return last == "assistant"


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
        sp = harnesses.get("claude").settings_path()
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
        configured = int(float(_option("CONTEXT_WINDOW", "0", config_key="context_window") or 0))
        size = configured or (_LARGE_WINDOW if "[1m]" in _settings_model() else _SMALL_WINDOW)
    return max(size, _LARGE_WINDOW) if tokens and tokens > _SMALL_WINDOW else size


def context_pct(sid: str, transcript_path: str) -> float | None:
    tokens = context_tokens(transcript_path)
    return None if tokens is None else 100.0 * tokens / window_size(sid, tokens)


def _flag(sid: str) -> Path:
    return state_dir() / f"{sid}.handoff"


def _spawned(sid: str) -> Path:
    return state_dir() / f"{sid}.spawned"


def _opening(sid: str) -> Path:
    return state_dir() / f"{sid}.opening"


def _turn_file(sid: str) -> Path:
    return state_dir() / f"{sid}.turn"


def _running(nxt: str, prev: str = "", harness: str = "claude") -> bool:
    return harnesses.get(harness).started_check(prev, nxt)()


def _log(**rec) -> None:
    with open(state_dir() / "autohandoff.log.jsonl", "a") as fh:
        fh.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}) + "\n")


def _mark(sid: str, args: str, who: str) -> None:
    if "nonext" in args.split():
        _flag(sid).unlink(missing_ok=True)
    else:
        _flag(sid).write_text(who)


def context_line(sid: str, transcript_path: str | None, harness: str = "claude") -> str | None:
    """`[context] 31.2% of 1,000,000 tokens (312,345) · this session: fooPart3` and, near the
    threshold, that the handoff is the hook's job, so the agent keeps working."""
    tokens = context_tokens(transcript_path) if transcript_path else None
    if tokens is None:
        return None
    h = harnesses.get(harness)
    size = window_size(sid, tokens)
    pct, at = 100.0 * tokens / size, threshold(harness)
    line = f"[context] {pct:.1f}% of {size:,} tokens ({tokens:,})"
    title = h.title_of(sid, Path(transcript_path))
    if title:
        line += f" · this session: {title}"
    if pct >= PLAN_FROM * at:
        line += f" · autohandoff takes over at {at:g}% ({threshold_source()}); keep working"
    return line


def launch_line(prev: str, nxt: str, harness: str = "claude") -> str:
    """The command that opens the next part, told to restate and wait rather than start working."""
    return harnesses.get(harness).launch_line(prev, nxt)


def terminal_spawn(prev: str, nxt: str, event: dict, harness: str = "claude") -> tuple[bool, str]:
    """Open the next part beside this session in its terminal (tmux, Ghostty, iTerm2, Terminal.app),
    or print the command where medulla cannot drive the terminal."""
    from medulla import terminals
    h = harnesses.get(harness)
    return terminals.open_beside(h.launch_line(prev, nxt), event.get("cwd") or os.getcwd(), title=prev,
                                 started=h.started_check(prev, nxt), fallback_title=h.pane_fallback_title)


def interrupted(sid: str) -> str | None:
    """An open the hook did not live to finish: confirmed quietly if the next part is running, else the
    command to open it by hand."""
    try:
        rec = json.loads(_opening(sid).read_text())
    except (OSError, ValueError):
        return None
    finally:
        _opening(sid).unlink(missing_ok=True)
    prev, nxt = rec["prev"], rec["next"]
    harness = rec.get("harness", "claude")
    running = _running(nxt) if harness == "claude" else _running(nxt, prev, harness)
    if running:
        _spawned(sid).write_text(nxt)
        _log(sid=sid, action="spawn", ok=True, prev=prev, next=nxt, detail="confirmed after an interrupted open")
        return None
    _log(sid=sid, action="spawn", ok=False, prev=prev, next=nxt, detail="interrupted")
    return (f"autohandoff: opening {nxt} was interrupted before it finished, and it is not running. "
            f"Open it with `medulla nextpart --from {prev}`, or run it yourself: "
            f"cd {shlex.quote(rec['cwd'])} && {launch_line(prev, nxt, harness)}")


def _turn_already_seen(sid: str, turn_key: str | None) -> bool:
    """Dedup hooks that fire multiple times within one user turn (e.g. Antigravity PreInvocation)."""
    if turn_key is None:
        return False
    if turn_key == "__skip__":
        return True
    tf = _turn_file(sid)
    if tf.exists() and tf.read_text() == turn_key:
        return True
    tf.write_text(turn_key)
    return False


def _handle_prompt(ev: HookEvent) -> HookOutcome | None:
    if _turn_already_seen(ev.sid, ev.turn_key):
        return None
    m = _HANDOFF_CMD.fullmatch(ev.prompt)
    if m:
        _mark(ev.sid, m.group(1) or "", "user")
    notice = interrupted(ev.sid)
    context = "\n".join(x for x in (notice, context_line(ev.sid, ev.transcript_path, ev.harness)) if x)
    if not notice and not context:
        return None
    return HookOutcome(system_message=notice, additional_context=context or None)


def _handle_tool(ev: HookEvent) -> None:
    if ev.tool_handoff_args is not None:
        _mark(ev.sid, ev.tool_handoff_args, "skill")


def _handle_stop(ev: HookEvent, raw_event: dict, spawn) -> HookOutcome | None:
    if ev.skip_stop or _spawned(ev.sid).exists() or ev.is_subagent:
        return None
    h = harnesses.get(ev.harness)
    if _flag(ev.sid).exists():
        if not _REPORT.search(ev.last_message or ""):
            return None
        _flag(ev.sid).unlink()
        title = h.title_of(ev.sid, Path(ev.transcript_path) if ev.transcript_path else None)
        if not title:
            _log(sid=ev.sid, action="skip", why="unnamed")
            return HookOutcome(system_message="autohandoff: this session has no /rename title, so no next part "
                                              "was opened. Rename it, then open the next part yourself.")
        nxt, prev = next_names(title)
        rec = {"prev": prev, "next": nxt, "cwd": ev.cwd or os.getcwd()}
        if ev.harness != "claude":
            rec["harness"] = ev.harness
        _opening(ev.sid).write_text(json.dumps(rec))
        _log(sid=ev.sid, action="opening", prev=prev, next=nxt)
        try:
            ok, detail = spawn(prev, nxt, raw_event)
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        _opening(ev.sid).unlink()
        _log(sid=ev.sid, action="spawn", ok=ok, prev=prev, next=nxt, detail=detail)
        if not ok:
            return HookOutcome(system_message=f"autohandoff: could not open {nxt}: {detail} "
                                              f"Or retry with `medulla nextpart --from {prev}`.")
        _spawned(ev.sid).write_text(nxt)
        return HookOutcome(system_message=f"autohandoff: {nxt}: {detail}")
    if ev.continuation_active or not ev.transcript_path:
        return None
    wait_for_reply(ev.transcript_path)
    pct = context_pct(ev.sid, ev.transcript_path)
    at = threshold(ev.harness)
    if pct is None or pct < at:
        return None
    _flag(ev.sid).write_text("hook")
    _log(sid=ev.sid, action="request", pct=round(pct, 1))
    return HookOutcome(decision="continue", reason=NUDGE.format(pct=pct, at=at, src=threshold_source()))


def on_prompt(event: dict) -> dict | None:
    """Claude-compatible prompt hook entry point (also used directly in unit tests)."""
    h = harnesses.detect_event(event) or harnesses.get("claude")
    ev = h.normalize_event({"hook_event_name": "UserPromptSubmit", **event} if "hook_event_name" not in event and "hookEventName" not in event else event)
    return h.format_output("prompt", _handle_prompt(ev)) if ev else None


def on_skill(event: dict) -> None:
    """Claude-compatible tool hook entry point (also used directly in unit tests)."""
    h = harnesses.detect_event(event) or harnesses.get("claude")
    ev = h.normalize_event({"hook_event_name": "PostToolUse", **event} if "hook_event_name" not in event and "hookEventName" not in event else event)
    if ev:
        _handle_tool(ev)


def on_stop(event: dict, spawn) -> dict | None:
    """The Stop decision. `spawn(prev, nxt, event) -> (ok, detail)` opens the next part."""
    h = harnesses.detect_event(event) or harnesses.get("claude")
    ev = h.normalize_event(event)
    return h.format_output("stop", _handle_stop(ev, event, spawn)) if ev else None


def current_session(harness: str | None = None) -> tuple[str | None, str | None]:
    """(id, title) of the active harness session this command runs in."""
    h = harnesses.get(harness) if harness else harnesses.active_harness()
    sid = os.environ.get(h.session_env)
    if not sid:
        return None, None
    return sid, h.title_of(sid, None)


def nextpart(from_title: str | None = None, dry_run: bool = False, spawn=None,
             harness: str | None = None) -> tuple[bool, str]:
    """Open the next part by hand, beside this session or the one titled from_title: after an open that
    failed or was interrupted, or whenever the user wants it now. Continuing this session marks it
    handed off, so autohandoff does not open a second one."""
    h = harnesses.get(harness) if harness else harnesses.active_harness()
    sid, own = current_session(h.name)
    title = from_title or own
    if not title:
        return False, "no session to continue: this one has no /rename title, so pass --from <title>"
    nxt, prev = next_names(title)
    cwd = os.getcwd()
    if dry_run:
        return True, f"would run: cd {shlex.quote(cwd)} && {h.launch_line(prev, nxt)}"
    opener = spawn or (lambda p, n, e: terminal_spawn(p, n, e, harness=h.name))
    ok, detail = opener(prev, nxt, {"cwd": cwd})
    _log(sid=sid, action="spawn", ok=ok, prev=prev, next=nxt, detail=f"nextpart: {detail}")
    if not ok:
        return False, f"could not open {nxt}: {detail}"
    if sid and title == own:
        _spawned(sid).write_text(nxt)
    return True, f"{nxt}: {detail}"


def _dispatch(raw: dict, spawn) -> dict | None:
    h = harnesses.detect_event(raw)
    if h is None:
        return None
    ev = h.normalize_event(raw)
    if ev is None:
        return None
    if ev.kind == "prompt":
        return h.format_output("prompt", _handle_prompt(ev))
    if ev.kind == "tool":
        _handle_tool(ev)
        return h.format_output("tool", None)
    opener = spawn if spawn is not terminal_spawn else (lambda p, n, e: terminal_spawn(p, n, e, harness=h.name))
    return h.format_output("stop", _handle_stop(ev, raw, opener))


def main(stdin=sys.stdin, stdout=sys.stdout, spawn=terminal_spawn) -> None:
    try:
        if not enabled():
            return
        raw = json.load(stdin)
        out = _dispatch(raw, spawn)
        if out is not None:
            stdout.write(json.dumps(out))
    except Exception:
        return


if __name__ == "__main__":  # pragma: no cover  (run by the plugin's hooks, in a subprocess)
    main()
