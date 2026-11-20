"""Agent harness registry: Claude Code and Antigravity CLI (`agy` / "gemini") share one interface.

Every harness difference — settings path, hook JSON contract, transcript token/role layout, title
lookup, next-part launch command, process check, and Ghostty fallback pane title — lives here as a
registry entry in `HARNESSES`. Orchestration (`autohandoff`, `statusline_setup`, `cli`) looks up a
`Harness` by name or event shape instead of branching on harness strings.
"""
from __future__ import annotations

import json
import os
import shlex
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from medulla import statusline, terminals

_TAIL_BYTES = 4 << 20
_PICKOFF_NOTE = "opened automatically: restate, then wait for my reply before doing any work"


@dataclass(frozen=True)
class HookEvent:
    harness: str
    kind: str                              # "prompt" | "tool" | "stop"
    sid: str
    transcript_path: str | None = None
    cwd: str | None = None
    prompt: str = ""
    tool_handoff_args: str | None = None   # non-None when PostToolUse invoked the handoff skill
    continuation_active: bool = False
    is_subagent: bool = False
    last_message: str | None = None
    turn_key: str | None = None            # dedup key for hooks firing multiple times per user turn
    skip_stop: bool = False


@dataclass(frozen=True)
class HookOutcome:
    decision: str | None = None            # "continue" -> block stop and nudge handoff
    reason: str | None = None
    system_message: str | None = None
    additional_context: str | None = None


@dataclass(frozen=True)
class Harness:
    name: str
    session_env: str
    default_threshold_pct: float
    settings_path: Callable[[], Path]
    pane_fallback_title: str | None
    launch_line: Callable[[str, str], str]
    started_check: Callable[[str, str], Callable[[], bool]]
    title_of: Callable[[str, Path | None], str | None]
    entry_usage: Callable[[dict], int | None]
    entry_role: Callable[[dict], str | None]
    normalize_event: Callable[[dict], HookEvent | None]
    format_output: Callable[[str, HookOutcome | None], dict | None]


# ── Claude Code ───────────────────────────────────────────────────────────────

def _claude_settings() -> Path:
    return Path(os.environ.get("CLAUDE_CONFIG_DIR") or Path.home() / ".claude") / "settings.json"


def _claude_launch(prev: str, nxt: str) -> str:
    pickoff = f"/medulla:pickoff {prev} {_PICKOFF_NOTE}"
    return f"claude -n {shlex.quote(nxt)} {shlex.quote(pickoff)}"


def _claude_started(prev: str, nxt: str) -> Callable[[], bool]:
    return terminals.claude_started(nxt)


def _claude_title(sid: str, path: Path | None) -> str | None:
    from medulla.episodic import scanner
    target = path
    if target is None:
        target = next(scanner.CLAUDE_PROJECTS_DIR.glob(f"*/{sid}.jsonl"), None)
    return scanner.title_of(target) if target and target.exists() else None


def _claude_usage(e: dict) -> int | None:
    msg = e.get("message")
    if e.get("type") != "assistant" or e.get("isSidechain") or not isinstance(msg, dict):
        return None
    u = msg.get("usage")
    if not u:
        return None
    return sum(u.get(k) or 0 for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"))


def _claude_role(e: dict) -> str | None:
    t = e.get("type")
    if t in ("user", "assistant") and not e.get("isSidechain"):
        return t
    return None


def _claude_normalize(raw: dict) -> HookEvent | None:
    name = raw.get("hook_event_name")
    sid = raw.get("session_id") or ""
    if name == "UserPromptSubmit":
        return HookEvent("claude", "prompt", sid, raw.get("transcript_path"), raw.get("cwd"),
                         prompt=raw.get("prompt") or "")
    if name == "PostToolUse":
        inp = raw.get("tool_input") or {}
        args = (inp.get("args") or "") if (
            raw.get("tool_name") == "Skill" and inp.get("skill") in ("handoff", "medulla:handoff")
        ) else None
        return HookEvent("claude", "tool", sid, raw.get("transcript_path"), raw.get("cwd"),
                         tool_handoff_args=args)
    if name == "Stop":
        return HookEvent("claude", "stop", sid, raw.get("transcript_path"), raw.get("cwd"),
                         continuation_active=bool(raw.get("stop_hook_active")),
                         is_subagent=bool(raw.get("agent_id")),
                         last_message=raw.get("last_assistant_message") or "")
    return None


def _claude_format(kind: str, outcome: HookOutcome | None) -> dict | None:
    if outcome is None or kind == "tool":
        return None
    if kind == "prompt":
        out: dict = {}
        if outcome.system_message:
            out["systemMessage"] = outcome.system_message
        if outcome.additional_context:
            out["hookSpecificOutput"] = {
                "hookEventName": "UserPromptSubmit",
                "additionalContext": outcome.additional_context,
            }
        return out or None
    if outcome.decision == "continue":
        return {"decision": "block", "reason": outcome.reason}
    return {"systemMessage": outcome.system_message} if outcome.system_message else None


# ── Antigravity CLI ("gemini") ────────────────────────────────────────────────

def _agy_dir() -> Path:
    return Path(os.environ.get("ANTIGRAVITY_CONFIG_DIR")
                or os.environ.get("ANTIGRAVITY_APP_DATA_DIR")
                or Path.home() / ".gemini" / "antigravity-cli")


def _gemini_settings() -> Path:
    return _agy_dir() / "settings.json"


def _gemini_launch(prev: str, nxt: str) -> str:
    pickoff = f"/pickoff {prev} {_PICKOFF_NOTE} (suggest `/rename {nxt}`)"
    return f"agy -i {shlex.quote(pickoff)}"


def _gemini_started(prev: str, nxt: str) -> Callable[[], bool]:
    return terminals.agy_started(prev)


def _gemini_title(sid: str, path: Path | None) -> str | None:
    try:
        name = json.loads((Path(statusline.state_dir()) / f"{sid}.ctx.json").read_text()).get("name")
        if name:
            return str(name)
    except (OSError, ValueError, AttributeError):
        pass
    db = _agy_dir() / "conversation_summaries.db"
    if not db.exists():
        return None
    try:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = conn.execute("SELECT title FROM conversation_summaries WHERE conversation_id = ?", (sid,)).fetchone()
            return row[0] if row and row[0] else None
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _gemini_usage(e: dict) -> int | None:
    if e.get("type") != "PLANNER_RESPONSE":
        return None
    if e.get("input_tokens") is None and e.get("cache_read_tokens") is None:
        return None
    return int(e.get("input_tokens") or 0) + int(e.get("cache_read_tokens") or 0)


def _gemini_role(e: dict) -> str | None:
    t = e.get("type")
    if t == "USER_INPUT":
        return "user"
    if t == "PLANNER_RESPONSE":
        return "assistant"
    return None


def _last_gemini_reply(transcript_path: str | None) -> str:
    if not transcript_path:
        return ""
    try:
        with open(transcript_path, "rb") as fh:
            size = fh.seek(0, os.SEEK_END)
            fh.seek(max(0, size - _TAIL_BYTES))
            if size > _TAIL_BYTES:
                fh.readline()
            last = ""
            for line in fh:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("type") == "PLANNER_RESPONSE" and e.get("content"):
                    last = str(e["content"])
            return last
    except OSError:
        return ""


def _gemini_normalize(raw: dict) -> HookEvent | None:
    name = raw.get("hookEventName")
    sid = raw.get("conversationId") or ""
    tp = raw.get("transcriptPath")
    cwd = raw.get("cwd")
    if name == "PreInvocation":
        turn_key = str(raw["initialNumSteps"]) if "initialNumSteps" in raw else (
            None if raw.get("invocationNum", 1) == 1 else "__skip__"
        )
        return HookEvent("gemini", "prompt", sid, tp, cwd,
                         prompt=raw.get("lastUserInput") or "", turn_key=turn_key)
    if name == "PostToolUse":
        tc = raw.get("toolCall") or {}
        path = str((tc.get("args") or {}).get("Path") or "")
        args = "" if tc.get("name") == "read_file" and path.endswith("/skills/handoff/SKILL.md") else None
        return HookEvent("gemini", "tool", sid, tp, cwd, tool_handoff_args=args)
    if name == "Stop":
        reason = raw.get("terminationReason") or "NO_TOOL_CALL"
        return HookEvent("gemini", "stop", sid, tp, cwd,
                         continuation_active=bool(raw.get("continuationActive")),
                         last_message=_last_gemini_reply(tp),
                         skip_stop=(reason != "NO_TOOL_CALL"))
    return None


def _gemini_format(kind: str, outcome: HookOutcome | None) -> dict:
    if kind == "tool" or outcome is None:
        return {}
    if kind == "prompt":
        if not outcome.additional_context:
            return {}
        return {"injectSteps": [{"systemMessage": {"systemMessage": outcome.additional_context}}]}
    if outcome.decision == "continue":
        return {"decision": "continue", "reason": outcome.reason}
    return {}


# ── Registry ──────────────────────────────────────────────────────────────────

HARNESSES: dict[str, Harness] = {
    "claude": Harness(
        name="claude",
        session_env="CLAUDE_CODE_SESSION_ID",
        default_threshold_pct=93.0,
        settings_path=_claude_settings,
        pane_fallback_title=None,
        launch_line=_claude_launch,
        started_check=_claude_started,
        title_of=_claude_title,
        entry_usage=_claude_usage,
        entry_role=_claude_role,
        normalize_event=_claude_normalize,
        format_output=_claude_format,
    ),
    "gemini": Harness(
        name="gemini",
        session_env="ANTIGRAVITY_CONVERSATION_ID",
        default_threshold_pct=45.0,
        settings_path=_gemini_settings,
        pane_fallback_title="agy",
        launch_line=_gemini_launch,
        started_check=_gemini_started,
        title_of=_gemini_title,
        entry_usage=_gemini_usage,
        entry_role=_gemini_role,
        normalize_event=_gemini_normalize,
        format_output=_gemini_format,
    ),
}


def get(name: str) -> Harness:
    if name not in HARNESSES:
        raise ValueError(f"unknown harness: {name!r} (expected {', '.join(HARNESSES)})")
    return HARNESSES[name]


def detect_event(raw: dict) -> Harness | None:
    if "hookEventName" in raw:
        return HARNESSES["gemini"]
    if "hook_event_name" in raw:
        return HARNESSES["claude"]
    return None


def active_harness(env=None) -> Harness:
    env = os.environ if env is None else env
    for h in HARNESSES.values():
        if env.get(h.session_env):
            return h
    return HARNESSES["claude"]


def entry_usage(entry: dict) -> int | None:
    for h in HARNESSES.values():
        val = h.entry_usage(entry)
        if val is not None:
            return val
    return None


def entry_role(entry: dict) -> str | None:
    for h in HARNESSES.values():
        val = h.entry_role(entry)
        if val is not None:
            return val
    return None
