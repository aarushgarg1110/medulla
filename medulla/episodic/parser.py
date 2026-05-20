"""Parse Claude Code (and stub Kiro/Codex/Gemini) JSONL session files."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


MAX_FIRST_MSG = 500
SUBAGENT_PATH_RE = re.compile(r"/subagents/agent-[^/]+\.jsonl$")


@dataclass
class ParsedSession:
    session_id: str
    source: str  # claude | kiro | codex | gemini
    project_dir: str | None
    git_branch: str | None
    slug: str | None
    model: str | None
    started_at: str | None
    ended_at: str | None
    turn_count: int
    tool_call_count: int
    tool_names: list[str]
    files: list[str]
    first_message: str | None
    all_user_text: str  # complete, no cap
    user_messages: list[str]  # ordered, for chunker


@dataclass
class ParsedAgentSession:
    agent_id: str
    parent_session_id: str | None
    agent_slug: str | None
    project_dir: str | None
    cwd: str | None
    model: str | None
    turn_count: int
    tool_call_count: int
    tool_names: list[str]
    first_message: str | None
    all_user_text: str
    message_count: int
    first_seen_at: str | None
    last_updated_at: str | None


def is_subagent_file(path: Path) -> bool:
    return bool(SUBAGENT_PATH_RE.search(str(path)))


def parse_session(path: Path) -> ParsedSession | None:
    """Detect format and parse."""
    if is_subagent_file(path):
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    if not text.strip():
        return None

    slug = path.stem
    return _parse_claude(path, slug, text)


def parse_agent_session(path: Path) -> ParsedAgentSession | None:
    """Parse a subagent JSONL file."""
    if not is_subagent_file(path):
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    return _parse_agent_claude(path, text)


# ── Claude format ─────────────────────────────────────────────────────────────

def _parse_claude(path: Path, slug: str, body: str) -> ParsedSession | None:
    session_id: str = path.stem
    project_dir: str | None = None
    git_branch: str | None = None
    model: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    turn_count = 0
    tool_call_count = 0
    tool_names: set[str] = set()
    files: set[str] = set()
    first_message: str | None = None
    user_messages: list[str] = []

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        node = _read_json(line)
        if node is None:
            continue

        if project_dir is None and "cwd" in node:
            project_dir = node["cwd"]
        if "sessionId" in node and node["sessionId"]:
            session_id = node["sessionId"]
        if git_branch is None and "gitBranch" in node:
            git_branch = node["gitBranch"]

        ts = node.get("timestamp")
        if ts:
            if started_at is None:
                started_at = ts
            ended_at = ts

        msg = node.get("message", {})
        role = msg.get("role") or node.get("type", "")

        # User / human turn
        if role in ("user", "human"):
            text = _extract_text(msg)
            if text:
                if first_message is None:
                    first_message = text[:MAX_FIRST_MSG]
                user_messages.append(text)

        # Assistant turn
        if role == "assistant":
            turn_count += 1
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_call_count += 1
                        name = item.get("name", "")
                        if name:
                            tool_names.add(name)
                        _extract_paths(item.get("input", {}), files)

        # Model field
        if model is None and "model" in node:
            model = node["model"]

    if not user_messages:
        return None

    all_user_text = " ".join(user_messages)

    return ParsedSession(
        session_id=session_id,
        source="claude",
        project_dir=project_dir,
        git_branch=git_branch,
        slug=slug,
        model=model,
        started_at=started_at,
        ended_at=ended_at,
        turn_count=turn_count,
        tool_call_count=tool_call_count,
        tool_names=sorted(tool_names),
        files=sorted(files),
        first_message=first_message,
        all_user_text=all_user_text,
        user_messages=user_messages,
    )


def _parse_agent_claude(path: Path, body: str) -> ParsedAgentSession | None:
    # agent-<agentId>.jsonl inside .../subagents/
    agent_id = path.stem.removeprefix("agent-")
    parent_session_id: str | None = None
    agent_slug: str | None = None
    project_dir: str | None = None
    cwd: str | None = None
    model: str | None = None
    first_seen_at: str | None = None
    last_updated_at: str | None = None
    turn_count = 0
    tool_call_count = 0
    tool_names: set[str] = set()
    first_message: str | None = None
    user_messages: list[str] = []
    message_count = 0

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        node = _read_json(line)
        if node is None:
            continue

        if "sessionId" in node and node["sessionId"]:
            parent_session_id = node["sessionId"]
        if cwd is None and "cwd" in node:
            cwd = node["cwd"]
        if project_dir is None and "cwd" in node:
            project_dir = str(Path(node["cwd"]).parent)

        ts = node.get("timestamp")
        if ts:
            if first_seen_at is None:
                first_seen_at = ts
            last_updated_at = ts

        msg = node.get("message", {})
        role = msg.get("role") or node.get("type", "")
        message_count += 1

        if role in ("user", "human"):
            text = _extract_text(msg)
            if text:
                if first_message is None:
                    first_message = text[:MAX_FIRST_MSG]
                user_messages.append(text)

        if role == "assistant":
            turn_count += 1
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_call_count += 1
                        name = item.get("name", "")
                        if name:
                            tool_names.add(name)

        if model is None and "model" in node:
            model = node["model"]

    if not user_messages:
        return None

    return ParsedAgentSession(
        agent_id=agent_id,
        parent_session_id=parent_session_id,
        agent_slug=agent_slug,
        project_dir=project_dir,
        cwd=cwd,
        model=model,
        turn_count=turn_count,
        tool_call_count=tool_call_count,
        tool_names=sorted(tool_names),
        first_message=first_message,
        all_user_text=" ".join(user_messages),
        message_count=message_count,
        first_seen_at=first_seen_at,
        last_updated_at=last_updated_at,
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(line: str) -> dict | None:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_text(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif item.get("type") == "tool_result":
                    # skip tool outputs in user messages
                    pass
        return " ".join(p for p in parts if p).strip()
    return ""


_PATH_RE = re.compile(r"[\"']?(/(?:[^\"',\s\])}]+))[\"']?")

def _extract_paths(obj: object, acc: set[str]) -> None:
    if isinstance(obj, str):
        for m in _PATH_RE.finditer(obj):
            p = m.group(1)
            if len(p) > 3 and not p.startswith("/proc"):
                acc.add(p)
    elif isinstance(obj, dict):
        for v in obj.values():
            _extract_paths(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            _extract_paths(v, acc)
