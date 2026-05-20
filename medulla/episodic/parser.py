"""Parse Claude Code (and stub Kiro/Codex/Gemini) JSONL session files.

Both user AND assistant text blocks are indexed — assistant messages contain
the most valuable content (analysis, findings, code) and must be searchable.
Only tool_use and tool_result blocks are excluded (noisy / already in tool_events).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


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
    all_user_text: str       # full conversation text (user + assistant), no cap
    messages: list[str]      # ordered interleaved user+assistant text, for chunker


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
    all_user_text: str       # full conversation text (user + assistant), no cap
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
    messages: list[str] = []   # interleaved user + assistant text
    has_user_message = False

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

        if role in ("user", "human"):
            text = _extract_user_text(msg)
            if text:
                if first_message is None:
                    first_message = text[:MAX_FIRST_MSG]
                messages.append(text)
                has_user_message = True

        elif role == "assistant":
            turn_count += 1
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "tool_use":
                            tool_call_count += 1
                            name = item.get("name", "")
                            if name:
                                tool_names.add(name)
                            _extract_paths(item.get("input", {}), files)
            # Index assistant text blocks (the analysis, findings, code explanations)
            text = _extract_assistant_text(msg)
            if text:
                messages.append(text)

        if model is None and "model" in node:
            model = node["model"]

    if not has_user_message:
        return None

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
        all_user_text=" ".join(messages),
        messages=messages,
    )


def _parse_agent_claude(path: Path, body: str) -> ParsedAgentSession | None:
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
    messages: list[str] = []
    message_count = 0
    has_user_message = False

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
            text = _extract_user_text(msg)
            if text:
                if first_message is None:
                    first_message = text[:MAX_FIRST_MSG]
                messages.append(text)
                has_user_message = True

        elif role == "assistant":
            turn_count += 1
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_call_count += 1
                        name = item.get("name", "")
                        if name:
                            tool_names.add(name)
            text = _extract_assistant_text(msg)
            if text:
                messages.append(text)

        if model is None and "model" in node:
            model = node["model"]

    if not has_user_message:
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
        all_user_text=" ".join(messages),
        message_count=message_count,
        first_seen_at=first_seen_at,
        last_updated_at=last_updated_at,
    )


# ── Text extraction ───────────────────────────────────────────────────────────

def _extract_user_text(msg: dict) -> str:
    """Extract text from a user message. Skips tool_result blocks (noisy)."""
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
                # skip tool_result — raw tool output, not human intent
        return " ".join(p for p in parts if p).strip()
    return ""


def _extract_assistant_text(msg: dict) -> str:
    """Extract text blocks from an assistant message. Skips tool_use blocks."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            # skip tool_use (captured as tool_events) and tool_result
        return " ".join(p for p in parts if p).strip()
    return ""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_json(line: str) -> dict | None:
    try:
        obj = json.loads(line)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


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
