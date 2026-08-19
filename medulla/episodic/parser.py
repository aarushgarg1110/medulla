"""Parse Claude Code (and stub Kiro/Codex/Gemini) JSONL session files.

Both user AND assistant text blocks are indexed — assistant messages contain
the most valuable content (analysis, findings, code) and must be searchable.
Only tool_use and tool_result blocks are excluded (noisy / already in tool_events).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from medulla.episodic.scrub import scrub_secrets


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


@dataclass
class ToolEvent:
    session_id: str
    project_dir: str | None
    event_ts: str | None
    tool: str
    command: str
    description: str
    output_preview: str
    is_error: bool
    event_hash: str
    interrupted: bool = False


def is_subagent_file(path: Path) -> bool:
    return bool(SUBAGENT_PATH_RE.search(str(path)))


# ── tool-event extraction (backfill of command history) ─────────────────────────

_TRIVIAL_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")


def _derive_command(name: str, inp: object) -> str:
    """Best-effort human-meaningful command string for any tool."""
    if not isinstance(inp, dict):
        return f"{name} {str(inp)[:200]}".strip()
    cmd = inp.get("command")
    if isinstance(cmd, str) and cmd.strip():
        return cmd
    for k in ("file_path", "path", "pattern", "query", "url", "notebook_path"):
        v = inp.get(k)
        if isinstance(v, str) and v.strip():
            return f"{name} {v}"
    return f"{name} {json.dumps(inp, default=str)[:200]}"


def _is_trivial_command(cmd: str) -> bool:
    """Single-line `cd …` or bare VAR=value — low recall value, skip."""
    lines = [ln for ln in cmd.strip().splitlines() if ln.strip()]
    if len(lines) != 1:
        return False  # multi-line commands are substantive (kept)
    ln = lines[0].strip()
    return ln.startswith("cd ") or bool(_TRIVIAL_ASSIGN_RE.match(ln))


def _result_output(block: dict, tool_use_result: object) -> str:
    if isinstance(tool_use_result, dict):
        return str(tool_use_result.get("stdout") or tool_use_result.get("stderr") or "")
    return _block_text(block)


def _block_text(block: dict) -> str:
    content = block.get("content")
    if isinstance(content, list):
        return " ".join(x.get("text", "") for x in content if isinstance(x, dict))
    return str(content or "")


_BG_ID_RE = re.compile(r"running in background with ID:\s*(\w+)")
_TASK_ID_RE = re.compile(r"<task-id>(\w+)</task-id>")
_EXIT_RE = re.compile(r"exit code (\d+)")
_ABORT_MARKERS = (
    "tool use was rejected", "user doesn't want to proceed",
    "[Request interrupted", "Request interrupted by user",
)


def _looks_aborted(text: str) -> bool:
    return any(m in text for m in _ABORT_MARKERS)


def _task_notifications(nodes: list[dict]) -> dict[str, bool]:
    """task_id → is_error, parsed from <task-notification> messages (exit code != 0)."""
    out: dict[str, bool] = {}
    for node in nodes:
        msg = node.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else None
        texts: list[str] = []
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts += [b.get("text", "") for b in content if isinstance(b, dict)]
        for t in texts:
            if "task-notification" not in t:
                continue
            tid = _TASK_ID_RE.search(t)
            if tid:
                ec = _EXIT_RE.search(t)
                out[tid.group(1)] = bool(ec and ec.group(1) != "0")
    return out


def extract_tool_events(path: Path) -> list[ToolEvent]:
    """Harvest tool_use/tool_result pairs from a session JSONL into ToolEvents.

    Outcomes are corrected beyond the raw tool_result:
      - background/Monitor commands take their real exit status from the matching
        <task-notification> (joined by task-id);
      - user-aborted/rejected/interrupted calls are flagged `interrupted` and NOT
        counted as `is_error` (an abort is a choice, not a fixable failure).
    Commands/outputs are secret-scrubbed; trivial `cd`/assignments are dropped;
    subagent files are skipped (v1).
    """
    if is_subagent_file(path):
        return []
    try:
        body = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    nodes = [n for n in (_read_json(ln.strip()) for ln in body.splitlines() if ln.strip()) if n]
    notifs = _task_notifications(nodes)

    session_id = path.stem
    uses: dict[str, tuple[str, object, str | None, str | None]] = {}
    events: list[ToolEvent] = []

    for node in nodes:
        if node.get("sessionId"):
            session_id = node["sessionId"]
        ts = node.get("timestamp")
        project = node.get("cwd")
        tool_use_result = node.get("toolUseResult")

        msg = node.get("message", {})
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            btype = block.get("type")
            if btype == "tool_use":
                uses[block.get("id")] = (block.get("name", ""), block.get("input", {}), ts, project)
            elif btype == "tool_result":
                tid = block.get("tool_use_id")
                if tid not in uses:
                    continue
                name, inp, use_ts, use_proj = uses.pop(tid)
                cmd = _derive_command(name, inp)
                if _is_trivial_command(cmd):
                    continue
                desc = inp.get("description", "") if isinstance(inp, dict) else ""
                result_text = _block_text(block)

                is_error = bool(block.get("is_error"))
                interrupted = bool(isinstance(tool_use_result, dict)
                                   and tool_use_result.get("interrupted"))
                if _looks_aborted(result_text):
                    interrupted = True

                bg = _BG_ID_RE.search(result_text)
                if bg:
                    # the "started" result says nothing — real outcome is the notification
                    is_error = notifs.get(bg.group(1), False)
                    interrupted = False
                if interrupted:
                    is_error = False   # an abort is not a failure to learn from

                events.append(ToolEvent(
                    session_id=session_id,
                    project_dir=use_proj,
                    event_ts=use_ts,
                    tool=name,
                    command=scrub_secrets(cmd)[:500],
                    description=(desc or "")[:200],
                    output_preview=scrub_secrets(_result_output(block, tool_use_result))[:200],
                    is_error=is_error,
                    interrupted=interrupted,
                    event_hash=hashlib.sha256(f"{session_id}:{tid}".encode()).hexdigest(),
                ))
    return events


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
    if _is_kiro_format(text):
        return _parse_kiro(path, slug, text)
    return _parse_claude(path, slug, text)


def _is_kiro_format(body: str) -> bool:
    """Detect Kiro JSONL format by checking first non-empty line."""
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        node = _read_json(line)
        if node and node.get("version") == "v1" and "kind" in node:
            return True
        return False  # first parseable line is not Kiro
    return False


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
    custom_title: str | None = None
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
        if node.get("type") == "custom-title" and node.get("customTitle"):
            custom_title = node["customTitle"]

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
        slug=custom_title or slug,
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


# ── Kiro format ───────────────────────────────────────────────────────────────

def _parse_kiro(path: Path, slug: str, body: str) -> ParsedSession | None:
    """Parse Kiro CLI session JSONL.

    Format per line:
      kind: "Prompt"          → user message (data.content[].kind=="text")
      kind: "AssistantMessage" → assistant response + tool calls
      kind: "ToolResults"     → tool outputs (skipped)
    """
    from datetime import datetime, timezone

    session_id = path.stem
    started_at: str | None = None
    ended_at: str | None = None
    turn_count = 0
    tool_call_count = 0
    tool_names: set[str] = set()
    first_message: str | None = None
    messages: list[str] = []
    has_user_message = False

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        node = _read_json(line)
        if node is None or node.get("version") != "v1":
            continue

        kind = node.get("kind", "")
        data = node.get("data", {})

        # Timestamp
        ts_unix = data.get("meta", {}).get("timestamp")
        if ts_unix:
            ts = datetime.fromtimestamp(ts_unix, tz=timezone.utc).isoformat()
            if started_at is None:
                started_at = ts
            ended_at = ts

        content = data.get("content", [])

        if kind == "Prompt":
            # User message
            text_parts = [
                c["data"] for c in content
                if isinstance(c, dict) and c.get("kind") == "text" and c.get("data")
            ]
            text = _sanitize_text(" ".join(text_parts))
            if text:
                if first_message is None:
                    first_message = text[:MAX_FIRST_MSG]
                messages.append(text)
                has_user_message = True

        elif kind == "AssistantMessage":
            turn_count += 1
            # Extract text blocks
            text_parts = [
                c["data"] for c in content
                if isinstance(c, dict) and c.get("kind") == "text" and c.get("data")
            ]
            text = _sanitize_text(" ".join(text_parts))
            if text:
                messages.append(text)
            # Extract tool calls
            for c in content:
                if isinstance(c, dict) and c.get("kind") == "toolUse":
                    tool_call_count += 1
                    name = c.get("data", {}).get("name", "")
                    if name:
                        tool_names.add(name)

    if not has_user_message:
        return None

    return ParsedSession(
        session_id=session_id,
        source="kiro",
        project_dir=None,   # Kiro CLI sessions don't expose cwd
        git_branch=None,
        slug=slug,
        model=None,
        started_at=started_at,
        ended_at=ended_at,
        turn_count=turn_count,
        tool_call_count=tool_call_count,
        tool_names=sorted(tool_names),
        files=[],
        first_message=first_message,
        all_user_text=" ".join(messages),
        messages=messages,
    )


# ── Text extraction ───────────────────────────────────────────────────────────

# ── conversation-text sanitization ──────────────────────────────────────────────
# Machine boilerplate that lands inside kept user/assistant text and pollutes
# chunks/embeddings/excerpts. Stripped at parse time; a message that is entirely
# boilerplate collapses to "" and is dropped from the messages list.
_SANITIZE_RES = [
    __import__("re").compile(p, __import__("re").DOTALL) for p in (
        r"<local-command-caveat>.*?</local-command-caveat>",
        r"<command-name>.*?</command-name>",
        r"<command-message>.*?</command-message>",
        r"<command-args>.*?</command-args>",
        r"<local-command-stdout>.*?</local-command-stdout>",
        r"<system-reminder>.*?</system-reminder>",
        r"\[Image #\d+\]",
        r"\[Image: source:[^\]]*\]",
    )
]


def _sanitize_text(text: str) -> str:
    """Strip machine boilerplate; preserve real content (incl. internal newlines)."""
    if not text:
        return text
    for r in _SANITIZE_RES:
        text = r.sub("", text)
    text = re.sub(r"[ \t]+\n", "\n", text)     # trailing spaces on lines
    text = re.sub(r"\n{3,}", "\n\n", text)     # collapse gaps left by removed blocks
    return text.strip()


def _extract_user_text(msg: dict) -> str:
    """Extract text from a user message. Skips tool_result blocks (noisy)."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _sanitize_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                # skip tool_result — raw tool output, not human intent
        return _sanitize_text(" ".join(p for p in parts if p))
    return ""


def _extract_assistant_text(msg: dict) -> str:
    """Extract text blocks from an assistant message. Skips tool_use blocks."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return _sanitize_text(content)
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(item.get("text", ""))
            # skip tool_use (captured as tool_events) and tool_result
        return _sanitize_text(" ".join(p for p in parts if p))
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
