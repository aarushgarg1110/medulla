"""Parse Claude Code (and stub Kiro/Codex/Gemini) JSONL session files.

Both user AND assistant text blocks are indexed — assistant messages contain
the most valuable content (analysis, findings, code) and must be searchable.
Only tool_use and tool_result blocks are excluded (noisy / already in tool_events).
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from medulla.episodic.scrub import scrub_secrets


MAX_FIRST_MSG = 500
SUBAGENT_PATH_RE = re.compile(r"/subagents/agent-[^/]+\.jsonl$")
GEMINI_TRANSCRIPT_RE = re.compile(r"/brain/([^/]+)/\.system_generated/logs/transcript\.jsonl$")


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


def is_gemini_transcript(path: Path) -> bool:
    return bool(GEMINI_TRANSCRIPT_RE.search(str(path)))


def gemini_session_id(path: Path) -> str:
    m = GEMINI_TRANSCRIPT_RE.search(str(path))
    return m.group(1) if m else path.stem


def gemini_summary_meta(path: Path, sid: str) -> dict:
    """Read title, project_dir, parent_conversation_id and nesting_depth from Antigravity's
    `conversation_summaries.db` (read-only). Fails open to {} when absent or unreadable."""
    if not is_gemini_transcript(path):
        return {}
    db = path.parents[4] / "conversation_summaries.db"
    try:
        if not db.exists():
            return {}
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            row = conn.execute(
                "SELECT title, workspace_uris, parent_conversation_id, nesting_depth "
                "FROM conversation_summaries WHERE conversation_id = ?",
                (sid,),
            ).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return {}
    if not row:
        return {}
    title, uris_raw, parent_id, depth = row
    project_dir = None
    try:
        uris = json.loads(uris_raw) if uris_raw else []
        if isinstance(uris, list) and uris and isinstance(uris[0], str):
            project_dir = unquote(uris[0].removeprefix("file://"))
    except ValueError:
        pass
    return {
        "title": title or None,
        "project_dir": project_dir,
        "parent_conversation_id": parent_id or None,
        "nesting_depth": int(depth or 0),
    }


# ── tool-event extraction (backfill of command history) ─────────────────────────

_TRIVIAL_ASSIGN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S*$")


def _derive_command(name: str, inp: object) -> str:
    """Best-effort human-meaningful command string for any tool."""
    if not isinstance(inp, dict):
        return f"{name} {str(inp)[:200]}".strip()
    cmd = inp.get("command") or inp.get("CommandLine")
    if isinstance(cmd, str) and cmd.strip():
        return cmd
    for k in ("file_path", "Path", "path", "pattern", "query", "url", "notebook_path"):
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
    "user denied permission", "Tool execution was canceled",
)
_AGY_BG_ID_RE = re.compile(r"background task with task id:\s*(\S+)")
_AGY_TASK_DONE_RE = re.compile(r'Task id "([^"]+)" (?:finished with result:|was canceled)')
_AGY_EXIT_RE = re.compile(r"Command exited with code (\d+)")
_USER_REQUEST_RE = re.compile(r"<USER_REQUEST>\s*(.*?)\s*</USER_REQUEST>", re.DOTALL)
_AGY_STRIP_RES = [
    re.compile(r"<ADDITIONAL_METADATA>.*?</ADDITIONAL_METADATA>", re.DOTALL),
    re.compile(r"<USER_SETTINGS_CHANGE>.*?</USER_SETTINGS_CHANGE>", re.DOTALL),
]


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
    if is_gemini_transcript(path) or _is_gemini_nodes(nodes):
        return _extract_gemini_tool_events(path, nodes)
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
    if is_gemini_transcript(path) or _is_gemini_format(text):
        return _parse_gemini(path, text)
    if _is_kiro_format(text):
        return _parse_kiro(path, slug, text)
    return _parse_claude(path, slug, text)


def _is_gemini_nodes(nodes: list[dict]) -> bool:
    return bool(nodes) and "step_index" in nodes[0] and nodes[0].get("type") in (
        "USER_INPUT", "PLANNER_RESPONSE", "GENERIC", "SYSTEM_MESSAGE",
    )


def _is_gemini_format(body: str) -> bool:
    for line in body.splitlines():
        line = line.strip()
        if not line:
            continue
        node = _read_json(line)
        return bool(node) and "step_index" in node and node.get("type") in (
            "USER_INPUT", "PLANNER_RESPONSE", "GENERIC", "SYSTEM_MESSAGE",
        )
    return False


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
    """Parse a subagent JSONL file (Claude subagent or Antigravity nested conversation)."""
    gemini = is_gemini_transcript(path)
    if not is_subagent_file(path) and not gemini:
        return None
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if not text.strip():
        return None
    if gemini:
        return _parse_agent_gemini(path, text)
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
    has_reply = False

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
        role = _turn_role(node)

        if role in ("user", "human"):
            text = _extract_user_text(msg)
            if text and not _is_skill_body(node, text):
                if first_message is None:
                    first_message = text[:MAX_FIRST_MSG]
                messages.append(text)
                has_user_message = True

        elif role == "assistant":
            turn_count += 1
            # Claude Code writes a failed request (502, bad model name) as a "<synthetic>" reply
            if msg.get("model") != "<synthetic>":
                has_reply = True
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

    if not has_user_message or not has_reply:
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
        role = _turn_role(node)
        message_count += 1

        if role in ("user", "human"):
            text = _extract_user_text(msg)
            if text and not _is_skill_body(node, text):
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


# ── Gemini / Antigravity CLI format ───────────────────────────────────────────

def _unquote_agy_args(args: object) -> dict:
    """Antigravity writes tool_calls[].args values as JSON-encoded strings (`"\"ls -la\""`)."""
    if not isinstance(args, dict):
        return {}
    out = {}
    for k, v in args.items():
        if isinstance(v, str):
            try:
                out[k] = json.loads(v)
            except ValueError:
                out[k] = v
        else:
            out[k] = v
    return out


def _gemini_tool_name(name: str, args: dict) -> str:
    if name == "call_mcp_tool" and args.get("ToolName"):
        srv = args.get("ServerName") or ""
        return f"mcp:{srv}/{args['ToolName']}" if srv else str(args["ToolName"])
    return name


def _extract_gemini_user_text(content: str) -> str:
    if not content:
        return ""
    m = _USER_REQUEST_RE.search(content)
    text = m.group(1) if m else content
    if not m:
        for rx in _AGY_STRIP_RES:
            text = rx.sub("", text)
    return _sanitize_text(text)


def _agy_bg_outcomes(nodes: list[dict]) -> dict[str, tuple[bool, bool]]:
    """task_id -> (is_error, interrupted) parsed from SYSTEM_MESSAGE completion notifications."""
    out: dict[str, tuple[bool, bool]] = {}
    for node in nodes:
        if node.get("type") != "SYSTEM_MESSAGE":
            continue
        content = str(node.get("content") or "")
        m = _AGY_TASK_DONE_RE.search(content)
        if not m:
            continue
        canceled = "was canceled" in content
        ec = _AGY_EXIT_RE.search(content)
        is_error = (not canceled) and bool(ec and ec.group(1) != "0")
        out[m.group(1)] = (is_error, canceled)
    return out


def _extract_gemini_tool_events(path: Path, nodes: list[dict]) -> list[ToolEvent]:
    sid = gemini_session_id(path)
    project_dir = gemini_summary_meta(path, sid).get("project_dir")
    bg_outcomes = _agy_bg_outcomes(nodes)
    events: list[ToolEvent] = []

    for i, node in enumerate(nodes):
        if node.get("type") != "PLANNER_RESPONSE":
            continue
        tcs = node.get("tool_calls") or []
        if not isinstance(tcs, list) or not tcs:
            continue
        ts = node.get("created_at")
        step_idx = node.get("step_index", i)

        # Pair ONLY with consecutive GENERIC steps immediately following this PLANNER_RESPONSE:
        # when a user denies permission on call #1 of a parallel batch, agy omits sibling GENERIC steps.
        results: list[dict] = []
        j = i + 1
        while j < len(nodes) and len(results) < len(tcs) and nodes[j].get("type") == "GENERIC":
            results.append(nodes[j])
            j += 1

        for idx, (tc, res) in enumerate(zip(tcs, results)):
            if not isinstance(tc, dict):
                continue
            args = _unquote_agy_args(tc.get("args"))
            tool = _gemini_tool_name(str(tc.get("name") or ""), args)
            cmd = _derive_command(tool, args)
            if _is_trivial_command(cmd):
                continue
            desc = str(args.get("toolSummary") or args.get("description") or "")
            res_content = str(res.get("content") or "")
            err_text = str(res.get("error") or "")

            bg = _AGY_BG_ID_RE.search(res_content)
            if bg and bg.group(1) in bg_outcomes:
                is_error, interrupted = bg_outcomes[bg.group(1)]
            else:
                interrupted = _looks_aborted(f"{err_text}\n{res_content}")
                ec = _AGY_EXIT_RE.search(res_content)
                is_error = (not interrupted) and (
                    res.get("status") == "ERROR" or bool(ec and ec.group(1) != "0")
                )

            preview = err_text or res_content
            events.append(ToolEvent(
                session_id=sid,
                project_dir=project_dir,
                event_ts=ts,
                tool=tool,
                command=scrub_secrets(cmd)[:500],
                description=desc[:200],
                output_preview=scrub_secrets(preview)[:200],
                is_error=is_error,
                interrupted=interrupted,
                event_hash=hashlib.sha256(f"{sid}:{step_idx}:{idx}".encode()).hexdigest(),
            ))
    return events


def _parse_gemini(path: Path, body: str) -> ParsedSession | None:
    """Parse Antigravity CLI (`agy`) `transcript.jsonl` into a ParsedSession."""
    sid = gemini_session_id(path)
    meta = gemini_summary_meta(path, sid)
    model: str | None = None
    started_at: str | None = None
    ended_at: str | None = None
    turn_count = 0
    tool_call_count = 0
    tool_names: set[str] = set()
    files: set[str] = set()
    first_message: str | None = None
    messages: list[str] = []
    has_user_message = False
    has_reply = False

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        node = _read_json(line)
        if node is None:
            continue

        ts = node.get("created_at")
        if ts:
            if started_at is None:
                started_at = ts
            ended_at = ts

        kind = node.get("type")
        if kind == "USER_INPUT" and node.get("source") == "USER_EXPLICIT":
            text = _extract_gemini_user_text(str(node.get("content") or ""))
            if text:
                if first_message is None:
                    first_message = text[:MAX_FIRST_MSG]
                messages.append(text)
                has_user_message = True

        elif kind == "PLANNER_RESPONSE":
            turn_count += 1
            has_reply = True
            if model is None and node.get("model"):
                model = str(node["model"])
            for tc in (node.get("tool_calls") or []):
                if isinstance(tc, dict):
                    tool_call_count += 1
                    args = _unquote_agy_args(tc.get("args"))
                    name = _gemini_tool_name(str(tc.get("name") or ""), args)
                    if name:
                        tool_names.add(name)
                    _extract_paths(args, files)
            # Visible reply only — inner `thinking` is excluded from search chunks
            text = _sanitize_text(str(node.get("content") or ""))
            if text:
                messages.append(text)

    if not has_user_message or not has_reply:
        return None

    return ParsedSession(
        session_id=sid,
        source="gemini",
        project_dir=meta.get("project_dir"),
        git_branch=None,
        slug=meta.get("title") or sid,
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


def _parse_agent_gemini(path: Path, body: str) -> ParsedAgentSession | None:
    session = _parse_gemini(path, body)
    if session is None:
        return None
    meta = gemini_summary_meta(path, session.session_id)
    return ParsedAgentSession(
        agent_id=session.session_id,
        parent_session_id=meta.get("parent_conversation_id"),
        agent_slug=session.slug,
        project_dir=session.project_dir,
        cwd=session.project_dir,
        model=session.model,
        turn_count=session.turn_count,
        tool_call_count=session.tool_call_count,
        tool_names=session.tool_names,
        first_message=session.first_message,
        all_user_text=session.all_user_text,
        message_count=len(session.messages),
        first_seen_at=session.started_at,
        last_updated_at=session.ended_at,
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


_TURN_TYPES = ("user", "assistant", "human")


def _turn_role(node: dict) -> str:
    """The role of an entry that is a conversation turn, else "". Other entry types can carry a message
    too: api-request-blob (written since 2026-10) is a copy of the API request with role "user"."""
    kind = node.get("type")
    if kind and kind not in _TURN_TYPES:
        return ""
    return (node.get("message") or {}).get("role") or kind or ""


def _is_skill_body(node: dict, text: str) -> bool:
    """A skill's instructions, injected as a meta user message when it runs: the same text in every session
    that uses the skill, and the handoff skill's reads like a handoff report."""
    return bool(node.get("isMeta")) and text.startswith("Base directory for this skill:")


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
