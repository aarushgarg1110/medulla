"""Markdown / plain text ingestion."""
from __future__ import annotations

import re
from pathlib import Path


def extract(path: Path, max_chars: int = 50_000) -> tuple[str, str]:
    """Read markdown/text file. Returns (title, text)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    title = _extract_title(text) or path.stem.replace("-", " ").replace("_", " ").title()
    return title, text[:max_chars]


def _extract_title(text: str) -> str:
    """Extract title: prefer YAML frontmatter title over H1 heading.

    Perplexity exports have a truncated user question as frontmatter title and
    '# You' as the first H1. Neither is useful — the stem fallback wins instead.
    Frontmatter title is only accepted if it looks like a real title (not a
    truncated question ending with '...' and not a single generic word).
    """
    # 1. Try YAML frontmatter title: field
    m = re.search(r"^title:\s*[\"']?(.+?)[\"']?\s*$", text, re.MULTILINE)
    if m:
        title = m.group(1).strip().strip('"').strip("'")
        # Skip truncated titles (Perplexity exports the user's question, cut off)
        truncated = title.endswith("...") or title.endswith('..."') or title.endswith("...'")
        too_short = len(title) <= 3
        generic = title.lower() in ("untitled", "none", "null")
        if title and not truncated and not too_short and not generic:
            return title[:120]
    # 2. Fall back to first H1 that isn't a Perplexity/chat speaker label
    _SKIP = {"you", "assistant", "human", "user", "perplexity", "claude"}
    for m in re.finditer(r"^#\s+(.+)$", text, re.MULTILINE):
        candidate = m.group(1).strip()
        if candidate.lower() not in _SKIP:
            return candidate
    return ""
