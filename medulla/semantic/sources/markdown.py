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
    """Extract first H1 heading as title."""
    m = re.match(r"^#\s+(.+)$", text.strip(), re.MULTILINE)
    return m.group(1).strip() if m else ""
