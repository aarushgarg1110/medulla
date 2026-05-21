"""URL ingestion — fetch and extract article text."""
from __future__ import annotations

import re


def extract(url: str, max_chars: int = 50_000) -> tuple[str, str]:
    """Fetch URL and extract readable text. Returns (title, text)."""
    import httpx
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Medulla/0.1)"}
    response = httpx.get(url, headers=headers, follow_redirects=True, timeout=30.0)
    response.raise_for_status()
    html = response.text
    title = _extract_title(html)
    text = _html_to_text(html)
    return title, text[:max_chars]


def _extract_title(html: str) -> str:
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    m = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    return "Untitled"


def _html_to_text(html: str) -> str:
    # Remove script and style blocks
    html = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    # Remove all tags
    text = re.sub(r"<[^>]+>", " ", html)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    return text.strip()
