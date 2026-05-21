"""URL ingestion — fetch and extract article text using trafilatura.

trafilatura extracts main article content (strips nav, ads, footers).
Falls back to basic HTML stripping if trafilatura returns nothing.
SSL_CERT_FILE / REQUESTS_CA_BUNDLE respected for corporate CA setups.
"""
from __future__ import annotations

import re


def _ssl_verify():
    """Respect SSL_CERT_FILE / REQUESTS_CA_BUNDLE (e.g. Zscaler).
    Falls back to True (certifi default) for users without a custom CA bundle.
    """
    import os
    return os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE") or True


def extract(url: str, max_chars: int = 50_000) -> tuple[str, str]:
    """Fetch URL and extract main article text. Returns (title, text)."""
    import httpx
    headers = {"User-Agent": "Mozilla/5.0 (compatible; Medulla/0.1)"}
    response = httpx.get(url, headers=headers, follow_redirects=True, timeout=30.0, verify=_ssl_verify())
    response.raise_for_status()
    html = response.text

    title = _extract_title(html)
    text = _extract_text(html)
    return title, text[:max_chars]


def _extract_title(html: str) -> str:
    """Extract a clean title — prefer H1 over <title> to avoid site suffixes."""
    # Try H1 first (article headline, no " | Site Name" suffix)
    m = re.search(r"<h1[^>]*>([^<]+)</h1>", html, re.IGNORECASE)
    if m:
        return re.sub(r"<[^>]+>", "", m.group(1)).strip()
    # Fall back to <title> but trim common suffixes
    m = re.search(r"<title[^>]*>([^<]+)</title>", html, re.IGNORECASE)
    if m:
        title = m.group(1).strip()
        # Strip " | Site", " - Site", " — Site" suffixes
        title = re.sub(r"\s*[\|—\-]\s*.{3,50}$", "", title).strip()
        return title
    return "Untitled"


def _extract_text(html: str) -> str:
    """Extract main article body using trafilatura, falling back to HTML stripping."""
    try:
        import trafilatura
        text = trafilatura.extract(html, include_comments=False, include_tables=False)
        if text and len(text) > 200:
            return text
    except Exception:
        pass
    # Fallback: basic HTML stripping
    cleaned = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", cleaned)
    return re.sub(r"\s+", " ", text).strip()
