"""PDF text extraction via PyMuPDF."""
from __future__ import annotations

from pathlib import Path


def extract(path: Path, max_chars: int = 50_000) -> str:
    """Extract plain text from a PDF. Returns first max_chars characters."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(path))
    parts = []
    total = 0
    for page in doc:
        text = page.get_text()
        parts.append(text)
        total += len(text)
        if total >= max_chars:
            break
    doc.close()
    return "\n".join(parts)[:max_chars]
