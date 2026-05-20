"""Split ordered messages into topic-coherent chunks for FTS indexing.

Sprint 1: fixed window of TURNS_PER_CHUNK turns.
Sprint 2: vocabulary-divergence topic detection — cuts at actual topic shifts
          using Jaccard similarity on content words between consecutive windows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TURNS_PER_CHUNK = 20
MAX_CHUNK_CHARS = 4000
TOPIC_SHIFT_THRESHOLD = 0.12   # Jaccard below this → new chunk
MIN_CHUNK_TURNS = 5            # never cut before this many turns in a chunk

_STOP_WORDS = {
    "this", "that", "with", "from", "have", "will", "been", "they", "them",
    "what", "when", "where", "which", "your", "their", "there", "about",
    "would", "could", "should", "also", "just", "like", "more", "some",
    "then", "than", "into", "over", "after", "before", "were", "here",
    "these", "those", "such", "very", "much", "make", "made", "need",
    "want", "know", "look", "going", "doing", "being", "getting",
}


@dataclass
class Chunk:
    chunk_index: int
    chunk_text: str
    turn_start: int
    turn_end: int


def chunk_messages(
    messages: list[str],
    turns_per_chunk: int = TURNS_PER_CHUNK,
    use_topic_shift: bool = True,
) -> list[Chunk]:
    """Split messages into chunks. Uses topic-shift detection by default."""
    if not messages:
        return []
    if use_topic_shift and len(messages) > MIN_CHUNK_TURNS * 2:
        return _chunk_by_topic(messages)
    return _chunk_fixed(messages, turns_per_chunk)


def _chunk_fixed(messages: list[str], turns_per_chunk: int) -> list[Chunk]:
    """Fixed-window fallback."""
    chunks: list[Chunk] = []
    i = 0
    while i < len(messages):
        window = messages[i: i + turns_per_chunk]
        text = " ".join(window)
        chunks.append(Chunk(
            chunk_index=len(chunks),
            chunk_text=text[:MAX_CHUNK_CHARS],
            turn_start=i,
            turn_end=min(i + turns_per_chunk, len(messages)) - 1,
        ))
        i += turns_per_chunk
    return chunks


def _chunk_by_topic(messages: list[str]) -> list[Chunk]:
    """Vocabulary-divergence chunking.

    Slides a window of MIN_CHUNK_TURNS across messages. When the Jaccard
    similarity between the current window's vocabulary and the next window's
    vocabulary drops below TOPIC_SHIFT_THRESHOLD, we cut a new chunk.
    """
    chunks: list[Chunk] = []
    chunk_start = 0
    i = MIN_CHUNK_TURNS  # start evaluating cuts after minimum turns

    while i < len(messages):
        current_window = messages[chunk_start:i]
        next_window = messages[i: i + MIN_CHUNK_TURNS]

        if not next_window:
            break

        sim = _jaccard(
            _content_words(" ".join(current_window)),
            _content_words(" ".join(next_window)),
        )

        if sim < TOPIC_SHIFT_THRESHOLD or (i - chunk_start) >= TURNS_PER_CHUNK:
            # Topic shift detected or window too large — cut here
            text = " ".join(messages[chunk_start:i])
            chunks.append(Chunk(
                chunk_index=len(chunks),
                chunk_text=text[:MAX_CHUNK_CHARS],
                turn_start=chunk_start,
                turn_end=i - 1,
            ))
            chunk_start = i

        i += 1

    # Flush remaining
    if chunk_start < len(messages):
        text = " ".join(messages[chunk_start:])
        chunks.append(Chunk(
            chunk_index=len(chunks),
            chunk_text=text[:MAX_CHUNK_CHARS],
            turn_start=chunk_start,
            turn_end=len(messages) - 1,
        ))

    return chunks if chunks else _chunk_fixed(messages, TURNS_PER_CHUNK)


def _content_words(text: str) -> frozenset[str]:
    """Extract meaningful words: length > 3, not stop words, lowercased."""
    words = re.findall(r"\b[a-zA-Z]\w+\b", text)
    return frozenset(
        w.lower() for w in words
        if len(w) > 3 and w.lower() not in _STOP_WORDS
    )


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union > 0 else 0.0
