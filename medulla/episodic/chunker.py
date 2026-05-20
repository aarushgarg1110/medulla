"""Split ordered user messages into topic-coherent chunks for FTS indexing.

Sprint 1: fixed window of TURNS_PER_CHUNK turns.
Sprint 2: vocabulary-divergence detection.
"""
from __future__ import annotations

from dataclasses import dataclass

TURNS_PER_CHUNK = 20
MAX_CHUNK_CHARS = 4000  # soft ceiling per chunk


@dataclass
class Chunk:
    chunk_index: int
    chunk_text: str
    turn_start: int
    turn_end: int


def chunk_messages(user_messages: list[str], turns_per_chunk: int = TURNS_PER_CHUNK) -> list[Chunk]:
    """Split messages into fixed-window chunks. Each chunk has its own FTS row."""
    if not user_messages:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0

    while i < len(user_messages):
        window = user_messages[i : i + turns_per_chunk]
        text = " ".join(window)

        # If a single message exceeds the soft ceiling, still keep it as one chunk
        chunks.append(Chunk(
            chunk_index=chunk_index,
            chunk_text=text[:MAX_CHUNK_CHARS] if len(text) > MAX_CHUNK_CHARS else text,
            turn_start=i,
            turn_end=min(i + turns_per_chunk, len(user_messages)) - 1,
        ))

        chunk_index += 1
        i += turns_per_chunk

    return chunks
