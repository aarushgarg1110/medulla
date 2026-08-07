"""Split ordered messages into topic-coherent chunks for FTS indexing.

Chunks are **size-primary, topic-aware**: a chunk grows toward TARGET_CHUNK_CHARS
and hard-cuts at MAX_CHUNK_CHARS. Within that budget it cuts early at real topic
shifts so chunks stay coherent:

  * a *strong* shift (near-disjoint vocabulary, Jaccard < STRONG_SHIFT_THRESHOLD)
    cuts as soon as the chunk clears a small floor — this catches abrupt topic
    changes and keeps the "two clearly different topics → two chunks" guarantee.
  * a *weak* shift (Jaccard < TOPIC_SHIFT_THRESHOLD) only cuts once the chunk has
    reached MIN_CHUNK_CHARS *and* the dip is sustained for SHIFT_DEBOUNCE windows —
    this stops transient vocabulary dips from shredding a session into tiny chunks.

Undersized trailing chunks are merged back into their predecessor, but only when
the merge fits under the cap — a merge that would overflow is declined, since a
slightly-small chunk is better than a truncated one.

Content is never dropped: a span whose joined text still exceeds the cap (a single
oversized message, or a fixed-window join) is *split* into consecutive chunks
rather than truncated. Sibling pieces share the turn range they came from and
differ only by chunk_index.

History: Sprint 1 used a fixed TURNS_PER_CHUNK window; Sprint 2 switched to
vocabulary-divergence cuts but over-segmented badly (one real session produced
1,562 chunks). This size-primary rewrite targets ~5-15 coherent chunks/session.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

TURNS_PER_CHUNK = 20           # fixed-window fallback size
MAX_CHUNK_CHARS = 4000         # hard ceiling — always cut here
TARGET_CHUNK_CHARS = 2500      # soft target — weak shifts past MIN may cut before this
MIN_CHUNK_CHARS = 1200         # a weak topic shift may only cut once past this
MIN_STRONG_CHARS = 400         # a strong (disjoint) shift may cut once past this
MERGE_FLOOR_CHARS = 250        # chunks smaller than this are folded into a neighbor
TOPIC_SHIFT_THRESHOLD = 0.10   # Jaccard below this → weak shift (needs size + debounce)
STRONG_SHIFT_THRESHOLD = 0.02  # Jaccard below this → strong shift (near-disjoint)
SHIFT_DEBOUNCE = 2             # consecutive weak-shift windows before a weak cut
MIN_CHUNK_TURNS = 3            # never cut before this many turns in a chunk
COMPARE_WINDOW = 5             # window size (turns) for vocabulary comparison

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
    """Split messages into chunks. Uses size-primary topic-aware chunking by default."""
    if not messages:
        return []
    if use_topic_shift and len(messages) > MIN_CHUNK_TURNS * 2:
        return _chunk_by_topic(messages)
    return _chunk_fixed(messages, turns_per_chunk)


def _split_oversized(text: str, max_chars: int) -> list[str]:
    """Break text into consecutive pieces of at most max_chars, losing no content.

    Cuts at the last whitespace inside the window so words stay intact; falls back
    to a hard cut when a single run has no whitespace (minified JSON, a base64 blob).
    Only the whitespace character at a cut point is consumed.
    """
    if len(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    n = len(text)
    while True:                    # always exits below, once the remainder fits
        if n - start <= max_chars:
            pieces.append(text[start:])
            break
        # rfind lower bound is start+1 so a cut always advances past `start`.
        cut = text.rfind(" ", start + 1, start + max_chars + 1)
        if cut == -1:
            cut = start + max_chars
            pieces.append(text[start:cut])
            start = cut
        else:
            pieces.append(text[start:cut])
            start = cut + 1        # drop the single space we cut on
    return pieces


def _materialize(
    messages: list[str],
    bounds: list[tuple[int, int]],
    max_chars: int,
) -> list[Chunk]:
    """Turn (start, end_exclusive) spans into Chunks, splitting any span over the cap.

    The single place chunk_text is produced, so no code path can truncate. Pieces of
    one span share turn_start/turn_end and take consecutive chunk_index values.
    """
    chunks: list[Chunk] = []
    for s, e in bounds:
        text = " ".join(messages[s:e])
        for piece in _split_oversized(text, max_chars):
            chunks.append(Chunk(
                chunk_index=len(chunks),
                chunk_text=piece,
                turn_start=s,
                turn_end=e - 1,
            ))
    return chunks


def _chunk_fixed(messages: list[str], turns_per_chunk: int) -> list[Chunk]:
    """Fixed-window fallback."""
    bounds = [
        (i, min(i + turns_per_chunk, len(messages)))
        for i in range(0, len(messages), turns_per_chunk)
    ]
    return _materialize(messages, bounds, MAX_CHUNK_CHARS)


def _chunk_by_topic(
    messages: list[str],
    *,
    max_chars: int = MAX_CHUNK_CHARS,
    min_chars: int = MIN_CHUNK_CHARS,
    min_strong_chars: int = MIN_STRONG_CHARS,
    merge_floor: int = MERGE_FLOOR_CHARS,
    weak_threshold: float = TOPIC_SHIFT_THRESHOLD,
    strong_threshold: float = STRONG_SHIFT_THRESHOLD,
    debounce: int = SHIFT_DEBOUNCE,
    min_turns: int = MIN_CHUNK_TURNS,
    window: int = COMPARE_WINDOW,
) -> list[Chunk]:
    """Size-primary, topic-aware chunking.

    Grows a chunk message-by-message; cuts at a hard char ceiling, at a strong
    (near-disjoint) topic shift past a small floor, or at a sustained weak shift
    once past a real minimum size. Parameters are exposed for offline tuning.
    """
    n = len(messages)
    bounds: list[tuple[int, int]] = []   # (start, end_exclusive)
    start = 0
    cur_chars = 0
    weak_streak = 0

    for i in range(n):
        msg_len = len(messages[i]) + 1
        # Close the current chunk *before* a message that would overflow the cap,
        # so the joined text never exceeds max_chars (no silent truncation loss).
        # A single message larger than the cap becomes its own chunk (truncated).
        if cur_chars > 0 and cur_chars + msg_len > max_chars:
            bounds.append((start, i))
            start = i
            cur_chars = 0
            weak_streak = 0

        cur_chars += msg_len
        turns = i - start + 1
        next_start = i + 1

        # Topic-shift cut *after* this message.
        if turns >= min_turns and next_start < n:
            prev_w = messages[max(start, next_start - window):next_start]
            next_w = messages[next_start:next_start + window]
            sim = _jaccard(
                _content_words(" ".join(prev_w)),
                _content_words(" ".join(next_w)),
            )
            if sim < weak_threshold:
                weak_streak += 1
            else:
                weak_streak = 0

            cut = (
                (sim < strong_threshold and cur_chars >= min_strong_chars)  # abrupt shift
                or (weak_streak >= debounce and cur_chars >= min_chars)     # sustained drift
            )
            if cut:
                bounds.append((start, next_start))
                start = next_start
                cur_chars = 0
                weak_streak = 0

    if start < n:
        bounds.append((start, n))

    bounds = _merge_undersized(messages, bounds, merge_floor, max_chars)

    chunks = _materialize(messages, bounds, max_chars)
    return chunks if chunks else _chunk_fixed(messages, TURNS_PER_CHUNK)


def _merge_undersized(
    messages: list[str],
    bounds: list[tuple[int, int]],
    min_chars: int,
    max_chars: int,
) -> list[tuple[int, int]]:
    """Fold any chunk below min_chars into its predecessor (or successor if first).

    A merge that would push the combined span past max_chars is declined — it would
    only be split straight back apart, and pre-#58 it silently truncated the tail.
    """
    if len(bounds) <= 1:
        return bounds

    def span_size(s: int, e: int) -> int:
        return sum(len(messages[k]) + 1 for k in range(s, e))

    merged: list[tuple[int, int]] = []
    for s, e in bounds:
        if merged and span_size(s, e) < min_chars:
            ps, _ = merged[-1]
            if span_size(ps, e) <= max_chars:
                merged[-1] = (ps, e)      # extend previous chunk
                continue
        merged.append((s, e))
    # If the very first chunk was undersized it stayed as-is; fold it forward.
    if len(merged) > 1:
        s0, e0 = merged[0]
        s1, e1 = merged[1]
        if span_size(s0, e0) < min_chars and span_size(s0, e1) <= max_chars:
            merged[0:2] = [(s0, e1)]
    return merged


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
