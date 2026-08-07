"""Tests for medulla.episodic.chunker — fixed window + topic-shift."""
import pytest

from medulla.episodic.chunker import (
    MAX_CHUNK_CHARS,
    MIN_CHUNK_TURNS,
    TOPIC_SHIFT_THRESHOLD,
    TURNS_PER_CHUNK,
    Chunk,
    _chunk_fixed,
    _chunk_by_topic,
    _content_words,
    _jaccard,
    _merge_undersized,
    _split_oversized,
    chunk_messages,
)


# ── fixed window (existing tests, kept green) ─────────────────────────────────

def test_empty_messages_returns_empty():
    assert chunk_messages([]) == []


def test_single_message_one_chunk():
    chunks = chunk_messages(["hello world"], use_topic_shift=False)
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].turn_start == 0
    assert chunks[0].turn_end == 0
    assert "hello world" in chunks[0].chunk_text


def test_exactly_window_size_is_one_chunk():
    messages = [f"msg {i}" for i in range(TURNS_PER_CHUNK)]
    chunks = chunk_messages(messages, use_topic_shift=False)
    assert len(chunks) == 1


def test_one_over_window_creates_two_chunks():
    messages = [f"msg {i}" for i in range(TURNS_PER_CHUNK + 1)]
    chunks = chunk_messages(messages, use_topic_shift=False)
    assert len(chunks) == 2
    assert chunks[1].turn_start == TURNS_PER_CHUNK


def test_chunk_indices_are_sequential():
    messages = [f"msg {i}" for i in range(TURNS_PER_CHUNK * 3 + 5)]
    chunks = chunk_messages(messages, use_topic_shift=False)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_all_messages_covered():
    messages = [f"unique-marker-{i}" for i in range(45)]
    chunks = chunk_messages(messages, use_topic_shift=False)
    all_text = " ".join(c.chunk_text for c in chunks)
    for msg in messages:
        assert msg in all_text


def test_very_long_single_message_splits_instead_of_truncating():
    """A message past the cap becomes N chunks (#58) — it used to lose the tail."""
    body = "X" * (MAX_CHUNK_CHARS + 5000)
    chunks = chunk_messages([body], use_topic_shift=False)
    assert len(chunks) == 3                    # 9000 chars / 4000 cap
    assert all(len(c.chunk_text) <= MAX_CHUNK_CHARS for c in chunks)
    assert "".join(c.chunk_text for c in chunks) == body
    assert [c.chunk_index for c in chunks] == [0, 1, 2]
    # Both pieces came from turn 0, so they share its bounds.
    assert all((c.turn_start, c.turn_end) == (0, 0) for c in chunks)


def test_custom_window_size():
    messages = [f"msg {i}" for i in range(10)]
    chunks = _chunk_fixed(messages, turns_per_chunk=3)
    assert len(chunks) == 4


def test_turn_boundaries_are_correct():
    messages = [f"msg {i}" for i in range(25)]
    chunks = _chunk_fixed(messages, turns_per_chunk=10)
    assert chunks[0].turn_start == 0
    assert chunks[0].turn_end == 9
    assert chunks[1].turn_start == 10
    assert chunks[2].turn_end == 24


def test_chunk_text_joins_messages():
    chunks = _chunk_fixed(["hello", "world"], turns_per_chunk=5)
    assert chunks[0].chunk_text == "hello world"


# ── _jaccard ──────────────────────────────────────────────────────────────────

def test_jaccard_identical_sets():
    s = frozenset(["logD", "compoundx", "batch"])
    assert _jaccard(s, s) == 1.0


def test_jaccard_disjoint_sets():
    a = frozenset(["logD", "compoundx"])
    b = frozenset(["pKa", "acidic"])
    assert _jaccard(a, b) == 0.0


def test_jaccard_partial_overlap():
    a = frozenset(["logD", "compoundx", "batch"])
    b = frozenset(["logD", "pKa", "result"])
    result = _jaccard(a, b)
    assert 0.0 < result < 1.0


def test_jaccard_both_empty():
    assert _jaccard(frozenset(), frozenset()) == 1.0


def test_jaccard_one_empty():
    assert _jaccard(frozenset(["logD"]), frozenset()) == 0.0


# ── _content_words ────────────────────────────────────────────────────────────

def test_content_words_filters_stop_words():
    words = _content_words("this is just about logD prediction from an internal database")
    assert "logD" in words or "logd" in words
    assert "this" not in words
    assert "just" not in words
    assert "about" not in words


def test_content_words_filters_short_words():
    words = _content_words("the logD and pKa are key")
    # "and", "are", "the" are short or stop words
    assert all(len(w) > 3 for w in words)


def test_content_words_lowercased():
    words = _content_words("LogD COMPOUNDX Batch")
    assert "logd" in words
    assert "compoundx" in words


def test_content_words_empty_string():
    assert _content_words("") == frozenset()


# ── topic-shift chunker ───────────────────────────────────────────────────────

def _logd_msg(i: int) -> str:
    """Clearly logD-domain message — no words shared with pKa domain (unique CMPD prefix)."""
    return f"lipophilicity compoundx chromatographic octanol partition coefficient outlier batch CMPD{i:04d}"


def _pka_msg(i: int) -> str:
    """Clearly pKa-domain message — no words shared with logD domain (unique PKAMSG prefix)."""
    return f"protonation ionization acidic triazole nitrogen titration endpoint PKAMSG{i:04d}"


def test_topic_shift_creates_new_chunk_on_clear_shift():
    """Two completely different topics should produce at least 2 chunks."""
    logd_msgs = [_logd_msg(i) for i in range(MIN_CHUNK_TURNS + 2)]
    pka_msgs = [_pka_msg(i) for i in range(MIN_CHUNK_TURNS + 2)]
    chunks = _chunk_by_topic(logd_msgs + pka_msgs)
    assert len(chunks) >= 2


def test_topic_shift_same_topic_stays_together():
    """Same-topic messages should stay in fewer chunks than fixed-window would produce."""
    messages = [_logd_msg(i) for i in range(TURNS_PER_CHUNK - 2)]
    chunks = _chunk_by_topic(messages)
    assert len(chunks) <= 3


def test_topic_shift_all_messages_covered():
    """Every unique marker must appear in the combined chunk text."""
    logd_msgs = [_logd_msg(i) for i in range(MIN_CHUNK_TURNS + 2)]
    pka_msgs = [_pka_msg(i) for i in range(MIN_CHUNK_TURNS + 2)]
    messages = logd_msgs + pka_msgs
    chunks = _chunk_by_topic(messages)
    all_text = " ".join(c.chunk_text for c in chunks)
    # Each message has a unique prefix marker — check those
    for i in range(len(logd_msgs)):
        assert f"CMPD{i:04d}" in all_text
    for i in range(len(pka_msgs)):
        assert f"PKAMSG{i:04d}" in all_text


def test_topic_shift_chunk_indices_sequential():
    messages = [_logd_msg(i) for i in range(MIN_CHUNK_TURNS + 5)]
    chunks = _chunk_by_topic(messages)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_topic_shift_falls_back_for_small_sessions():
    """Sessions smaller than MIN_CHUNK_TURNS * 2 use fixed window."""
    messages = ["short message"] * (MIN_CHUNK_TURNS - 1)
    chunks = chunk_messages(messages, use_topic_shift=True)
    assert len(chunks) >= 1
    all_text = " ".join(c.chunk_text for c in chunks)
    assert "short message" in all_text


def test_topic_shift_respects_max_chunk_size():
    """No chunk should exceed MAX_CHUNK_CHARS."""
    messages = ["A" * 500] * 50
    chunks = _chunk_by_topic(messages)
    for c in chunks:
        assert len(c.chunk_text) <= MAX_CHUNK_CHARS


def test_chunk_messages_uses_topic_shift_by_default():
    """chunk_messages with use_topic_shift=True (default) should work."""
    messages = [_logd_msg(i) for i in range(TURNS_PER_CHUNK + 5)]
    chunks = chunk_messages(messages)
    assert len(chunks) >= 1


def test_topic_shift_no_content_loss_with_large_messages():
    """Every message survives even when chunks brush the char cap (no truncation loss)."""
    # Each message ~1.2k chars; several exceed a chunk when combined.
    messages = [f"UNIQUEMARK{i:03d} " + "padding content words here " * 45 for i in range(30)]
    chunks = _chunk_by_topic(messages)
    combined = " ".join(c.chunk_text for c in chunks)
    for i in range(30):
        assert f"UNIQUEMARK{i:03d}" in combined
    for c in chunks:
        assert len(c.chunk_text) <= MAX_CHUNK_CHARS


# ── _split_oversized / no-truncation invariants (#58) ─────────────────────────

def test_split_oversized_returns_text_unchanged_under_cap():
    assert _split_oversized("short text", 4000) == ["short text"]


def test_split_oversized_exact_cap_is_one_piece():
    text = "X" * 100
    assert _split_oversized(text, 100) == [text]


def test_split_oversized_cuts_on_whitespace_keeping_words_intact():
    # 10 words of 9 chars + spaces; cap forces a cut mid-run.
    text = " ".join(["wordwords"] * 10)
    pieces = _split_oversized(text, 30)
    assert all(len(p) <= 30 for p in pieces)
    # No word was sliced in half.
    for p in pieces:
        for word in p.split():
            assert word == "wordwords"


def test_split_oversized_hard_cuts_when_no_whitespace():
    """A whitespace-free run (minified JSON, base64) still splits, never truncates."""
    text = "A" * 250
    pieces = _split_oversized(text, 100)
    assert [len(p) for p in pieces] == [100, 100, 50]
    assert "".join(pieces) == text


def test_split_oversized_exact_multiple_of_cap():
    """Whitespace-free text that divides evenly ends on a boundary, no empty tail."""
    pieces = _split_oversized("A" * 200, 100)
    assert [len(p) for p in pieces] == [100, 100]


def test_split_oversized_loses_only_cut_whitespace():
    text = " ".join(f"token{i:04d}" for i in range(200))
    pieces = _split_oversized(text, 100)
    assert "".join(p.replace(" ", "") for p in pieces) == text.replace(" ", "")


def test_split_oversized_never_emits_empty_pieces():
    text = "word" + " " * 300 + "tail"
    pieces = _split_oversized(text, 50)
    assert all(p for p in pieces)


def test_oversized_message_mid_session_splits_and_keeps_neighbours():
    """A giant turn between normal ones: nothing before, in, or after it is lost."""
    huge = "HUGEMARK " + "giant payload content here " * 400   # ~10.8k chars
    messages = (
        [f"BEFORE{i:02d} some ordinary conversation text here" for i in range(8)]
        + [huge]
        + [f"AFTER{i:02d} more ordinary conversation text here" for i in range(8)]
    )
    chunks = _chunk_by_topic(messages)
    combined = " ".join(c.chunk_text for c in chunks)
    for i in range(8):
        assert f"BEFORE{i:02d}" in combined
        assert f"AFTER{i:02d}" in combined
    assert combined.count("giant payload content here") == 400
    assert all(len(c.chunk_text) <= MAX_CHUNK_CHARS for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_merge_undersized_declines_merge_that_would_overflow_cap():
    """Merging a tiny chunk into a nearly-full predecessor must not truncate it."""
    messages = ["F" * 3900, "t" * 100]
    bounds = [(0, 1), (1, 2)]
    merged = _merge_undersized(messages, bounds, min_chars=250, max_chars=MAX_CHUNK_CHARS)
    assert merged == [(0, 1), (1, 2)]      # declined — would have been 4002 chars


def test_merge_undersized_still_merges_when_it_fits():
    messages = ["F" * 1000, "t" * 100]
    merged = _merge_undersized(
        messages, [(0, 1), (1, 2)], min_chars=250, max_chars=MAX_CHUNK_CHARS
    )
    assert merged == [(0, 2)]


def test_merge_undersized_first_chunk_folds_forward_when_it_fits():
    messages = ["t" * 100, "F" * 1000]
    merged = _merge_undersized(
        messages, [(0, 1), (1, 2)], min_chars=250, max_chars=MAX_CHUNK_CHARS
    )
    assert merged == [(0, 2)]


def test_merge_undersized_first_chunk_forward_fold_respects_cap():
    """An undersized *first* chunk folds forward only if the result fits."""
    messages = ["t" * 100, "F" * 3950]
    merged = _merge_undersized(
        messages, [(0, 1), (1, 2)], min_chars=250, max_chars=MAX_CHUNK_CHARS
    )
    assert merged == [(0, 1), (1, 2)]


def test_no_content_loss_across_varied_message_sizes():
    """Invariant: every non-whitespace char of every message reaches some chunk."""
    sizes = [50, 4500, 200, 12000, 80, 3999, 4001, 120]
    messages = [f"MARK{i:02d}" + "z" * sizes[i] for i in range(len(sizes))]
    for chunks in (_chunk_by_topic(messages), _chunk_fixed(messages, 3)):
        combined = "".join(c.chunk_text for c in chunks).replace(" ", "")
        assert combined == "".join(messages).replace(" ", "")
        assert all(len(c.chunk_text) <= MAX_CHUNK_CHARS for c in chunks)


def test_fixed_window_splits_oversized_window():
    """The fixed-window fallback path splits too (it used to slice at the cap)."""
    messages = ["Q" * 2000] * 5              # 10k chars in one 20-turn window
    chunks = _chunk_fixed(messages, TURNS_PER_CHUNK)
    assert len(chunks) > 1
    assert all(len(c.chunk_text) <= MAX_CHUNK_CHARS for c in chunks)
    assert "".join(c.chunk_text for c in chunks).replace(" ", "") == "Q" * 10000
    # All pieces describe the same window.
    assert {(c.turn_start, c.turn_end) for c in chunks} == {(0, 4)}


def test_topic_shift_produces_substantial_chunks():
    """Same-topic content packs into few large chunks, not many tiny ones."""
    # 60 short same-topic messages (~30 chars each) → well under one chunk.
    messages = [f"consistent topic message about logd prediction number {i}" for i in range(60)]
    chunks = _chunk_by_topic(messages)
    # Old vocabulary-divergence chunker shredded this into ~10+ tiny chunks;
    # size-primary packing keeps it to a couple.
    assert len(chunks) <= 3
