"""Tests for medulla.episodic.chunker."""
import pytest

from medulla.episodic.chunker import MAX_CHUNK_CHARS, TURNS_PER_CHUNK, Chunk, chunk_messages


def test_empty_messages_returns_empty():
    assert chunk_messages([]) == []


def test_single_message_one_chunk():
    chunks = chunk_messages(["hello world"])
    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].turn_start == 0
    assert chunks[0].turn_end == 0
    assert "hello world" in chunks[0].chunk_text


def test_exactly_window_size_is_one_chunk():
    messages = [f"msg {i}" for i in range(TURNS_PER_CHUNK)]
    chunks = chunk_messages(messages)
    assert len(chunks) == 1
    assert chunks[0].turn_start == 0
    assert chunks[0].turn_end == TURNS_PER_CHUNK - 1


def test_one_over_window_creates_two_chunks():
    messages = [f"msg {i}" for i in range(TURNS_PER_CHUNK + 1)]
    chunks = chunk_messages(messages)
    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert chunks[1].turn_start == TURNS_PER_CHUNK
    assert chunks[1].turn_end == TURNS_PER_CHUNK


def test_chunk_indices_are_sequential():
    messages = [f"msg {i}" for i in range(TURNS_PER_CHUNK * 3 + 5)]
    chunks = chunk_messages(messages)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks)))


def test_all_messages_covered():
    """Every message should appear in exactly one chunk."""
    messages = [f"unique-marker-{i}" for i in range(45)]
    chunks = chunk_messages(messages)
    all_text = " ".join(c.chunk_text for c in chunks)
    for i, msg in enumerate(messages):
        assert msg in all_text, f"message {i} not found in any chunk"


def test_very_long_single_message_truncated():
    long_msg = "X" * (MAX_CHUNK_CHARS + 5000)
    chunks = chunk_messages([long_msg])
    assert len(chunks) == 1
    assert len(chunks[0].chunk_text) <= MAX_CHUNK_CHARS


def test_normal_messages_not_truncated():
    messages = ["short message"] * 10
    chunks = chunk_messages(messages)
    for chunk in chunks:
        assert len(chunk.chunk_text) <= MAX_CHUNK_CHARS


def test_custom_window_size():
    messages = [f"msg {i}" for i in range(10)]
    chunks = chunk_messages(messages, turns_per_chunk=3)
    assert len(chunks) == 4  # 3+3+3+1


def test_turn_boundaries_are_correct():
    messages = [f"msg {i}" for i in range(25)]
    chunks = chunk_messages(messages, turns_per_chunk=10)
    # chunk 0: turns 0-9, chunk 1: turns 10-19, chunk 2: turns 20-24
    assert chunks[0].turn_start == 0
    assert chunks[0].turn_end == 9
    assert chunks[1].turn_start == 10
    assert chunks[1].turn_end == 19
    assert chunks[2].turn_start == 20
    assert chunks[2].turn_end == 24


def test_chunk_text_joins_messages_with_space():
    chunks = chunk_messages(["hello", "world"], turns_per_chunk=5)
    assert chunks[0].chunk_text == "hello world"
