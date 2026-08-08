"""Protocol-level tests: drive the real Server over an in-memory transport.

test_mcp_tools.py only calls _dispatch/_HANDLERS, so it stays green even when the
SDK-facing layer is dead. These cover registration, handler signatures and result
shapes, which is what an SDK upgrade actually breaks.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import anyio
import pytest
from mcp import ClientSession
from mcp.shared.memory import create_client_server_memory_streams

import medulla.mcp as mcp_mod


@asynccontextmanager
async def _connected_session():
    """Run the real medulla Server against an in-process client session."""
    async with create_client_server_memory_streams() as (client_streams, server_streams):
        client_read, client_write = client_streams
        server_read, server_write = server_streams

        async with anyio.create_task_group() as tg:

            async def _run_server() -> None:
                await mcp_mod._server.run(
                    server_read,
                    server_write,
                    mcp_mod._server.create_initialization_options(),
                    # surface handler bugs as test failures instead of error payloads
                    raise_exceptions=True,
                )

            tg.start_soon(_run_server)

            async with ClientSession(client_read, client_write) as session:
                await session.initialize()
                yield session

            tg.cancel_scope.cancel()


@pytest.fixture
def patched_conn(db, monkeypatch):
    """Without this, _get_conn() would memoise a real connect() to the live medulla.db."""
    monkeypatch.setattr(mcp_mod, "_conn", db)
    return db


def test_initialize_and_list_tools(patched_conn):
    """The handshake completes and every registered tool is advertised."""

    async def _run():
        async with _connected_session() as session:
            return await session.list_tools()

    result = anyio.run(_run)

    names = {t.name for t in result.tools}
    assert len(result.tools) == len(mcp_mod._TOOLS)
    # every tool medulla dispatches must actually be advertised over the wire
    assert names == set(mcp_mod._HANDLERS)
    assert "medulla_search" in names


def test_every_advertised_tool_has_a_handler(patched_conn):
    """A tool listed but not dispatchable is a dead entry in the client's menu."""

    async def _run():
        async with _connected_session() as session:
            return await session.list_tools()

    result = anyio.run(_run)

    for tool in result.tools:
        assert tool.name in mcp_mod._HANDLERS, f"{tool.name} advertised but not dispatchable"
        assert tool.description, f"{tool.name} has no description"
        # 2.0 exposes the field as input_schema; "inputSchema" remains the wire alias
        assert tool.input_schema.get("type") == "object"


def test_call_tool_returns_text_content(patched_conn):
    """A normal call round-trips through the protocol as text, not an error."""

    async def _run():
        async with _connected_session() as session:
            return await session.call_tool("medulla_stats", {})

    result = anyio.run(_run)

    assert result.is_error is False
    assert result.content
    assert result.content[0].type == "text"
    assert "Episodic:" in result.content[0].text


def test_unknown_tool_is_reported_not_raised(patched_conn):
    """An unknown name comes back as a readable result, not a transport error."""

    async def _run():
        async with _connected_session() as session:
            return await session.call_tool("medulla_does_not_exist", {})

    result = anyio.run(_run)

    assert "Unknown tool" in result.content[0].text


def test_handler_exception_becomes_error_result(patched_conn, monkeypatch):
    """A raising handler yields is_error and leaves the session usable."""

    def _boom(conn, args):
        raise RuntimeError("synthetic failure")

    monkeypatch.setitem(mcp_mod._HANDLERS, "medulla_stats", _boom)

    async def _run():
        async with _connected_session() as session:
            failed = await session.call_tool("medulla_stats", {})
            # the session must still be usable afterwards
            listed = await session.list_tools()
            return failed, listed

    failed, listed = anyio.run(_run)

    assert failed.is_error is True
    assert "synthetic failure" in failed.content[0].text
    assert len(listed.tools) == len(mcp_mod._TOOLS)
