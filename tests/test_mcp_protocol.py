"""Protocol-level tests for the MCP server.

test_mcp_tools.py covers tool *logic* by calling `_dispatch`/`_HANDLERS`
directly. That left the SDK-facing layer — tool registration, the handler
signatures, the result shapes — completely untested. When mcp 2.0 removed the
`@server.list_tools()`/`@server.call_tool()` decorators, `medulla/mcp.py` began
raising AttributeError at import time and `medulla mcp` died before answering
`initialize`, yet the entire suite stayed green because nothing here ever built
a Server.

These tests drive the real Server object over an in-memory transport, so the
next time the SDK moves the registration API underneath us it fails loudly and
in CI rather than silently in every user's editor.
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
    """Point the server's module-global connection at the test DB.

    `_dispatch` resolves its connection through `_get_conn()`, which memoises a
    real `connect()`. Without this the protocol tests would read the developer's
    live medulla.db.
    """
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
    """A raising handler yields is_error, keeping the session alive.

    If this leaked as a protocol error the client would drop the connection and
    every subsequent tool call in that session would fail too.
    """

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
