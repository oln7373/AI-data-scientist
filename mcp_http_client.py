"""Minimal MCP JSON-RPC HTTP client.

Uses the same HTTP calls verified with curl.
Avoids the official streamable_http_client which hangs in this environment.
"""

import json

import httpx
import structlog

from config import get_config

logger = structlog.get_logger(__name__)


def _parse_response(resp: httpx.Response) -> dict:
    """Parse a response that may be plain JSON or SSE-wrapped JSON.

    Args:
        resp: The raw httpx response.

    Returns:
        Parsed response as a dict.

    Raises:
        RuntimeError: If an SSE response contains no data line.
    """
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:"):].strip())
        raise RuntimeError(f"No data line found in SSE response: {resp.text!r}")
    return resp.json()


class MCPHttpClient:
    """Minimal MCP JSON-RPC client over plain HTTP."""

    def __init__(self, url: str) -> None:
        """Initialise the client.

        Args:
            url: Full URL of the MCP server endpoint (e.g. http://127.0.0.1:8005/mcp).
        """
        self.url = url.rstrip("/")
        self.session_id: str | None = None
        self._timeout = get_config().mcp.http_client_timeout_seconds
        self._protocol_version = get_config().mcp.protocol_version

    async def initialize(self) -> dict:
        """Perform the MCP handshake and store the session ID.

        Returns:
            The parsed initialize response dict.

        Raises:
            RuntimeError: If the server does not return a session ID header.
            httpx.HTTPStatusError: On non-2xx responses.
        """
        async with httpx.AsyncClient(trust_env=False, timeout=self._timeout) as client:
            resp = await client.post(
                self.url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": self._protocol_version,
                        "clientInfo": {"name": "fastapi", "version": "0.1"},
                        "capabilities": {},
                    },
                },
            )
            resp.raise_for_status()
            sid = resp.headers.get("mcp-session-id") or resp.headers.get("Mcp-Session-Id")
            if not sid:
                raise RuntimeError("Missing mcp-session-id header on initialize response")
            self.session_id = sid
            logger.info("mcp_session_initialized", session_id=sid)
            return _parse_response(resp)

    async def tools_list(self) -> dict:
        """List tools available on the MCP server.

        Returns:
            Parsed tools/list response dict.
        """
        if not self.session_id:
            await self.initialize()

        async with httpx.AsyncClient(trust_env=False, timeout=self._timeout) as client:
            resp = await client.post(
                self.url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": self.session_id,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {"cursor": None},
                },
            )
            resp.raise_for_status()
            return _parse_response(resp)

    async def tool_call(self, name: str, arguments: dict) -> dict:
        """Invoke a named tool on the MCP server.

        Args:
            name: The tool name as registered on the MCP server.
            arguments: Keyword arguments to pass to the tool.

        Returns:
            Parsed tools/call response dict.
        """
        if not self.session_id:
            await self.initialize()

        async with httpx.AsyncClient(trust_env=False, timeout=self._timeout) as client:
            resp = await client.post(
                self.url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "Mcp-Session-Id": self.session_id,
                },
                json={
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            resp.raise_for_status()
            return _parse_response(resp)
