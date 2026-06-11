"""Asynchronous MCP connection manager using the fastmcp client library."""

import asyncio

import structlog
from fastmcp import Client

logger = structlog.get_logger(__name__)


class MCPHTTPManager:
    """Manages the lifecycle of a persistent fastmcp HTTP connection."""

    def __init__(self, url: str, headers: dict[str, str] | None = None) -> None:
        """Initialise the manager.

        Args:
            url: MCP server URL (e.g. ``http://127.0.0.1:8005/mcp``).
            headers: Optional extra HTTP headers merged with the default Accept header.
        """
        self.url = url
        base_headers: dict[str, str] = {"Accept": "application/json"}
        if headers:
            base_headers.update(headers)
        self.headers = base_headers
        self.client: Client | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Open the MCP connection and run the protocol handshake."""
        if self.client is None:
            self.client = Client(self.url)
        await self.client.__aenter__()
        logger.info("mcp_manager_started", url=self.url)

    async def stop(self) -> None:
        """Close the MCP connection."""
        if self.client is not None:
            await self.client.__aexit__(None, None, None)
            self.client = None
            logger.info("mcp_manager_stopped")

    async def list_tools(self) -> list:
        """Return the list of tools available on the MCP server.

        Returns:
            List of tool descriptors from the MCP server.

        Raises:
            RuntimeError: If the manager has not been started.
        """
        if self.client is None:
            raise RuntimeError("MCPHTTPManager not started")
        async with self._lock:
            return await self.client.list_tools()

    async def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Invoke a named tool on the MCP server.

        Args:
            name: The registered tool name.
            arguments: Keyword arguments forwarded to the tool.

        Returns:
            The tool result dict.

        Raises:
            RuntimeError: If the manager has not been started.
        """
        if self.client is None:
            raise RuntimeError("MCPHTTPManager not started")
        async with self._lock:
            return await self.client.call_tool(name, arguments or {})


# Backwards-compatible alias expected by backend.py
MCPManager = MCPHTTPManager
