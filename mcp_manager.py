import asyncio
from typing import Any, Dict, Optional

from fastmcp import Client


class MCPHTTPManager:
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        # url should be like: "http://127.0.0.1:8005/mcp"
        self.url = url

        # Ensure MCP server gets an Accept header it likes
        base_headers = {"Accept": "application/json"}
        if headers:
            base_headers.update(headers)

        self.headers = base_headers
        self.client: Optional[Client] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        # Create client with headers ONCE and enter it
        if self.client is None:
            self.client = Client(self.url)

        await self.client.__aenter__()   # opens connection + initializes

    async def stop(self) -> None:
        if self.client is not None:
            await self.client.__aexit__(None, None, None)
            self.client = None

    async def list_tools(self):
        if self.client is None:
            raise RuntimeError("MCPHTTPManager not started")
        async with self._lock:
            return await self.client.list_tools()

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None):
        if self.client is None:
            raise RuntimeError("MCPHTTPManager not started")
        async with self._lock:
            return await self.client.call_tool(name, arguments or {})


# Backwards-compatible alias expected by backend.py
MCPManager = MCPHTTPManager