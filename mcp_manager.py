import asyncio
from typing import Any, Dict, Optional

from fastmcp import Client


class MCPHTTPManager:
    def __init__(self, url: str, headers: Optional[Dict[str, str]] = None):
        # url should be like: "http://127.0.0.1:8000/mcp"
        self.url = url
        self.headers = headers or {}
        self.client: Optional[Client] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        # FastMCP client auto-selects HTTP transport from URL
        # (or you can explicitly use StreamableHttpTransport, but this is simplest)
        self.client = Client(self.url)  # :contentReference[oaicite:3]{index=3}
        await self.client.__aenter__()   # opens connection + initializes

    async def stop(self) -> None:
        if self.client:
            await self.client.__aexit__(None, None, None)
            self.client = None

    async def list_tools(self):
        if not self.client:
            raise RuntimeError("MCPHTTPManager not started")
        async with self._lock:
            return await self.client.list_tools()

    async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None):
        if not self.client:
            raise RuntimeError("MCPHTTPManager not started")
        async with self._lock:
            return await self.client.call_tool(name, arguments or {})

# Backwards-compatible alias expected by backend.py
MCPManager = MCPHTTPManager