# mcp_http_client.py
import httpx
import json





def _parse_response(resp: httpx.Response) -> dict:
    """Handle both plain JSON and SSE-wrapped JSON responses."""
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        # Extract the data: line from the SSE payload
        for line in resp.text.splitlines():
            if line.startswith("data:"):
                return json.loads(line[len("data:"):].strip())
        raise RuntimeError(f"No data line found in SSE response: {resp.text!r}")
    return resp.json()


class MCPHttpClient:
    """
    Minimal MCP JSON-RPC client using the same HTTP calls you verified with curl.
    Avoids the official streamable_http_client (which is hanging in this environment).
    """

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.session_id: str | None = None

    async def initialize(self) -> dict:
        async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
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
                        "protocolVersion": "2025-06-18",
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
            return _parse_response(resp)

    async def tools_list(self) -> dict:
        if not self.session_id:
            await self.initialize()

        async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
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
        if not self.session_id:
            await self.initialize()

        async with httpx.AsyncClient(trust_env=False, timeout=30.0) as client:
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
