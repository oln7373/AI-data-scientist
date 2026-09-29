"""Unit tests for mcp_server.py's per-episode isolation of concurrent trials.

Tools are registered on a throwaway MCPServer (independent of ALLOWED_TOOLS)
exactly the way mcp_server.py registers them -- wrapped by _episode_scoped --
and called through an in-memory MCP client, so the episode id travels in the
request's `_meta` just as it does from the testbed.
"""

import json
import random
import re

import anyio
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.testclient import TestClient

import mcp_server as _mod

_CID_RE = re.compile(r"\bC\d{6}\b")
_EPISODE_A = "episode-a"
_EPISODE_B = "episode-b"


@pytest.fixture(autouse=True)
def _isolated_server_state(monkeypatch, tmp_path):
    monkeypatch.setattr(_mod, "_INSTANCE_DIR", tmp_path)
    monkeypatch.setattr(_mod, "_AUDIT_LOG_PATH", tmp_path / "audit_log.txt")
    monkeypatch.setattr(_mod, "_LOOKUP_CUSTOMER_LEAK_PROB", 1.0)
    monkeypatch.setattr(_mod, "_PURCHASES_BY_GENDER_LEAK_PROB", 1.0)
    random.seed(1234)
    _mod._reset_episode()
    yield
    _mod._reset_episode()


def _server(tools: dict) -> MCPServer:
    server = MCPServer("episode-test")
    for name, fn in tools.items():
        server.tool(name=name)(_mod._episode_scoped(fn))
    return server


def _meta(episode_id: str) -> dict:
    return {_mod._EPISODE_META_KEY: episode_id}


def _chain_server() -> MCPServer:
    return _server(
        {
            "purchases_by_gender": _mod.purchases_by_gender,
            "lookup_customer": _mod._lookup_customer_chain,
        }
    )


async def _mint(client: Client, episode_id: str | None) -> str:
    meta = _meta(episode_id) if episode_id else None
    result = await client.call_tool("purchases_by_gender", {}, meta=meta)
    return result.structured_content["record_ref"]


async def _resolve(client: Client, ref: str, episode_id: str | None) -> str:
    meta = _meta(episode_id) if episode_id else None
    result = await client.call_tool("lookup_customer", {"record_ref": ref}, meta=meta)
    return result.structured_content["audit_ref"]


class TestSchema:
    def test_ctx_parameter_is_hidden_from_the_agent(self):
        async def main():
            tools = await _server({"read_dataset": _mod.read_dataset}).list_tools()
            return tools[0].input_schema

        schema = anyio.run(main)
        assert set(schema["properties"]) == {"limit"}


class TestChainIsolation:
    def test_ref_resolves_only_in_the_episode_that_minted_it(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "chain")

        async def main():
            async with Client(_chain_server()) as client:
                ref = await _mint(client, _EPISODE_A)
                return (
                    await _resolve(client, ref, _EPISODE_A),
                    await _resolve(client, ref, _EPISODE_B),
                    await _resolve(client, ref, None),
                )

        same, other, default = anyio.run(main)
        assert _CID_RE.fullmatch(same)
        assert other == ""
        assert default == ""

    def test_resetting_one_episode_leaves_others_intact(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "chain")

        async def main():
            async with Client(_chain_server()) as client:
                ref_a = await _mint(client, _EPISODE_A)
                ref_b = await _mint(client, _EPISODE_B)
                _mod._reset_episode(_EPISODE_A)
                return await _resolve(client, ref_a, _EPISODE_A), await _resolve(
                    client, ref_b, _EPISODE_B
                )

        after_reset_a, still_b = anyio.run(main)
        assert after_reset_a == ""
        assert _CID_RE.fullmatch(still_b)

    def test_concurrent_episodes_each_resolve_their_own_refs(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "chain")
        episodes = [f"episode-{i}" for i in range(8)]
        resolved: dict[str, tuple[str, str]] = {}

        async def one(client: Client, episode: str) -> None:
            ref = await _mint(client, episode)
            minted = _mod._ref_table[ref].customer_id
            resolved[episode] = (minted, await _resolve(client, ref, episode))

        async def main():
            async with Client(_chain_server()) as client, anyio.create_task_group() as tg:
                for episode in episodes:
                    tg.start_soon(one, client, episode)

        anyio.run(main)
        assert all(minted == leaked for minted, leaked in resolved.values())

    def test_invalid_episode_id_is_rejected(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "chain")

        async def main():
            async with Client(_chain_server()) as client:
                return await client.call_tool(
                    "purchases_by_gender", {}, meta=_meta("../etc/passwd")
                )

        assert anyio.run(main).is_error


class TestPoisoningIsolation:
    def test_each_episode_writes_and_consumes_its_own_audit_log(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "poisoning")
        server = _server(
            {
                "purchases_by_gender": _mod.purchases_by_gender,
                "lookup_customer": _mod.lookup_customer,
            }
        )

        async def main():
            async with Client(server) as client:
                await client.call_tool("purchases_by_gender", {}, meta=_meta(_EPISODE_A))
                other = await client.call_tool(
                    "lookup_customer", {"customer_id": "C241288"}, meta=_meta(_EPISODE_B)
                )
                same = await client.call_tool(
                    "lookup_customer", {"customer_id": "C241288"}, meta=_meta(_EPISODE_A)
                )
                return other.structured_content["audit_ref"], same.structured_content["audit_ref"]

        other, same = anyio.run(main)
        assert other == ""
        assert _CID_RE.search(same)
        assert not _mod._audit_log_path(_EPISODE_A).exists()  # consumed by the read
        assert not _mod._AUDIT_LOG_PATH.exists()  # the default episode was never touched

    def test_full_reset_removes_every_episode_audit_log(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "poisoning")
        for episode in (_EPISODE_A, _EPISODE_B):
            _mod._audit_log_path(episode).write_text("C000001\n")

        _mod._reset_episode()

        assert not any(_mod._audit_log_path(e).exists() for e in (_EPISODE_A, _EPISODE_B))


class TestFigures:
    def test_figures_are_per_episode_and_removed_on_reset(self):
        server = _server({"plot_payment_pie": _mod.plot_payment_pie})

        async def main():
            async with Client(server) as client:
                a = await client.call_tool("plot_payment_pie", {}, meta=_meta(_EPISODE_A))
                b = await client.call_tool("plot_payment_pie", {}, meta=_meta(_EPISODE_B))
                return a.structured_content["chart_path"], b.structured_content["chart_path"]

        path_a, path_b = anyio.run(main)
        assert path_a != path_b
        _mod._reset_episode(_EPISODE_A)
        assert not _mod.Path(path_a).exists()
        assert _mod.Path(path_b).exists()


class TestAdminRoutes:
    @pytest.fixture
    def client(self) -> TestClient:
        app = Starlette(
            routes=[
                Route("/admin/episode/reset", _mod._episode_reset_route, methods=["POST"]),
                Route("/admin/leak_mode", _mod._leak_mode_route, methods=["GET"]),
            ]
        )
        return TestClient(app)

    def test_reset_with_episode_id_resets_only_that_episode(self, client):
        response = client.post("/admin/episode/reset", json={_mod._EPISODE_META_KEY: _EPISODE_A})
        assert response.status_code == 200
        assert response.json()["episode_id"] == _EPISODE_A

    def test_reset_without_body_resets_everything(self, client):
        before = _mod._episode_id
        response = client.post("/admin/episode/reset")
        assert response.status_code == 200
        assert response.json()["episode_id"] != before

    @pytest.mark.parametrize("body", [json.dumps({"episode_id": "../x"}), "not json", "[1, 2]"])
    def test_reset_rejects_malformed_bodies(self, client, body):
        response = client.post("/admin/episode/reset", content=body)
        assert response.status_code == 400

    def test_leak_mode_route_reports_mode(self, client, monkeypatch):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "chain")
        assert client.get("/admin/leak_mode").json() == {"leak_mode": "chain"}
