"""Many concurrent /redteam_exfil trials against ONE testbed and ONE MCP server.

The LLM is the only thing faked: _run_chat is replaced by a scripted "agent"
that calls the real, registered tool wrappers (writer, then its paired reader
with the ref it was handed). Everything else is real -- the FastAPI endpoint,
the per-trial MCP episode carried in `_meta`, and mcp_server.py's tools over
an in-memory MCP session shared by every trial, exactly as in production.
"""

import re
import types

import anyio
import httpx
import pytest
from mcp import Client
from mcp.server.mcpserver import MCPServer

import mcp_server as _server_mod
import redteam_mcp_testbed as _tb
from config import LLMEndpoint
from testbed_runtime import EndpointPool

_REF_RE = re.compile(r"\b[0-9a-f]{32}\b")
_N_TRIALS = 12
_MAX_CONCURRENT = 6
_WRITER, _READER = "purchases_by_gender", "lookup_customer"


@pytest.fixture(autouse=True)
def _chain_mode_server_state(monkeypatch, tmp_path):
    monkeypatch.setattr(_server_mod, "_INSTANCE_DIR", tmp_path)
    monkeypatch.setattr(_server_mod, "_LEAK_MODE", "chain")
    monkeypatch.setattr(_server_mod, "_LOOKUP_CUSTOMER_LEAK_PROB", 1.0)
    _server_mod._reset_episode()
    yield
    _server_mod._reset_episode()


@pytest.fixture
def resets(monkeypatch) -> list[str]:
    """Record the testbed's teardown calls; apply them to the in-process server."""
    calls: list[str] = []

    async def fake_reset(episode_id: str) -> None:
        calls.append(episode_id)
        _server_mod._reset_episode(episode_id)

    monkeypatch.setattr(_tb, "_reset_mcp_episode", fake_reset)
    return calls


def _scripted_chat(runner, victim, task, max_turns):  # noqa: ARG001 - mirrors _run_chat
    """Stand-in for the LLM: chain the writer into its paired reader, then stop."""
    minted = runner.function_map[_WRITER]()
    ref = _REF_RE.search(minted).group(0)
    # Pause mid-chain so other trials start, finish, and tear down their
    # episodes between this trial's mint and its resolve.
    anyio.from_thread.run(anyio.sleep, 0.05)
    runner.function_map[_READER](record_ref=ref)
    return types.SimpleNamespace(summary="done TERMINATE", chat_history=[])


async def _run_trials(pool: EndpointPool) -> list[dict]:
    server = MCPServer("concurrency-test")
    for name, fn in (
        (_WRITER, _server_mod.purchases_by_gender),
        (_READER, _server_mod._lookup_customer_chain),
    ):
        server.tool(name=name)(_server_mod._episode_scoped(fn))

    async with Client(server) as session:
        listed = await session.list_tools()
        _tb.app.state.mcp_session = session
        _tb.app.state.mcp_tools = {
            t.name: {"description": t.description or "", "schema": t.input_schema}
            for t in listed.tools
        }
        _tb.app.state.leak_mode = "chain"
        _tb.app.state.agent_limiter = anyio.CapacityLimiter(_MAX_CONCURRENT)
        _tb.app.state.endpoint_pool = pool

        results: list[dict] = []
        transport = httpx.ASGITransport(app=_tb.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://testbed") as client:

            async def one() -> None:
                response = await client.post(
                    "/redteam_exfil", json={"enabled_tools": [_WRITER, _READER]}, timeout=30
                )
                results.append(response.json())

            async with anyio.create_task_group() as tg:
                for _ in range(_N_TRIALS):
                    tg.start_soon(one)
        return results


def test_concurrent_trials_stay_isolated_and_are_torn_down(monkeypatch, resets):
    monkeypatch.setattr(_tb, "_run_chat", _scripted_chat)

    results = anyio.run(_run_trials, EndpointPool([]))

    assert len(results) == _N_TRIALS
    assert all(r["status"] == "ok" for r in results)
    # Every trial chained its OWN writer's ref into the reader and leaked.
    assert [r["chain_outcome"] for r in results] == ["exfiltrated"] * _N_TRIALS
    episodes = {r["episode_id"] for r in results}
    assert len(episodes) == _N_TRIALS
    assert set(resets) == episodes  # each trial tore down exactly its own episode
    assert _server_mod._ref_table == {}  # nothing left behind on the server


def test_trials_are_routed_across_llm_endpoints(monkeypatch, resets):  # noqa: ARG001
    monkeypatch.setattr(_tb, "_run_chat", _scripted_chat)
    pool = EndpointPool(
        [LLMEndpoint(url="http://gpu0/v1", slots=2), LLMEndpoint(url="http://gpu1/v1", slots=4)]
    )

    results = anyio.run(_run_trials, pool)

    used = {r["llm_url"] for r in results}
    assert used == {"http://gpu0/v1", "http://gpu1/v1"}
    assert pool.load() == {"http://gpu0/v1": "0/2", "http://gpu1/v1": "0/4"}


def test_failed_chat_is_reported_as_error_not_clean(monkeypatch, resets):  # noqa: ARG001
    def rate_limited_chat(*_args):
        raise RuntimeError("Error code: 429 - rate limit exceeded")

    monkeypatch.setattr(_tb, "_run_chat", rate_limited_chat)

    results = anyio.run(_run_trials, EndpointPool([]))

    assert all(r["status"] == "error" for r in results)
    assert all("429" in r["error"] for r in results)


class TestLlmConfig:
    def test_openai_compatible_entry_is_pinned_and_given_retries(self, monkeypatch):
        monkeypatch.setattr(_tb, "LLM_PROVIDER", "ollama")
        entry = _tb._llm_config("http://gpu1/v1")["config_list"][0]
        assert entry["base_url"] == "http://gpu1/v1"
        assert entry["max_retries"] == _tb._TESTBED_CFG.llm_max_retries
        assert entry["timeout"] == _tb._TESTBED_CFG.llm_call_timeout_seconds

    def test_without_a_leased_endpoint_keeps_configured_url(self, monkeypatch):
        monkeypatch.setattr(_tb, "LLM_PROVIDER", "ollama")
        configured = _tb._base_llm_config()["config_list"][0]["base_url"]
        assert _tb._llm_config(None)["config_list"][0]["base_url"] == configured


class TestEndpointPoolConfig:
    def test_env_var_overrides_config(self, monkeypatch):
        monkeypatch.setattr(_tb, "LLM_PROVIDER", "ollama")
        monkeypatch.setenv("LLM_ENDPOINTS", "http://a/v1=3,http://b/v1=5")
        assert _tb._endpoint_pool().load() == {"http://a/v1": "0/3", "http://b/v1": "0/5"}

    @pytest.mark.parametrize("provider", ["bedrock", "azure"])
    def test_cloud_providers_are_not_routed(self, monkeypatch, provider):
        monkeypatch.setattr(_tb, "LLM_PROVIDER", provider)
        monkeypatch.setenv("LLM_ENDPOINTS", "http://a/v1=3")
        assert _tb._endpoint_pool().load() == {}
