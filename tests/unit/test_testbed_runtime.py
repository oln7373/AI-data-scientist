"""Unit tests for testbed_runtime: LLM endpoint routing and the bounded chat runner."""

import threading
import time

import anyio
import pytest

from config import LLMEndpoint
from testbed_runtime import EndpointPool, parse_endpoints, run_agent_chat

# Short, test-only budgets: long enough to be reliable, short enough to keep the suite fast.
_SHORT_TIMEOUT = 0.2
_LONG_TIMEOUT = 5.0


def _pool(*slots: int) -> EndpointPool:
    return EndpointPool(
        [LLMEndpoint(url=f"http://gpu{i}/v1", slots=n) for i, n in enumerate(slots)]
    )


# ---------------------------------------------------------------------------
# parse_endpoints
# ---------------------------------------------------------------------------


class TestParseEndpoints:
    def test_parses_url_and_slots_pairs(self):
        endpoints = parse_endpoints("http://a:1/v1=8, http://b:2/v1=16", default_slots=4)
        assert [(e.url, e.slots) for e in endpoints] == [
            ("http://a:1/v1", 8),
            ("http://b:2/v1", 16),
        ]

    def test_missing_slots_use_default(self):
        endpoints = parse_endpoints("http://a:1/v1", default_slots=4)
        assert [(e.url, e.slots) for e in endpoints] == [("http://a:1/v1", 4)]

    def test_blank_entries_are_skipped(self):
        assert parse_endpoints(" , ", default_slots=4) == []

    def test_non_integer_slots_raise(self):
        with pytest.raises(ValueError, match="slots must be an integer"):
            parse_endpoints("http://a:1/v1=many", default_slots=4)

    def test_non_positive_slots_raise(self):
        with pytest.raises(ValueError):
            parse_endpoints("http://a:1/v1=0", default_slots=4)


# ---------------------------------------------------------------------------
# EndpointPool
# ---------------------------------------------------------------------------


class TestEndpointPool:
    def test_empty_pool_yields_none(self):
        async def main():
            async with EndpointPool([]).lease() as endpoint:
                return endpoint

        assert anyio.run(main) is None

    def test_spreads_trials_in_proportion_to_slots(self):
        pool = _pool(2, 4)

        async def main():
            async with (
                pool.lease() as first,
                pool.lease() as second,
                pool.lease() as third,
            ):
                return [first.url, second.url, third.url], pool.load()

        urls, load = anyio.run(main)
        # Ties go to the earlier endpoint; then the lower in_flight/slots wins.
        assert urls == ["http://gpu0/v1", "http://gpu1/v1", "http://gpu1/v1"]
        assert load == {"http://gpu0/v1": "1/2", "http://gpu1/v1": "2/4"}

    def test_releases_slot_even_when_trial_raises(self):
        pool = _pool(1)

        async def main():
            with pytest.raises(RuntimeError):
                async with pool.lease():
                    raise RuntimeError("trial blew up")
            return pool.load()

        assert anyio.run(main) == {"http://gpu0/v1": "0/1"}

    def test_waits_when_every_endpoint_is_full(self):
        pool = _pool(1)
        order: list[str] = []

        async def trial(name: str, hold: float) -> None:
            async with pool.lease():
                order.append(f"{name}-start")
                await anyio.sleep(hold)
                order.append(f"{name}-end")

        async def main():
            async with anyio.create_task_group() as tg:
                tg.start_soon(trial, "a", _SHORT_TIMEOUT)
                await anyio.sleep(0)  # let "a" take the only slot first
                tg.start_soon(trial, "b", 0)

        anyio.run(main)
        assert order == ["a-start", "a-end", "b-start", "b-end"]


# ---------------------------------------------------------------------------
# run_agent_chat
# ---------------------------------------------------------------------------


class TestRunAgentChat:
    def test_returns_result_on_success(self):
        async def main():
            return await run_agent_chat(
                lambda: "chat-result",
                limiter=anyio.CapacityLimiter(1),
                timeout_seconds=_LONG_TIMEOUT,
                cancelled=threading.Event(),
            )

        outcome = anyio.run(main)
        assert (outcome.result, outcome.status, outcome.error) == ("chat-result", "ok", None)

    def test_timeout_returns_promptly_and_cancels(self):
        release = threading.Event()
        cancelled = threading.Event()

        async def main():
            return await run_agent_chat(
                lambda: release.wait(_LONG_TIMEOUT),  # a chat that hangs
                limiter=anyio.CapacityLimiter(1),
                timeout_seconds=_SHORT_TIMEOUT,
                cancelled=cancelled,
            )

        start = time.perf_counter()
        outcome = anyio.run(main)
        elapsed = time.perf_counter() - start
        release.set()  # let the abandoned thread finish
        assert outcome.status == "timed_out"
        assert cancelled.is_set()
        assert elapsed < _LONG_TIMEOUT / 2

    def test_chat_exception_is_reported_not_raised(self):
        def failing_chat():
            raise ConnectionError("429 Too Many Requests")

        cancelled = threading.Event()

        async def main():
            return await run_agent_chat(
                failing_chat,
                limiter=anyio.CapacityLimiter(1),
                timeout_seconds=_LONG_TIMEOUT,
                cancelled=cancelled,
            )

        outcome = anyio.run(main)
        assert outcome.status == "error"
        assert outcome.error == "ConnectionError: 429 Too Many Requests"
        assert cancelled.is_set()

    def test_limiter_caps_concurrency_and_queue_time_is_not_timed(self):
        limiter = anyio.CapacityLimiter(1)
        running = 0
        peak = 0
        lock = threading.Lock()

        def chat() -> str:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(_SHORT_TIMEOUT / 2)
            with lock:
                running -= 1
            return "done"

        outcomes = []

        async def one() -> None:
            # Each chat fits its timeout, but three queued behind one slot would
            # not if waiting for the limiter counted against it.
            outcomes.append(
                await run_agent_chat(
                    chat,
                    limiter=limiter,
                    timeout_seconds=_SHORT_TIMEOUT,
                    cancelled=threading.Event(),
                )
            )

        async def main():
            async with anyio.create_task_group() as tg:
                for _ in range(3):
                    tg.start_soon(one)

        anyio.run(main)
        assert peak == 1
        assert [o.status for o in outcomes] == ["ok", "ok", "ok"]
