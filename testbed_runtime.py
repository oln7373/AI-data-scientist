"""Concurrency plumbing that lets one testbed serve many red-team trials at once.

Two pieces, both independent of AutoGen so they can be unit-tested directly:

- EndpointPool routes each trial to one of several OpenAI-compatible LLM
  servers (typically one Ollama per GPU), picking the server with the lowest
  in_flight / slots. A trial waits in the testbed when every server is full,
  so it goes to whichever one frees a slot first.
- run_agent_chat runs a blocking AutoGen chat on a worker thread under a
  capacity limiter, with a timeout that actually returns on time.
"""

import threading
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import anyio
import structlog

from config import LLMEndpoint

logger = structlog.get_logger(__name__)


@dataclass
class Endpoint:
    """One LLM server in the pool, with its live load.

    Attributes:
        url: OpenAI-compatible base URL.
        slots: Trials this server may run at once.
        in_flight: Trials currently pinned to this server.
    """

    url: str
    slots: int
    in_flight: int = 0


def parse_endpoints(raw: str, default_slots: int) -> list[LLMEndpoint]:
    """Parse the LLM_ENDPOINTS env var ("url=slots,url=slots").

    Args:
        raw: Comma-separated endpoints; "=slots" is optional per entry.
        default_slots: Slots for an entry that does not give its own.

    Returns:
        The validated endpoints, in the order given.

    Raises:
        ValueError: If an entry's slot count is not a positive integer.
    """
    endpoints = []
    for entry in (part.strip() for part in raw.split(",")):
        if not entry:
            continue
        url, sep, slots = entry.rpartition("=") if "=" in entry else (entry, "", "")
        try:
            count = int(slots) if sep else default_slots
        except ValueError as err:
            raise ValueError(f"LLM_ENDPOINTS entry {entry!r}: slots must be an integer") from err
        endpoints.append(LLMEndpoint(url=url.strip(), slots=count))
    return endpoints


class EndpointPool:
    """Least-loaded router over a fixed set of LLM endpoints.

    An empty pool routes nothing: lease() yields None and the caller keeps its
    provider's default URL (the single-server, Bedrock, and Azure cases).

    Attributes:
        endpoints: The endpoints, in configured order (earlier wins ties).
    """

    def __init__(self, endpoints: list[LLMEndpoint]) -> None:
        """Build the pool.

        Args:
            endpoints: Endpoints to route across; may be empty.
        """
        self.endpoints = [Endpoint(url=e.url, slots=e.slots) for e in endpoints]
        total = sum(e.slots for e in self.endpoints)
        self._capacity = anyio.Semaphore(total) if total else None

    @asynccontextmanager
    async def lease(self) -> AsyncGenerator[Endpoint | None, None]:
        """Pin one trial to the least-loaded endpoint for its whole duration.

        Pinning a whole trial (not each LLM call) keeps its growing prompt on
        one server, whose prompt cache can then reuse it turn after turn.

        Yields:
            The chosen endpoint, or None for an empty pool.
        """
        if self._capacity is None:
            yield None
            return
        async with self._capacity:
            # The semaphore guarantees a free slot somewhere, and nothing awaits
            # between choosing and incrementing, so this is race-free on the loop.
            endpoint = min(
                (e for e in self.endpoints if e.in_flight < e.slots),
                key=lambda e: e.in_flight / e.slots,
            )
            endpoint.in_flight += 1
            try:
                yield endpoint
            finally:
                endpoint.in_flight -= 1

    def load(self) -> dict[str, str]:
        """Return each endpoint's current in_flight/slots, for logging."""
        return {e.url: f"{e.in_flight}/{e.slots}" for e in self.endpoints}


@dataclass
class ChatOutcome:
    """How one agent chat ended.

    Attributes:
        result: AutoGen's ChatResult, or None if the chat did not finish.
        status: "ok", "timed_out", or "error".
        error: "<ExceptionType>: <message>" when status is "error".
    """

    result: Any  # Any: autogen.ChatResult is not re-exported with a stable type
    status: str
    error: str | None = None


async def run_agent_chat(
    chat: Callable[[], Any],
    *,
    limiter: anyio.CapacityLimiter,
    timeout_seconds: float,
    cancelled: threading.Event,
) -> ChatOutcome:
    """Run a blocking chat on a worker thread, bounded in count and in time.

    Waiting for a limiter token does not count against the timeout, so a
    queued trial is never reported as timed out before it starts. On timeout
    the thread is abandoned so the request returns promptly, and
    ``cancelled`` is set so the chat stops at its next turn or tool call
    rather than running on in the background.

    Args:
        chat: Zero-argument callable that runs the whole chat.
        limiter: Caps how many chats run at once.
        timeout_seconds: Wall-clock budget for the chat once it has started.
        cancelled: Set when the chat ends abnormally; the chat's tools and
            termination check watch it.

    Returns:
        The chat's outcome. Never raises for a failed chat: the error is
        reported so the caller can tell it apart from a clean, leak-free trial.
    """
    async with limiter:
        try:
            with anyio.fail_after(timeout_seconds):
                # A private one-token limiter: the shared one is already held
                # above, and anyio's default (40 threads) must not cap us.
                result = await anyio.to_thread.run_sync(
                    chat, abandon_on_cancel=True, limiter=anyio.CapacityLimiter(1)
                )
            return ChatOutcome(result=result, status="ok")
        except TimeoutError:
            cancelled.set()
            return ChatOutcome(result=None, status="timed_out")
        except Exception as err:  # noqa: BLE001 - a failed chat is reported, never scored clean
            cancelled.set()
            logger.error("agent_chat_error", error=str(err), error_type=type(err).__name__)
            return ChatOutcome(result=None, status="error", error=f"{type(err).__name__}: {err}")
