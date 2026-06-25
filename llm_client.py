"""Unified LLM client factory supporting all OpenAI-compatible providers and Amazon Bedrock.

All modules that need an LLM client import from here instead of computing
LLM_BASE_URL / LLM_API_KEY / LLM_MODEL themselves. Centralising the logic here
means Bedrock support (which requires SigV4 signing and a custom AutoGen model
client) is wired up in one place.

Environment variables
---------------------
LLM_BASE_URL     API endpoint (default: Ollama localhost).
LLM_API_KEY      API key ("ollama" for Ollama; leave unset for Bedrock — AWS
                 credentials are handled by boto3 via the standard AWS env vars).
LLM_MODEL        Model name exactly as the provider expects it.
LLM_PROVIDER     Set to "bedrock" to enable Amazon Bedrock. All other values (or
                 unset) use the OpenAI-compatible path via LLM_BASE_URL.
AWS_REGION       AWS region for Bedrock (default: us-east-1).
OLLAMA_PORT      Fallback Ollama port when LLM_BASE_URL is unset (default: 11434).

Standard AWS credential env vars honoured by boto3:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN, AWS_PROFILE.

Bedrock prerequisites (not pre-installed)::

    pip install boto3 botocore
"""

import os
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog
from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAI

from config import get_config

load_dotenv()

logger = structlog.get_logger(__name__)
_cfg = get_config()

LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "").strip().lower()

_ollama_port = os.getenv("OLLAMA_PORT", "11434")
LLM_BASE_URL: str = os.getenv("LLM_BASE_URL") or f"http://localhost:{_ollama_port}/v1"
LLM_API_KEY: str = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "ollama"
LLM_MODEL: str = (
    os.getenv("LLM_MODEL")
    or os.getenv("OPENAI_MODEL")
    or os.getenv("OLLAMA_MODEL")
    or "llama3.2"
)

AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_BASE_URL: str = f"https://bedrock-runtime.{AWS_REGION}.amazonaws.com/openai/v1"

LLM_URL: str = (
    f"{BEDROCK_BASE_URL}/chat/completions"
    if LLM_PROVIDER == "bedrock"
    else f"{LLM_BASE_URL}/chat/completions"
)


def _require_boto3() -> None:
    """Raise a clear ImportError if boto3 / botocore are not installed.

    Raises:
        ImportError: With install instructions when boto3 is absent.
    """
    try:
        import boto3  # noqa: F401
        import botocore  # noqa: F401
    except ImportError as err:
        raise ImportError(
            "Amazon Bedrock support requires boto3 and botocore. "
            "Install them with:  pip install boto3 botocore"
        ) from err


class _BedRockSigV4Auth(httpx.Auth):
    """httpx Auth handler that signs every request with AWS SigV4 for Bedrock.

    Attributes:
        _session: boto3 Session used to obtain frozen credentials.
        _region: AWS region string.
    """

    def __init__(self, region: str) -> None:
        """Initialise the SigV4 auth handler.

        Args:
            region: AWS region string (e.g. "us-east-1").

        Raises:
            ImportError: If boto3 / botocore are not installed.
        """
        _require_boto3()
        import boto3
        self._session = boto3.Session()
        self._region = region

    def auth_flow(self, request: httpx.Request):  # noqa: ANN201
        """Sign the outgoing request in-place using AWS SigV4 and yield it.

        Args:
            request: The httpx request to sign.

        Yields:
            The signed request.
        """
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        creds = self._session.get_credentials().get_frozen_credentials()
        aws_req = AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content or b"",
        )
        for k, v in request.headers.items():
            aws_req.headers[k] = v
        SigV4Auth(creds, "bedrock", self._region).add_auth(aws_req)
        for k, v in aws_req.headers.items():
            request.headers[k] = v
        yield request


def get_sync_client() -> OpenAI:
    """Return a configured synchronous OpenAI-compatible LLM client.

    For Bedrock (LLM_PROVIDER=bedrock): points to the Bedrock OpenAI-compatible
    endpoint and attaches a SigV4 signing transport via boto3.
    For all other providers: uses LLM_BASE_URL and LLM_API_KEY from the
    environment.

    Returns:
        Configured OpenAI client instance.

    Raises:
        ImportError: If LLM_PROVIDER=bedrock and boto3 is not installed.
    """
    if LLM_PROVIDER == "bedrock":
        return OpenAI(
            base_url=BEDROCK_BASE_URL,
            api_key="bedrock",
            http_client=httpx.Client(auth=_BedRockSigV4Auth(AWS_REGION), trust_env=False),
        )
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def get_async_client() -> AsyncOpenAI:
    """Return a configured async OpenAI-compatible LLM client.

    For Bedrock (LLM_PROVIDER=bedrock): points to the Bedrock OpenAI-compatible
    endpoint and attaches a SigV4 signing transport via boto3.
    For all other providers: uses LLM_BASE_URL and LLM_API_KEY from the
    environment.

    Returns:
        Configured AsyncOpenAI client instance.

    Raises:
        ImportError: If LLM_PROVIDER=bedrock and boto3 is not installed.
    """
    if LLM_PROVIDER == "bedrock":
        return AsyncOpenAI(
            base_url=BEDROCK_BASE_URL,
            api_key="bedrock",
            http_client=httpx.AsyncClient(auth=_BedRockSigV4Auth(AWS_REGION), trust_env=False),
        )
    return AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


# ---------------------------------------------------------------------------
# AutoGen 0.2 custom model client for Amazon Bedrock (Converse API)
# ---------------------------------------------------------------------------


@dataclass
class _BedrockMessage:
    content: str | None = None
    role: str = "assistant"
    function_call: Any = None
    tool_calls: list = field(default_factory=list)


@dataclass
class _BedrockChoice:
    message: _BedrockMessage = field(default_factory=_BedrockMessage)
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class _BedrockUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class _BedrockResponse:
    """OpenAI-compatible response wrapper for boto3 Converse API output."""

    choices: list = field(default_factory=list)
    usage: _BedrockUsage = field(default_factory=_BedrockUsage)
    model: str = ""
    id: str = ""


class BedrockAutoGenClient:
    """AutoGen 0.2 custom model client backed by Amazon Bedrock Converse API.

    Converts OpenAI-style message params to Bedrock Converse format and wraps
    the response in an OpenAI-compatible object that AutoGen can consume.

    Registration — call once before creating any AutoGen agents::

        import autogen
        from llm_client import BedrockAutoGenClient

        autogen.AssistantAgent.register_model_client(
            model_client_cls=BedrockAutoGenClient
        )

    Then include ``"model_client_cls": "BedrockAutoGenClient"`` in each
    config_list entry (see :func:`get_autogen_llm_config`).

    Attributes:
        _model_id: Bedrock model ID.
        _region: AWS region.
        _temperature: Sampling temperature loaded from config.
        _client: boto3 bedrock-runtime client.
    """

    def __init__(self, config: dict, **kwargs: Any) -> None:
        """Initialise from an AutoGen config dict.

        Args:
            config: Dict from AutoGen's config_list. Must contain ``model``
                (Bedrock model ID). Optionally ``aws_region`` and
                ``temperature``.
            **kwargs: Forwarded by AutoGen; unused.

        Raises:
            ImportError: If boto3 is not installed.
        """
        _require_boto3()
        import boto3

        self._model_id: str = config.get("model", LLM_MODEL)
        self._region: str = config.get("aws_region", AWS_REGION)
        self._temperature: float = float(
            config.get("temperature", _cfg.llm.temperature_agent)
        )
        self._client = boto3.client("bedrock-runtime", region_name=self._region)
        logger.info(
            "bedrock_autogen_client_init",
            model=self._model_id,
            region=self._region,
        )

    @staticmethod
    def _to_bedrock_messages(
        openai_messages: list[dict],
    ) -> tuple[list[dict], str | None]:
        """Convert OpenAI-format messages to Bedrock Converse format.

        Args:
            openai_messages: List of OpenAI-format message dicts.

        Returns:
            Tuple of (bedrock_messages, system_prompt_str or None).
        """
        bedrock_messages: list[dict] = []
        system_text: str | None = None

        for msg in openai_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            content = str(content or "")

            if role == "system":
                system_text = content
            elif role in ("user", "assistant"):
                bedrock_messages.append({"role": role, "content": [{"text": content}]})

        # Bedrock requires turns to alternate and start with "user".
        if bedrock_messages and bedrock_messages[0]["role"] != "user":
            bedrock_messages.insert(
                0, {"role": "user", "content": [{"text": "Begin."}]}
            )

        return bedrock_messages, system_text

    def create(self, params: dict) -> _BedrockResponse:
        """Call the Bedrock Converse API and return an OpenAI-compatible response.

        Args:
            params: AutoGen params dict containing at minimum ``messages``.

        Returns:
            OpenAI-compatible response wrapped in :class:`_BedrockResponse`.
        """
        messages, system_text = self._to_bedrock_messages(params.get("messages", []))
        converse_kwargs: dict = {
            "modelId": self._model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": self._temperature,
                "maxTokens": 4096,
            },
        }
        if system_text:
            converse_kwargs["system"] = [{"text": system_text}]

        raw = self._client.converse(**converse_kwargs)

        output_text = ""
        for block in raw.get("output", {}).get("message", {}).get("content", []):
            if "text" in block:
                output_text += block["text"]

        usage_meta = raw.get("usage", {})
        response = _BedrockResponse(
            choices=[_BedrockChoice(message=_BedrockMessage(content=output_text))],
            usage=_BedrockUsage(
                prompt_tokens=usage_meta.get("inputTokens", 0),
                completion_tokens=usage_meta.get("outputTokens", 0),
                total_tokens=usage_meta.get("totalTokens", 0),
            ),
            model=self._model_id,
        )
        logger.info(
            "bedrock_completion",
            model=self._model_id,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )
        return response

    def message_retrieval(self, response: _BedrockResponse) -> list[str]:
        """Extract message strings from a Bedrock response.

        Args:
            response: Response returned by :meth:`create`.

        Returns:
            List of content strings, one per choice.
        """
        return [c.message.content or "" for c in response.choices]

    def cost(self, response: _BedrockResponse) -> float:  # noqa: ARG002
        """Return cost estimate (always 0; Bedrock is billed directly by AWS).

        Args:
            response: Response returned by :meth:`create`.

        Returns:
            0.0.
        """
        return 0.0

    @staticmethod
    def get_usage(response: _BedrockResponse) -> dict:
        """Return token usage information from the response.

        Args:
            response: Response returned by :meth:`create`.

        Returns:
            Dict with ``prompt_tokens``, ``completion_tokens``, ``total_tokens``.
        """
        return {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }


def get_autogen_llm_config() -> dict:
    """Return an AutoGen llm_config dict for the active LLM provider.

    For Bedrock: the config_list entry includes ``model_client_cls`` pointing to
    :class:`BedrockAutoGenClient`. Callers must also register the client class
    before creating any agents::

        autogen.AssistantAgent.register_model_client(
            model_client_cls=BedrockAutoGenClient
        )

    For all other providers: returns a standard OpenAI-compatible config_list
    using LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL from the environment.

    Returns:
        Dict with a ``config_list`` key suitable for AutoGen agents/managers.
    """
    if LLM_PROVIDER == "bedrock":
        return {
            "config_list": [
                {
                    "model": LLM_MODEL,
                    "model_client_cls": "BedrockAutoGenClient",
                    "aws_region": AWS_REGION,
                    "temperature": _cfg.llm.temperature_agent,
                    "price": [0, 0],
                }
            ]
        }
    return {
        "config_list": [
            {
                "model": LLM_MODEL,
                "base_url": LLM_BASE_URL,
                "api_key": LLM_API_KEY,
                "temperature": _cfg.llm.temperature_agent,
                "price": [0, 0],
            }
        ]
    }
