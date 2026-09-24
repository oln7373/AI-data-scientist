"""Configuration and connectivity checker for AI-data-scientist.

Verifies that configs/default.toml is valid, the .env variables required by the
configured LLM provider are present, and the LLM responds. Provider handling
(OpenAI-compatible, Azure OpenAI, Amazon Bedrock) comes from llm_client.py, so
the checker exercises the same client the testbed uses. Run this before
starting the testbed to catch misconfiguration early.

Usage:
    python check_config.py
"""

import os
import socket
import sys
from pathlib import Path

# Ensure project root is on the path regardless of working directory.
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv

load_dotenv()

_PASS = "[ OK ]"
_FAIL = "[FAIL]"
_SKIP = "[SKIP]"

# Connectivity probe settings.
_PROBE_PROMPT = "Reply with the single word: pong"
_PROBE_MAX_TOKENS = 8
_PROBE_TIMEOUT_SECONDS = 20

# Variables every non-Bedrock provider needs. Bedrock authenticates with AWS
# credentials instead, so it needs only LLM_MODEL (see _check_env_vars).
_REQUIRED_KEY_VARS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")


def _check(label: str, fn) -> bool:
    """Run fn(), print a pass/fail line, and return success."""
    try:
        detail = fn()
        suffix = f"  ({detail})" if detail else ""
        print(f"  {_PASS}  {label}{suffix}")
        return True
    except Exception as exc:
        print(f"  {_FAIL}  {label}: {exc}")
        return False


# ---------------------------------------------------------------------------
# Individual checks
# ---------------------------------------------------------------------------

def _check_toml() -> str:
    from config import get_config
    cfg = get_config()
    sections = (cfg.llm, cfg.data, cfg.mcp, cfg.privacy, cfg.redteam, cfg.response_extractor)
    # model_fields must be read from the class, not the instance (deprecated in Pydantic 2.11).
    n = sum(len(type(section).model_fields) for section in sections)
    return f"{n} parameters validated"


def _provider() -> str:
    """Return the active LLM provider as resolved by llm_client.

    Returns:
        "bedrock", "azure", or "openai" (any other or unset LLM_PROVIDER is
        treated as a plain OpenAI-compatible endpoint, as llm_client does).
    """
    import llm_client

    return llm_client.LLM_PROVIDER if llm_client.LLM_PROVIDER in ("bedrock", "azure") else "openai"


def _check_aws_credentials() -> str:
    """Confirm boto3 is installed and AWS credentials can be resolved.

    Returns:
        A short description of where the credentials come from (never the
        credential values themselves).

    Raises:
        ImportError: If boto3 is not installed.
        EnvironmentError: If boto3 cannot find any AWS credentials.
    """
    try:
        import boto3
    except ImportError as err:
        raise ImportError(
            "Amazon Bedrock support requires boto3 and botocore. "
            "Install them with:  pip install boto3 botocore"
        ) from err
    credentials = boto3.Session().get_credentials()
    if credentials is None:
        raise EnvironmentError(
            "no AWS credentials found: set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY, "
            "AWS_PROFILE, or run under an IAM role"
        )
    return f"credentials={credentials.method or 'unknown'}"


def _check_env_vars() -> str:
    provider = _provider()
    model = os.getenv("LLM_MODEL")
    if provider == "bedrock":
        if not model:
            raise EnvironmentError("missing: LLM_MODEL")
        region = os.getenv("AWS_REGION", "us-east-1")
        return f"provider=bedrock  model={model}  region={region}  {_check_aws_credentials()}"
    missing = [v for v in _REQUIRED_KEY_VARS if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"missing: {', '.join(missing)}")
    return f"provider={provider}  model={model}  url={os.getenv('LLM_BASE_URL')}"


def _check_llm() -> str:
    import llm_client

    if _provider() == "bedrock":
        # The agent talks to Bedrock through the Converse API, so probe that path.
        client = llm_client.BedrockAutoGenClient({"model": llm_client.LLM_MODEL})
        response = client.create({"messages": [{"role": "user", "content": _PROBE_PROMPT}]})
        reply = client.message_retrieval(response)[0].strip()
    else:
        resp = llm_client.get_sync_client().chat.completions.create(
            model=llm_client.LLM_MODEL,
            messages=[{"role": "user", "content": _PROBE_PROMPT}],
            max_tokens=_PROBE_MAX_TOKENS,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
        reply = resp.choices[0].message.content.strip()
    return f"response={reply!r}"


def _check_mcp_port() -> str:
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8005"))
    with socket.create_connection((host, port), timeout=5):
        pass
    return f"something is listening on {host}:{port}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 56)
    print("  AI-data-scientist — configuration check")
    print("=" * 56)

    results: list[bool] = []

    print("\nConfig file (configs/default.toml):")
    results.append(_check("TOML loads and validates via Pydantic", _check_toml))

    print("\nEnvironment variables (.env):")
    results.append(_check("Required LLM settings present for the configured provider", _check_env_vars))

    print("\nLLM connectivity:")
    results.append(_check("LLM endpoint responds to a completion request", _check_llm))

    print("\nMCP server (optional — only if mcp_server.py is running):")
    try:
        ok = _check("MCP server port is open", _check_mcp_port)
        results.append(ok)
    except Exception:
        print(f"  {_SKIP}  MCP server not checked — start mcp_server.py first")

    passed = sum(results)
    total = len(results)
    print("\n" + "=" * 56)
    if passed == total:
        print(f"  All {total} checks passed.")
    else:
        print(f"  {passed}/{total} checks passed — fix the failures above.")
    print("=" * 56)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
