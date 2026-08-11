"""Configuration and connectivity checker for AI-data-scientist.

Verifies that configs/default.toml is valid, required .env variables are
present, and the configured LLM endpoint responds. Run this before starting
the testbed to catch misconfiguration early.

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
    n = (
        len(cfg.llm.model_fields)
        + len(cfg.data.model_fields)
        + len(cfg.mcp.model_fields)
        + len(cfg.privacy.model_fields)
        + len(cfg.redteam.model_fields)
        + len(cfg.response_extractor.model_fields)
    )
    return f"{n} parameters validated"


def _check_env_vars() -> str:
    required = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    missing = [v for v in required if not os.getenv(v)]
    if missing:
        raise EnvironmentError(f"missing: {', '.join(missing)}")
    return f"model={os.getenv('LLM_MODEL')}  url={os.getenv('LLM_BASE_URL')}"


def _check_llm() -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=os.getenv("LLM_API_KEY", "ollama"),
    )
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL"),
        messages=[{"role": "user", "content": "Reply with the single word: pong"}],
        max_tokens=8,
        timeout=20,
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
    results.append(_check("Required LLM variables present", _check_env_vars))

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
