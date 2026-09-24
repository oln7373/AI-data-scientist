"""Unit tests for check_config.py's provider-aware LLM checks.

No network and no real AWS: llm_client's provider is monkeypatched and boto3 is
replaced by a stub where a check would otherwise look for real credentials.
"""

import sys
import types

import pytest

import check_config
import llm_client


@pytest.fixture(autouse=True)
def _clean_llm_env(monkeypatch):
    for name in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL", "AWS_REGION"):
        monkeypatch.delenv(name, raising=False)


def _set_provider(monkeypatch, provider: str) -> None:
    monkeypatch.setattr(llm_client, "LLM_PROVIDER", provider)


def _stub_boto3(monkeypatch, credentials) -> None:
    """Install a fake boto3 whose Session().get_credentials() returns credentials."""
    fake = types.SimpleNamespace(
        Session=lambda: types.SimpleNamespace(get_credentials=lambda: credentials)
    )
    monkeypatch.setitem(sys.modules, "boto3", fake)


# ---------------------------------------------------------------------------
# _provider
# ---------------------------------------------------------------------------


class TestProvider:
    @pytest.mark.parametrize(
        "configured,expected",
        [("", "openai"), ("ollama", "openai"), ("azure", "azure"), ("bedrock", "bedrock")],
    )
    def test_maps_llm_provider(self, monkeypatch, configured, expected):
        _set_provider(monkeypatch, configured)
        assert check_config._provider() == expected


# ---------------------------------------------------------------------------
# _check_env_vars
# ---------------------------------------------------------------------------


class TestEnvVarsOpenAiAndAzure:
    @pytest.mark.parametrize("provider", ["", "azure"])
    def test_passes_with_all_three_vars(self, monkeypatch, provider):
        _set_provider(monkeypatch, provider)
        monkeypatch.setenv("LLM_BASE_URL", "http://example.test/v1")
        monkeypatch.setenv("LLM_API_KEY", "key")
        monkeypatch.setenv("LLM_MODEL", "some-model")
        detail = check_config._check_env_vars()
        assert f"provider={provider or 'openai'}" in detail
        assert "some-model" in detail

    @pytest.mark.parametrize("provider", ["", "azure"])
    def test_names_the_missing_variable(self, monkeypatch, provider):
        _set_provider(monkeypatch, provider)
        monkeypatch.setenv("LLM_BASE_URL", "http://example.test/v1")
        monkeypatch.setenv("LLM_MODEL", "some-model")
        with pytest.raises(EnvironmentError, match="LLM_API_KEY"):
            check_config._check_env_vars()


class TestEnvVarsBedrock:
    def test_passes_without_api_key_or_base_url(self, monkeypatch):
        """Regression: Bedrock is documented with LLM_API_KEY unset."""
        _set_provider(monkeypatch, "bedrock")
        monkeypatch.setenv("LLM_MODEL", "us.amazon.nova-pro-v1:0")
        _stub_boto3(monkeypatch, types.SimpleNamespace(method="env"))
        detail = check_config._check_env_vars()
        assert "provider=bedrock" in detail
        assert "region=us-east-1" in detail
        assert "credentials=env" in detail

    def test_uses_configured_region(self, monkeypatch):
        _set_provider(monkeypatch, "bedrock")
        monkeypatch.setenv("LLM_MODEL", "m")
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        _stub_boto3(monkeypatch, types.SimpleNamespace(method="shared-credentials-file"))
        assert "region=eu-west-1" in check_config._check_env_vars()

    def test_requires_model(self, monkeypatch):
        _set_provider(monkeypatch, "bedrock")
        _stub_boto3(monkeypatch, types.SimpleNamespace(method="env"))
        with pytest.raises(EnvironmentError, match="LLM_MODEL"):
            check_config._check_env_vars()

    def test_fails_when_no_aws_credentials(self, monkeypatch):
        _set_provider(monkeypatch, "bedrock")
        monkeypatch.setenv("LLM_MODEL", "m")
        _stub_boto3(monkeypatch, None)
        with pytest.raises(EnvironmentError, match="no AWS credentials"):
            check_config._check_env_vars()

    def test_fails_clearly_when_boto3_missing(self, monkeypatch):
        _set_provider(monkeypatch, "bedrock")
        monkeypatch.setenv("LLM_MODEL", "m")
        monkeypatch.setitem(sys.modules, "boto3", None)  # makes `import boto3` raise
        with pytest.raises(ImportError, match="pip install boto3"):
            check_config._check_env_vars()

    def test_detail_never_contains_key_material(self, monkeypatch):
        _set_provider(monkeypatch, "bedrock")
        monkeypatch.setenv("LLM_MODEL", "m")
        secret = types.SimpleNamespace(
            method="env", access_key="AKIA-SECRET", secret_key="TOP-SECRET-VALUE"
        )
        _stub_boto3(monkeypatch, secret)
        detail = check_config._check_env_vars()
        assert "AKIA-SECRET" not in detail
        assert "TOP-SECRET-VALUE" not in detail


# ---------------------------------------------------------------------------
# _check_llm dispatch
# ---------------------------------------------------------------------------


def _chat_client(reply: str):
    message = types.SimpleNamespace(content=f" {reply} ")
    response = types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])
    completions = types.SimpleNamespace(create=lambda **_: response)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions))


class TestCheckLlm:
    @pytest.mark.parametrize("provider", ["", "azure"])
    def test_openai_and_azure_use_the_apps_sync_client(self, monkeypatch, provider):
        _set_provider(monkeypatch, provider)
        monkeypatch.setattr(llm_client, "get_sync_client", lambda: _chat_client("pong"))
        assert check_config._check_llm() == "response='pong'"

    def test_openai_path_sends_the_configured_model(self, monkeypatch):
        _set_provider(monkeypatch, "azure")
        monkeypatch.setattr(llm_client, "LLM_MODEL", "my-deployment")
        seen = {}

        def create(**kwargs):
            seen.update(kwargs)
            message = types.SimpleNamespace(content="pong")
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=create))
        )
        monkeypatch.setattr(llm_client, "get_sync_client", lambda: client)
        check_config._check_llm()
        assert seen["model"] == "my-deployment"

    def test_bedrock_uses_the_converse_client(self, monkeypatch):
        _set_provider(monkeypatch, "bedrock")
        monkeypatch.setattr(llm_client, "LLM_MODEL", "us.amazon.nova-pro-v1:0")
        calls = {}

        class FakeBedrockClient:
            def __init__(self, config):
                calls["config"] = config

            def create(self, params):
                calls["params"] = params
                return "raw"

            def message_retrieval(self, response):
                assert response == "raw"
                return [" pong "]

        monkeypatch.setattr(llm_client, "BedrockAutoGenClient", FakeBedrockClient)
        monkeypatch.setattr(
            llm_client,
            "get_sync_client",
            lambda: pytest.fail("Bedrock must not use the OpenAI-compatible client"),
        )
        assert check_config._check_llm() == "response='pong'"
        assert calls["config"] == {"model": "us.amazon.nova-pro-v1:0"}
        assert calls["params"]["messages"][0]["role"] == "user"
