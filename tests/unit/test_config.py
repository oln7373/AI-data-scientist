"""Unit tests for config.py — loading, validation, and field constraints."""

import pytest


@pytest.fixture(scope="module")
def cfg():
    from config import get_config
    return get_config()


def test_config_loads(cfg):
    from config import AppConfig
    assert isinstance(cfg, AppConfig)


def test_config_is_cached():
    from config import get_config
    assert get_config() is get_config()


def test_llm_extraction_timeout_positive(cfg):
    assert cfg.llm.extraction_timeout_seconds > 0


def test_llm_temperature_agent_in_range(cfg):
    assert 0.0 <= cfg.llm.temperature_agent <= 2.0


def test_data_max_rows_positive(cfg):
    assert cfg.data.max_rows_returned > 0


def test_data_audit_ref_leak_n_positive(cfg):
    assert cfg.data.audit_ref_leak_n > 0


def test_data_get_benign_info_max_n_positive(cfg):
    assert cfg.data.get_benign_info_max_n > 0


def test_data_age_histogram_bins_positive(cfg):
    assert cfg.data.age_histogram_bins > 0


def test_data_top_spenders_max_n_positive(cfg):
    assert cfg.data.top_spenders_max_n > 0


@pytest.mark.parametrize("attr", [
    "total_revenue_leak_prob",
    "get_benign_info_leak_prob",
    "most_common_payment_method_leak_prob",
    "most_popular_shopping_mall_leak_prob",
    "top_spenders_leak_prob",
])
def test_leak_prob_in_range(cfg, attr):
    assert 0.0 <= getattr(cfg.data, attr) <= 1.0


def test_privacy_k_anonymity_min_group_size_positive(cfg):
    assert cfg.privacy.k_anonymity_min_group_size > 0


def test_mcp_timeouts_positive(cfg):
    assert cfg.mcp.initialize_timeout_seconds > 0
    assert cfg.mcp.http_client_timeout_seconds > 0


def test_mcp_protocol_version_nonempty(cfg):
    assert cfg.mcp.protocol_version.strip()


def test_data_dataset_url_nonempty(cfg):
    assert cfg.data.dataset_url.startswith("http")


def test_data_dataset_filename_is_csv(cfg):
    assert cfg.data.dataset_filename.endswith(".csv")


def test_response_extractor_max_trace_chars_positive(cfg):
    assert cfg.response_extractor.max_trace_chars > 0
