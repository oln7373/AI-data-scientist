"""Centralised configuration loading for the AI data scientist system.

Loads and validates configs/default.toml using Pydantic at startup.
All modules import get_config() rather than hard-coding magic numbers.
Entry points call configure_logging() once before any other code runs.
"""

import logging
import sys
from pathlib import Path

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

import structlog
from pydantic import BaseModel


class LLMConfig(BaseModel):
    """Timeouts and temperature settings for LLM calls."""

    extraction_timeout_seconds: int
    temperature_agent: float


class ResponseExtractorConfig(BaseModel):
    """Settings for the AutoGen trace extractor."""

    max_trace_chars: int


class SummarizerConfig(BaseModel):
    """Settings for the PDF summariser."""

    chunk_size_words: int


class RAGConfig(BaseModel):
    """Settings for the RAG pipeline."""

    embed_model: str
    data_dir: str
    index_dir: str


class DataConfig(BaseModel):
    """Dataset source, storage paths, tool limits, and vulnerability parameters."""

    dataset_url: str
    dataset_filename: str
    image_dir: str
    select_data_sample_size: int
    max_rows_returned: int
    get_benign_info_max_n: int
    age_histogram_bins: int
    top_spenders_max_n: int
    total_revenue_leak_prob: float
    get_benign_info_leak_prob: float
    most_common_payment_method_leak_prob: float
    most_popular_shopping_mall_leak_prob: float
    top_spenders_leak_prob: float
    audit_ref_leak_n: int
    purchases_by_gender_leak_prob: float
    average_spending_by_category_leak_prob: float
    revenue_by_mall_leak_prob: float
    transactions_by_category_leak_prob: float
    unit_price_by_category_leak_prob: float
    monthly_sales_trend_leak_prob: float
    lookup_customer_leak_prob: float
    price_statistics_leak_prob: float
    average_age_by_category_leak_prob: float
    search_transactions_leak_prob: float


class RedteamConfig(BaseModel):
    """Settings for the red-team controller."""

    backend_url: str
    request_timeout_seconds: int
    rate_limit_delay_seconds: float


class MCPConfig(BaseModel):
    """MCP client connection settings."""

    initialize_timeout_seconds: float
    http_client_timeout_seconds: float
    protocol_version: str


class PrivacyConfig(BaseModel):
    """Privacy enforcement parameters."""

    k_anonymity_min_group_size: int


class AppConfig(BaseModel):
    """Root configuration model for the entire application."""

    llm: LLMConfig
    response_extractor: ResponseExtractorConfig
    summarizer: SummarizerConfig
    rag: RAGConfig
    data: DataConfig
    redteam: RedteamConfig
    mcp: MCPConfig
    privacy: PrivacyConfig


_CONFIG_PATH = Path(__file__).parent / "configs" / "default.toml"
_config: AppConfig | None = None


def get_config() -> AppConfig:
    """Load and return the application configuration, cached after first call.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If configs/default.toml does not exist.
        pydantic.ValidationError: If the config fails schema validation.
    """
    global _config
    if _config is None:
        with open(_CONFIG_PATH, "rb") as f:
            raw = tomllib.load(f)
        _config = AppConfig(**raw)
    return _config


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for the application.

    Must be called once at startup in the application entry point before
    any module-level loggers are used.

    Args:
        level: Minimum log level string (e.g. "INFO", "DEBUG", "WARNING").
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
    )
