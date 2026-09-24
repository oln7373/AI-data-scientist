"""Unit tests for the testbed's per-trial token-usage summary."""

from redteam_mcp_testbed import _usage_from_summary


def _summary(actual: dict) -> dict:
    """Wrap an actual-usage block the way autogen.gather_usage_summary does."""
    return {"usage_including_cached_inference": actual, "usage_excluding_cached_inference": actual}


def test_sums_every_model_and_ignores_total_cost():
    usage = _usage_from_summary(
        _summary(
            {
                "total_cost": 0.5,
                "gpt-oss:20b": {"cost": 0, "prompt_tokens": 100, "completion_tokens": 40, "total_tokens": 140},
                "gpt-4o": {"cost": 0.5, "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        )
    )
    assert usage == {"prompt_tokens": 110, "completion_tokens": 45, "total_tokens": 155}


def test_excludes_cached_inference():
    summary = {
        "usage_including_cached_inference": {
            "total_cost": 0,
            "m": {"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100},
        },
        "usage_excluding_cached_inference": {
            "total_cost": 0,
            "m": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        },
    }
    assert _usage_from_summary(summary) == {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}


def test_no_reported_usage_returns_none():
    assert _usage_from_summary(_summary({"total_cost": 0})) is None
    assert _usage_from_summary({}) is None
