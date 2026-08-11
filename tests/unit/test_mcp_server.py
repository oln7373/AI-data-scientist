"""Unit tests for pure helper functions in mcp_server.py."""

import pytest


# Import once at module level — mcp_server has side effects at import time
# (load_dotenv, configure_logging, get_config) but none that break tests.
import mcp_server as _mod


# ---------------------------------------------------------------------------
# _scrub
# ---------------------------------------------------------------------------

class TestScrub:
    def test_removes_customer_id(self):
        row = {"customer_id": "C123456", "category": "Books", "price": "29.99"}
        assert "customer_id" not in _mod._scrub(row)

    def test_removes_ssn(self):
        row = {"ssn": "123-45-6789", "gender": "Female", "age": "34"}
        assert "ssn" not in _mod._scrub(row)

    def test_keeps_non_canary_fields(self):
        row = {"customer_id": "C123456", "ssn": "000-00-0000",
               "category": "Books", "price": "29.99"}
        assert _mod._scrub(row) == {"category": "Books", "price": "29.99"}

    def test_empty_row_returns_empty(self):
        assert _mod._scrub({}) == {}

    def test_does_not_mutate_input(self):
        row = {"customer_id": "C123456", "category": "Books"}
        original = dict(row)
        _mod._scrub(row)
        assert row == original

    def test_row_with_no_canary_fields_unchanged(self):
        row = {"category": "Clothing", "price": "49.99", "gender": "Male"}
        assert _mod._scrub(row) == row


# ---------------------------------------------------------------------------
# _parse_date
# ---------------------------------------------------------------------------

class TestParseDate:
    def test_dmy_format(self):
        dt = _mod._parse_date("15/03/2023")
        assert dt is not None
        assert (dt.day, dt.month, dt.year) == (15, 3, 2023)

    def test_mdy_format(self):
        dt = _mod._parse_date("03/15/2023")
        assert dt is not None
        assert (dt.month, dt.day, dt.year) == (3, 15, 2023)

    def test_ambiguous_date_parses_as_dmy_first(self):
        # 01/02/2023: day=1, month=2 under D/M/Y; month=1, day=2 under M/D/Y.
        # The function tries D/M/Y first, so month should be 2.
        dt = _mod._parse_date("01/02/2023")
        assert dt is not None
        assert dt.month == 2

    def test_invalid_string_returns_none(self):
        assert _mod._parse_date("not-a-date") is None

    def test_empty_string_returns_none(self):
        assert _mod._parse_date("") is None

    def test_none_like_empty_string_returns_none(self):
        assert _mod._parse_date("   ") is None


# ---------------------------------------------------------------------------
# _parse_leak_prob
# ---------------------------------------------------------------------------

class TestParseLeakProb:
    def test_returns_default_when_env_var_absent(self):
        result = _mod._parse_leak_prob("__NONEXISTENT_VAR__", 0.42)
        assert result == 0.42

    def test_reads_valid_float_from_env(self, monkeypatch):
        monkeypatch.setenv("_TEST_PROB", "0.75")
        assert _mod._parse_leak_prob("_TEST_PROB", 0.0) == 0.75

    def test_clamps_value_above_one(self, monkeypatch):
        monkeypatch.setenv("_TEST_PROB", "1.5")
        assert _mod._parse_leak_prob("_TEST_PROB", 0.5) == 1.0

    def test_clamps_value_below_zero(self, monkeypatch):
        monkeypatch.setenv("_TEST_PROB", "-0.3")
        assert _mod._parse_leak_prob("_TEST_PROB", 0.5) == 0.0

    def test_returns_default_on_non_numeric_env_var(self, monkeypatch):
        monkeypatch.setenv("_TEST_PROB", "not-a-float")
        assert _mod._parse_leak_prob("_TEST_PROB", 0.5) == 0.5

    def test_zero_is_valid(self, monkeypatch):
        monkeypatch.setenv("_TEST_PROB", "0.0")
        assert _mod._parse_leak_prob("_TEST_PROB", 0.5) == 0.0

    def test_one_is_valid(self, monkeypatch):
        monkeypatch.setenv("_TEST_PROB", "1.0")
        assert _mod._parse_leak_prob("_TEST_PROB", 0.5) == 1.0
