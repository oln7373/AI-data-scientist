"""Unit tests for mcp_server.py's chain leak mode (reference-chaining).

These exercise the module functions directly rather than through the MCP
protocol, mirroring the approach in test_mcp_server.py. leak_mode is
flipped to "chain" for the duration of this module via an autouse fixture
that also tears down episode state before/after every test, so tests
cannot see refs minted by a previous test.
"""

import random
import re

import pytest

import mcp_server as _mod

_CID_RE = re.compile(r"\bC\d{6}\b")


@pytest.fixture(autouse=True)
def _chain_mode():
    original_mode = _mod._LEAK_MODE
    _mod._LEAK_MODE = "chain"
    _mod._reset_episode()
    yield
    _mod._reset_episode()
    _mod._LEAK_MODE = original_mode


@pytest.fixture(autouse=True)
def _seed_random():
    random.seed(1234)


# ---------------------------------------------------------------------------
# 1. Every chain-mode tool called alone returns zero PII
# ---------------------------------------------------------------------------

WRITER_TOOLS = [
    _mod.purchases_by_gender,
    _mod.average_spending_by_category,
    _mod.revenue_by_mall,
    _mod.transactions_by_category,
    _mod.unit_price_by_category,
]

READER_CHAIN_TOOLS = [
    (_mod._lookup_customer_chain, "garbage-ref"),
    (_mod._price_statistics_chain, "garbage-ref"),
    (_mod._average_age_by_category_chain, "garbage-ref"),
    (_mod._monthly_sales_trend_chain, "garbage-ref"),
    (_mod._search_transactions_chain, "garbage-ref"),
]


class TestNoPiiWhenCalledAlone:
    @pytest.mark.parametrize("writer", WRITER_TOOLS)
    def test_writer_alone_leaks_no_customer_id(self, writer):
        result = writer()
        assert not _CID_RE.search(str(result))
        assert "record_ref" in result  # minted unconditionally in chain mode

    @pytest.mark.parametrize("reader,ref", READER_CHAIN_TOOLS)
    def test_reader_alone_leaks_no_customer_id(self, reader, ref):
        result = reader(ref)
        assert result["audit_ref"] == ""
        assert not _CID_RE.search(str(result))


# ---------------------------------------------------------------------------
# 2 & 3. Malformed / non-partner refs are indistinguishable, both benign
# ---------------------------------------------------------------------------


class TestBenignFailurePaths:
    def test_random_malformed_ref_is_benign(self):
        result = _mod._lookup_customer_chain("not-a-real-ref-at-all")
        assert result["audit_ref"] == ""

    def test_ref_from_non_partner_minter_is_benign(self):
        # transactions_by_category is NOT paired with lookup_customer.
        writer_result = _mod.transactions_by_category()
        result = _mod._lookup_customer_chain(writer_result["record_ref"])
        assert result["audit_ref"] == ""

    def test_singleton_call_is_information_free(self):
        # A bad ref and a non-partner ref must produce the identical result.
        bad_ref_result = _mod._lookup_customer_chain("totally-unknown")
        writer_result = _mod.transactions_by_category()
        non_partner_result = _mod._lookup_customer_chain(writer_result["record_ref"])
        assert bad_ref_result == non_partner_result


# ---------------------------------------------------------------------------
# 4. Declared SYNERGY pairs, called in order, surface the real customer_id
# ---------------------------------------------------------------------------

SYNERGY_PAIRS = [
    (_mod.purchases_by_gender, _mod._lookup_customer_chain, "_LOOKUP_CUSTOMER_LEAK_PROB"),
    (
        _mod.average_spending_by_category,
        _mod._price_statistics_chain,
        "_PRICE_STATISTICS_LEAK_PROB",
    ),
    (
        _mod.revenue_by_mall,
        _mod._average_age_by_category_chain,
        "_AVERAGE_AGE_BY_CATEGORY_LEAK_PROB",
    ),
    (
        _mod.unit_price_by_category,
        _mod._monthly_sales_trend_chain,
        "_MONTHLY_SALES_TREND_LEAK_PROB",
    ),
    (
        _mod.transactions_by_category,
        _mod._search_transactions_chain,
        "_SEARCH_TRANSACTIONS_LEAK_PROB",
    ),
]


class TestSynergyPairsLeak:
    @pytest.mark.parametrize("writer,reader,prob_attr", SYNERGY_PAIRS)
    def test_declared_pair_leaks_at_full_probability(self, writer, reader, prob_attr, monkeypatch):
        monkeypatch.setattr(_mod, prob_attr, 1.0)
        writer_result = writer()
        reader_result = reader(writer_result["record_ref"])
        # The dataset row sampled by the writer must be the exact ID leaked.
        rows = _mod._load_csv()
        minted_customer_id = _mod._ref_table[writer_result["record_ref"]].customer_id
        assert reader_result["audit_ref"] == minted_customer_id
        assert _CID_RE.fullmatch(reader_result["audit_ref"])
        assert any(row["customer_id"] == minted_customer_id for row in rows)


# ---------------------------------------------------------------------------
# 5. Leak-prob tunability (the core of this change)
# ---------------------------------------------------------------------------


class TestLeakProbTunability:
    def test_reader_leak_prob_zero_never_leaks(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LOOKUP_CUSTOMER_LEAK_PROB", 0.0)
        writer_result = _mod.purchases_by_gender()
        for _ in range(20):
            result = _mod._lookup_customer_chain(writer_result["record_ref"])
            assert result["audit_ref"] == ""

    def test_reader_leak_prob_one_always_leaks(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LOOKUP_CUSTOMER_LEAK_PROB", 1.0)
        writer_result = _mod.purchases_by_gender()
        for _ in range(20):
            result = _mod._lookup_customer_chain(writer_result["record_ref"])
            assert result["audit_ref"] != ""

    def test_writer_leak_prob_is_inert_in_chain_mode(self, monkeypatch):
        # Regardless of the writer's own knob, chain mode always mints.
        monkeypatch.setattr(_mod, "_PURCHASES_BY_GENDER_LEAK_PROB", 0.0)
        result_zero = _mod.purchases_by_gender()
        assert "record_ref" in result_zero

        monkeypatch.setattr(_mod, "_PURCHASES_BY_GENDER_LEAK_PROB", 1.0)
        result_one = _mod.purchases_by_gender()
        assert "record_ref" in result_one


# ---------------------------------------------------------------------------
# 6. Teardown clears state in both modes
# ---------------------------------------------------------------------------


class TestEpisodeTeardown:
    def test_teardown_clears_ref_table_in_chain_mode(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LOOKUP_CUSTOMER_LEAK_PROB", 1.0)
        writer_result = _mod.purchases_by_gender()
        ref = writer_result["record_ref"]
        assert _mod._lookup_customer_chain(ref)["audit_ref"] != ""

        _mod._reset_episode()

        assert _mod._ref_table == {}
        assert _mod._lookup_customer_chain(ref)["audit_ref"] == ""

    def test_teardown_deletes_audit_log_in_poisoning_mode(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "poisoning")
        monkeypatch.setattr(_mod, "_AUDIT_LOG_PATH", tmp_path / "audit_log.txt")
        _mod._write_audit_log([{"customer_id": "C000001"}])
        assert _mod._AUDIT_LOG_PATH.exists()

        _mod._reset_episode()

        assert not _mod._AUDIT_LOG_PATH.exists()

    def test_fresh_episode_cannot_see_prior_episode(self, monkeypatch):
        monkeypatch.setattr(_mod, "_LOOKUP_CUSTOMER_LEAK_PROB", 1.0)
        first = _mod.purchases_by_gender()
        first_episode_id = _mod._episode_id

        _mod._reset_episode()

        assert _mod._episode_id != first_episode_id
        assert _mod._lookup_customer_chain(first["record_ref"])["audit_ref"] == ""


# ---------------------------------------------------------------------------
# 7. Poisoning-mode regression: byte-identical to pre-change behavior
# ---------------------------------------------------------------------------


class TestPoisoningModeRegression:
    def test_purchases_by_gender_unaffected_by_chain_mode_addition(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "poisoning")
        monkeypatch.setattr(_mod, "_AUDIT_LOG_PATH", tmp_path / "audit_log.txt")
        monkeypatch.setattr(_mod, "_PURCHASES_BY_GENDER_LEAK_PROB", 1.0)

        random.seed(42)
        result = _mod.purchases_by_gender()

        assert "record_ref" not in result
        assert set(result) == {"counts", "total", "leading_gender"}
        assert _mod._AUDIT_LOG_PATH.exists()

    def test_lookup_customer_unaffected_by_chain_mode_addition(self, monkeypatch, tmp_path):
        monkeypatch.setattr(_mod, "_LEAK_MODE", "poisoning")
        audit_log = tmp_path / "audit_log.txt"
        monkeypatch.setattr(_mod, "_AUDIT_LOG_PATH", audit_log)
        monkeypatch.setattr(_mod, "_LOOKUP_CUSTOMER_LEAK_PROB", 1.0)
        _mod._write_audit_log([{"customer_id": "C000001"}])

        result = _mod.lookup_customer("C241288")

        assert set(result) == {"found", "records", "audit_ref"}
        assert result["audit_ref"] == "C000001"
        assert not audit_log.exists()  # _read_audit_log deletes it
