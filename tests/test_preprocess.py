"""
tests/test_preprocess.py
========================
Unit tests for src/preprocess.py
"""

import pandas as pd

from src.preprocess import (
    clean_text,
    filter_narratives,
    map_products,
    TARGET_PRODUCTS,
)


# ── clean_text ────────────────────────────────────────────────────────────────


class TestCleanText:
    def test_lowercases(self):
        assert clean_text("HELLO WORLD") == "hello world"

    def test_removes_boilerplate_writing_to_complain(self):
        result = clean_text("I am writing to file a complaint about my card.")
        assert "i am writing to file a complaint" not in result

    def test_removes_xxxx_placeholders(self):
        result = clean_text("My account XXXX was charged incorrectly.")
        assert "xxxx" not in result

    def test_strips_special_characters(self):
        result = clean_text("Hello @world! #test $100 % discount")
        for char in ["@", "#", "$", "%"]:
            assert char not in result

    def test_collapses_whitespace(self):
        result = clean_text("too    many     spaces")
        assert "  " not in result

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_returns_empty(self):
        assert clean_text(None) == ""

    def test_preserves_meaningful_content(self):
        text = "The bank charged me twice for the same transaction."
        result = clean_text(text)
        assert "bank" in result
        assert "charged" in result
        assert "transaction" in result


# ── map_products ──────────────────────────────────────────────────────────────


class TestMapProducts:
    def _make_df(self, products):
        return pd.DataFrame({"Product": products})

    def test_maps_credit_card(self):
        df = self._make_df(["Credit Card"])
        result = map_products(df)
        assert result["product_category"].iloc[0] == "Credit Card"

    def test_maps_prepaid_card_to_credit_card(self):
        df = self._make_df(["Prepaid card"])
        result = map_products(df)
        assert result["product_category"].iloc[0] == "Credit Card"

    def test_maps_personal_loan(self):
        df = self._make_df(["Consumer Loan"])
        result = map_products(df)
        assert result["product_category"].iloc[0] == "Personal Loan"

    def test_maps_savings_account(self):
        df = self._make_df(["Checking or savings account"])
        result = map_products(df)
        assert result["product_category"].iloc[0] == "Savings Account"

    def test_maps_money_transfer(self):
        df = self._make_df(["Money transfer, virtual currency, or money service"])
        result = map_products(df)
        assert result["product_category"].iloc[0] == "Money Transfer"

    def test_drops_unknown_products(self):
        df = self._make_df(["Credit Card", "Unknown Product"])
        result = map_products(df)
        assert len(result) == 1
        assert result["product_category"].iloc[0] == "Credit Card"

    def test_all_target_products_map(self):
        df = self._make_df(list(TARGET_PRODUCTS.keys()))
        result = map_products(df)
        assert len(result) == len(TARGET_PRODUCTS)
        expected_categories = {
            "Credit Card",
            "Personal Loan",
            "Savings Account",
            "Money Transfer",
        }
        assert set(result["product_category"].unique()) == expected_categories


# ── filter_narratives ─────────────────────────────────────────────────────────


class TestFilterNarratives:
    COL = "Consumer complaint narrative"

    def _make_df(self, narratives):
        return pd.DataFrame({self.COL: narratives})

    def test_keeps_valid_narratives(self):
        df = self._make_df(["This is a valid complaint narrative."])
        result = filter_narratives(df)
        assert len(result) == 1

    def test_removes_nan(self):
        df = self._make_df([None, "Valid narrative"])
        result = filter_narratives(df)
        assert len(result) == 1

    def test_removes_empty_string(self):
        df = self._make_df(["", "Valid narrative"])
        result = filter_narratives(df)
        assert len(result) == 1

    def test_removes_whitespace_only(self):
        df = self._make_df(["   ", "Valid narrative"])
        result = filter_narratives(df)
        assert len(result) == 1

    def test_all_empty_returns_empty_df(self):
        df = self._make_df([None, "", "  "])
        result = filter_narratives(df)
        assert len(result) == 0

    def test_preserves_row_count_when_all_valid(self):
        df = self._make_df(["Complaint one", "Complaint two", "Complaint three"])
        result = filter_narratives(df)
        assert len(result) == 3
