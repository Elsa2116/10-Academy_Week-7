"""
tests/test_embed_and_index.py
==============================
Unit tests for src/embed_and_index.py
"""

import pandas as pd

from src.embed_and_index import chunk_text, build_chunks_dataframe, stratified_sample


# ── chunk_text ────────────────────────────────────────────────────────────────


class TestChunkText:
    def test_short_text_single_chunk(self):
        text = "This is a short complaint."
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_multiple_chunks(self):
        text = "A" * 1200
        chunks = chunk_text(text, chunk_size=500, overlap=50)
        assert len(chunks) >= 3

    def test_overlap_creates_shared_content(self):
        text = "A" * 600
        chunks = chunk_text(text, chunk_size=500, overlap=100)
        # Second chunk starts at 500-100=400, so chars 400-500 appear in both
        assert len(chunks) == 2
        assert len(chunks[0]) == 500
        # overlap: last 100 chars of chunk 0 should equal first 100 of chunk 1
        assert chunks[0][-100:] == chunks[1][:100]

    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_none_returns_empty_list(self):
        assert chunk_text(None) == []

    def test_chunk_size_respected(self):
        text = "B" * 2000
        chunks = chunk_text(text, chunk_size=300, overlap=30)
        for chunk in chunks[:-1]:  # all except possibly last
            assert len(chunk) <= 300

    def test_tiny_chunks_discarded(self):
        # Last fragment < 20 chars should be discarded
        text = "A" * 510  # 500 + 10 → second chunk = 10 chars → discarded
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        assert len(chunks) == 1

    def test_exact_chunk_size_boundary(self):
        text = "C" * 500
        chunks = chunk_text(text, chunk_size=500, overlap=0)
        assert len(chunks) == 1


# ── build_chunks_dataframe ────────────────────────────────────────────────────


class TestBuildChunksDataframe:
    def _sample_df(self):
        return pd.DataFrame(
            {
                "complaint_id": ["C001", "C002"],
                "product_category": ["Credit Card", "Personal Loan"],
                "product_raw": ["Credit Card", "Consumer Loan"],
                "issue": ["Billing dispute", "High interest rate"],
                "sub_issue": ["", ""],
                "company": ["Bank A", "Bank B"],
                "state": ["CA", "NY"],
                "date_received": ["2023-01-01", "2023-01-02"],
                "cleaned_narrative": [
                    "The bank charged me twice for the same purchase.",
                    (
                        "My interest rate was raised without any notification "
                        "from the lender."
                    ),
                ],
            }
        )

    def test_returns_dataframe(self):
        df = self._sample_df()
        result = build_chunks_dataframe(df)
        assert isinstance(result, pd.DataFrame)

    def test_has_required_columns(self):
        df = self._sample_df()
        result = build_chunks_dataframe(df)
        required = {
            "chunk_id",
            "complaint_id",
            "product_category",
            "text",
            "chunk_index",
        }
        assert required.issubset(set(result.columns))

    def test_chunk_count_positive(self):
        df = self._sample_df()
        result = build_chunks_dataframe(df)
        assert len(result) >= 2  # at least one chunk per complaint

    def test_metadata_preserved(self):
        df = self._sample_df()
        result = build_chunks_dataframe(df)
        assert "Credit Card" in result["product_category"].values
        assert "Personal Loan" in result["product_category"].values

    def test_chunk_index_starts_at_zero(self):
        df = self._sample_df()
        result = build_chunks_dataframe(df)
        assert result["chunk_index"].min() == 0

    def test_unique_chunk_ids(self):
        df = self._sample_df()
        result = build_chunks_dataframe(df)
        assert result["chunk_id"].nunique() == len(result)


# ── stratified_sample ─────────────────────────────────────────────────────────


class TestStratifiedSample:
    def _make_df(self, n_per_class=100):
        categories = [
            "Credit Card",
            "Personal Loan",
            "Savings Account",
            "Money Transfer",
        ]
        rows = []
        for cat in categories:
            for i in range(n_per_class):
                rows.append({"product_category": cat, "text": f"complaint {i}"})
        return pd.DataFrame(rows)

    def test_correct_sample_size(self):
        df = self._make_df(100)
        sample = stratified_sample(df, n=100)
        assert len(sample) == 100

    def test_all_categories_represented(self):
        df = self._make_df(100)
        sample = stratified_sample(df, n=100)
        assert set(sample["product_category"].unique()) == {
            "Credit Card",
            "Personal Loan",
            "Savings Account",
            "Money Transfer",
        }

    def test_proportional_representation(self):
        df = self._make_df(100)  # 400 total, balanced
        sample = stratified_sample(df, n=200)
        counts = sample["product_category"].value_counts()
        # Each category should be roughly 25% ± 5%
        for count in counts.values:
            assert 40 <= count <= 60, f"Expected ~50 per class, got {count}"

    def test_sample_larger_than_dataset_returns_full(self):
        df = self._make_df(10)  # 40 total
        sample = stratified_sample(df, n=200)
        assert len(sample) == len(df)

    def test_reproducible_with_seed(self):
        df = self._make_df(100)
        s1 = stratified_sample(df, n=100)
        s2 = stratified_sample(df, n=100)
        assert list(s1.index) == list(s2.index)
