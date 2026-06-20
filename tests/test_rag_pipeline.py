"""
tests/test_rag_pipeline.py
==========================
Unit tests for src/rag_pipeline.py
Uses mocking to avoid requiring a live vector store or LLM during CI.
"""

from unittest.mock import MagicMock, patch

from src.rag_pipeline import (
    RetrievedChunk,
    RAGResponse,
    build_prompt,
)


# ── build_prompt ──────────────────────────────────────────────────────────────


def _make_chunk(**kwargs) -> RetrievedChunk:
    defaults = dict(
        text="The customer reported a billing error on their credit card statement.",
        complaint_id="C001",
        product_category="Credit Card",
        issue="Billing dispute",
        company="Bank A",
        state="CA",
        distance=0.12,
        chunk_index=0,
    )
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


class TestBuildPrompt:
    def test_returns_string(self):
        chunks = [_make_chunk()]
        result = build_prompt("Why are customers unhappy?", chunks)
        assert isinstance(result, str)

    def test_contains_question(self):
        question = "Why are customers unhappy with Credit Cards?"
        chunks = [_make_chunk()]
        result = build_prompt(question, chunks)
        assert question in result

    def test_contains_chunk_text(self):
        chunk = _make_chunk(text="Customer was charged twice.")
        result = build_prompt("Test question?", [chunk])
        assert "Customer was charged twice." in result

    def test_contains_product_category_in_context(self):
        chunk = _make_chunk(product_category="Personal Loan")
        result = build_prompt("Test?", [chunk])
        assert "Personal Loan" in result

    def test_multiple_chunks_all_included(self):
        chunks = [
            _make_chunk(text="First complaint text."),
            _make_chunk(text="Second complaint text."),
            _make_chunk(text="Third complaint text."),
        ]
        result = build_prompt("What issues exist?", chunks)
        assert "First complaint text." in result
        assert "Second complaint text." in result
        assert "Third complaint text." in result

    def test_empty_chunks_list(self):
        result = build_prompt("Any issues?", [])
        assert "Any issues?" in result


# ── RetrievedChunk ────────────────────────────────────────────────────────────


class TestRetrievedChunk:
    def test_creation(self):
        chunk = _make_chunk()
        assert chunk.product_category == "Credit Card"
        assert chunk.issue == "Billing dispute"
        assert 0.0 <= chunk.distance <= 1.0

    def test_default_chunk_index(self):
        chunk = _make_chunk()
        assert chunk.chunk_index == 0


# ── RAGResponse ───────────────────────────────────────────────────────────────


class TestRAGResponse:
    def test_creation(self):
        response = RAGResponse(
            question="Why unhappy?",
            answer="Customers face billing errors.",
            sources=[_make_chunk()],
            model_used="test-model",
        )
        assert response.question == "Why unhappy?"
        assert len(response.sources) == 1

    def test_empty_sources_default(self):
        response = RAGResponse(question="Q", answer="A")
        assert response.sources == []
        assert response.model_used == ""


# ── VectorStore (mocked) ──────────────────────────────────────────────────────


class TestVectorStoreMocked:
    @patch("src.rag_pipeline.chromadb.PersistentClient")
    def test_search_returns_chunks(self, mock_client_cls):
        from src.rag_pipeline import VectorStore

        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "documents": [["Chunk text about billing error."]],
            "metadatas": [
                [
                    {
                        "complaint_id": "C001",
                        "product_category": "Credit Card",
                        "issue": "Billing",
                        "company": "Bank A",
                        "state": "CA",
                        "date_received": "2023-01-01",
                        "chunk_index": "0",
                        "total_chunks": "1",
                    }
                ]
            ],
            "distances": [[0.15]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        store = VectorStore(store_path="fake_path")
        results = store.search(query_embedding=[0.0] * 384, k=1)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].product_category == "Credit Card"

    @patch("src.rag_pipeline.chromadb.PersistentClient")
    def test_search_with_product_filter(self, mock_client_cls):
        from src.rag_pipeline import VectorStore

        mock_collection = MagicMock()
        mock_collection.count.return_value = 50
        mock_collection.query.return_value = {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_collection
        mock_client_cls.return_value = mock_client

        store = VectorStore(store_path="fake_path")
        store.search(query_embedding=[0.0] * 384, k=5, product_filter="Personal Loan")

        call_kwargs = mock_collection.query.call_args.kwargs
        assert call_kwargs["where"] == {"product_category": "Personal Loan"}


# ── RAGPipeline (mocked) ──────────────────────────────────────────────────────


class TestRAGPipelineMocked:
    @patch("src.rag_pipeline.LLMGenerator")
    @patch("src.rag_pipeline.VectorStore")
    @patch("src.rag_pipeline.EmbeddingEncoder")
    def test_answer_returns_rag_response(
        self, mock_encoder_cls, mock_store_cls, mock_llm_cls
    ):
        from src.rag_pipeline import RAGPipeline

        # Mock encoder
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.1] * 384
        mock_encoder_cls.return_value = mock_encoder

        # Mock vector store
        mock_store = MagicMock()
        mock_store.search.return_value = [_make_chunk()]
        mock_store_cls.return_value = mock_store

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Customers are unhappy due to billing errors."
        mock_llm.model_name = "test-model"
        mock_llm_cls.return_value = mock_llm

        pipeline = RAGPipeline(store_path="fake_path")
        response = pipeline.answer("Why are customers unhappy?")

        assert isinstance(response, RAGResponse)
        assert response.answer == "Customers are unhappy due to billing errors."
        assert len(response.sources) == 1
        assert response.question == "Why are customers unhappy?"

    @patch("src.rag_pipeline.LLMGenerator")
    @patch("src.rag_pipeline.VectorStore")
    @patch("src.rag_pipeline.EmbeddingEncoder")
    def test_empty_retrieval_returns_fallback(
        self, mock_encoder_cls, mock_store_cls, mock_llm_cls
    ):
        from src.rag_pipeline import RAGPipeline

        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = [0.1] * 384
        mock_encoder_cls.return_value = mock_encoder

        mock_store = MagicMock()
        mock_store.search.return_value = []  # no results
        mock_store_cls.return_value = mock_store

        mock_llm = MagicMock()
        mock_llm.model_name = "test-model"
        mock_llm_cls.return_value = mock_llm

        pipeline = RAGPipeline(store_path="fake_path")
        response = pipeline.answer("An obscure question with no matches")

        assert "could not find" in response.answer.lower()
        assert response.sources == []
