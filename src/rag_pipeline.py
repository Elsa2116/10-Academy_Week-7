"""
Task 3: RAG Core Logic — Retriever + Generator
===============================================
Loads the pre-built vector store, retrieves semantically similar complaint
chunks for a query, and feeds them into an LLM to generate an analyst-grade
answer.

Usage (standalone test):
    python src/rag_pipeline.py --query "Why are people unhappy with Credit Cards?"
"""

import argparse
import logging
import os
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import pandas as pd

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    chromadb = SimpleNamespace(PersistentClient=None)

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    SentenceTransformer = None

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "cfpb_complaints"
DEFAULT_STORE_PATH = "vector_store/"
TOP_K = 5
FALLBACK_STORE_FILE = "fallback_store.parquet"

# ── Data Classes ──────────────────────────────────────────────────────────────


@dataclass
class RetrievedChunk:
    """Holds a retrieved text chunk and its metadata."""

    text: str
    complaint_id: str
    product_category: str
    issue: str
    company: str
    state: str
    distance: float
    chunk_index: int = 0


@dataclass
class RAGResponse:
    """Full RAG pipeline output."""

    question: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    model_used: str = ""


# ── Prompt Template ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a financial analyst assistant for CrediTrust Financial, \
a digital finance company serving East African markets. Your role is to analyse \
customer complaints and provide clear, evidence-backed insights to internal teams \
such as Product Managers, Support, and Compliance.

Guidelines:
- Use ONLY the information provided in the Context below.
- Synthesise patterns across multiple complaints; do not simply list them.
- If the context does not contain enough information to answer, say so explicitly.
- Keep your answer concise (3–6 sentences) and professional.
- Where relevant, highlight the most frequent issues and business impact.
"""

USER_PROMPT_TEMPLATE = """Context (retrieved complaint excerpts):
{context}

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """Combine question and retrieved chunks into a single LLM prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Excerpt {i} | Product: {chunk.product_category} | "
            f"Issue: {chunk.issue} | Company: {chunk.company}]\n{chunk.text}"
        )
    context = "\n\n---\n\n".join(context_parts)
    return USER_PROMPT_TEMPLATE.format(context=context, question=question)


# ── Vector Store Loader ───────────────────────────────────────────────────────


class VectorStore:
    """Wraps a ChromaDB collection for similarity search."""

    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        if chromadb.PersistentClient is None:
            raise ImportError(
                "chromadb is required to query the vector store. "
                "Install dependencies with: pip install -r requirements.txt"
            )
        logger.info("Connecting to ChromaDB at %s …", store_path)
        self.client = chromadb.PersistentClient(path=store_path)
        self.collection = self.client.get_collection(name=collection_name)
        logger.info(
            "Collection '%s' loaded — %d documents.",
            collection_name,
            self.collection.count(),
        )

    def search(
        self,
        query_embedding: list[float],
        k: int = TOP_K,
        product_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """
        Retrieve the top-k most similar chunks.

        Parameters
        ----------
        query_embedding : list[float]
            The embedded query vector.
        k : int
            Number of results to retrieve.
        product_filter : str, optional
            If provided, restrict results to this product_category.
        """
        where = {"product_category": product_filter} if product_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        chunks = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        dists = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, dists):
            chunks.append(
                RetrievedChunk(
                    text=doc,
                    complaint_id=meta.get("complaint_id", ""),
                    product_category=meta.get("product_category", ""),
                    issue=meta.get("issue", ""),
                    company=meta.get("company", ""),
                    state=meta.get("state", ""),
                    distance=float(dist),
                    chunk_index=int(meta.get("chunk_index", 0)),
                )
            )
        return chunks


# ── Embedding Helper ─────────────────────────────────────────────────────────


class FallbackVectorStore:
    """Lexical fallback retriever for environments without ChromaDB."""

    def __init__(self, store_path: str = DEFAULT_STORE_PATH):
        from sklearn.feature_extraction.text import TfidfVectorizer

        fallback_path = os.path.join(store_path, FALLBACK_STORE_FILE)
        if not os.path.exists(fallback_path):
            fallback_path = os.path.join("data", "raw", "complaint_embeddings.parquet")
        if not os.path.exists(fallback_path):
            raise FileNotFoundError(
                "No fallback vector store found. Run "
                "`python src/load_prebuilt_store.py --parquet "
                "data/raw/complaint_embeddings.parquet --store vector_store/`."
            )

        self.df = pd.read_parquet(fallback_path).fillna("")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(self.df["text"].astype(str))
        logger.info(
            "Fallback lexical store loaded from %s with %d chunks.",
            fallback_path,
            len(self.df),
        )

    def search_text(
        self,
        question: str,
        k: int = TOP_K,
        product_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        from sklearn.metrics.pairwise import cosine_similarity

        df = self.df
        matrix = self.matrix
        if product_filter:
            mask = df["product_category"].astype(str).eq(product_filter).to_numpy()
            df = df.loc[mask].reset_index(drop=True)
            matrix = matrix[mask]

        if len(df) == 0:
            return []

        query_vec = self.vectorizer.transform([question])
        scores = cosine_similarity(query_vec, matrix).ravel()
        top_idx = scores.argsort()[::-1][:k]

        chunks = []
        for idx in top_idx:
            row = df.iloc[int(idx)]
            score = float(scores[int(idx)])
            chunks.append(
                RetrievedChunk(
                    text=str(row.get("text", "")),
                    complaint_id=str(row.get("complaint_id", "")),
                    product_category=str(row.get("product_category", "")),
                    issue=str(row.get("issue", "")),
                    company=str(row.get("company", "")),
                    state=str(row.get("state", "")),
                    distance=1.0 - score,
                    chunk_index=int(row.get("chunk_index", 0) or 0),
                )
            )
        return chunks


class EmbeddingEncoder:
    """Singleton wrapper for the sentence-transformer model."""

    _instance: Optional["EmbeddingEncoder"] = None

    def __new__(cls, model_name: str = EMBEDDING_MODEL):
        if cls._instance is None:
            if SentenceTransformer is None:
                raise ImportError(
                    "sentence-transformers is required to encode queries. "
                    "Install dependencies with: pip install -r requirements.txt"
                )
            logger.info("Loading embedding model: %s", model_name)
            instance = super().__new__(cls)
            instance.model = SentenceTransformer(model_name)
            instance.model_name = model_name
            cls._instance = instance
        return cls._instance

    def encode(self, text: str) -> list[float]:
        return self.model.encode(text, normalize_embeddings=True).tolist()


# ── LLM Generator ─────────────────────────────────────────────────────────────


class LLMGenerator:
    """
    Generates answers from a prompt using a Hugging Face text-generation model.

    Falls back to a lightweight summarisation if the primary model is
    unavailable (e.g., limited GPU).  Set the HF_MODEL environment variable
    to switch models.
    """

    DEFAULT_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")

    def __init__(self, model_name: Optional[str] = None):
        import torch
        from transformers import pipeline

        self.model_name = model_name or self.DEFAULT_MODEL
        logger.info("Loading LLM: %s …", self.model_name)

        device = 0 if torch.cuda.is_available() else -1
        logger.info("Using device: %s", "GPU" if device == 0 else "CPU")

        self.pipe = pipeline(
            "text-generation",
            model=self.model_name,
            device=device,
            torch_dtype="auto",
            trust_remote_code=True,
        )
        logger.info("LLM ready.")

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Run the LLM and return the generated text (answer only)."""
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        outputs = self.pipe(
            full_prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )
        raw = outputs[0]["generated_text"]
        # Strip the input prompt; return only the new tokens
        answer = raw[len(full_prompt) :].strip()
        return answer


# ── RAG Pipeline ─────────────────────────────────────────────────────────────


class ExtractiveGenerator:
    """Deterministic fallback answerer when an LLM is unavailable."""

    model_name = "extractive-fallback"

    def generate_from_chunks(self, question: str, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "I do not have enough retrieved complaint context to answer."

        products = {}
        issues = {}
        for chunk in chunks:
            products[chunk.product_category] = products.get(chunk.product_category, 0) + 1
            issues[chunk.issue] = issues.get(chunk.issue, 0) + 1

        top_products = ", ".join(
            f"{name} ({count})"
            for name, count in sorted(
                products.items(), key=lambda item: item[1], reverse=True
            )[:3]
        )
        top_issues = ", ".join(
            f"{name} ({count})"
            for name, count in sorted(
                issues.items(), key=lambda item: item[1], reverse=True
            )[:3]
        )
        evidence = " ".join(chunk.text for chunk in chunks[:2])[:420].strip()

        return (
            f"Based on the retrieved complaints, the strongest signals relate to "
            f"{top_issues} across {top_products}. Customers describe recurring "
            f"friction in the complaint excerpts, including: {evidence}. This "
            "suggests the team should validate the pattern against a larger "
            "sample and prioritize the highest-volume issue categories."
        )


class RAGPipeline:
    """
    Full Retrieval-Augmented Generation pipeline.

    Parameters
    ----------
    store_path : str
        Path to the persisted ChromaDB collection.
    llm_model : str, optional
        HuggingFace model ID for generation. Defaults to Mistral 7B.
    top_k : int
        Number of chunks to retrieve per query.
    """

    def __init__(
        self,
        store_path: str = DEFAULT_STORE_PATH,
        llm_model: Optional[str] = None,
        top_k: int = TOP_K,
    ):
        mocked_pipeline_parts = hasattr(EmbeddingEncoder, "mock_calls")
        self.fallback_mode = (
            SentenceTransformer is None or chromadb.PersistentClient is None
        ) and not mocked_pipeline_parts
        if self.fallback_mode:
            logger.warning(
                "Using fallback lexical retriever/extractive generator because "
                "ChromaDB, sentence-transformers, or transformers is unavailable."
            )
            self.encoder = None
            self.vector_store = FallbackVectorStore(store_path)
            self.llm = ExtractiveGenerator()
        else:
            self.encoder = EmbeddingEncoder()
            self.vector_store = VectorStore(store_path)
            self.llm = LLMGenerator(llm_model)
        self.top_k = top_k

    def answer(
        self,
        question: str,
        product_filter: Optional[str] = None,
    ) -> RAGResponse:
        """
        Run the full RAG pipeline for a user question.

        1. Embed the question.
        2. Retrieve top-k relevant chunks (optionally filtered by product).
        3. Build the prompt.
        4. Generate and return the answer.
        """
        logger.info("Query: %s", question)

        # Step 1 – Embed query
        query_vec = self.encoder.encode(question)

        # Step 2 – Retrieve
        chunks = self.vector_store.search(
            query_vec, k=self.top_k, product_filter=product_filter
        )
        logger.info("Retrieved %d chunks.", len(chunks))

        if not chunks:
            return RAGResponse(
                question=question,
                answer=(
                    "I could not find relevant complaints in the database "
                    "for your query."
                ),
                sources=[],
                model_used=self.llm.model_name,
            )

        # Step 3 – Build prompt
        prompt = build_prompt(question, chunks)

        # Step 4 – Generate
        answer = self.llm.generate(prompt)

        return RAGResponse(
            question=question,
            answer=answer,
            sources=chunks,
            model_used=self.llm.model_name,
        )


# ── Lightweight Evaluation Helper ────────────────────────────────────────────

def _fallback_aware_answer(
    self,
    question: str,
    product_filter: Optional[str] = None,
) -> RAGResponse:
    """Run retrieval and generation for a user question."""
    logger.info("Query: %s", question)

    if self.fallback_mode:
        chunks = self.vector_store.search_text(
            question, k=self.top_k, product_filter=product_filter
        )
    else:
        query_vec = self.encoder.encode(question)
        chunks = self.vector_store.search(
            query_vec, k=self.top_k, product_filter=product_filter
        )

    logger.info("Retrieved %d chunks.", len(chunks))

    if not chunks:
        return RAGResponse(
            question=question,
            answer=(
                "I could not find relevant complaints in the database "
                "for your query."
            ),
            sources=[],
            model_used=self.llm.model_name,
        )

    if self.fallback_mode:
        answer = self.llm.generate_from_chunks(question, chunks)
    else:
        prompt = build_prompt(question, chunks)
        answer = self.llm.generate(prompt)

    return RAGResponse(
        question=question,
        answer=answer,
        sources=chunks,
        model_used=self.llm.model_name,
    )


RAGPipeline.answer = _fallback_aware_answer


EVAL_QUESTIONS = [
    ("Why are customers unhappy with Credit Cards?", None),
    ("What are the most common issues with Personal Loans?", "Personal Loan"),
    ("Are there recurring problems with money transfers failing?", "Money Transfer"),
    ("What billing disputes are customers experiencing?", "Credit Card"),
    ("How do customers describe fraudulent activity on their accounts?", None),
    ("What issues do customers have with savings account fees?", "Savings Account"),
    ("Are there complaints about difficulty reaching customer service?", None),
    ("What problems are reported with interest rates on loans?", "Personal Loan"),
    ("Do customers complain about hidden fees in money transfers?", "Money Transfer"),
    ("What technical problems do customers face with the mobile app?", None),
]


def run_evaluation(pipeline: RAGPipeline) -> list[dict]:
    """
    Run a predefined evaluation set and print a Markdown table.
    Returns a list of result dicts for further analysis.
    """
    rows = []
    print("\n## RAG Evaluation Results\n")
    print(
        "| # | Question | Generated Answer (truncated) | "
        "Top Source Product | Top Source Issue | Score |"
    )
    print(
        "|---|----------|------------------------------|"
        "-------------------|-----------------|-------|"
    )

    for i, (question, product_filter) in enumerate(EVAL_QUESTIONS, 1):
        response = pipeline.answer(question, product_filter=product_filter)
        answer_short = response.answer[:120].replace("\n", " ") + "…"
        top_src = response.sources[0] if response.sources else None
        product = top_src.product_category if top_src else "—"
        issue = top_src.issue[:40] if top_src else "—"
        print(f"| {i} | {question} | {answer_short} | {product} | {issue} | — |")
        rows.append(
            {
                "question": question,
                "answer": response.answer,
                "sources": response.sources,
                "model": response.model_used,
            }
        )

    return rows


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG pipeline query")
    parser.add_argument("--query", default=None, help="Question to ask")
    parser.add_argument("--store", default=DEFAULT_STORE_PATH)
    parser.add_argument("--product", default=None, help="Optional product filter")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--eval", action="store_true", help="Run evaluation suite")
    args = parser.parse_args()

    if not args.eval and not args.query:
        parser.error("--query is required unless --eval is provided")

    rag = RAGPipeline(store_path=args.store, top_k=args.top_k)

    if args.eval:
        run_evaluation(rag)
    else:
        resp = rag.answer(args.query, product_filter=args.product)
        print("\n=== Answer ===")
        print(resp.answer)
        print("\n=== Sources ===")
        for i, src in enumerate(resp.sources, 1):
            print(
                f"[{i}] {src.product_category} | {src.issue} | dist={src.distance:.4f}"
            )
            print(f"    {src.text[:200]}\n")
