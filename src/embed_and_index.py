"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
===========================================================
Creates a stratified sample from the cleaned dataset, chunks the narratives,
generates sentence-transformer embeddings, and persists a ChromaDB vector store.

Usage:
    python src/embed_and_index.py \
        --input  data/processed/filtered_complaints.csv \
        --store  vector_store/ \
        --sample 12000
"""

import argparse
import logging
import os
import uuid

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

try:
    import chromadb
except ImportError:  # pragma: no cover - exercised only when optional deps are missing
    chromadb = None

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
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50  # characters
BATCH_SIZE = 256  # embeddings per batch
COLLECTION_NAME = "cfpb_complaints"


# ── Stratified Sampling ───────────────────────────────────────────────────────


def stratified_sample(
    df: pd.DataFrame, n: int, stratify_col: str = "product_category"
) -> pd.DataFrame:
    """
    Draw a stratified random sample of `n` rows, preserving the
    product-category distribution.
    """
    logger.info("Sampling %d rows (stratified by %s) …", n, stratify_col)
    if n >= len(df):
        logger.warning(
            "Requested sample (%d) >= dataset size (%d); returning full dataset.",
            n,
            len(df),
        )
        return df.reset_index(drop=True)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=n, random_state=42)
    # StratifiedShuffleSplit uses test_size as the SAMPLE; we want `n` rows
    # Use train_size = len(df) - n so test = n
    _, sample_idx = next(sss.split(df, df[stratify_col]))
    sample = df.iloc[sample_idx].reset_index(drop=True)

    logger.info(
        "Sample distribution:\n%s", sample[stratify_col].value_counts().to_string()
    )
    return sample


# ── Text Chunking ─────────────────────────────────────────────────────────────


def chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """
    Split text into overlapping character-level chunks.

    A character-level splitter is used (mirroring LangChain's
    RecursiveCharacterTextSplitter behavior) to keep chunks under a
    predictable token budget for the embedding model.
    """
    if not text or len(text) == 0:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if len(chunk) > 20:  # discard tiny trailing chunks
            chunks.append(chunk)
        if end == len(text):
            break
        start += chunk_size - overlap

    return chunks


def build_chunks_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explode each complaint narrative into individual text chunks,
    preserving all metadata per chunk.
    """
    logger.info("Chunking %d narratives …", len(df))
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
        chunks = chunk_text(str(row.get("cleaned_narrative", "")))
        for idx, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": str(uuid.uuid4()),
                    "complaint_id": str(row.get("complaint_id", "")),
                    "product_category": str(row.get("product_category", "")),
                    "product_raw": str(row.get("product_raw", "")),
                    "issue": str(row.get("issue", "")),
                    "sub_issue": str(row.get("sub_issue", "")),
                    "company": str(row.get("company", "")),
                    "state": str(row.get("state", "")),
                    "date_received": str(row.get("date_received", "")),
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                    "text": chunk,
                }
            )
    chunks_df = pd.DataFrame(records)
    logger.info("Created %d chunks from %d complaints.", len(chunks_df), len(df))
    return chunks_df


# ── Embedding ─────────────────────────────────────────────────────────────────


def load_embedding_model(model_name: str = EMBEDDING_MODEL):
    """Load the sentence-transformer model."""
    if SentenceTransformer is None:
        raise ImportError(
            "sentence-transformers is required for embedding. "
            "Install dependencies with: pip install -r requirements.txt"
        )
    logger.info("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    logger.info(
        "Model loaded (vector dim = %d).", model.get_sentence_embedding_dimension()
    )
    return model


def embed_chunks(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """Generate embeddings for a list of text strings."""
    logger.info("Embedding %d chunks (batch_size=%d) …", len(texts), batch_size)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity ready
        convert_to_numpy=True,
    )
    logger.info("Embeddings shape: %s", embeddings.shape)
    return embeddings


# ── Vector Store ──────────────────────────────────────────────────────────────


def build_chroma_store(
    chunks_df: pd.DataFrame,
    embeddings: np.ndarray,
    store_path: str,
    collection_name: str = COLLECTION_NAME,
):
    """
    Persist chunks + embeddings into a ChromaDB collection.
    Creates (or resets) the collection at `store_path`.
    """
    if chromadb is None:
        raise ImportError(
            "chromadb is required to build the vector store. "
            "Install dependencies with: pip install -r requirements.txt"
        )

    os.makedirs(store_path, exist_ok=True)
    logger.info("Initialising ChromaDB at %s …", store_path)

    client = chromadb.PersistentClient(path=store_path)

    # Drop existing collection if present (idempotent re-runs)
    try:
        client.delete_collection(name=collection_name)
        logger.info("Deleted existing collection '%s'.", collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Insert in batches to avoid memory spikes
    total = len(chunks_df)
    insert_batch = 500
    logger.info("Inserting %d chunks into ChromaDB …", total)

    for start in tqdm(range(0, total, insert_batch), desc="Indexing"):
        end = min(start + insert_batch, total)
        batch = chunks_df.iloc[start:end]
        batch_embeddings = embeddings[start:end].tolist()

        collection.add(
            ids=batch["chunk_id"].tolist(),
            embeddings=batch_embeddings,
            documents=batch["text"].tolist(),
            metadatas=batch[
                [
                    "complaint_id",
                    "product_category",
                    "issue",
                    "sub_issue",
                    "company",
                    "state",
                    "date_received",
                    "chunk_index",
                    "total_chunks",
                ]
            ].to_dict(orient="records"),
        )

    logger.info(
        "ChromaDB collection '%s' built — %d documents.",
        collection_name,
        collection.count(),
    )
    return collection


# ── Main ──────────────────────────────────────────────────────────────────────


def run_embedding_pipeline(
    input_path: str,
    store_path: str,
    n_sample: int = 12_000,
) -> None:
    """End-to-end pipeline: load → sample → chunk → embed → index."""
    # Load cleaned data
    logger.info("Loading %s …", input_path)
    df = pd.read_csv(input_path)
    logger.info("Loaded %d rows.", len(df))

    # Stratified sample
    df_sample = stratified_sample(df, n_sample)

    # Chunk
    chunks_df = build_chunks_dataframe(df_sample)

    # Embed
    model = load_embedding_model()
    embeddings = embed_chunks(chunks_df["text"].tolist(), model)

    # Index
    build_chroma_store(chunks_df, embeddings, store_path)

    # Save chunk metadata separately (useful for offline inspection)
    meta_path = os.path.join(store_path, "chunks_metadata.parquet")
    chunks_df.drop(columns=["text"]).to_parquet(meta_path, index=False)
    logger.info("Chunk metadata saved → %s", meta_path)

    logger.info("✅ Embedding pipeline complete.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk, embed, and index complaints")
    parser.add_argument("--input", default="data/filtered_complaints.csv")
    parser.add_argument("--store", default="vector_store/")
    parser.add_argument("--sample", type=int, default=12_000)
    args = parser.parse_args()
    run_embedding_pipeline(args.input, args.store, args.sample)
