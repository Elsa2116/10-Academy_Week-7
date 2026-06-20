"""
Vector Store Loader
===================
Utility to load the pre-built complaint_embeddings.parquet into a local
ChromaDB collection.  Run this ONCE before starting the RAG pipeline when
using the pre-built dataset provided in the challenge resources.

Usage:
    python src/load_prebuilt_store.py \
        --parquet  data/raw/complaint_embeddings.parquet \
        --store    vector_store/
"""

import argparse
import json
import logging
import os
import uuid

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import chromadb
except ImportError:  # pragma: no cover - fallback path depends on environment
    chromadb = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

COLLECTION_NAME = "cfpb_complaints"
INSERT_BATCH = 1000
FALLBACK_STORE_FILE = "fallback_store.parquet"
FALLBACK_MANIFEST_FILE = "manifest.json"


def load_prebuilt_store(parquet_path: str, store_path: str) -> None:
    """
    Read the pre-built parquet file and push all embeddings + metadata
    into a local ChromaDB persistent store.

    The parquet is expected to have at minimum:
        - text (str): chunk text
        - embedding (list[float] or np.ndarray): 384-dim vector
        - complaint_id, product_category, issue, sub_issue,
          company, state, date_received, chunk_index, total_chunks
    """
    logger.info("Loading parquet from %s …", parquet_path)
    df = pd.read_parquet(parquet_path)
    logger.info("Parquet shape: %s", df.shape)

    os.makedirs(store_path, exist_ok=True)
    if chromadb is None:
        fallback_path = os.path.join(store_path, FALLBACK_STORE_FILE)
        manifest_path = os.path.join(store_path, FALLBACK_MANIFEST_FILE)
        df.to_parquet(fallback_path, index=False)
        manifest = {
            "backend": "fallback_parquet",
            "collection_name": COLLECTION_NAME,
            "source_parquet": parquet_path,
            "rows": int(len(df)),
            "note": (
                "ChromaDB was not installed, so chunks and embeddings were "
                "persisted as parquet for the local fallback retriever."
            ),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.warning("Saved fallback vector store to %s.", fallback_path)
        return

    client = chromadb.PersistentClient(path=store_path)

    # Clean slate
    try:
        client.delete_collection(COLLECTION_NAME)
        logger.info("Deleted existing collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total = len(df)
    logger.info("Inserting %d chunks …", total)

    for start in tqdm(range(0, total, INSERT_BATCH), desc="Indexing"):
        end = min(start + INSERT_BATCH, total)
        batch = df.iloc[start:end]

        # Embeddings may be stored as bytes, list, or numpy array
        raw_embs = batch["embedding"].tolist()
        if isinstance(raw_embs[0], (bytes, bytearray)):
            embeddings = [np.frombuffer(e, dtype=np.float32).tolist() for e in raw_embs]
        else:
            embeddings = [
                e.tolist() if hasattr(e, "tolist") else list(e) for e in raw_embs
            ]

        ids = [str(uuid.uuid4()) for _ in range(len(batch))]
        documents = batch["text"].fillna("").tolist()

        meta_cols = [
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
        available = [c for c in meta_cols if c in batch.columns]
        metadatas = batch[available].fillna("").astype(str).to_dict(orient="records")

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    logger.info(
        "✅  Pre-built store loaded — collection '%s' has %d documents.",
        COLLECTION_NAME,
        collection.count(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet", required=True, help="Path to complaint_embeddings.parquet"
    )
    parser.add_argument("--store", default="vector_store/", help="ChromaDB directory")
    args = parser.parse_args()
    load_prebuilt_store(args.parquet, args.store)
