# Vector Store

This directory holds the persisted ChromaDB index after it is generated.

At the time of this audit, only this README is present. Build the index from the
Task 2 sample or load the provided pre-built embeddings before running the RAG
pipeline or app.

## Contents (generated — not committed to git)

```
vector_store/
├── chroma/                  # ChromaDB internal files
│   ├── chroma.sqlite3
│   └── <uuid>/
│       ├── data_level0.bin
│       ├── header.bin
│       ├── index_metadata.pickle
│       └── length.bin
└── chunks_metadata.parquet  # Chunk metadata (without embeddings)
```

## Rebuilding the Index

**From scratch (Task 2 sample):**
```bash
python src/embed_and_index.py --input data/filtered_complaints.csv --store vector_store/ --sample 12000
```

**From pre-built embeddings (Tasks 3–4, full dataset):**
```bash
python src/load_prebuilt_store.py --parquet data/raw/complaint_embeddings.parquet --store vector_store/
```
