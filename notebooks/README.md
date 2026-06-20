# Notebooks

This directory contains Jupyter notebooks for exploratory analysis and development.

## Notebooks

| Notebook                      | Description                                          | Task   |
| ----------------------------- | ---------------------------------------------------- | ------ |
| `01_eda_preprocessing.ipynb`  | Exploratory Data Analysis and data cleaning          | Task 1 |
| `02_chunking_embedding.ipynb` | Text chunking, embedding, and vector store creation  | Task 2 |
| `03_rag_pipeline_eval.ipynb`  | RAG pipeline construction and qualitative evaluation | Task 3 |

## Running Notebooks

```bash
# From the project root
jupyter notebook notebooks/
```

## Notes

- Run notebooks in order (01 → 02 → 03).
- Ensure the CFPB dataset is placed at `data/raw/complaints.csv` before running notebook 01.
- Notebook 03 uses the pre-built vector store from `complaint_embeddings.parquet`.
