# Deliverables Checklist

Audit date: 2026-06-20

## Overall Status

The code deliverables for Tasks 1-4 are present and now pass the local unit test
and lint checks. Two submission artifacts still need to be produced with the full
runtime dependencies/assets: the persisted ChromaDB vector store and the final UI
screenshot or GIF for the report.

## Project Structure

| Requirement | Status | Evidence |
| --- | --- | --- |
| `.vscode/settings.json` | Complete | Present |
| `.github/workflows/unittests.yml` | Complete | Present |
| `data/raw/` and cleaned data | Complete | `data/raw/complaints.csv`, `data/raw/complaint_embeddings.parquet`, `data/filtered_complaints.csv` |
| `vector_store/` | Needs generation | Directory exists, but currently only contains `README.md` |
| `notebooks/` | Complete | Task notebooks 01, 02, and 03 are present |
| `src/` modules | Complete | `preprocess.py`, `embed_and_index.py`, `load_prebuilt_store.py`, `rag_pipeline.py` |
| `tests/` | Complete | 54 tests passing |
| `app.py` | Complete | Gradio chat app with product filter, sources, and clear button |
| `requirements.txt`, `.gitignore`, `README.md` | Complete | Present |

## Task 1: EDA and Preprocessing

| Requirement | Status | Evidence |
| --- | --- | --- |
| Load CFPB complaint dataset | Complete | `src/preprocess.py`, `notebooks/01_eda_preprocessing.ipynb` |
| Product distribution analysis | Complete | `data/plot_product_distribution.png` |
| Narrative length analysis | Complete | `data/plot_narrative_length.png` |
| Count records with/without narratives | Complete | `data/plot_narrative_coverage.png` |
| Filter to target products and non-empty narratives | Complete | `map_products`, `filter_narratives` |
| Clean text narratives | Complete | `clean_text` |
| Save cleaned dataset | Complete | `data/filtered_complaints.csv` |
| Written 2-3 paragraph EDA summary | Partial | README has summary-level findings; expand in final report if submitting separately |

## Task 2: Chunking, Embedding, and Indexing

| Requirement | Status | Evidence |
| --- | --- | --- |
| Stratified sample of 10k-15k complaints | Complete in code | `stratified_sample(..., n_sample=12000)` |
| Chunking with chosen size/overlap | Complete | `chunk_text`, 500 chars, 50 overlap |
| Embedding model choice | Complete | `sentence-transformers/all-MiniLM-L6-v2` |
| Persist vector store | Needs generation | Run `python src/embed_and_index.py --input data/filtered_complaints.csv --store vector_store/ --sample 12000` after installing dependencies |
| Sampling/chunking/model rationale | Complete | README technical decisions section |

## Task 3: RAG Core Logic and Evaluation

| Requirement | Status | Evidence |
| --- | --- | --- |
| Load pre-built vector store | Complete in code | `src/load_prebuilt_store.py` |
| Retrieve top-k chunks with optional product filter | Complete | `VectorStore.search` |
| Grounded analyst prompt | Complete | `SYSTEM_PROMPT`, `build_prompt` |
| LLM generator | Complete | `LLMGenerator` |
| 5-10 representative questions | Complete | 10 questions in `EVAL_QUESTIONS` |
| Evaluation table with answers/sources/scores | Partial | Notebook can generate `data/processed/rag_evaluation.csv`; file is not currently present |

## Task 4: Interactive Chat Interface

| Requirement | Status | Evidence |
| --- | --- | --- |
| Gradio or Streamlit UI | Complete | `app.py` uses Gradio |
| Text input and Ask button | Complete | `question_box`, `submit_btn` |
| Answer display | Complete | `gr.Chatbot` |
| Retrieved sources display | Complete | `sources_display` |
| Clear button | Complete | `clear_btn` |
| Screenshot/GIF in final report | Needs capture | Run the app after loading vector store, then add screenshot/GIF to final report |

## Verification

Commands run successfully:

```bash
pytest tests/ -q
flake8 src/ tests/ app.py --max-line-length=88 --extend-ignore=E203,W503
python -m py_compile src/preprocess.py src/embed_and_index.py src/rag_pipeline.py src/load_prebuilt_store.py app.py
```

Current results:

- `54 passed`
- `flake8` clean
- Python compilation clean

## Remaining Before Final Submission

1. Install dependencies in a compatible Python environment.
2. Build or load the ChromaDB vector store into `vector_store/`.
3. Run the RAG evaluation notebook/script to create `data/processed/rag_evaluation.csv`.
4. Launch `app.py`, capture a screenshot or GIF, and include it in the final Medium-style report.
5. Expand the final report sections with concrete EDA findings and evaluation examples from your actual run.
