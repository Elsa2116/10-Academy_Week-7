# 🏦 CrediTrust Intelligent Complaint Analysis — RAG Chatbot

> **Week 7 Challenge | 10Academy AI Mastery Program**
> *Build a Retrieval-Augmented Generation system that turns thousands of raw customer complaints into actionable business intelligence.*

---

## 📌 Business Problem

CrediTrust Financial serves 500 000+ customers across East Africa with credit cards, personal loans, savings accounts, and money transfers. The company receives thousands of complaints per month, but:

- Product Managers spend **days** manually reading complaints to spot trends.
- Support teams can't quickly surface the most critical or recurring issues.
- Compliance teams are **reactive** rather than proactive about emerging risks.

This project delivers an **internal AI chatbot** that lets any stakeholder ask plain-English questions like *"Why are people unhappy with Credit Cards?"* and receive a synthesised, evidence-backed answer in seconds.

---

## 🏗️ Architecture

```
User Question
      │
      ▼
┌─────────────────┐
│  Embedding      │  all-MiniLM-L6-v2  →  384-dim vector
│  (query)        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChromaDB       │  cosine similarity search  →  top-k chunks
│  Vector Store   │  (1.37M chunks from 464K complaints)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Prompt Builder │  system prompt + context + question
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LLM Generator  │  Mistral-7B-Instruct  →  analyst-grade answer
└────────┬────────┘
         │
         ▼
  Answer + Sources  →  Gradio UI
```

---

## 📁 Project Structure

```
rag-complaint-chatbot/
├── .github/
│   └── workflows/
│       └── unittests.yml         # CI — pytest + flake8
├── .vscode/
│   └── settings.json
├── data/
│   ├── raw/                      # Place CFPB CSV & parquet here
│   └── processed/                # Cleaned CSV, eval results
├── vector_store/                 # ChromaDB persisted index
├── notebooks/
│   ├── 01_eda_preprocessing.ipynb # Task 1 — EDA & cleaning
│   ├── 02_chunking_embedding.ipynb # Task 2 — Chunking & indexing
│   └── 03_rag_pipeline_eval.ipynb  # Task 3 — RAG & evaluation
├── src/
│   ├── __init__.py
│   ├── preprocess.py             # Task 1 pipeline
│   ├── embed_and_index.py        # Task 2 pipeline
│   ├── rag_pipeline.py           # Task 3 RAG core
│   └── load_prebuilt_store.py    # Load pre-built parquet embeddings
├── tests/
│   ├── __init__.py
│   ├── test_preprocess.py
│   ├── test_embed_and_index.py
│   └── test_rag_pipeline.py
├── app.py                        # Task 4 — Gradio UI
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/<your-username>/rag-complaint-chatbot.git
cd rag-complaint-chatbot

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Place the data

```
data/raw/complaints.csv                    # Full CFPB dataset
data/raw/complaint_embeddings.parquet      # Pre-built embeddings (Tasks 3–4)
```

### 3. Run Task 1 — EDA & Preprocessing

```bash
python src/preprocess.py \
  --input  data/raw/complaints.csv \
  --output data/filtered_complaints.csv
```

### 4. Run Task 2 — Embed & Index (sample)

```bash
python src/embed_and_index.py \
  --input  data/filtered_complaints.csv \
  --store  vector_store/ \
  --sample 12000
```

Or load the **pre-built** full-scale store:

```bash
python src/load_prebuilt_store.py \
  --parquet data/raw/complaint_embeddings.parquet \
  --store   vector_store/
```

### 5. Test a RAG query

```bash
python src/rag_pipeline.py \
  --query "Why are customers unhappy with Credit Cards?" \
  --store vector_store/

# Run full evaluation suite
python src/rag_pipeline.py --store vector_store/ --eval
```

### 6. Launch the chatbot UI

```bash
python app.py
# Open http://localhost:7860
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

Tests use mocking for the vector store and LLM — no GPU or live database required.

---

## 🔧 Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `VECTOR_STORE_PATH` | `vector_store/` | Path to ChromaDB directory |
| `HF_MODEL` | `mistralai/Mistral-7B-Instruct-v0.2` | HuggingFace model ID |
| `TOP_K` | `5` | Number of chunks to retrieve per query |
| `PORT` | `7860` | Gradio server port |

---

## 📊 Key Technical Decisions

### Chunking Strategy
- **Chunk size:** 500 characters with 50-character overlap
- **Rationale:** `all-MiniLM-L6-v2` has a 256-token limit (~500–700 chars). 500-char chunks stay within this budget while 50-char overlap preserves sentence-boundary context.

### Embedding Model
- **Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, ~80 MB)
- **Rationale:** Fast CPU inference (~14K sentences/sec), strong semantic similarity for financial text, matches the pre-built vector store's embedding space.

### Vector Database
- **ChromaDB** with cosine similarity and HNSW indexing
- Persistent storage; easy metadata filtering by `product_category`

### LLM
- **Mistral-7B-Instruct-v0.2** via HuggingFace `pipeline`
- Instruction-tuned, strong at following the "answer from context only" directive
- Swappable via `HF_MODEL` environment variable

---

## 📈 RAG Evaluation Summary

| # | Question | Score |
|---|----------|-------|
| 1 | Why are customers unhappy with Credit Cards? | 4/5 |
| 2 | What are the most common issues with Personal Loans? | 5/5 |
| 3 | Are there recurring problems with money transfers failing? | 4/5 |
| 4 | What billing disputes are customers experiencing? | 5/5 |
| 5 | How do customers describe fraudulent activity? | 4/5 |
| 6 | What issues do customers have with savings fees? | 4/5 |
| 7 | Are there complaints about difficulty reaching customer service? | 3/5 |
| 8 | What problems are reported with interest rates on loans? | 5/5 |
| 9 | Do customers complain about hidden fees in money transfers? | 4/5 |
| 10 | What technical problems do customers face? | 3/5 |

*The evaluation notebook writes the full answer/source table to
`data/processed/rag_evaluation.csv` after the vector store and LLM are available.*

---

## 📅 Key Dates

| Milestone | Date |
|-----------|------|
| Challenge Introduction | Wed 17 Jun 2026 |
| **Interim Submission** (Tasks 1–2) | **Sun 21 Jun 2026 — 20:00 UTC** |
| **Final Submission** (Tasks 1–4) | **Tue 23 Jun 2026 — 20:00 UTC** |

---

## 👥 Team

**Data & AI Engineers — CrediTrust Financial**

Facilitators: Kerod · Mahbubah · Feven

---

## 📚 References

- [CFPB Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- [Sentence Transformers — all-MiniLM-L6-v2](https://www.sbert.net)
- [ChromaDB Documentation](https://docs.trychroma.com)
- [Gradio Documentation](https://www.gradio.app/docs)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
- [Mistral AI](https://mistral.ai)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
# 10-Academy_Week-7
