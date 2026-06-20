# Data Directory

## Structure

```
data/
├── filtered_complaints.csv       # Output of Task 1 preprocessing
├── plot_product_distribution.png
├── plot_narrative_length.png
├── plot_narrative_coverage.png
├── raw/                          # Source files
│   ├── complaints.csv            # Full CFPB dataset
│   └── complaint_embeddings.parquet  # Pre-built embeddings for Tasks 3-4
│
└── processed/                    # Optional generated evaluation artifacts
    └── rag_evaluation.csv        # Output of Task 3 evaluation notebook
```

## Data Sources

- **CFPB Complaint Dataset**: Download from the [Consumer Financial Protection Bureau](https://www.consumerfinance.gov/data-research/consumer-complaints/)
- **Pre-built Embeddings**: Provided by 10Academy in the challenge resources (complaint_embeddings.parquet)

## Notes

- Raw and processed large files should normally be git-ignored (see `.gitignore`)
- `data/filtered_complaints.csv` is the Task 1 cleaned dataset deliverable
- The vector store is saved separately in `vector_store/`
