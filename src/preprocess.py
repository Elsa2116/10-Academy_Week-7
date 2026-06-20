"""
Task 1: Exploratory Data Analysis and Data Preprocessing
=========================================================
Loads the CFPB complaint dataset, performs EDA, filters to the four
target product categories, and cleans the consumer narratives.

Usage:
    python src/preprocess.py --input data/raw/complaints.csv \
                              --output data/filtered_complaints.csv
"""

import argparse
import logging
import os
import re

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TARGET_PRODUCTS = {
    "Credit Card": "Credit Card",
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card",
    "Prepaid card": "Credit Card",
    "Personal Loan": "Personal Loan",
    "Consumer Loan": "Personal Loan",
    "Student loan": "Personal Loan",
    "Payday loan": "Personal Loan",
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Savings account": "Savings Account",
    "Checking or savings account": "Savings Account",
    "Bank account or service": "Savings Account",
    "Money transfer": "Money Transfer",
    "Money transfer, virtual currency, or money service": "Money Transfer",
    "Money transfers": "Money Transfer",
    "Virtual currency": "Money Transfer",
}

BOILERPLATE_PATTERNS = [
    r"i am writing to (file|submit|make) (a )?complaint.*?\.",
    r"to whom it may concern[,:]?",
    r"dear (sir|madam|consumer financial protection bureau|cfpb)[,:]?",
    r"sincerely[,.]?\s*\w+",
    r"xx+",  # redacted info placeholders
    r"\bXXXX\b",
    r"(?:account|card) (number|#)?\s*:?\s*\d+",
    r"(?:phone|tel)\.?\s*:?\s*[\d\-\(\) ]+",
]

# ── Helper Functions ──────────────────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
    """Load the CFPB CSV dataset."""
    logger.info("Loading dataset from %s …", filepath)
    df = pd.read_csv(filepath, low_memory=False)
    logger.info("Loaded %d rows, %d columns.", *df.shape)
    return df


def map_products(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw product names to the four target categories."""
    df = df.copy()
    df["product_category"] = df["Product"].map(TARGET_PRODUCTS)
    before = len(df)
    df = df[df["product_category"].notna()].reset_index(drop=True)
    logger.info(
        "Product filter: %d → %d rows (removed %d).",
        before,
        len(df),
        before - len(df),
    )
    return df


def filter_narratives(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows without a consumer narrative."""
    col = "Consumer complaint narrative"
    before = len(df)
    df = df[df[col].notna() & (df[col].str.strip() != "")].reset_index(drop=True)
    logger.info(
        "Narrative filter: %d → %d rows (removed %d).",
        before,
        len(df),
        before - len(df),
    )
    return df


def clean_text(text: str) -> str:
    """
    Clean a single complaint narrative:
      1. Lowercase
      2. Remove boilerplate phrases
      3. Strip special characters / excess whitespace
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)

    # Remove non-alphanumeric chars except basic punctuation
    text = re.sub(r"[^a-z0-9\s.,!?'\-]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def add_word_count(df: pd.DataFrame) -> pd.DataFrame:
    """Compute word count for each narrative."""
    df = df.copy()
    df["word_count"] = df["cleaned_narrative"].str.split().str.len()
    return df


# ── EDA Plots ─────────────────────────────────────────────────────────────────


def plot_product_distribution(df: pd.DataFrame, out_dir: str = "data/processed"):
    """Bar chart: complaint count per product category."""
    fig, ax = plt.subplots(figsize=(9, 5))
    counts = df["product_category"].value_counts()
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)
    ax.set_title("Complaint Distribution by Product Category", fontsize=14)
    ax.set_xlabel("Product Category")
    ax.set_ylabel("Number of Complaints")
    ax.bar_label(ax.containers[0], fmt="%d", padding=3)
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_product_distribution.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_narrative_length(df: pd.DataFrame, out_dir: str = "data/processed"):
    """Histogram: word-count distribution of cleaned narratives."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Overall distribution (capped at 500 words for readability)
    capped = df["word_count"].clip(upper=500)
    axes[0].hist(capped, bins=50, color="steelblue", edgecolor="white")
    axes[0].set_title("Word Count Distribution (capped at 500)")
    axes[0].set_xlabel("Word Count")
    axes[0].set_ylabel("Frequency")

    # Per-product boxplot
    sns.boxplot(
        data=df,
        x="product_category",
        y="word_count",
        palette="Set2",
        showfliers=False,
        ax=axes[1],
    )
    axes[1].set_title("Word Count by Product Category")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Word Count")
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    path = os.path.join(out_dir, "plot_narrative_length.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved → %s", path)


def plot_narratives_with_without(
    df_full: pd.DataFrame, out_dir: str = "data/processed"
):
    """Pie chart: complaints with vs. without narratives."""
    col = "Consumer complaint narrative"
    has = df_full[col].notna() & (df_full[col].str.strip() != "")
    counts = [has.sum(), (~has).sum()]
    labels = [f"With Narrative\n({counts[0]:,})", f"Without Narrative\n({counts[1]:,})"]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        counts,
        labels=labels,
        autopct="%1.1f%%",
        colors=["#4CAF50", "#F44336"],
        startangle=140,
    )
    ax.set_title("Complaints: With vs. Without Narratives")
    plt.tight_layout()
    path = os.path.join(out_dir, "plot_narrative_coverage.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved → %s", path)


# ── Main Pipeline ─────────────────────────────────────────────────────────────


def run_preprocessing(input_path: str, output_path: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline.

    Returns the cleaned, filtered DataFrame and saves it to output_path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out_dir = os.path.dirname(output_path)

    # 1. Load
    df_raw = load_data(input_path)

    # 2. EDA on raw data (narrative coverage)
    plot_narratives_with_without(df_raw, out_dir)

    # 3. Filter products
    df = map_products(df_raw)

    # 4. Filter narratives
    df = filter_narratives(df)

    # 5. Clean text
    logger.info("Cleaning narratives …")
    df["cleaned_narrative"] = df["Consumer complaint narrative"].apply(clean_text)

    # Remove empty after cleaning
    before = len(df)
    df = df[df["cleaned_narrative"].str.len() > 20].reset_index(drop=True)
    logger.info("Post-clean empty filter: %d → %d rows.", before, len(df))

    # 6. Add word count
    df = add_word_count(df)

    # 7. EDA plots (post-filter)
    plot_product_distribution(df, out_dir)
    plot_narrative_length(df, out_dir)

    # 8. Summary stats
    logger.info("\n=== EDA Summary ===")
    logger.info("Total complaints after filtering: %d", len(df))
    logger.info(
        "Product distribution:\n%s", df["product_category"].value_counts().to_string()
    )
    logger.info(
        "Narrative word count — mean: %.0f | median: %.0f | min: %d | max: %d",
        df["word_count"].mean(),
        df["word_count"].median(),
        df["word_count"].min(),
        df["word_count"].max(),
    )
    logger.info("Very short (<20 words): %d", (df["word_count"] < 20).sum())
    logger.info("Very long (>500 words): %d", (df["word_count"] > 500).sum())

    # 9. Select and rename columns for downstream use
    output_cols = {
        "Complaint ID": "complaint_id",
        "Product": "product_raw",
        "product_category": "product_category",
        "Issue": "issue",
        "Sub-issue": "sub_issue",
        "Consumer complaint narrative": "original_narrative",
        "cleaned_narrative": "cleaned_narrative",
        "word_count": "word_count",
        "Company": "company",
        "State": "state",
        "Date received": "date_received",
    }
    existing_cols = {k: v for k, v in output_cols.items() if k in df.columns}
    df_out = df.rename(columns=existing_cols)[list(existing_cols.values())]

    # 10. Save
    df_out.to_csv(output_path, index=False)
    logger.info("Saved cleaned dataset → %s  (%d rows)", output_path, len(df_out))

    return df_out


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CFPB EDA & Preprocessing")
    parser.add_argument(
        "--input",
        default="data/raw/complaints.csv",
        help="Path to raw CFPB CSV",
    )
    parser.add_argument(
        "--output",
        default="data/filtered_complaints.csv",
        help="Output path for cleaned CSV",
    )
    args = parser.parse_args()
    run_preprocessing(args.input, args.output)
