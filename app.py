"""
app.py — CrediTrust Intelligent Complaint Chatbot
==================================================
Interactive Gradio interface for the RAG complaint-analysis system.

Run:
    python app.py
    # or
    gradio app.py
"""

import logging
import os

import gradio as gr

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
STORE_PATH = os.getenv("VECTOR_STORE_PATH", "vector_store/")
LLM_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
TOP_K = int(os.getenv("TOP_K", "5"))

PRODUCT_CHOICES = [
    "All Products",
    "Credit Card",
    "Personal Loan",
    "Savings Account",
    "Money Transfer",
]

EXAMPLE_QUESTIONS = [
    "Why are customers unhappy with Credit Cards?",
    "What are the most common issues with Personal Loans?",
    "Are there recurring problems with money transfers failing?",
    "What billing disputes are customers experiencing?",
    "How do customers describe fraudulent activity on their accounts?",
    "What issues do customers have with savings account fees?",
]

# ── Lazy-load the pipeline (avoid slow import at module level) ────────────────
_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from src.rag_pipeline import RAGPipeline

        logger.info("Initialising RAG pipeline …")
        _pipeline = RAGPipeline(store_path=STORE_PATH, llm_model=LLM_MODEL, top_k=TOP_K)
        logger.info("Pipeline ready.")
    return _pipeline


# ── Core query function ───────────────────────────────────────────────────────


def query_rag(
    question: str,
    product_filter: str,
    history: list,
) -> tuple[list, str]:
    """
    Called by Gradio on every submission.

    Returns
    -------
    history : list of [user, assistant] pairs  (for gr.Chatbot)
    sources_md : Markdown string showing retrieved source excerpts
    """
    if not question.strip():
        return history, "⚠️ Please enter a question."

    product = None if product_filter == "All Products" else product_filter

    try:
        pipeline = get_pipeline()
        response = pipeline.answer(question, product_filter=product)
    except Exception as exc:
        logger.exception("Pipeline error: %s", exc)
        history = history + [[question, f"❌ Error: {exc}"]]
        return history, ""

    # Build sources markdown
    source_lines = []
    for i, src in enumerate(response.sources, 1):
        source_lines.append(
            f"**Source {i}** | 📦 `{src.product_category}` | 🏷️ `{src.issue}` "
            f"| 🏢 `{src.company}` | 📍 `{src.state}`\n"
            f"> {src.text[:300]}{'…' if len(src.text) > 300 else ''}\n"
        )
    sources_md = (
        "\n---\n".join(source_lines) if source_lines else "_No sources retrieved._"
    )

    history = history + [[question, response.answer]]
    return history, sources_md


def clear_all():
    return [], "", "All Products", ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
#title {
    text-align: center;
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}
#subtitle { text-align: center; color: #666; margin-bottom: 1rem; }
#chatbot { min-height: 420px; }
#sources { background: #f8f9fa; border-radius: 8px; padding: 12px; }
footer { display: none !important; }
"""

with gr.Blocks(
    css=CSS, title="CrediTrust Complaint Chatbot", theme=gr.themes.Soft()
) as demo:

    # ── Header ──────────────────────────────────────────────────────────────
    gr.Markdown("# 🏦 CrediTrust Complaint Intelligence", elem_id="title")
    gr.Markdown(
        "Ask plain-English questions about customer complaints across Credit Cards, "
        "Personal Loans, Savings Accounts, and Money Transfers.",
        elem_id="subtitle",
    )

    with gr.Row():
        # ── Left column: chat ────────────────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                elem_id="chatbot",
            )

            with gr.Row():
                question_box = gr.Textbox(
                    placeholder="e.g. Why are customers unhappy with Credit Cards?",
                    label="Your Question",
                    lines=2,
                    scale=5,
                )
                product_dropdown = gr.Dropdown(
                    choices=PRODUCT_CHOICES,
                    value="All Products",
                    label="Filter by Product",
                    scale=1,
                )

            with gr.Row():
                submit_btn = gr.Button("🔍 Ask", variant="primary", scale=3)
                clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=question_box,
                label="Example Questions",
            )

        # ── Right column: sources ────────────────────────────────────────────
        with gr.Column(scale=2):
            gr.Markdown("### 📄 Retrieved Sources")
            sources_display = gr.Markdown(
                value="_Sources will appear here after you ask a question._",
                elem_id="sources",
            )

    # ── State ────────────────────────────────────────────────────────────────
    state_history = gr.State([])

    # ── Event handlers ────────────────────────────────────────────────────────
    submit_btn.click(
        fn=query_rag,
        inputs=[question_box, product_dropdown, state_history],
        outputs=[chatbot, sources_display],
    ).then(
        fn=lambda h: h,
        inputs=[chatbot],
        outputs=[state_history],
    ).then(
        fn=lambda: "",
        outputs=[question_box],
    )

    question_box.submit(
        fn=query_rag,
        inputs=[question_box, product_dropdown, state_history],
        outputs=[chatbot, sources_display],
    ).then(
        fn=lambda h: h,
        inputs=[chatbot],
        outputs=[state_history],
    ).then(
        fn=lambda: "",
        outputs=[question_box],
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[chatbot, sources_display, product_dropdown, question_box],
    ).then(
        fn=lambda: [],
        outputs=[state_history],
    )

    # ── Footer info ───────────────────────────────────────────────────────────
    gr.Markdown(
        "---\n"
        "🔎 Powered by **all-MiniLM-L6-v2** embeddings + **ChromaDB** retrieval + "
        f"**{LLM_MODEL.split('/')[-1]}** generation | "
        "Data: CFPB Consumer Complaints"
    )


# ── Launch ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        share=False,
        show_error=True,
    )
