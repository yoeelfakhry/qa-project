import streamlit as st

from src.models.predict import QAModel

MODEL_SOURCE = "saved_models/roberta_qa_full_best"
HF_MODEL_URL = "https://huggingface.co/yoeel/roberta-qa-squad"
GITHUB_URL = "https://github.com/yoeelfakhry/qa-project"

st.set_page_config(
    page_title="Extractive QA - RoBERTa",
    page_icon="🔎",
    layout="wide",
)


@st.cache_resource
def load_model():
    return QAModel(MODEL_SOURCE)


EXAMPLES = {
    "Select an example...": None,
    "History - Notre Dame": {
        "context": (
            "The Basilica of the Sacred Heart at Notre Dame is beside the Main Building. "
            "Immediately behind the basilica is the Grotto, a Marian place of prayer and "
            "reflection. It is a replica of the grotto at Lourdes, France where the Virgin "
            "Mary reputedly appeared to Saint Bernadette Soubirous in 1858."
        ),
        "question": "To whom did the Virgin Mary allegedly appear in 1858?",
    },
    "Geography - Eiffel Tower": {
        "context": (
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, "
            "France. It is named after the engineer Gustave Eiffel, whose company designed "
            "and built the tower. Locally nicknamed 'La dame de fer', it was constructed "
            "from 1887 to 1889."
        ),
        "question": "Who is the Eiffel Tower named after?",
    },
    "Technology - Python": {
        "context": (
            "Python is a high-level, general-purpose programming language. Its design "
            "philosophy emphasizes code readability with the use of significant "
            "indentation. Python was created by Guido van Rossum and first released "
            "in 1991."
        ),
        "question": "Who created Python?",
    },
}

# ---------- Header ----------
st.title("🔎 Extractive Question Answering")
st.markdown(
    "Fine-tuned **RoBERTa-base** on **SQuAD v1.1** — give it a passage and a question, "
    "and it extracts the answer directly from the text itself (no generation, "
    "no hallucinated facts)."
)

metric_col1, metric_col2, metric_col3 = st.columns(3)
metric_col1.metric("Exact Match", "72.38%")
metric_col2.metric("F1 Score", "85.26%")
metric_col3.metric("Validation Examples", "8,654")

st.markdown(
    f"[📦 Model on Hugging Face]({HF_MODEL_URL}) &nbsp;&nbsp;|&nbsp;&nbsp; "
    f"[💻 Source code on GitHub]({GITHUB_URL})"
)
st.divider()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("About this project")
    st.markdown(
        "Part of a controlled comparison of three transformer architectures "
        "(ALBERT, BERT, RoBERTa) fine-tuned for extractive QA. RoBERTa-base "
        "was selected as the production model after a fair, matched-config "
        "comparison on a held-out subset."
    )

    st.subheader("Model details")
    st.markdown(
        "- **Base checkpoint:** `FacebookAI/roberta-base`\n"
        "- **Parameters:** 124M\n"
        "- **Training data:** SQuAD v1.1 (~87K examples)\n"
        "- **Task:** extractive span selection"
    )

    st.subheader("Pipeline")
    st.markdown(
        "1. Data cleaning + leakage-safe splitting\n"
        "2. Config-driven training (`src/models/train.py`)\n"
        "3. Evaluation via Exact Match / F1\n"
        "4. Deployment via this Streamlit app"
    )

# ---------- Example picker + input ----------
st.subheader("Try it")
example_choice = st.selectbox("Load an example (optional)", list(EXAMPLES.keys()))

default_context = ""
default_question = ""
if EXAMPLES[example_choice] is not None:
    default_context = EXAMPLES[example_choice]["context"]
    default_question = EXAMPLES[example_choice]["question"]

input_col, tip_col = st.columns([2, 1])

with input_col:
    context = st.text_area(
        "Context / Passage",
        value=default_context,
        height=220,
        placeholder="Paste a passage here...",
    )
    question = st.text_input(
        "Question",
        value=default_question,
        placeholder="Ask something about the passage above",
    )
    ask_clicked = st.button("Get Answer", type="primary", use_container_width=True)

with tip_col:
    st.info(
        "**Note:** this model extracts a direct span from your passage — "
        "it can't answer questions the passage doesn't actually cover."
    )

# ---------- Load model (cached, runs once) ----------
with st.spinner("Loading model (first run only)..."):
    qa_model = load_model()

# ---------- Answer ----------
if ask_clicked:
    if not context.strip() or not question.strip():
        st.warning("Please provide both a context and a question.")
    else:
        with st.spinner("Thinking..."):
            result = qa_model.answer(question, context)

        if result["answer"]:
            st.success(f"**Answer:** {result['answer']}")
        else:
            st.error("No confident answer found in this passage.")

        with st.expander("Details"):
            st.write(f"Raw model score (start + end logits): {result['score']:.2f}")
            st.write(f"Context chunks processed: {result['number_of_chunks']}")

st.divider()
st.caption(
    "Built as part of a QA model comparison project (ALBERT / BERT / RoBERTa). "
    f"[View source on GitHub]({GITHUB_URL})."
)