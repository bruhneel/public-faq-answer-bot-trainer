"""Streamlit entry point for the fictional Alderbridge FAQ demo."""
from pathlib import Path

import streamlit as st

from app.llm import DEFAULT_BASE, generate_answer, load_adapted, load_base

ADAPTER = Path("artifacts/adapter.pt")


@st.cache_resource
def get_model(mode: str, source: str, stamp: int):
    """Cache one loaded CPU model per mode and artifact revision."""
    import torch

    torch.set_num_threads(4)
    if mode == "LoRA adapter":
        return load_adapted(Path(source))
    return load_base(source)


def render() -> None:
    """Render the question form and local model settings."""
    st.set_page_config(page_title="Public FAQ Answer Bot Trainer")
    st.title("Public FAQ Answer Bot Trainer")
    st.warning("SAMPLE data from the fictional town of Alderbridge. These answers are not government policy.")
    mode = st.sidebar.selectbox("Model", ("LoRA adapter", "Base model"))
    max_tokens = st.sidebar.slider("Max new tokens", 8, 96, 48)
    temperature = st.sidebar.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.0, 0.05)
    source = ADAPTER if mode == "LoRA adapter" else None
    available = source is None or source.is_file()
    st.caption(f"Selected model: {mode}")
    if not available:
        st.info("No adapter has been trained. Run `python -m app.train` from the repository root, or choose Base model.")
    with st.form("ask"):
        question = st.text_input("Question", placeholder="What documents do I need to update my address?")
        submitted = st.form_submit_button("Generate answer", disabled=not available)
    if submitted:
        if not question.strip():
            st.warning("Enter a question.")
            return
        try:
            with st.spinner("Loading the local model and generating an answer…"):
                model, tokenizer = get_model(
                    mode, str(source) if source else DEFAULT_BASE,
                    source.stat().st_mtime_ns if source else 0,
                )
                answer = generate_answer(model, tokenizer, question, max_new_tokens=max_tokens,
                                         temperature=temperature)
            st.subheader("Generated answer")
            st.write(answer or "The model returned no answer. Try rephrasing the question.")
            st.caption(f"Loaded model: {mode}")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not generate an answer: {exc}")


if __name__ == "__main__":
    render()
