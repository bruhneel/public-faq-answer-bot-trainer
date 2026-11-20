"""Streamlit entry point for the fictional Alderbridge FAQ demo."""
import json
from pathlib import Path
import sys

import streamlit as st

# Streamlit adds the script directory (app/) to sys.path. Add the package root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.llm import DEFAULT_BASE, generate_answer, load_adapted, load_base

ADAPTER = Path("artifacts/adapter.pt")
MERGED = Path("artifacts/merged")


@st.cache_resource
def get_model(mode: str, source: str, stamp: int):
    """Cache one loaded CPU model per mode and artifact revision."""
    import torch

    torch.set_num_threads(4)
    if mode == "LoRA adapter":
        return load_adapted(Path(source))
    return load_base(source, local_only=mode == "Merged model")


def _answer_tab() -> None:
    """Render the question form and local model settings."""
    mode = st.sidebar.selectbox("Model", ("LoRA adapter", "Merged model", "Base model"))
    max_tokens = st.sidebar.slider("Max new tokens", 8, 96, 48)
    temperature = st.sidebar.slider("Temperature (0 = greedy)", 0.0, 1.5, 0.0, 0.05)

    source = {"LoRA adapter": ADAPTER, "Merged model": MERGED}.get(mode)
    manifest = MERGED / "config.json" if mode == "Merged model" else source
    available = manifest is None or manifest.is_file()
    st.caption(f"Selected model: {mode}")
    if not available and mode == "Merged model":
        st.info("No merged model found. Run `python -m app.merge` after training.")
    elif not available:
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
                    manifest.stat().st_mtime_ns if manifest else 0,
                )
                answer = generate_answer(model, tokenizer, question, max_new_tokens=max_tokens,
                                         temperature=temperature)
            st.subheader("Generated answer")
            st.write(answer or "The model returned no answer. Try rephrasing the question.")
            st.caption(f"Loaded model: {mode}")
        except (OSError, ValueError, RuntimeError) as exc:
            st.error(f"Could not generate an answer: {exc}")


def _evaluation_tab() -> None:
    """Show the recorded raw base and adapter answers with fact scores."""
    path = Path("results/evaluation.json")
    if not path.is_file():
        st.info("No evaluation found. Run `python -m app.evaluate` after training.")
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    st.caption("A fact hit requires every reference phrase. It does not guarantee correctness.")
    left, right = st.columns(2)
    left.metric("Base fact hits", f"{report['base_hits']} / {report['total']}")
    right.metric("LoRA fact hits", f"{report['lora_hits']} / {report['total']}")
    st.dataframe([
        {"Question": row["question"], "Reference": row["reference"], "Base": row["base"],
         "LoRA": row["lora"], "Base hit": row["base_hit"], "LoRA hit": row["lora_hit"]}
        for row in report["rows"]
    ], hide_index=True)


def render() -> None:
    """Render the local demo and its recorded evidence."""
    st.set_page_config(page_title="Public FAQ Answer Bot Trainer")
    st.title("Public FAQ Answer Bot Trainer")
    st.warning("SAMPLE data from the fictional town of Alderbridge. These answers are not government policy.")
    answer, evaluation = st.tabs(("Ask a question", "Evaluation"))
    with answer:
        _answer_tab()
    with evaluation:
        _evaluation_tab()


if __name__ == "__main__":
    render()
