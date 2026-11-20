"""Dataset preparation and LoRA training for the Alderbridge FAQ demo."""
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class FAQ:
    topic: str
    question: str
    answer: str
    facts: tuple[str, ...]
    split: str


def format_prompt(question: str) -> str:
    """Format the question in the same form used for instruction tuning."""
    normalized = " ".join(question.split())
    if not normalized:
        raise ValueError("Enter a nonempty question.")
    return f"### Question: {normalized}\n### Answer:"


def format_example(row: FAQ, end_token: str = "<|endoftext|>") -> str:
    """Format an FAQ example and append the model end token."""
    return f"{format_prompt(row.question)} {row.answer}{end_token}"


def split_faqs(rows: Sequence[FAQ]) -> tuple[list[FAQ], list[FAQ]]:
    """Honor fixed split labels and reject duplicate questions across the dataset."""
    seen = set()
    train, validation = [], []
    for row in rows:
        key = " ".join(row.question.casefold().split()).rstrip("?.!")
        if not key or not row.answer.strip() or not row.facts:
            raise ValueError("Each FAQ needs a question, an answer, and fact probes.")
        if key in seen:
            raise ValueError(f"Duplicate FAQ question: {row.question}")
        seen.add(key)
        if row.split == "train":
            train.append(row)
        elif row.split == "validation":
            validation.append(row)
        else:
            raise ValueError(f"Unknown FAQ split: {row.split}")
    if not train or not validation:
        raise ValueError("Both FAQ splits must contain questions.")
    return train, validation


# --- Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
# Hint: Apply the `lfllm-lflora` technique for LoRA adapter injection and training.
def milestone_2():  # TODO(milestone 2)
    raise NotImplementedError("TODO(milestone 2)")


# --- Milestone 3: Build a Streamlit interface where users can ask questions and get answers.
# Hint: Use Streamlit for building an interactive web application interface.
def milestone_3():  # TODO(milestone 3)
    raise NotImplementedError("TODO(milestone 3)")


# --- Milestone 4: Verify the accuracy and relevance of the generated answers.
# Hint: Perform qualitative evaluation of the model's generated responses for accuracy and relevance.
def milestone_4():  # TODO(milestone 4)
    raise NotImplementedError("TODO(milestone 4)")


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Use the `lfllm-lflora` technique to merge the LoRA adapter for efficient inference.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
