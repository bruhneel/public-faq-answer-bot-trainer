"""Regression tests for the FAQ training pipeline."""
import pytest

def test_dataset_formatting_and_holdout():
    from dataclasses import replace

    import pytest

    from app.core import format_example, format_prompt, split_faqs
    from app.storage import load_faqs

    rows = load_faqs()
    train, val = split_faqs(rows)
    assert len(rows) == 80 and (len(train), len(val)) == (60, 20)
    assert {row.topic for row in train} == {row.topic for row in val}
    assert {row.question for row in train}.isdisjoint(row.question for row in val)
    assert format_prompt("  What\n now? ") == "### Question: What now?\n### Answer:"
    assert format_example(val[0], "<end>").endswith(val[0].answer + "<end>")
    with pytest.raises(ValueError, match="Duplicate"):
        split_faqs([*rows, replace(train[0], split="validation")])


# Milestone 2: Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).
@pytest.mark.skip(reason="remove this line when you start milestone 2")
def test_milestone_2():
    raise NotImplementedError("write the test for milestone 2")


# Milestone 3: Build a Streamlit interface where users can ask questions and get answers.
@pytest.mark.skip(reason="remove this line when you start milestone 3")
def test_milestone_3():
    raise NotImplementedError("write the test for milestone 3")


# Milestone 4: Verify the accuracy and relevance of the generated answers.
@pytest.mark.skip(reason="remove this line when you start milestone 4")
def test_milestone_4():
    raise NotImplementedError("write the test for milestone 4")


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

