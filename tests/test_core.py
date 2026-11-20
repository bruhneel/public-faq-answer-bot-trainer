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


def test_lora_initialization_training_and_roundtrip(tmp_path):
    import copy

    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from app.core import LoRAConfig, encode_answer, inject_lora, parameter_counts, train_adapter
    from app.storage import apply_adapter, read_adapter, save_adapter

    torch.set_num_threads(1)
    torch.manual_seed(42)
    base = GPT2LMHeadModel(GPT2Config(
        vocab_size=24, n_positions=32, n_embd=24, n_layer=1, n_head=4,
        resid_pdrop=0, embd_pdrop=0, attn_pdrop=0,
    )).eval()
    model = copy.deepcopy(base)
    config = LoRAConfig("test-only", 4, 8, ("c_attn", "c_proj"))
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    inject_lora(model, config)
    model.eval()
    torch.testing.assert_close(model(input_ids).logits, base(input_ids).logits, atol=0, rtol=0)
    assert {n.rsplit(".", 1)[-1] for n, p in model.named_parameters() if p.requires_grad} == {"A", "B"}
    assert 0 < parameter_counts(model)[0] < parameter_counts(model)[1]
    example = encode_answer([1, 2], [3, 4], 5)
    assert example[1] == [-100, -100, 3, 4, 5]
    history = train_adapter(model, [example] * 4, [example], pad=0, epochs=12, batch_size=1,
                            learning_rate=0.025, seed=42, log=False)
    assert history[-1]["val_loss"] < history[0]["val_loss"]
    path = tmp_path / "adapter.pt"
    save_adapter(path, model, config)
    restored_config, weights = read_adapter(path)
    assert restored_config == config and all(key.endswith((".A", ".B")) for key in weights)
    restored = apply_adapter(base, restored_config, weights)
    torch.testing.assert_close(restored(input_ids).logits, model(input_ids).logits, atol=0, rtol=0)


def test_generation_with_download_free_model():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    from app.llm import generate_answer

    torch.set_num_threads(1)
    torch.manual_seed(42)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1, "[PAD]": 2, "fee": 3}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       eos_token="[EOS]", pad_token="[PAD]")
    model = GPT2LMHeadModel(GPT2Config(vocab_size=4, n_positions=128, n_embd=16, n_layer=1, n_head=2,
                                      eos_token_id=1, pad_token_id=2))
    a = generate_answer(model, tokenizer, "Which fee?", max_new_tokens=3)
    assert isinstance(a, str) and len(a.split()) <= 3
    assert a == generate_answer(model, tokenizer, "Which fee?", max_new_tokens=3)


def test_streamlit_renders_without_adapter(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    entry = Path("app/main.py").resolve()
    monkeypatch.chdir(tmp_path)
    page = AppTest.from_file(str(entry)).run()
    assert not page.exception
    assert page.title[0].value == "Public FAQ Answer Bot Trainer"
    assert "python -m app.train" in page.info[0].value
    assert page.button[0].disabled


def test_fact_matching_requires_every_whole_phrase():
    from app.core import contains_facts

    assert contains_facts("Form RC–4 costs 14 CROWNS.", ("RC-4", "14 crowns"))
    assert not contains_facts("Form RC-4 costs 114 crowns.", ("RC-4", "14 crowns"))
    assert not contains_facts("The fee is 14 crowns.", ("RC-4", "14 crowns"))
    assert contains_facts("From 9 a.m. to 4 p.m.", ("9 a.m.", "4 p.m."))
    assert not contains_facts("Nothing to score", ())


# Milestone 5: Implement adapter merging for inference.
@pytest.mark.skip(reason="remove this line when you start milestone 5")
def test_milestone_5():
    raise NotImplementedError("write the test for milestone 5")

