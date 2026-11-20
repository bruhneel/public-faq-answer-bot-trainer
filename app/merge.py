"""Merge the saved LoRA into a standalone CPU model and verify equivalence."""
import argparse
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from app.core import LoRALinear, merge_lora, split_faqs
from app.llm import generate_answer, load_adapted, prompt_ids
from app.storage import load_faqs


def _latency(model, tokenizer, questions: list[str], repeats: int = 3) -> float:
    """Return median per-question generation latency in milliseconds."""
    generate_answer(model, tokenizer, questions[0])
    runs = []
    for _ in range(repeats):
        start = perf_counter()
        for question in questions:
            generate_answer(model, tokenizer, question)
        runs.append((perf_counter() - start) * 1000 / len(questions))
    return median(runs)


@torch.inference_mode()
def merge_run(adapter: Path, output: Path, dataset: Path) -> dict:
    """Measure pre/post-merge logits, generations, and local CPU latency."""
    torch.set_num_threads(4)
    model, tokenizer = load_adapted(adapter)
    _, validation = split_faqs(load_faqs(dataset))
    questions = [row.question for row in validation[:5]]
    ids = [torch.tensor([prompt_ids(tokenizer, question)]) for question in questions]
    before = [model(tokens).logits for tokens in ids]
    adapter_answers = [generate_answer(model, tokenizer, question) for question in questions]
    adapter_ms = _latency(model, tokenizer, questions)
    merge_lora(model)
    after = [model(tokens).logits for tokens in ids]
    for previous, merged in zip(before, after, strict=True):
        torch.testing.assert_close(previous, merged, atol=0.0003, rtol=0.0001)
    max_error = max((previous - merged).abs().max().item()
                    for previous, merged in zip(before, after, strict=True))
    merged_answers = [generate_answer(model, tokenizer, question) for question in questions]
    merged_ms = _latency(model, tokenizer, questions)
    assert not any(isinstance(module, LoRALinear) for module in model.modules())
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    return {
        "max_absolute_logit_difference": max_error, "logit_atol": 0.0003, "logit_rtol": 0.0001,
        "questions": len(questions), "identical_greedy_answers": sum(
            a == b for a, b in zip(adapter_answers, merged_answers, strict=True)),
        "remaining_lora_wrappers": 0, "warmup_questions_per_variant": 1,
        "timing_repeats": 3, "cpu_threads": 4, "max_new_tokens": 48,
        "adapter_ms_per_answer": adapter_ms, "merged_ms_per_answer": merged_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/adapter.pt"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/merged"))
    parser.add_argument("--dataset", type=Path, default=Path("data/faqs.jsonl"))
    parser.add_argument("--report", type=Path, default=Path("results/merge.json"))
    args = parser.parse_args()
    result = merge_run(args.adapter, args.output, args.dataset)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
