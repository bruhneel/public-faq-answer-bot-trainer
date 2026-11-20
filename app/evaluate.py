"""Compare the pretrained base and adapter on held-out fictional FAQs."""
import argparse
import json
from pathlib import Path

import torch

from app.core import FAQ, contains_facts, split_faqs
from app.llm import generate_answer, load_base
from app.storage import apply_adapter, load_faqs, read_adapter


def generate_all(model, tokenizer, questions: list[FAQ], max_tokens: int) -> list[str]:
    """Generate the same greedy, bounded answer for each held-out question."""
    return [
        generate_answer(model, tokenizer, row.question, max_new_tokens=max_tokens)
        for row in questions
    ]


def evaluate(adapter: Path, dataset: Path, max_tokens: int = 48) -> dict:
    """Capture raw answers and required-phrase hits under identical decoding."""
    torch.set_num_threads(4)
    _, validation = split_faqs(load_faqs(dataset))
    config, weights = read_adapter(adapter)
    model, tokenizer = load_base(config.base_model)
    base = generate_all(model, tokenizer, validation, max_tokens)
    apply_adapter(model, config, weights)
    tuned = generate_all(model, tokenizer, validation, max_tokens)
    rows = [
        {"topic": row.topic, "question": row.question, "reference": row.answer,
         "required_facts": row.facts, "base": first, "lora": second,
         "base_hit": contains_facts(first, row.facts), "lora_hit": contains_facts(second, row.facts)}
        for row, first, second in zip(validation, base, tuned, strict=True)
    ]
    return {"model": config.base_model, "rank": config.rank, "max_new_tokens": max_tokens,
            "scoring": "All required normalized fact phrases must occur in the generated answer.",
            "base_hits": sum(row["base_hit"] for row in rows),
            "lora_hits": sum(row["lora_hit"] for row in rows), "total": len(rows), "rows": rows}


def write_report(report: dict, directory: Path) -> None:
    """Write machine-readable answers and a Markdown table."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "evaluation.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    def cell(text: str) -> str:
        return " ".join(text.split()).replace("|", r"\|").replace("<", "&lt;") or "(empty)"

    lines = [
        "# Held-out FAQ evaluation", "",
        f'Base matched **{report["base_hits"]}/{report["total"]}** fact sets. '
        f'LoRA matched **{report["lora_hits"]}/{report["total"]}**.',
        "", "One held-out paraphrase per fictional policy. Both variants used the same prompt "
        "and greedy decoding. The policies appeared in training under other phrasings. "
        "A hit can still contain errors, and a correctly phrased synonym can miss.", "",
        "| Question | Reference | Base answer | LoRA answer | Fact hit (base / LoRA) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        content = " | ".join(cell(row[key]) for key in ("question", "reference", "base", "lora"))
        lines.append(f'| {content} | {int(row["base_hit"])} / {int(row["lora_hit"])} |')
    (directory / "evaluation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, default=Path("artifacts/adapter.pt"))
    parser.add_argument("--dataset", type=Path, default=Path("data/faqs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--max-tokens", type=int, default=48)
    args = parser.parse_args()
    report = evaluate(args.adapter, args.dataset, args.max_tokens)
    write_report(report, args.output)
    print(f'Base: {report["base_hits"]}/{report["total"]} | LoRA: {report["lora_hits"]}/{report["total"]}')


if __name__ == "__main__":
    main()
