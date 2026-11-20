"""Command-line CPU training for the fictional FAQ adapter."""
import argparse
import json
from pathlib import Path

import torch

from app.core import LoRAConfig, inject_lora, parameter_counts, split_faqs, train_adapter
from app.llm import DEFAULT_BASE, encode_faq, load_base
from app.storage import load_faqs, save_adapter


def train_run(
    config: LoRAConfig, destination: Path, *, epochs: int, batch_size: int,
    learning_rate: float, seed: int, threads: int = 4, dataset: Path = Path("data/faqs.jsonl"),
) -> dict:
    """Fine-tune a local adapter and record its held-out loss curve."""
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    train, validation = split_faqs(load_faqs(dataset))
    model, tokenizer = load_base(config.base_model)
    inject_lora(model, config)
    trainable, total = parameter_counts(model)
    print(f"Trainable parameters: {trainable:,} / {total:,} ({100 * trainable / total:.3f}%)", flush=True)
    history = train_adapter(
        model, [encode_faq(tokenizer, row) for row in train],
        [encode_faq(tokenizer, row) for row in validation],
        pad=tokenizer.pad_token_id, epochs=epochs, batch_size=batch_size,
        learning_rate=learning_rate, seed=seed,
    )
    save_adapter(destination, model, config)
    return {"base_model": config.base_model, "rank": config.rank, "alpha": config.alpha,
            "targets": config.targets, "seed": seed, "batch_size": batch_size,
            "learning_rate": learning_rate, "cpu_threads": threads,
            "trainable": trainable, "total": total, "adapter_bytes": destination.stat().st_size,
            "train_questions": len(train), "validation_questions": len(validation), "loss": history}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--targets", nargs="+", help="projection leaf names, e.g. c_attn for GPT-2")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--dataset", type=Path, default=Path("data/faqs.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/adapter.pt"))
    parser.add_argument("--log", type=Path, default=Path("results/training.json"))
    args = parser.parse_args()
    targets = tuple(args.targets or LoRAConfig(args.base_model).targets)
    config = LoRAConfig(args.base_model, args.rank, args.alpha if args.alpha is not None else 2 * args.rank, targets)
    report = train_run(
        config, args.output, epochs=args.epochs, batch_size=args.batch_size,
        learning_rate=args.learning_rate, seed=args.seed, threads=args.threads, dataset=args.dataset,
    )
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
