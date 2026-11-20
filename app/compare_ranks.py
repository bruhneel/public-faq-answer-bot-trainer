"""Compare LoRA rank, adapter size, and held-out fact-match rate."""
import argparse
import json
from pathlib import Path

from app.core import LoRAConfig, contains_facts, split_faqs
from app.evaluate import generate_all
from app.llm import DEFAULT_BASE, load_adapted
from app.storage import load_faqs
from app.train import train_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    _, questions = split_faqs(load_faqs())
    rows = []
    for rank in (2, 4, 8, 16):
        print(f"Rank {rank}", flush=True)
        destination = Path("artifacts/ranks") / f"rank-{rank}.pt"
        metadata = train_run(
            LoRAConfig(DEFAULT_BASE, rank=rank, alpha=2 * rank), destination,
            epochs=args.epochs, batch_size=4, learning_rate=0.002, seed=args.seed, threads=args.threads,
        )
        model, tokenizer = load_adapted(destination)
        answers = generate_all(model, tokenizer, questions, 48)
        del model, tokenizer
        hits = sum(contains_facts(answer, row.facts) for row, answer in zip(questions, answers, strict=True))
        summary = {"rank": rank, "matched": hits, "total": len(questions),
                   "adapter_bytes": metadata["adapter_bytes"], "trainable": metadata["trainable"],
                   "validation_loss": metadata["loss"][-1]["val_loss"]}
        rows.append(summary)
        print(summary, flush=True)
    Path("results").mkdir(exist_ok=True)
    Path("results/rank_comparison.json").write_text(
        json.dumps({"epochs": args.epochs, "seed": args.seed, "ranks": rows}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
