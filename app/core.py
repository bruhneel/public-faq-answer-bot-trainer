"""Dataset preparation and LoRA training for the Alderbridge FAQ demo."""
from dataclasses import dataclass
from collections.abc import Sequence
import math
import random
import re

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers.pytorch_utils import Conv1D


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


@dataclass(frozen=True)
class LoRAConfig:
    base_model: str
    rank: int = 8
    alpha: float = 16.0
    targets: tuple[str, ...] = ("q_proj", "v_proj", "o_proj", "down_proj")
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.rank < 1 or not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("LoRA rank and alpha must be positive.")
        if not 0 <= self.dropout < 1 or not self.targets:
            raise ValueError("Provide target names and dropout in [0, 1).")


class LoRALinear(nn.Module):
    """Wrap either nn.Linear or GPT-2 Conv1D with a low-rank residual."""

    def __init__(self, base: nn.Linear | Conv1D, config: LoRAConfig) -> None:
        super().__init__()
        self.base = base
        self.conv1d = isinstance(base, Conv1D)
        if not self.conv1d and not isinstance(base, nn.Linear):
            raise TypeError("LoRA expects a Linear or GPT-2 Conv1D projection.")
        in_features, out_features = base.weight.shape if self.conv1d else base.weight.T.shape
        self.base.requires_grad_(False)
        self.A = nn.Parameter(base.weight.new_empty(config.rank, in_features))
        self.B = nn.Parameter(base.weight.new_zeros(out_features, config.rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.scaling = config.alpha / config.rank
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        update = F.linear(F.linear(self.dropout(x), self.A), self.B)
        return self.base(x) + update * self.scaling


def inject_lora(model: nn.Module, config: LoRAConfig) -> nn.Module:
    """Freeze the base and replace the selected leaf projections."""
    if any(isinstance(module, LoRALinear) for module in model.modules()):
        raise ValueError("This model already has LoRA projections.")
    matches = [
        (name, module) for name, module in model.named_modules()
        if name.split(".")[-1] in config.targets and isinstance(module, (nn.Linear, Conv1D))
    ]
    found = {name.rsplit(".", 1)[-1] for name, _ in matches}
    if found != set(config.targets):
        raise ValueError(f"Missing LoRA targets: {sorted(set(config.targets) - found)}")
    model.requires_grad_(False)
    for name, module in matches:
        model.set_submodule(name, LoRALinear(module, config))
    return model


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return trainable and total parameter counts after injection."""
    return (sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for p in model.parameters()))


def encode_answer(
    prefix_ids: list[int], answer_ids: list[int], eos: int
) -> tuple[list[int], list[int]]:
    """Build causal labels that ignore the instruction prefix."""
    if not prefix_ids or not answer_ids:
        raise ValueError("A training example needs both a prompt and an answer.")
    completion = [*answer_ids, eos]
    return [*prefix_ids, *completion], [-100] * len(prefix_ids) + completion


def _batch(examples: Sequence[tuple[list[int], list[int]]], pad: int) -> dict[str, Tensor]:
    length = max(len(ids) for ids, _ in examples)
    ids, labels, masks = [], [], []
    for tokens, targets in examples:
        padding = length - len(tokens)
        ids.append(tokens + [pad] * padding)
        labels.append(targets + [-100] * padding)
        masks.append([1] * len(tokens) + [0] * padding)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks)}


def _epoch_loss(
    model: nn.Module, rows: Sequence[tuple[list[int], list[int]]], batch_size: int,
    pad: int, optimizer: torch.optim.Optimizer | None = None,
) -> float:
    model.train(optimizer is not None)
    loss_sum, count = 0.0, 0
    with torch.set_grad_enabled(optimizer is not None):
        for start in range(0, len(rows), batch_size):
            batch = _batch(rows[start:start + batch_size], pad)
            loss = model(**batch, use_cache=False).loss
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), 1.0
                )
                optimizer.step()
            tokens = int(batch["labels"][:, 1:].ne(-100).sum())
            loss_sum += loss.item() * tokens
            count += tokens
    return loss_sum / count


def train_adapter(
    model: nn.Module, train: Sequence[tuple[list[int], list[int]]],
    validation: Sequence[tuple[list[int], list[int]]], *, pad: int,
    epochs: int, batch_size: int, learning_rate: float, seed: int, log: bool = True,
) -> list[dict[str, float | int]]:
    """Optimize adapter tensors only and log answer-token losses."""
    if min(epochs, batch_size, len(train), len(validation)) < 1:
        raise ValueError("Need positive epochs and batch size and two nonempty data splits.")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Learning rate must be positive and finite.")
    params = [p for name, p in model.named_parameters()
              if p.requires_grad and name.rsplit(".", 1)[-1] in {"A", "B"}]
    if not params or len(params) != sum(p.requires_grad for p in model.parameters()):
        raise ValueError("Inject LoRA and freeze all base parameters before training.")
    rng = random.Random(seed)
    torch.manual_seed(seed)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.0)
    history = []
    for epoch in range(1, epochs + 1):
        rows = list(train)
        rng.shuffle(rows)
        train_loss = _epoch_loss(model, rows, batch_size, pad, optimizer)
        val_loss = _epoch_loss(model, validation, batch_size, pad)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if log:
            print(f"epoch {epoch:2d}  train {train_loss:.4f}  val {val_loss:.4f}", flush=True)
    return history


def _fact_words(text: str) -> str:
    """Keep lexical boundaries while normalizing punctuated codes and times."""
    return " ".join(re.findall(r"[^\W_]+", text.casefold()))


def contains_facts(answer: str, expected: Sequence[str]) -> bool:
    """Check that all reference fact phrases occur as whole normalized words."""
    words = f" {_fact_words(answer)} "
    return bool(expected) and all(
        (fact_words := _fact_words(fact)) and f" {fact_words} " in words
        for fact in expected
    )


# --- Milestone 5: Implement adapter merging for inference.
# Hint: Use the `lfllm-lflora` technique to merge the LoRA adapter for efficient inference.
def milestone_5():  # TODO(milestone 5)
    raise NotImplementedError("TODO(milestone 5)")
