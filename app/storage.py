"""A tiny key-value store in one JSON file (no server, easy to read). Values can be anything JSON can hold."""
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor, nn
    from app.core import FAQ, LoRAConfig


class Store:
    def __init__(self, path: str | None = None):
        self.path = Path(path or os.environ.get("DATA_PATH", "data/app.json"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("{}")

    def _read(self) -> dict:
        return json.loads(self.path.read_text() or "{}")

    def put(self, key: str, value) -> None:
        data = self._read()
        data[key] = value
        self.path.write_text(json.dumps(data, indent=2))

    def get(self, key: str, default=None):
        return self._read().get(key, default)

    def all(self) -> dict:
        return dict(sorted(self._read().items()))


def load_faqs(path: Path = Path("data/faqs.jsonl")) -> list["FAQ"]:
    """Read the labeled FAQ records from JSON Lines."""
    from app.core import FAQ

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            record = json.loads(line)
            rows.append(FAQ(
                topic=record["topic"], question=record["question"], answer=record["answer"],
                facts=tuple(record["facts"]), split=record["split"],
            ))
    return rows


def save_adapter(path: Path, model: "nn.Module", config: "LoRAConfig") -> None:
    """Persist only adapter matrices plus the primitive configuration record."""
    from dataclasses import asdict

    import torch

    from app.core import LoRALinear

    tensors = {
        f"{name}.{key}": getattr(layer, key).detach().cpu()
        for name, layer in model.named_modules() if isinstance(layer, LoRALinear)
        for key in ("A", "B")
    }
    if not tensors:
        raise ValueError("No LoRA projections to save.")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "weights": tensors}, path)


def read_adapter(path: Path) -> tuple["LoRAConfig", dict[str, "Tensor"]]:
    """Read only tensor and data records from a saved adapter."""
    import torch

    from app.core import LoRAConfig

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return LoRAConfig(**payload["config"]), payload["weights"]


def apply_adapter(model: "nn.Module", config: "LoRAConfig", weights: dict[str, "Tensor"]) -> "nn.Module":
    """Validate the saved adapter against a newly wrapped base model."""
    import torch

    from app.core import LoRALinear, inject_lora

    inject_lora(model, config)
    targets = {
        f"{name}.{key}": getattr(layer, key)
        for name, layer in model.named_modules() if isinstance(layer, LoRALinear)
        for key in ("A", "B")
    }
    if targets.keys() != weights.keys() or any(
        tensor.shape != weights[name].shape for name, tensor in targets.items()
    ):
        raise ValueError("Adapter weights do not match the supplied base model.")
    with torch.no_grad():
        for name, tensor in targets.items():
            tensor.copy_(weights[name])
    return model.eval()
