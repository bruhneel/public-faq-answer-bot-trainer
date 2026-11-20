"""One chat call that works with Ollama, Groq or Gemini: change three settings in .env, not the code."""
import os
from pathlib import Path

import httpx


def _load_env() -> None:
    """Read KEY=VALUE lines from .env (if there is one) without overriding real environment variables."""
    path = Path(".env")
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def settings() -> tuple[str, str, str]:
    _load_env()
    return (os.environ.get("LLM_BASE_URL") or "http://localhost:11434/v1",
            os.environ.get("LLM_API_KEY", ""), os.environ.get("LLM_MODEL", ""))


def chat(messages: list[dict], **options) -> str:
    """Send chat messages, get the reply text. `options` go straight to the API (temperature, max_tokens...)."""
    base, key, model = settings()
    if not model:
        raise RuntimeError("Set LLM_MODEL in .env (see .env.example).")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    r = httpx.post(f"{base.rstrip('/')}/chat/completions", headers=headers, timeout=120,
                   json={"model": model, "messages": messages, **options})
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


DEFAULT_BASE = "HuggingFaceTB/SmolLM2-135M-Instruct"
SYSTEM_PROMPT = (
    "Answer FAQs for the fictional Alderbridge municipal services office. "
    "Give a short direct answer in one sentence."
)


def load_base(model_name: str = DEFAULT_BASE, *, local_only: bool = False):
    """Load an anonymous public model through the Hugging Face cache on CPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=False, local_files_only=local_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, use_safetensors=True,
        token=False, local_files_only=local_only,
    )
    return model.eval(), tokenizer


def prompt_ids(tokenizer, question: str) -> list[int]:
    """Use the model's dialogue template when it provides one."""
    from app.core import format_prompt

    text = format_prompt(question)
    if tokenizer.chat_template:
        return list(tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True,
        ))
    return tokenizer.encode(f"{SYSTEM_PROMPT}\n{text}", add_special_tokens=False)


def encode_faq(tokenizer, row) -> tuple[list[int], list[int]]:
    """Keep prompt and answer tokenization consistent with inference."""
    from app.core import encode_answer

    answer = tokenizer.encode(row.answer, add_special_tokens=False)
    return encode_answer(prompt_ids(tokenizer, row.question), answer, tokenizer.eos_token_id)
