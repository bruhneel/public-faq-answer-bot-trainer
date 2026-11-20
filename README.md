# Public FAQ Answer Bot Trainer

[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/b6n79ky6cs/badge.svg)](https://msrit.ask.thili.ai/b/b6n79ky6cs)

A local Streamlit demonstration of training and merging a LoRA adapter for a municipal FAQ assistant. **SAMPLE only:** Alderbridge, its forms, fees, and rules are fictional. Generated answers are not government policy or advice.

## Install and run

From the repository root with Python 3.12 and Git:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
# Linux users can install the CPU-only PyTorch wheel first:
# pip install 'torch>=2.6,<3' --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env
python -m app.train
python -m app.evaluate
python -m app.merge
streamlit run app/main.py
```

The first training or base-model load downloads the public [SmolLM2-135M-Instruct checkpoint](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct) and tokenizer into the Hugging Face cache. The workflow uses CPU, needs no key, and can run with `HF_HUB_OFFLINE=1` once cached. Keep `.env` local. The starter's Ollama, Groq, and Gemini helper remains optional and is separate from the Streamlit model.

`python -m app.train --help` shows the dataset, optimizer, projection, and artifact options. Training saves only the LoRA matrices and metadata to `artifacts/adapter.pt`. The merge command saves a standalone checkpoint and tokenizer to `artifacts/merged/`. Both paths are Git-ignored. Choose Base, LoRA, or Merged in the app sidebar.

To run tests without model downloads, use `HF_HUB_OFFLINE=1 pytest -q`. They use randomly initialized tiny models. The Dockerfile runs Streamlit on port 8080 by default. For a trained adapter or merged model, mount your local artifacts at `/srv/artifacts`; for base or adapter mode, the public base must be available in the container cache or downloadable.

## Dataset and training

The [dataset](data/README.md) contains 80 original pairs across 20 invented policies, with 60 training questions and 20 held-out paraphrases. Each example uses an instruction-formatted question and an end token. The loss ignores the question, instruction prefix, and padded positions. AdamW updates only the adapter matrices, with seed 42.

The default rank-8 run trained **1,244,160 / 135,759,168 parameters (0.916%)**. The saved adapter was **5,051,423 bytes**. Over eight epochs the answer-token training loss fell from 2.2098 to 0.0163, and validation loss from 1.3897 to 0.1901. See the [recorded training curve](results/training.json).

## How LoRA works

The frozen projection receives a residual update `(alpha/r) × B × A`. A starts with Kaiming initialization and B starts at zero, so the wrapped model initially produces identical logits. SmolLM2 targets its query, value, attention output, and feed-forward down projections. The wrapper also handles GPT-2 Conv1D's transposed weight layout, using a download-free GPT-2 test. LoRA was implemented directly in PyTorch following the drill's injection and merging steps.

Merging adds the residual to each base weight and replaces the wrapper with the original projection. Across 5 tested prompts, the maximum absolute logit difference was **0.00031**, all 5 greedy answers matched, and zero wrappers remained. On this laptop, median per-answer generation latency was **396.3 ms with adapters** and **371.4 ms merged**, measured with four CPU threads, one warm-up per model and three passes over five prompts. This small timing comparison excludes load time and varies with system load. [Merge report](results/merge.json)

## Held-out evaluation

With identical greedy prompts, the pretrained base matched every expected fact phrase for **1/20** questions. The rank-8 adapter matched **13/20**. This is a small qualitative check on held-out phrasings of learned fictional policies. Phrase hits are neither human accuracy grades nor evidence of generalization to new policies.

The adapted model confused counter hours with payment methods, gave appointment instructions for the kiosk, answered complaint acknowledgement with the filing window, and omitted form MP-2 for the market permit. It invented a missing credit card in the lost resident-card case and swapped the two bulky-waste policies. Even among hits, it said to file a parking notice rather than appeal it and added an appointment desk to the interpreter instructions. The base guessed many unsupported details. Review the raw [evaluation answers](results/evaluation.md) before drawing conclusions.

For example, the adapted model answered the held-out resident-card question with “A resident card renewal costs 14 crowns and requires form RC-4.” It can still invent, omit, or confuse details. This demo should not answer real public-service inquiries.

## Rank comparison

Run `python -m app.compare_ranks` to repeat the four CPU runs. It uses the same eight epochs, seed 42, target projections, decoding, and held-out questions at every rank. Alpha is twice the rank, so scaling stays fixed.

| Rank | Fact hits | Trainable parameters | Adapter file (bytes) | Final val loss |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 10/20 | 311,040 | 1,312,873 | 0.3277 |
| 4 | 16/20 | 622,080 | 2,557,033 | 0.2736 |
| 8 | 13/20 | 1,244,160 | 5,045,353 | 0.1901 |
| 16 | 13/20 | 2,488,320 | 10,028,063 | 0.1918 |

The rank sweep reuses the validation set for comparison. The scores are descriptive and vary by phrasing. Final loss and exact fact matches can rank models differently. The default adapter and merged benchmark both use rank 8. [Rank report](results/rank_comparison.json)

The Streamlit **Evaluation** tab shows the held-out questions, references, raw base and adapter answers, and their fact matches.
