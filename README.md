# Public FAQ Answer Bot Trainer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/b6n79ky6cs/badge.svg)](https://ask.thili.ai/b/b6n79ky6cs)

This project, "Public FAQ Answer Bot Trainer," enables public service administrators to train a language model using LoRA. The goal is to provide accurate answers to frequently asked questions based on specific departmental guidelines, thereby reducing inquiry load. It demonstrates the LoRA fine-tuning technique by generating accurate FAQ answers after training and merging the adapter.

## What you are building

**Goal.** You will have built a bot that public service administrators can train using LoRA to provide accurate answers to frequently asked questions based on specific departmental guidelines, reducing inquiry load.

**Done when.** Your bot will generate accurate and relevant answers to public service FAQs after training a pre-trained LLM with a LoRA adapter and merging it for inference, proving the effectiveness of the LoRA fine-tuning technique.

## Milestones

Tick these off as you go, and commit after each one.

- [ ] **1. Create a small dataset of public service FAQs and their answers.**  
      Hint: Use data collection and formatting techniques to prepare your dataset.
- [ ] **2. Implement LoRA adapter injection and training on a pre-trained LLM (e.g., GPT-2).**  
      Hint: Apply the `lfllm-lflora` technique for LoRA adapter injection and training.
- [ ] **3. Build a Streamlit interface where users can ask questions and get answers.**  
      Hint: Use Streamlit for building an interactive web application interface.
- [ ] **4. Verify the accuracy and relevance of the generated answers.**  
      Hint: Perform qualitative evaluation of the model's generated responses for accuracy and relevance.
- [ ] **5. Implement adapter merging for inference.**  
      Hint: Use the `lfllm-lflora` technique to merge the LoRA adapter for efficient inference.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
streamlit run app/main.py
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Expand the dataset to include FAQs from multiple public service departments.
- Add functionality for administrators to easily update or add new FAQ entries through the Streamlit interface.
- Implement basic logging to track the types of questions users are asking.
