# Alderbridge municipal services — SAMPLE data

Alderbridge, its office, forms, eligibility rules, deadlines and currency are fictional. These are original examples, not government policy or advice. The canonical answers in `build_faqs.py` are the fictional departmental guidelines for this demo.

The JSONL file includes 80 records across 20 topics. Each record includes a question, reference answer, required fact phrases, and split label. Three phrasings per topic are used for training and one different paraphrase for validation. The facts themselves are present in training. This split tests whether the model can retrieve a learned fictional policy when it hears a new phrasing, not whether it can infer an unseen policy.

Regenerate with `python data/build_faqs.py` from the repository root.
