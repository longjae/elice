---
name: run-rag-eval
description: Build, validate, or execute the Part B RAG evaluation harness over the Gold Set, producing retrieval and generation metrics plus JSON and CSV reports. Do not change retrieval algorithms while evaluating them.
---

# Run the RAG evaluation

1. Validate every Gold item before spending API credits.
2. Require an ID, question type, answerability label, expected evidence, and either a reference answer or acceptance criteria.
3. Keep Recall@K and MRR formulas directly visible in `eval/metrics.py`.
4. Evaluate answerability and citations independently from retrieval ranking.
5. Continue after per-question errors and record each error in the report.
6. Record model IDs, configuration, local seed, timestamp, token usage, and raw outputs.
7. Write detailed JSON plus a compact CSV summary under `reports/`.
8. Do not tune thresholds, prompts, or Top-K during the same run.
9. Run offline metric tests before any live evaluation.

If an LLM judge is used, persist its prompt, model, raw response, parsed score, and rubric.
State that remote generation is not guaranteed deterministic.
