---
name: build-rag-baseline
description: Implement or repair the Part A Kubernetes RAG baseline including ingest, dense retrieval, grounded answers, citations, abstention, and FastAPI endpoints. Do not use for evaluation-only or hybrid-retrieval work.
---

# Build the RAG baseline

1. Inspect the current config, schemas, and persisted data before editing.
2. Keep the pipeline explicit: download -> parse -> chunk -> embed -> index -> retrieve -> answer.
3. Preserve `doc_id`, title, section, URL, category, and `chunk_id` through every stage.
4. Use OpenAI Embeddings for chunks and queries with the same recorded model.
5. Use Elice GPT-5.6 Luna through the OpenAI-compatible Chat Completions API.
6. Build citations only from retrieved metadata.
7. Return an explicit insufficient-evidence response when the threshold is not met.
8. Add offline tests with mocked external clients.
9. Run the smallest relevant ingest, import, endpoint, and test checks.

Do not add provider abstractions, service layers, generic helpers, hybrid retrieval,
LLM judges, or streaming during baseline work.

Before finishing, confirm that a reproducible baseline can be saved before Part C begins.
