# Elice RAG Eval Harness

## Project

Build an explainable Kubernetes-document RAG QA service for the Elice AI Engineer
(Platform) mini-project. Part A is a working MVP. Part B evaluation and Part C
experiments are the priority.

Stack: Python 3.11+, FastAPI, Elice GPT-5.6 Luna, OpenAI Embeddings, NumPy/FAISS,
BM25, JSONL evaluation data.

Expected code layout:

```text
app/            FastAPI entrypoint
rag/            ingest, index, retrieval, answer generation
eval/           metrics and evaluation runner
config/         corpus and experiment settings
data/           raw, processed, index, Gold Set
reports/        machine-readable evaluation outputs
tests/          offline tests with mocked APIs
.agents/skills/ reusable project workflows
.codex/agents/  project-scoped subagent profiles
.codex/hooks/   deterministic safety checks
```

## Commands

Run commands from the repository root.

```bash
pip install -r requirements.txt
python scripts/download_corpus.py
python -m rag.ingest
python -m rag.index
uvicorn app.main:app --reload
python -m eval.run_eval --config config/baseline.yaml
pytest -q
```

When these commands change, update this file and `README.md` together.

## Coding rules

- Keep execution paths readable from top to bottom.
- Prefer plain functions and explicit data over classes.
- Create a helper only for a meaningful operation or real repetition.
- Do not create `utils.py`, `common.py`, `helpers.py`, or `base.py`.
- Do not add factories, registries, DI containers, repository/service layers,
  abstract base classes, or interfaces with one implementation.
- Keep prompts, API calls, retrieval scores, thresholds, and metric formulas visible.
- Keep RAG runtime and evaluation code separate without a shared abstraction layer.
- Do not use LangChain or LlamaIndex for the baseline.
- Preserve unrelated user changes. Do not commit unless explicitly requested.
- Never use `git add -A`, `git add .`, destructive git commands, or broad deletes.

## Model and data rules

- Elice client uses `ELICE_BASE_URL`, `ELICE_API_KEY`, and `ELICE_LLM_MODEL`.
- `ELICE_BASE_URL` includes `/v1`; use the provider-qualified model ID `openai/gpt-5.6-luna` and normalize the documented short alias at the call site.
- Chat Completions uses `max_completion_tokens`, not `max_tokens`.
- Baseline uses `temperature=0`, `reasoning_effort="none"`, and no streaming.
- Do not send `seed` to Luna; record a local evaluation seed instead.
- OpenAI embeddings use `OPENAI_API_KEY` and `OPENAI_EMBEDDING_MODEL`.
- Never read, print, log, edit, or commit `.env`, keys, tokens, or certificate files.
- Unit tests must mock external APIs. Live calls require an explicit user request.
- Corpus is a controlled set of about 30 official Kubernetes documents.
- Citations come from retrieved metadata, never from LLM-generated URLs.
- Return `answerable=false` when the retrieved evidence is insufficient.

## Evaluation rules

- Use at least 20 Gold Set questions with expected evidence or acceptance criteria.
- Include factual, comparison, summary, reasoning, troubleshooting, and unanswerable items.
- Evaluate retrieval and generation separately.
- Minimum metrics: Recall@K, MRR, Answerable Accuracy, Citation Accuracy.
- Record models, config, local seed, timestamp, token usage, and per-question outputs.
- Save baseline results before implementing or measuring the improved retriever.
- Compare baseline and hybrid retrieval with the same Gold Set and configuration.

## Skills

- `$build-rag-baseline`: implement or repair Part A baseline RAG.
- `$run-rag-eval`: build and run Part B evaluation without changing retrieval behavior.
- `$report-rag-experiment`: compare saved Part C runs and update the report.

Skills live under `.agents/skills/<name>/SKILL.md` and may be invoked explicitly.

## Agents

- `corpus_researcher`: read-only corpus and source verification.
- `eval_reviewer`: read-only review of metrics, leakage, and experiment validity.

Custom profiles live under `.codex/agents/`. Delegate only bounded, independent work;
parallel agents must not edit the same files.

## Hooks

`.codex/hooks.json` runs `.codex/hooks/pre_tool_guard.mjs` before shell commands and
patches. It blocks sensitive-file edits, broad staging, destructive git commands,
and generic helper modules. Review and trust project hooks with `/hooks` before use.

## Completion check

Before reporting completion, run the smallest relevant commands above and report:

- files changed
- checks executed and results
- external calls made, if any
- remaining limitations
