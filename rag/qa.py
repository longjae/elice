import json
import os
import re
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def retrieve(question: str, config_path: Path | None = None, top_k: int | None = None) -> list[dict[str, object]]:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    config_path = config_path or PROJECT_ROOT / "config" / "baseline.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    index_dir = PROJECT_ROOT / "data" / "index"
    embeddings_path = index_dir / "embeddings.npy"
    chunks_path = index_dir / "chunks.json"
    metadata_path = index_dir / "metadata.json"
    if not embeddings_path.exists() or not chunks_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("Local index is incomplete. Run python -m rag.index first.")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["embedding_model"] != model:
        raise RuntimeError(f"Embedding model mismatch: index={metadata['embedding_model']} environment={model}")

    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    embeddings = np.load(embeddings_path)
    response = OpenAI(api_key=api_key).embeddings.create(model=model, input=[question])
    query = np.asarray(response.data[0].embedding, dtype=np.float32)
    query /= max(float(np.linalg.norm(query)), 1e-12)
    dense_scores = embeddings @ query
    count = min(top_k or config["top_k"], len(chunks))
    dense_order = np.argsort(dense_scores)[::-1]

    retrieval_scores: dict[int, float] = {}
    dense_ranks: dict[int, int] = {}
    bm25_ranks: dict[int, int] = {}
    if config["retrieval_mode"] == "dense":
        indices = dense_order[:count]
    elif config["retrieval_mode"] == "hybrid_rrf":
        candidate_count = min(int(config["candidate_k"]), len(chunks))
        rrf_k = int(config["rrf_k"])
        dense_weight = float(config["dense_weight"])
        bm25_weight = float(config["bm25_weight"])

        for rank, index in enumerate(dense_order[:candidate_count], start=1):
            dense_ranks[int(index)] = rank
            retrieval_scores[int(index)] = dense_weight / (rrf_k + rank)

        token_pattern = re.compile(r"[a-z0-9][a-z0-9._/-]*")
        tokenized_corpus = [token_pattern.findall(f"{chunk['title']} {chunk['section']} {chunk['text']}".lower()) for chunk in chunks]
        query_tokens = token_pattern.findall(question.lower())
        if query_tokens:
            bm25_scores = BM25Okapi(tokenized_corpus).get_scores(query_tokens)
            positive_indices = [int(index) for index in np.argsort(bm25_scores)[::-1] if bm25_scores[int(index)] > 0][:candidate_count]
            for rank, index in enumerate(positive_indices, start=1):
                bm25_ranks[index] = rank
                retrieval_scores[index] = retrieval_scores.get(index, 0.0) + bm25_weight / (rrf_k + rank)

        indices = np.asarray(sorted(retrieval_scores, key=lambda index: retrieval_scores[index], reverse=True)[:count])
    else:
        raise RuntimeError(f"Unsupported retrieval_mode: {config['retrieval_mode']}")

    results: list[dict[str, object]] = []
    for rank, index in enumerate(indices, start=1):
        result = dict(chunks[int(index)])
        result["score"] = float(dense_scores[int(index)])
        result["rank"] = rank
        if config["retrieval_mode"] == "hybrid_rrf":
            result["retrieval_score"] = retrieval_scores[int(index)]
            result["dense_rank"] = dense_ranks.get(int(index))
            result["bm25_rank"] = bm25_ranks.get(int(index))
        results.append(result)
    return results


def answer_question(question: str, config_path: Path | None = None, top_k: int | None = None) -> dict[str, object]:
    load_dotenv(PROJECT_ROOT / ".env")
    config_path = config_path or PROJECT_ROOT / "config" / "baseline.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    results = retrieve(question, config_path=config_path, top_k=top_k)
    if not results or float(results[0]["score"]) < float(config["minimum_similarity"]):
        return {"question": question, "answer": "제공된 Kubernetes 문서에서 답변에 필요한 근거를 찾지 못했습니다.", "answerable": False, "citations": [], "retrieval": results, "usage": {}}

    api_key = os.getenv("ELICE_API_KEY")
    base_url = os.getenv("ELICE_BASE_URL")
    model = os.getenv("ELICE_LLM_MODEL", "gpt-5.6-luna")
    if model == "gpt-5.6-luna":
        model = "openai/gpt-5.6-luna"
    missing = [name for name, value in {"ELICE_API_KEY": api_key, "ELICE_BASE_URL": base_url}.items() if not value]
    if missing:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")
    if not base_url.rstrip("/").endswith("/v1"):
        base_url = f"{base_url.rstrip('/')}/v1"

    context = "\n\n".join(f"[{number}] {item['title']} > {item['section']}\n{item['text']}" for number, item in enumerate(results, start=1))
    response = OpenAI(base_url=base_url, api_key=api_key).chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "당신은 Kubernetes 공식 문서 기반 QA 시스템입니다. 제공된 CONTEXT만 사용해 한국어로 간결하게 답하세요. 근거가 부족하면 추측하지 말고 정보가 부족하다고 답하세요. URL이나 출처를 직접 만들지 마세요."},
            {"role": "user", "content": f"CONTEXT\n{context}\n\nQUESTION\n{question}"},
        ],
        temperature=config["temperature"],
        max_completion_tokens=config["max_completion_tokens"],
        reasoning_effort=config["reasoning_effort"],
    )

    citations: list[dict[str, str]] = []
    seen: set[str] = set()
    for result in results:
        doc_id = str(result["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        citations.append({"doc_id": doc_id, "title": str(result["title"]), "section": str(result["section"]), "url": str(result["url"])})
        if len(citations) == 3:
            break

    usage = response.usage
    return {
        "question": question,
        "answer": response.choices[0].message.content or "",
        "answerable": True,
        "model": response.model,
        "citations": citations,
        "retrieval": results,
        "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", None), "completion_tokens": getattr(usage, "completion_tokens", None), "cached_prompt_tokens": getattr(usage, "cached_prompt_tokens", None)},
    }
