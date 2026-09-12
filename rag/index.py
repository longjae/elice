import json
import os
from pathlib import Path

import numpy as np
import yaml
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    if not api_key:
        raise SystemExit("Missing OPENAI_API_KEY")
    with (PROJECT_ROOT / "config" / "baseline.yaml").open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    chunks_path = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
    if not chunks_path.exists():
        raise SystemExit("Missing data/processed/chunks.jsonl. Run python -m rag.ingest first.")

    chunks = [json.loads(line) for line in chunks_path.read_text(encoding="utf-8").splitlines() if line]
    client = OpenAI(api_key=api_key)
    vectors: list[list[float]] = []
    batch_size = config["embedding_batch_size"]
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        response = client.embeddings.create(model=model, input=[chunk["text"] for chunk in batch])
        vectors.extend(item.embedding for item in response.data)
        print(f"embedded {min(start + batch_size, len(chunks))}/{len(chunks)}")

    embeddings = np.asarray(vectors, dtype=np.float32)
    embeddings /= np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    index_dir = PROJECT_ROOT / "data" / "index"
    index_dir.mkdir(parents=True, exist_ok=True)
    np.save(index_dir / "embeddings.npy", embeddings)
    (index_dir / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    (index_dir / "metadata.json").write_text(json.dumps({"embedding_model": model, "chunk_count": len(chunks), "dimensions": int(embeddings.shape[1]), "chunk_size": config["chunk_size"], "chunk_overlap": config["chunk_overlap"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"indexed={len(chunks)} model={model} output={index_dir}")


if __name__ == "__main__":
    main()
