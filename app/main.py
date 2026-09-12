import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag.qa import answer_question, retrieve as retrieve_chunks


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "embeddings.npy"
load_dotenv(PROJECT_ROOT / ".env")

app = FastAPI(title="Kubernetes Citation RAG", version="0.2.0", description="Evaluation-first RAG over an allowlist of official Kubernetes documents.")


class RetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class QARequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


def readiness() -> dict[str, bool]:
    return {"elice_configured": all(os.getenv(name) for name in ("ELICE_API_KEY", "ELICE_BASE_URL")), "openai_configured": bool(os.getenv("OPENAI_API_KEY")), "index_ready": INDEX_PATH.is_file()}


@app.get("/health")
def health() -> dict[str, object]:
    checks = readiness()
    return {"status": "ready" if all(checks.values()) else "setup_required", "checks": checks}


@app.post("/retrieve")
def retrieve(request: RetrieveRequest) -> dict[str, object]:
    try:
        return {"question": request.question, "results": retrieve_chunks(request.question, top_k=request.top_k)}
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.post("/qa")
def qa(request: QARequest) -> dict[str, object]:
    try:
        return answer_question(request.question, top_k=request.top_k)
    except (FileNotFoundError, RuntimeError) as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
