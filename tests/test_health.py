from fastapi.testclient import TestClient
import pytest

import app.main as main


client = TestClient(main.app)


def test_health_works_without_api_keys() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ready", "setup_required"}
    assert set(body["checks"]) == {
        "elice_configured",
        "openai_configured",
        "index_ready",
    }


def test_routes_report_missing_index(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing_index(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("test index is missing")

    monkeypatch.setattr(main, "retrieve_chunks", missing_index)
    monkeypatch.setattr(main, "answer_question", missing_index)
    retrieve = client.post("/retrieve", json={"question": "What is a Pod?"})
    qa = client.post("/qa", json={"question": "What is a Pod?"})

    assert retrieve.status_code == 503
    assert qa.status_code == 503
