from fastapi.testclient import TestClient

import rag.api as api

FAKE_STATE = {
    "final_answer": "Japanese companies supply most of the turbines [47341].",
    "grade": "grounded",
    "retrieved": [
        (47341, 3, "Japan and Iceland agree on geothermal energy cooperation", "chunk text", 0.569),
    ],
    "usage": {"in": 2063, "out": 18},
}


class FakeGraph:
    """Stands in for the real LangGraph pipeline: no OpenAI, no database."""

    def __init__(self, state=None, error=None):
        self.state, self.error = state, error

    def invoke(self, inputs, config=None):
        if self.error:
            raise self.error
        return self.state


def client_with(monkeypatch, graph):
    monkeypatch.setattr(api, "rag_graph", graph)
    return TestClient(api.app)


def test_ask_returns_answer_and_sources(monkeypatch):
    client = client_with(monkeypatch, FakeGraph(state=FAKE_STATE))
    response = client.post("/ask", json={"question": "Where do Iceland's turbines come from?"})

    assert response.status_code == 200
    body = response.json()
    assert body["grade"] == "grounded"
    assert body["sources"][0]["article_id"] == 47341
    assert body["tokens_in"] == 2063


def test_ask_rejects_too_short_question(monkeypatch):
    client = client_with(monkeypatch, FakeGraph(state=FAKE_STATE))
    response = client.post("/ask", json={"question": "hi"})
    assert response.status_code == 422


def test_ask_returns_502_when_pipeline_fails(monkeypatch):
    client = client_with(monkeypatch, FakeGraph(error=RuntimeError("OpenAI is down")))
    response = client.post("/ask", json={"question": "What is green hydrogen?"})

    assert response.status_code == 502
    assert "OpenAI" not in response.text  # internal details never leak to users