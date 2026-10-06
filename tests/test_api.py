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

def client_with(monkeypatch, graph, logged=None):
    monkeypatch.setattr(api, "rag_graph", graph)
    # Capture log calls instead of writing to the real database
    monkeypatch.setattr(api, "log_query", lambda *a, **k: logged.append((a, k)) if logged is not None else None)
    monkeypatch.setattr(api, "ACCESS_CODE", "")  # tests run without a code unless they set one
    api._recent_asks.clear()                      # each test starts with a fresh count
    monkeypatch.setattr(api, "MAX_ASK_PER_HOUR", 100)
    monkeypatch.setattr(api, "MAX_ASK_PER_DAY", 100)
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


def test_failed_request_is_logged_as_error(monkeypatch):
    logged = []
    client = client_with(monkeypatch, FakeGraph(error=RuntimeError("OpenAI is down")), logged)
    client.post("/ask", json={"question": "What is green hydrogen?"})

    assert len(logged) == 1
    args, kwargs = logged[0]
    assert args[2] == "error"
    assert "OpenAI is down" in kwargs["error"]


def test_ask_rejects_missing_access_code(monkeypatch):
    client = client_with(monkeypatch, FakeGraph(state=FAKE_STATE))
    monkeypatch.setattr(api, "ACCESS_CODE", "friends-only")
    response = client.post("/ask", json={"question": "What is green hydrogen?"})
    assert response.status_code == 401


def test_ask_accepts_correct_access_code(monkeypatch):
    client = client_with(monkeypatch, FakeGraph(state=FAKE_STATE))
    monkeypatch.setattr(api, "ACCESS_CODE", "friends-only")
    response = client.post(
        "/ask",
        json={"question": "What is green hydrogen?"},
        headers={"X-Access-Code": "friends-only"},
    )
    assert response.status_code == 200

def test_ask_returns_429_when_hourly_limit_hit(monkeypatch):
    client = client_with(monkeypatch, FakeGraph(state=FAKE_STATE))
    monkeypatch.setattr(api, "MAX_ASK_PER_HOUR", 2)
    body = {"question": "What is green hydrogen?"}

    assert client.post("/ask", json=body).status_code == 200
    assert client.post("/ask", json=body).status_code == 200
    assert client.post("/ask", json=body).status_code == 429