import logging
import time

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()  # must run before importing rag modules that read settings

from rag.db import connect
from rag.graph import _settings, build_graph
from rag.query_log import log_query

logger = logging.getLogger("rag.api")
app = FastAPI(title="CleanTech RAG")
rag_graph = build_graph()  # built once at startup, reused for every request


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    article_id: int
    title: str
    similarity: float


class AskResponse(BaseModel):
    answer: str
    grade: str
    sources: list[Source]
    tokens_in: int
    tokens_out: int


@app.get("/health")
def health():
    """Alive AND able to reach the database. Kubernetes will use this later."""
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {e}")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    start = time.perf_counter()
    settings = _settings({})  # the effective setup: top_k, rerank, etc.

    try:
        state = rag_graph.invoke({"query": req.question})
    except Exception as e:
        logger.exception("ask failed")
        log_query(req.question, int((time.perf_counter() - start) * 1000), "error",
                  error=f"{type(e).__name__}: {e}"[:1000], settings=settings)
        raise HTTPException(status_code=502, detail="Upstream model or database error")

    log_query(
        req.question, int((time.perf_counter() - start) * 1000), "ok",
        answer=state["final_answer"], grade=state["grade"],
        tokens_in=state["usage"]["in"], tokens_out=state["usage"]["out"],
        article_ids=[r[0] for r in state["retrieved"]], settings=settings,
    )

    return AskResponse(
        answer=state["final_answer"],
        grade=state["grade"],
        sources=[
            Source(article_id=r[0], title=r[2], similarity=round(float(r[4]), 3))
            for r in state["retrieved"]
        ],
        tokens_in=state["usage"]["in"],
        tokens_out=state["usage"]["out"],
    )