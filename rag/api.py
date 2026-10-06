import logging
import os
import secrets
import time
from contextlib import asynccontextmanager
import threading

from pathlib import Path
from fastapi.responses import FileResponse
from collections import deque
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Response, Security
from fastapi.security import APIKeyHeader
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

load_dotenv()  # must run before importing rag modules that read settings

from rag.db import connect
from rag.graph import _settings, build_graph
from rag.query_log import log_query

logger = logging.getLogger("rag.api")
@asynccontextmanager
async def lifespan(app):
    # Warm up before serving traffic, so no user pays the first-request cost
    if _settings({})["rerank"]:
        from rag.rerank import get_model
        get_model().predict([("warm up", "warm up")])
        logger.info("reranker warmed up")
    yield


app = FastAPI(title="CleanTech RAG", lifespan=lifespan)
rag_graph = build_graph()  # built once at startup, reused for every request

INDEX_HTML = Path(__file__).parent / "static" / "index.html"

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_HTML)

ACCESS_CODE = os.environ.get("ACCESS_CODE", "")
access_code_header = APIKeyHeader(name="X-Access-Code", auto_error=False)


def require_access_code(code: str | None = Security(access_code_header)):
    """Reject /ask requests without the right code. Disabled when ACCESS_CODE is empty (local dev)."""
    if ACCESS_CODE and not (code and secrets.compare_digest(code, ACCESS_CODE)):
        raise HTTPException(status_code=401, detail="Missing or invalid access code")

MAX_ASK_PER_HOUR = int(os.environ.get("MAX_ASK_PER_HOUR", "20"))
MAX_ASK_PER_DAY = int(os.environ.get("MAX_ASK_PER_DAY", "100"))
_recent_asks = deque()          # timestamps of /ask calls in the last 24 hours
_rate_lock = threading.Lock()   # requests run in parallel threads


def within_limits():
    """Global caps that protect the OpenAI budget. Returns False if a limit is hit."""
    now = time.time()
    with _rate_lock:
        while _recent_asks and now - _recent_asks[0] > 86400:
            _recent_asks.popleft()
        last_hour = sum(1 for t in _recent_asks if now - t <= 3600)
        if last_hour >= MAX_ASK_PER_HOUR or len(_recent_asks) >= MAX_ASK_PER_DAY:
            return False
        _recent_asks.append(now)
        return True

ASK_REQUESTS = Counter("rag_ask_requests_total", "Number of /ask requests", ["status"])
ASK_LATENCY = Histogram(
    "rag_ask_latency_seconds", "End-to-end /ask latency",
    buckets=(0.5, 1, 2, 5, 10, 15, 20, 30, 60),
)
ASK_TOKENS = Counter("rag_tokens_total", "LLM tokens used by /ask", ["direction"])
ASK_GRADES = Counter("rag_answer_grades_total", "Grader verdicts", ["grade"])



@app.get("/metrics", include_in_schema=False)
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    article_id: int
    title: str
    similarity: float
    url: str | None = None


class AskResponse(BaseModel):
    answer: str
    grade: str
    sources: list[Source]
    tokens_in: int
    tokens_out: int



@app.get("/health", include_in_schema=False)
def health():
    """Alive AND able to reach the database. Kubernetes will use this later."""
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {e}")
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, _: None = Depends(require_access_code)):
    if not within_limits():
        ASK_REQUESTS.labels(status="rate_limited").inc()
        raise HTTPException(status_code=429, detail="Too many questions right now. Please try again later.")
    start = time.perf_counter()
    settings = _settings({})

    try:
        state = rag_graph.invoke({"query": req.question})
    except Exception as e:
        elapsed = time.perf_counter() - start
        logger.exception("ask failed")
        ASK_REQUESTS.labels(status="error").inc()
        ASK_LATENCY.observe(elapsed)
        log_query(req.question, int(elapsed * 1000), "error",
                  error=f"{type(e).__name__}: {e}"[:1000], settings=settings)
        raise HTTPException(status_code=502, detail="Upstream model or database error")

    elapsed = time.perf_counter() - start
    ASK_REQUESTS.labels(status="ok").inc()
    ASK_LATENCY.observe(elapsed)
    ASK_TOKENS.labels(direction="in").inc(state["usage"]["in"])
    ASK_TOKENS.labels(direction="out").inc(state["usage"]["out"])
    ASK_GRADES.labels(grade=state["grade"]).inc()

    log_query(
        req.question, int(elapsed * 1000), "ok",
        answer=state["final_answer"], grade=state["grade"],
        tokens_in=state["usage"]["in"], tokens_out=state["usage"]["out"],
        article_ids=[r[0] for r in state["retrieved"]], settings=settings,
    )

    return AskResponse(
        answer=state["final_answer"],
        grade=state["grade"],
        sources=[
            Source(article_id=r[0], title=r[2], similarity=round(float(r[4]), 3),
                   url=r[5] if len(r) > 5 else None)
            for r in state["retrieved"]
        ],
        tokens_in=state["usage"]["in"],
        tokens_out=state["usage"]["out"],
    )