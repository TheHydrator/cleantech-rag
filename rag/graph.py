import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag.db import connect
from rag.generation import generate_answer
from rag.grader import grade_answer
from rag.hyde import generate_hypothesis
from rag.retrieval import embed_query, merge_by_article, search

UNVERIFIED_WARNING = (
    "\n\nWarning: parts of this answer could not be verified against the source articles."
)


class RAGState(TypedDict, total=False):
    query: str
    retrieved: list
    answer: str          # raw answer from the generator
    final_answer: str    # what the user sees (may include a warning)
    usage: dict
    hypothesis: str
    hyde_used: bool
    hyde_error: str
    grade: str           # grounded / not_grounded / grader_error / skipped
    grade_reasoning: str


def _settings(config):
    c = config.get("configurable", {})
    return {
        "use_hyde": c.get("use_hyde", False),
        "top_k": c.get("top_k", int(os.environ.get("TOP_K", 5))),
        "rerank": c.get("rerank", os.environ.get("RERANK", "false").lower() == "true"),
        "fetch_k": c.get("fetch_k", int(os.environ.get("FETCH_K", 20))),
        "use_grader": c.get("use_grader", os.environ.get("GRADER", "true").lower() == "true"),
    }


def retrieve_node(state: RAGState, config) -> dict:
    s = _settings(config)
    fetch_k = s["fetch_k"] if s["rerank"] else s["top_k"]
    update = {"hyde_used": False}

    with connect() as conn:
        results = search(conn, embed_query(state["query"]), k=fetch_k)
        if s["use_hyde"]:
            try:
                hypothesis = generate_hypothesis(state["query"])
                hyde_results = search(conn, embed_query(hypothesis), k=fetch_k)
                results = merge_by_article(results, hyde_results)
                update.update(hypothesis=hypothesis, hyde_used=True)
            except Exception as e:
                update["hyde_error"] = str(e)

    if s["rerank"]:
        from rag.rerank import rerank
        results = rerank(state["query"], results, top_n=s["top_k"])

    update["retrieved"] = results
    return update


def generate_node(state: RAGState) -> dict:
    answer, usage = generate_answer(state["query"], state["retrieved"])
    return {
        "answer": answer,
        "usage": {"in": usage.prompt_tokens, "out": usage.completion_tokens},
    }


def grade_node(state: RAGState, config) -> dict:
    if not _settings(config)["use_grader"]:
        return {"grade": "skipped", "final_answer": state["answer"]}

    try:
        grounded, reasoning = grade_answer(state["retrieved"], state["answer"])
        grade = "grounded" if grounded else "not_grounded"
    except Exception as e:
        # Fail CLOSED: a broken grader never approves an answer
        grade, reasoning = "grader_error", f"Grader failed: {e}"

    final = state["answer"] if grade == "grounded" else state["answer"] + UNVERIFIED_WARNING
    return {"grade": grade, "grade_reasoning": reasoning, "final_answer": final}


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade", grade_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_edge("grade", END)
    return graph.compile()