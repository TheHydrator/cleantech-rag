from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag.db import connect
from rag.generation import generate_answer
from rag.retrieval import embed_query, search

from rag.hyde import generate_hypothesis
from rag.retrieval import embed_query, merge_by_article, search

class RAGState(TypedDict, total=False):
    query: str
    retrieved: list
    final_answer: str
    usage: dict
    hypothesis: str
    hyde_used: bool
    hyde_error: str


def retrieve_node(state: RAGState, config) -> dict:
    use_hyde = config.get("configurable", {}).get("use_hyde", False)

    with connect() as conn:
        results = search(conn, embed_query(state["query"]))
        if not use_hyde:
            return {"retrieved": results, "hyde_used": False}

        try:
            hypothesis = generate_hypothesis(state["query"])
            hyde_results = search(conn, embed_query(hypothesis))
        except Exception as e:
            # Fall back to normal search, but record that it happened
            return {"retrieved": results, "hyde_used": False, "hyde_error": str(e)}

    return {
        "retrieved": merge_by_article(results, hyde_results),
        "hypothesis": hypothesis,
        "hyde_used": True,
    }


def generate_node(state: RAGState) -> dict:
    answer, usage = generate_answer(state["query"], state["retrieved"])
    return {
        "final_answer": answer,
        "usage": {"in": usage.prompt_tokens, "out": usage.completion_tokens},
    }


def build_graph():
    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()