from typing import TypedDict

from langgraph.graph import END, StateGraph

from rag.db import connect
from rag.generation import generate_answer
from rag.retrieval import embed_query, search


class RAGState(TypedDict, total=False):
    query: str
    retrieved: list
    final_answer: str
    usage: dict


def retrieve_node(state: RAGState) -> dict:
    query_vec = embed_query(state["query"])
    with connect() as conn:
        results = search(conn, query_vec)
    return {"retrieved": results}


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