from functools import lru_cache

from langgraph.graph import StateGraph, START, END

from app.graph.nodes import (
    check_sufficiency_node,
    fallback_node,
    generate_node,
    rerank_node,
    retrieve_node,
)
from app.graph.state import GraphState


def _route_on_sufficiency(state: GraphState) -> str:
    return "generate" if state.get("sufficient") else "fallback"


@lru_cache
def get_compiled_graph():
    """
    retrieve -> rerank -> check_sufficiency -> (generate | fallback) -> END

    Compiled once and reused across requests. Every node function above is
    plain and testable in isolation — that's the main win over the old flat
    LCEL chain, which had no branch point at all between retrieval and
    generation.
    """
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("rerank", rerank_node)
    graph.add_node("check_sufficiency", check_sufficiency_node)
    graph.add_node("generate", generate_node)
    graph.add_node("fallback", fallback_node)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency",
        _route_on_sufficiency,
        {"generate": "generate", "fallback": "fallback"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()
