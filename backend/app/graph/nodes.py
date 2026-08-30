from langchain_core.prompts import ChatPromptTemplate

from app.config import settings
from app.graph.state import GraphState
from app.services.embeddings import get_embeddings
from app.services.llm import get_llm
from app.services.parent_store import get_parent_chunks
from app.services.reranker import rerank
from app.services.vector_store import search_children

COMPLIANCE_PROMPT = ChatPromptTemplate.from_template(
    """You are a senior financial compliance auditor analyzing an official filing.
Answer the user's question using ONLY the provided context below.

STRICT AUDIT RULES:
1. Base every claim directly on the context provided.
2. If specific financial figures, dates, or percentages are mentioned, quote them accurately.
3. Do NOT extrapolate or use outside knowledge.

Context Excerpts:
{context}

Question:
{question}

Answer with precise citations (e.g., [Page X]):"""
)


def retrieve_node(state: GraphState) -> GraphState:
    """
    Vector search over child chunks, then resolve to their parent chunks.
    RETRIEVAL_TOP_K is intentionally wide (20 by default) — this node's job
    is recall, not precision. rerank_node below narrows it down.
    """
    embeddings = get_embeddings()
    query_vector = embeddings.embed_query(state["question"])

    matches = search_children(
        query_vector=query_vector,
        top_k=settings.retrieval_top_k,
        document_id=state.get("document_id"),
    )

    # Dedupe by parent_id, keeping the best child score per parent —
    # multiple matching children in the same parent shouldn't duplicate context.
    best_score_by_parent: dict[str, float] = {}
    for match in matches:
        pid = match["parent_id"]
        if pid not in best_score_by_parent or match["score"] > best_score_by_parent[pid]:
            best_score_by_parent[pid] = match["score"]

    parents = get_parent_chunks(list(best_score_by_parent.keys()))

    context_chunks = [
        {
            "document_id": parent["document_id"],
            "page": parent["page"],
            "text": parent["text"],
            "score": score,
        }
        for parent_id, score in best_score_by_parent.items()
        if (parent := parents.get(parent_id)) is not None
    ]
    context_chunks.sort(key=lambda c: c["score"], reverse=True)

    return {"context_chunks": context_chunks}


def rerank_node(state: GraphState) -> GraphState:
    """
    Cross-encodes the question against each candidate parent chunk's text
    and keeps only the top RERANKER_TOP_N by that score, replacing the
    vector-similarity score used for ordering/thresholding downstream.
    This is the fix for the exact failure mode you hit earlier in this
    project — a semantically-close-but-wrong chunk outranking the actually
    relevant one — since a cross-encoder judges query and candidate
    together instead of comparing two independent embeddings.

    Toggle via ENABLE_RERANKER if you want to A/B against vector-only
    ranking, or if you haven't provisioned inference for it yet.
    """
    if not settings.enable_reranker:
        return {}

    chunks = state.get("context_chunks", [])
    if not chunks:
        return {}

    scores = rerank(state["question"], [c["text"] for c in chunks])
    reranked = sorted(
        (({**chunk, "score": score}) for chunk, score in zip(chunks, scores)),
        key=lambda c: c["score"],
        reverse=True,
    )

    return {"context_chunks": reranked[: settings.reranker_top_n]}


def check_sufficiency_node(state: GraphState) -> GraphState:
    """
    Programmatic gate instead of relying on the LLM to self-police — this is
    the difference from the original app, where INSUFFICIENT_DATA was only
    ever a prompt instruction the model could ignore.
    """
    chunks = state.get("context_chunks", [])
    sufficient = bool(chunks) and chunks[0]["score"] >= settings.min_relevance_score
    return {"sufficient": sufficient}


def generate_node(state: GraphState) -> GraphState:
    context_text = "\n\n".join(
        f"--- [Excerpt from Page {c['page']}] ---\n{c['text']}"
        for c in state["context_chunks"]
    )

    chain = COMPLIANCE_PROMPT | get_llm()
    response = chain.invoke({"context": context_text, "question": state["question"]})

    return {"answer": response.content}


def fallback_node(state: GraphState) -> GraphState:
    return {
        "answer": (
            "INSUFFICIENT_DATA: The provided filing excerpts do not contain "
            "enough context to answer this query."
        )
    }
