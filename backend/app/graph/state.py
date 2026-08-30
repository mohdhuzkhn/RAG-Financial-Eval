from typing import TypedDict


class ContextChunk(TypedDict):
    document_id: str
    page: int | str
    text: str
    score: float


class GraphState(TypedDict, total=False):
    question: str
    document_id: str | None  # optional filter, from the API request

    context_chunks: list[ContextChunk]
    sufficient: bool

    answer: str
