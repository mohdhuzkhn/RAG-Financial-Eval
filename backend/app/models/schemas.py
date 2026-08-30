from pydantic import BaseModel, Field


class DocumentIngestResponse(BaseModel):
    document_id: str
    filename: str
    status: str  # "processing" | "ready" | "failed"


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    chunk_count: int | None = None
    error: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    document_id: str | None = Field(
        default=None,
        description="Restrict retrieval to one indexed document. Omit to search all.",
    )


class Citation(BaseModel):
    document_id: str
    page: int | str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sufficient: bool
    citations: list[Citation]
