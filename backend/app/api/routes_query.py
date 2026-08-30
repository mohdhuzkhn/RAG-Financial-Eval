from fastapi import APIRouter

from app.graph.build_graph import get_compiled_graph
from app.models.schemas import Citation, QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def run_query(request: QueryRequest):
    graph = get_compiled_graph()

    result = graph.invoke(
        {"question": request.question, "document_id": request.document_id}
    )

    citations = [
        Citation(document_id=c["document_id"], page=c["page"], snippet=c["text"][:280])
        for c in result.get("context_chunks", [])
    ] if result.get("sufficient") else []

    return QueryResponse(
        answer=result["answer"],
        sufficient=result.get("sufficient", False),
        citations=citations,
    )
