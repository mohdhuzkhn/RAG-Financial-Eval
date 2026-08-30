from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes_documents import router as documents_router
from app.api.routes_query import router as query_router
from app.services.document_registry import init_document_registry
from app.services.parent_store import init_parent_store
from app.services.vector_store import ensure_collection


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_parent_store()
    init_document_registry()
    ensure_collection()
    yield


app = FastAPI(
    title="Financial & Compliance RAG API",
    description="FastAPI + LangGraph backend — phase 1 of the god-stack migration.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(documents_router)
app.include_router(query_router)


@app.get("/health")
async def health():
    return {"status": "ok"}
