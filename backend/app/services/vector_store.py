import uuid
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams, Filter, FieldCondition, MatchValue

from app.config import settings


@lru_cache
def get_qdrant() -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)


def ensure_collection() -> None:
    """Idempotent — safe to call on every app startup."""
    client = get_qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if settings.qdrant_collection not in existing:
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config=VectorParams(
                size=settings.embedding_dim,
                distance=Distance.COSINE,
            ),
        )


def upsert_child_chunks(
    document_id: str,
    child_texts: list[str],
    child_vectors: list[list[float]],
    parent_ids: list[str],
    pages: list[int | str],
) -> None:
    """
    Each point stores just enough payload to (a) filter by document_id at
    query time and (b) look the parent chunk up in the parent store after a
    vector match — the heavy parent text itself never goes into Qdrant.
    """
    client = get_qdrant()
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=vector,
            payload={
                "document_id": document_id,
                "parent_id": parent_id,
                "page": page,
                "child_text": text,
            },
        )
        for text, vector, parent_id, page in zip(child_texts, child_vectors, parent_ids, pages)
    ]
    client.upsert(collection_name=settings.qdrant_collection, points=points)


def search_children(
    query_vector: list[float],
    top_k: int,
    document_id: str | None = None,
) -> list[dict]:
    client = get_qdrant()
    query_filter = None
    if document_id:
        query_filter = Filter(
            must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))]
        )

    results = client.query_points(
        collection_name=settings.qdrant_collection,
        query=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    ).points

    return [
        {
            "score": point.score,
            "parent_id": point.payload["parent_id"],
            "document_id": point.payload["document_id"],
            "page": point.payload["page"],
            "child_text": point.payload["child_text"],
        }
        for point in results
    ]
