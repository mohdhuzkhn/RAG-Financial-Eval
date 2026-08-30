from functools import lru_cache

from langchain_openai import OpenAIEmbeddings

from app.config import settings


@lru_cache
def get_embeddings() -> OpenAIEmbeddings:
    """
    Single embedding client, reused across ingestion and query.

    NOTE: swapping to a self-hosted BGE-M3 model later means replacing this
    function's body only — nothing else in the app should import
    OpenAIEmbeddings directly. That's the whole point of isolating it here.
    """
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
