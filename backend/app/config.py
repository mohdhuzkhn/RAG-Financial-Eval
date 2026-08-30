from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central config. Every value here is env-driven on purpose — the whole
    point of phase 1 is that swapping an LLM provider or model (the thing
    that broke the Streamlit app when gemini-1.5-flash was retired) is a
    config change, not a code change.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: str = "openai"  # openai | anthropic | google_genai
    llm_model: str = "gpt-5"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    google_api_key: str | None = None

    # Embeddings
    embedding_model: str = "text-embedding-3-large"
    embedding_dim: int = 3072

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "financial_child_chunks"

    # Retrieval
    retrieval_top_k: int = 20  # cast a wider net now that a reranker narrows it
    min_relevance_score: float = 0.35

    # Reranker
    enable_reranker: bool = True
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_top_n: int = 5  # candidates kept after reranking, sent to the LLM

    # Parent store (sqlite placeholder — becomes Postgres in phase 4)
    parent_store_path: str = "./parent_store.db"


settings = Settings()
