import math
from functools import lru_cache

from app.config import settings


@lru_cache
def _get_model():
    """
    Loaded lazily and cached — importing sentence-transformers/torch at
    module load time would slow down every app startup, including requests
    that never hit retrieval (e.g. /health). First call downloads the model
    weights from the Hugging Face Hub if not already cached locally; bake
    this into the Docker image build step for production so a cold request
    doesn't pay that cost.
    """
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model, max_length=512)


def rerank(query: str, candidate_texts: list[str]) -> list[float]:
    """
    Cross-encodes (query, candidate) pairs and returns a relevance score per
    candidate, in the same order as candidate_texts. Unlike vector
    similarity — which compares two independently-computed embeddings — a
    cross-encoder sees the query and each candidate together, which is why
    it catches cases plain vector search misses (a chunk that's topically
    close but doesn't actually answer the question).

    Scores are squashed to 0-1 via sigmoid so they're comparable to the
    existing MIN_RELEVANCE_SCORE threshold used for the sufficiency check.
    """
    if not candidate_texts:
        return []

    model = _get_model()
    pairs = [(query, text) for text in candidate_texts]
    raw_scores = model.predict(pairs)

    return [1 / (1 + math.exp(-score)) for score in raw_scores]
