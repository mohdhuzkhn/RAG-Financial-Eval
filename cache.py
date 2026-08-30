import hashlib
import json
from functools import lru_cache

import redis

from app.config import settings


@lru_cache
def get_redis() -> redis.Redis:
    return redis.from_url(settings.redis_url, decode_responses=True)


def _query_cache_key(question: str, document_id: str | None) -> str:
    raw = f"{document_id or 'all'}::{question.strip().lower()}"
    return "query_response:" + hashlib.sha256(raw.encode()).hexdigest()


def get_cached_response(question: str, document_id: str | None) -> dict | None:
    """
    Exact-match cache only — same question text (case/whitespace-insensitive)
    against the same document. Not semantic/near-duplicate matching; that's
    a reasonable next step later but adds real complexity (embedding the
    query just to check the cache, picking a similarity threshold) that
    isn't worth it until exact-match hit rate is measured and found lacking.
    """
    raw = get_redis().get(_query_cache_key(question, document_id))
    return json.loads(raw) if raw else None


def set_cached_response(question: str, document_id: str | None, response: dict) -> None:
    get_redis().setex(
        _query_cache_key(question, document_id),
        settings.query_cache_ttl_seconds,
        json.dumps(response),
    )


def invalidate_document_cache_note() -> str:
    """
    NOTE, not yet wired up: cached answers for a document should be
    invalidated when that document is re-ingested (a filing gets amended,
    a corrected PDF is uploaded under the same document_id). Since cache
    keys are hashed per-question there's no cheap way to purge "everything
    for document X" without either a secondary index of keys per document
    or a short TTL as the safety net. For now the TTL (query_cache_ttl_seconds)
    is that safety net — keep it short if documents get re-ingested often.
    """
    raise NotImplementedError


def check_rate_limit(client_key: str, limit: int, window_seconds: int) -> bool:
    """
    Fixed-window counter. Returns True if the request is allowed, False if
    the client is over the limit. Good enough for a single-instance API;
    if you scale to multiple FastAPI replicas this still works correctly
    since the counter lives in Redis, not in-process.
    """
    r = get_redis()
    key = f"rate_limit:{client_key}:{window_seconds}"
    count = r.incr(key)
    if count == 1:
        r.expire(key, window_seconds)
    return count <= limit