from contextlib import contextmanager
from functools import lru_cache

from psycopg_pool import ConnectionPool

from app.config import settings


@lru_cache
def get_pool() -> ConnectionPool:
    """
    One pool for the process lifetime. `open=False` here + explicit
    `.open()` in the FastAPI lifespan (see main.py) avoids psycopg_pool's
    warning about opening a pool outside an event loop/async context at
    import time.
    """
    return ConnectionPool(conninfo=settings.database_url, open=False, min_size=1, max_size=10)


@contextmanager
def get_conn():
    with get_pool().connection() as conn:
        yield conn