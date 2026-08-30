import sqlite3
from contextlib import contextmanager

from app.config import settings


@contextmanager
def _connect():
    conn = sqlite3.connect(settings.parent_store_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_parent_store() -> None:
    """
    Sqlite is a deliberate stand-in, not a design choice — it's here so
    parent chunks survive an app restart during phase 1 without pulling in
    Postgres before it's actually scheduled (phase 4 in the migration plan).
    Swap this module for a Postgres-backed one later; nothing else in the
    app should need to change since callers only see get/set by parent_id.
    """
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS parent_chunks (
                parent_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                page TEXT,
                text TEXT NOT NULL
            )
            """
        )


def save_parent_chunk(parent_id: str, document_id: str, page: int | str, text: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO parent_chunks (parent_id, document_id, page, text) "
            "VALUES (?, ?, ?, ?)",
            (parent_id, document_id, str(page), text),
        )


def get_parent_chunks(parent_ids: list[str]) -> dict[str, dict]:
    if not parent_ids:
        return {}
    placeholders = ",".join("?" for _ in parent_ids)
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT parent_id, document_id, page, text FROM parent_chunks "
            f"WHERE parent_id IN ({placeholders})",
            parent_ids,
        ).fetchall()
    return {
        row[0]: {"document_id": row[1], "page": row[2], "text": row[3]}
        for row in rows
    }
