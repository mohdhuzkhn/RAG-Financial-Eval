from app.services.parent_store import _connect


def init_document_registry() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                document_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                status TEXT NOT NULL,
                chunk_count INTEGER,
                error TEXT
            )
            """
        )


def create_document(document_id: str, filename: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO documents (document_id, filename, status) VALUES (?, ?, 'processing')",
            (document_id, filename),
        )


def mark_ready(document_id: str, chunk_count: int) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET status = 'ready', chunk_count = ? WHERE document_id = ?",
            (chunk_count, document_id),
        )


def mark_failed(document_id: str, error: str) -> None:
    with _connect() as conn:
        conn.execute(
            "UPDATE documents SET status = 'failed', error = ? WHERE document_id = ?",
            (error, document_id),
        )


def get_document(document_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT document_id, status, chunk_count, error FROM documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    if row is None:
        return None
    return {"document_id": row[0], "status": row[1], "chunk_count": row[2], "error": row[3]}
