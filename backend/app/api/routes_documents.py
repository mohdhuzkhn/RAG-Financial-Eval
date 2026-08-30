import shutil
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile

from app.ingestion.pipeline import ingest_pdf
from app.models.schemas import DocumentIngestResponse, DocumentStatusResponse
from app.services.document_registry import create_document, get_document, mark_failed, mark_ready

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = Path("./uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


def _run_ingestion(pdf_path: str, document_id: str) -> None:
    try:
        chunk_count = ingest_pdf(pdf_path, document_id)
        mark_ready(document_id, chunk_count)
    except Exception as exc:  # noqa: BLE001 — surfaced via status endpoint, not raised
        mark_failed(document_id, str(exc))
    finally:
        Path(pdf_path).unlink(missing_ok=True)


@router.post("", response_model=DocumentIngestResponse)
async def upload_document(file: UploadFile, background_tasks: BackgroundTasks):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported right now.")

    document_id = str(uuid.uuid4())
    dest_path = UPLOAD_DIR / f"{document_id}.pdf"
    with dest_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    create_document(document_id, file.filename)

    # Ingestion (parse + chunk + embed + upsert) happens off the request
    # thread — a large 10-K shouldn't block the HTTP response.
    background_tasks.add_task(_run_ingestion, str(dest_path), document_id)

    return DocumentIngestResponse(document_id=document_id, filename=file.filename, status="processing")


@router.get("/{document_id}/status", response_model=DocumentStatusResponse)
async def get_status(document_id: str):
    doc = get_document(document_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Unknown document_id")
    return DocumentStatusResponse(**doc)
