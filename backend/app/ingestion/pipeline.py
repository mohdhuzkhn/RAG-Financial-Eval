import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.services.embeddings import get_embeddings
from app.services.parent_store import save_parent_chunk
from app.services.vector_store import upsert_child_chunks

PARENT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
CHILD_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)


def ingest_pdf(pdf_path: str, document_id: str) -> int:
    """
    Same parent-child idea as the original ingestion.py, restructured so
    parents and children land in two different, independently-scalable
    services instead of one process's memory:
      - children -> embedded -> Qdrant
      - parents  -> sqlite (parent_store), keyed by parent_id

    Returns the number of child chunks indexed.
    """
    pages = PyPDFLoader(pdf_path).load()

    child_texts: list[str] = []
    child_parent_ids: list[str] = []
    child_pages: list[int | str] = []

    for page_doc in pages:
        page_number = page_doc.metadata.get("page", "N/A")

        for parent_chunk in PARENT_SPLITTER.split_text(page_doc.page_content):
            parent_id = str(uuid.uuid4())
            save_parent_chunk(parent_id, document_id, page_number, parent_chunk)

            for child_chunk in CHILD_SPLITTER.split_text(parent_chunk):
                child_texts.append(child_chunk)
                child_parent_ids.append(parent_id)
                child_pages.append(page_number)

    if not child_texts:
        return 0

    embeddings = get_embeddings()
    child_vectors = embeddings.embed_documents(child_texts)

    upsert_child_chunks(
        document_id=document_id,
        child_texts=child_texts,
        child_vectors=child_vectors,
        parent_ids=child_parent_ids,
        pages=child_pages,
    )

    return len(child_texts)
