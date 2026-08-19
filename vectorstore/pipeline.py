from .loaders import Convert, document_to_lists
from .db import get_client, DEFAULT_DB, DEFAULT_COLLECTION
from .vector_store import VectorStore

import os
from dotenv import load_dotenv

from datetime import datetime, timezone
from pymongo.errors import BulkWriteError
from pydantic import BaseModel, ConfigDict, Field
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# nomic-embed-text hard input cap = 2048 tokens (~6000-8000 chars
# depending on content density; dense tables/OCR push lower).
# Stay conservative: 1500 chars ≈ 400 tokens, safe margin.
_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=100,
)

# Hard ceiling for any single embed input. Markdown tables / OCR blobs
# sometimes exceed chunk_size because Recursive splitter can't find
# a separator inside them. Force re-split oversized pieces.
_MAX_CHARS = 1500


def _force_split(text: str) -> list[str]:
    """Re-split any chunk still above _MAX_CHARS, recursively."""
    if len(text) <= _MAX_CHARS:
        return [text]
    mid = len(text) // 2
    return _force_split(text[:mid]) + _force_split(text[mid:])

class Source(BaseModel):
    filename: str
    path: str
    page: int


class DocumentChunk(BaseModel):
    id: str = Field(alias="_id")
    document_id: str
    content: str
    source: Source
    chunk_id: int
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    model_config = ConfigDict(
        populate_by_name=True
    )


def ingest_pdfs(docs_path: str, replace: bool = True):

    print("Loading VectorStore....")
    vector_store = VectorStore()
    print("Loaded VectorStore.")

    collection_name = os.environ.get(
        "CHROMA_COLLECTION",
        DEFAULT_COLLECTION
    )

    print("Loading Documents....")
    docs = Convert.langchain_load_pdf(
        path=docs_path,
        format="markdown"
    )
    print("Loaded Documents.")

    # Split each page-level Document into sub-chunks small enough for
    # the embedding model. Pages from PDFs frequently exceed Ollama's
    # embed context window, causing "input length exceeds context length".
    sub_docs: list = []
    for doc in docs:
        pieces = _SPLITTER.split_text(doc.page_content)
        flat: list[str] = []
        for piece in pieces:
            flat.extend(_force_split(piece))
        for idx, piece in enumerate(flat):
            sub_meta = dict(doc.metadata)
            sub_meta["chunk_id"] = idx
            sub_docs.append(
                Document(
                    page_content=piece,
                    metadata=sub_meta,
                )
            )

    # data
    content, metadatas = document_to_lists(sub_docs)

    ids = []
    # MongoDB documents for injestion
    mongo_documents = []
    source_paths = set()

    for doc in sub_docs:

        source = str(doc.metadata["source"])

        page = doc.metadata.get("page", 1)

        chunk_id = doc.metadata.get("chunk_id", 0)

        # ID format is load-bearing for dedup + external references.
        # Do not change without coordinating a full reindex.
        doc_id = (
            f"{source}"
            f"_page{page}"
            f"_chunk_{chunk_id}"
        )

        ids.append(doc_id)
        source_paths.add(source)

        document_chunk = DocumentChunk(
            id=doc_id,
            document_id=source,
            content=doc.page_content,
            source=Source(
                filename=source,
                path=os.path.join(docs_path, source),
                page=page,
            ),
            chunk_id=chunk_id,
        )

        # Convert Pydantic model -> MongoDB document
        mongo_documents.append(
            document_chunk.model_dump(
                by_alias=True
            )
        )

    print("Saving Documents to MongoDB....")
    client = get_client()
    db = client[DEFAULT_DB]
    collection = db[DEFAULT_COLLECTION]

    if replace and source_paths:
        deleted = collection.delete_many(
            {"source.filename": {"$in": list(source_paths)}}
        )
        print(f"Deleted {deleted.deleted_count} existing chunks for re-ingest.")

    if mongo_documents:
        try:
            collection.insert_many(
                mongo_documents,
                ordered=False
            )
        except BulkWriteError as e:
            # Only reachable when replace=False and a duplicate _id appears.
            skipped = len(e.details.get("writeErrors", []))
            print(f"insert_many: {skipped} duplicates skipped.")

    print("Saved Documents to MongoDB.")

    print("Ingesting Documents into VectorStore....")
    vector_store.ingest(
        collection_name=collection_name,
        ids=ids,
        content=content,
        metadatas=metadatas
    )
    print("Ingested Documents.")


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "./docs/pdfs/"
    ingest_pdfs(path)