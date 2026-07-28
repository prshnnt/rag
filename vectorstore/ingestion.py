"""Ingestion: split loaded docs into chunks and upsert into ChromaDB.

Each chunk gets a DocumentMetadata (frozen Pydantic) with fresh UUIDs.
Embeddings via langchain_ollama (Ollama /api/embeddings).
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import chromadb
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from schemas.document_schema import DocumentMetadata

from .loader import load as load_doc

DEFAULT_COLLECTION = "rag"
CONFIG_PATH = Path(__file__).parent / "config.json"


def _max_chunk_length() -> int:
    import json
    try:
        return int(json.loads(CONFIG_PATH.read_text()).get("MAX_CHUNK_LENGTH", 500))
    except Exception:
        return 500


def _splitter() -> RecursiveCharacterTextSplitter:
    size = _max_chunk_length()
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=max(20, size // 10),
        length_function=len,
    )


def _embeddings() -> OllamaEmbeddings:
    base = os.getenv("OLLAMA_BASE_URL", "https://api.ollama.com")
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=model, base_url=base)


def _chroma(path: str | Path | None = None) -> chromadb.PersistentClient:
    db_path = path or os.getenv("CHROMA_PATH", "./vectorstore/chroma")
    return chromadb.PersistentClient(path=str(db_path))


def _doc_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower().lstrip(".")
    return ext if ext in {"pdf", "docx", "txt", "xlsx", "pptx"} else "txt"


def ingest_file(
    path: str | Path,
    user_id: uuid.UUID,
    document_id: uuid.UUID | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    chroma_path: str | Path | None = None,
) -> list[DocumentMetadata]:
    """Load + split + embed + upsert one file. Returns metadata for every chunk."""
    path = Path(path)
    document_id = document_id or uuid.uuid4()
    doc_type = _doc_type(path)

    pages = load_doc(path)
    splitter = _splitter()

    texts: list[str] = []
    metas: list[dict] = []
    chunk_metas: list[DocumentMetadata] = []
    idx = 0
    for page in pages:
        for chunk in splitter.split_text(page["text"]):
            if not chunk.strip():
                continue
            chunk_id = uuid.uuid4()
            meta = DocumentMetadata(
                user_id=user_id,
                document_id=document_id,
                document_name=path.name,
                document_type=doc_type,  # type: ignore[arg-type]
                chunk_id=chunk_id,
                chunk_index=idx,
            )
            chunk_metas.append(meta)
            texts.append(chunk)
            metas.append({
                "user_id": str(meta.user_id),
                "document_id": str(meta.document_id),
                "document_name": meta.document_name,
                "document_type": meta.document_type,
                "chunk_id": str(meta.chunk_id),
                "chunk_index": meta.chunk_index,
                **({"page": page["page"]} if "page" in page else {}),
            })
            idx += 1

    if not texts:
        return []

    client = _chroma(chroma_path)
    coll = client.get_or_create_collection(
        collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    # Embed in one shot; small corpora for now.
    vectors = _embeddings().embed_documents(texts)
    coll.upsert(
        ids=[m["chunk_id"] for m in metas],
        documents=texts,
        embeddings=vectors,
        metadatas=metas,
    )
    return chunk_metas
