"""Vector retrieval: embed query, hit ChromaDB, optional BM25 rerank, optional user filter."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from langchain_ollama import OllamaEmbeddings

from .reranker import rerank

DEFAULT_COLLECTION = "rag"


def _embeddings() -> OllamaEmbeddings:
    base = os.getenv("OLLAMA_BASE_URL", "https://api.ollama.com")
    model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    return OllamaEmbeddings(model=model, base_url=base)


def _client(path: str | Path | None = None):
    import chromadb
    db_path = path or os.getenv("CHROMA_PATH", "./vectorstore/chroma")
    return chromadb.PersistentClient(path=str(db_path))


def retrieve(
    query: str,
    top_k: int = 5,
    user_id: uuid.UUID | str | None = None,
    document_id: uuid.UUID | str | None = None,
    collection_name: str = DEFAULT_COLLECTION,
    chroma_path: str | Path | None = None,
    rerank_top_k: int | None = None,
    use_bm25: bool = True,
) -> list[dict]:
    """Return list of {text, metadata, score} dicts, ordered by relevance."""
    client = _client(chroma_path)
    coll = client.get_or_create_collection(collection_name)

    where: dict | None = None
    conds: list[dict] = []
    if user_id is not None:
        conds.append({"user_id": str(user_id)})
    if document_id is not None:
        conds.append({"document_id": str(document_id)})
    if conds:
        where = {"$and": conds} if len(conds) > 1 else conds[0]

    # Pull a wider pool then rerank.
    pool = max(top_k * 4, top_k)
    vec = _embeddings().embed_query(query)
    res = coll.query(
        query_embeddings=[vec],
        n_results=pool,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    chunks = [
        {"text": d, "metadata": m, "score": 1.0 - dist}
        for d, m, dist in zip(docs, metas, dists, strict=True)
    ]

    if use_bm25 and chunks:
        k = rerank_top_k if rerank_top_k is not None else top_k
        ranked = rerank(query, chunks, top_k=k, text_key="text")
        # Preserve distance-based score from original chunk.
        return ranked[:top_k]
    return chunks[:top_k]
