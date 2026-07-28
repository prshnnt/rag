"""LangChain tools exposing vector search to the deepagent.

`vector_search` returns formatted context; `list_documents` returns per-user
ingested docs from the metadata stored in Chroma.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

from langchain_core.tools import tool

from .retriever import DEFAULT_COLLECTION, retrieve

_CHROMA_PATH = os.getenv("CHROMA_PATH", "./vectorstore/chroma")
_TOP_K = int(os.getenv("VECTOR_TOP_K", "5"))


def _format(chunks: list[dict]) -> str:
    if not chunks:
        return "No matching chunks found."
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        meta = c.get("metadata") or {}
        name = meta.get("document_name", "?")
        page = meta.get("page")
        idx = meta.get("chunk_index", "?")
        loc = f"{name}#{idx}" if page is None else f"{name} p{page} #{idx}"
        text = (c.get("text") or "").strip()
        lines.append(f"[{i}] ({loc})\n{text}")
    return "\n\n".join(lines)


@tool
def vector_search(
    query: str,
    user_id: str,
    top_k: int | None = None,
    document_id: str | None = None,
) -> str:
    """Search the user's ingested documents. Returns ranked chunks with citations.

    Args:
        query: Natural language question.
        user_id: UUID of the requesting user (required for tenant isolation).
        top_k: Number of chunks to return (default 5).
        document_id: Optional UUID to scope to a single document.
    """
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        return f"Invalid user_id: {user_id!r}"
    did = None
    if document_id:
        try:
            did = uuid.UUID(document_id)
        except ValueError:
            return f"Invalid document_id: {document_id!r}"

    chunks = retrieve(
        query=query,
        top_k=top_k or _TOP_K,
        user_id=uid,
        document_id=did,
        collection_name=DEFAULT_COLLECTION,
        chroma_path=_CHROMA_PATH,
    )
    return _format(chunks)


@tool
def list_documents(user_id: str) -> str:
    """List documents previously ingested for a user.

    Args:
        user_id: UUID of the requesting user.
    """
    try:
        import chromadb
        uid = uuid.UUID(user_id)
    except ValueError:
        return f"Invalid user_id: {user_id!r}"

    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    coll = client.get_or_create_collection(DEFAULT_COLLECTION)
    res = coll.get(where={"user_id": str(uid)}, include=["metadatas"])

    seen: dict[str, dict] = {}
    for m in res.get("metadatas") or []:
        did = m.get("document_id")
        if did and did not in seen:
            seen[did] = {
                "document_id": did,
                "document_name": m.get("document_name"),
                "document_type": m.get("document_type"),
            }
    if not seen:
        return "No documents ingested for this user."
    lines = [
        f"- {d['document_name']} ({d['document_type']}) id={d['document_id']}"
        for d in seen.values()
    ]
    return "Documents:\n" + "\n".join(lines)


def get_tools() -> list:
    """Extension-point list for agents/agent.py."""
    return [vector_search, list_documents]
