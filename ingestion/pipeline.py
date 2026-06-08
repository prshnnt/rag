"""Ingestion pipeline.

Builds ingestion-ready records from raw LangChain Documents:

    record = {
        "document":  Document,
        "metadata":  dict,    # flattened for Chroma
        "id":        str,     # deterministic from source + page + chunk
    }

Records are handed to `vectordatabase.vectorstore.VectorStore.add_documents`.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Iterable, List, Optional, Sequence

from langchain_core.documents import Document


def _stable_id(*parts: str) -> str:
    """Deterministic id from arbitrary string parts."""
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h


def build_records(
    documents: Sequence[Document],
    source_tag: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Convert Documents into ingestion records.

    `Document.metadata` should already contain `source` and (ideally) `page`.
    `source_tag` overrides `source` for all records in this batch.
    """
    records: List[Dict[str, Any]] = []
    for doc in documents:
        meta = dict(doc.metadata or {})
        src = source_tag or meta.get("source", "unknown")
        page = meta.get("page")
        rid = _stable_id(str(src), str(page) if page is not None else "", doc.page_content[:64])
        meta.setdefault("source", src)
        records.append(
            {
                "document": Document(page_content=doc.page_content, metadata=meta),
                "metadata": meta,
                "id": rid,
            }
        )
    return records


def ingest(
    documents: Iterable[Document],
    store: Any,
    source_tag: Optional[str] = None,
    batch_size: int = 256,
) -> int:
    """Ingest Documents into a VectorStore. Returns count written."""
    records = build_records(list(documents), source_tag=source_tag)
    total = 0
    for i in range(0, len(records), batch_size):
        total += store.add_documents(records[i : i + batch_size])
    return total
