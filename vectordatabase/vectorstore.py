"""Vector store module.

Wraps a persistent ChromaDB collection with:
  * `add_documents(records)` — upsert LangChain Document records.
  * `query(text, k, ...)`     — similarity search.

Records accepted by `add_documents` follow:
    {"document": Document, "metadata": dict, "id": str}

Embedding function defaults to Chroma's built-in (no model needed for
local dev). Swap `embedding_fn` in `VectorStore(...)` for production.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document


# Default persistent directory; override via VectorStore(persist_directory=...).
DEFAULT_PERSIST_DIR = "./rag"
DEFAULT_COLLECTION = "documents"


@dataclass
class QueryHit:
    """A single retrieval result."""

    document: Document
    score: Optional[float]
    id: str


class VectorStore:
    """Thin wrapper over a ChromaDB persistent collection."""

    def __init__(
        self,
        persist_directory: Union[str, Path] = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_fn: Optional[Any] = None,
    ) -> None:
        self.persist_directory = str(persist_directory)
        self.collection_name = collection_name

        client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )
        self._client = client
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_fn,
        )

    # ---------- writes ----------

    def add_documents(self, records: Sequence[Dict[str, Any]]) -> int:
        """Upsert records into the collection.

        Each record must be `{"document": Document, "metadata": dict, "id": str}`.

        Returns the number of records accepted.
        """
        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for rec in records:
            doc = rec.get("document")
            meta = rec.get("metadata") or {}
            rid = rec.get("id")
            if not isinstance(doc, Document):
                raise TypeError(f"record[{rid}] 'document' must be a Document")
            if not rid:
                raise ValueError(f"record missing 'id': {rec}")
            ids.append(str(rid))
            texts.append(doc.page_content)
            # Chroma metadata values must be scalar; coerce safely.
            clean_meta = {
                k: ("" if v is None else str(v) if not isinstance(v, (str, int, float, bool)) else v)
                for k, v in meta.items()
            }
            metadatas.append(clean_meta)

        if not ids:
            return 0
        self._collection.upsert(ids=ids, documents=texts, metadatas=metadatas)
        return len(ids)

    # ---------- reads ----------

    def query(
        self,
        text: str,
        k: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[QueryHit]:
        """Return top-k similar documents to `text`."""
        res = self._collection.query(
            query_texts=[text],
            n_results=k,
            where=where,
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        hits: List[QueryHit] = []
        for i, doc_text in enumerate(docs):
            hit_id = ids[i] if i < len(ids) else ""
            hit_meta = metas[i] if i < len(metas) else {}
            hit_score = dists[i] if i < len(dists) else None
            hits.append(
                QueryHit(
                    document=Document(page_content=doc_text, metadata=hit_meta or {}),
                    score=hit_score,
                    id=hit_id,
                )
            )
        return hits

    def count(self) -> int:
        return self._collection.count()

    def delete_collection(self) -> None:
        self._client.delete_collection(self.collection_name)
