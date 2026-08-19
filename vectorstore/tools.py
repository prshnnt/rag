"""LangChain @tool wrappers over the vectorstore for use in the agent.

Exposes:
- vector_search(query, k=5, filename=None): vector + BM25 rerank, returns
  formatted chunks. Optional filename filter restricts retrieval to a single
  document.
- list_documents(): unique source documents available in the store.
- search_by_page(filename, page, k=5): retrieve chunks from one specific
  page of a document. Use when the user cites a page number or when
  cross-referencing an answer back to a source page.
- search_by_filename(filename, query, k=5): semantic + lexical search
  restricted to a single document. Use when the user asks a question
  scoped to a named document.
- get_document_outline(filename): list every page in a document with a
  short preview. Use to pick a page before calling search_by_page.

All tools return strings; the agent LLM consumes them directly.
"""

from typing import List

from langchain.tools import tool

from .db import get_client, DEFAULT_DB, DEFAULT_COLLECTION
from .pipeline import DocumentChunk
from .retriever import search as rerank_search


def _format_chunk(chunk: DocumentChunk) -> str:
    src = chunk.source
    location = f"{src.filename} (page {src.page})" if src.path else src.filename
    return (
        f"[source: {location}]\n"
        f"[chunk_id: {chunk.id}]\n"
        f"{chunk.content}"
    )


def _join_chunks(chunks: List[DocumentChunk]) -> str:
    if not chunks:
        return "No relevant documents found."
    return "\n\n---\n\n".join(_format_chunk(c) for c in chunks)


@tool
def vector_search(query: str, k: int = 5, filename: str | None = None) -> str:
    """Search the vectorstore for chunks relevant to a query.

    Runs vector retrieval on Chroma, fetches matching chunks from MongoDB,
    and reranks with BM25. Returns the top-k chunks as a single formatted
    string. Use this when the user asks questions about documents that have
    been ingested into the vectorstore.

    Args:
        query: natural-language question or keywords.
        k: number of chunks to return (1-20, default 5).
        filename: optional. When set, restrict retrieval to chunks whose
            source filename matches exactly. Use this when the user
            references a specific document by name.

    IMPORTANT: Call this tool AT MOST ONCE per user question. If it returns
    chunks, answer using those chunks. If it returns "No relevant documents
    found." tell the user the query did not match any indexed content --
    do NOT call this tool again with a rephrased query.
    """
    k = max(1, min(int(k), 20))  # clamp to a sane range
    try:
        results: List[DocumentChunk] = rerank_search(query=query, k=k, filename=filename)
    except Exception as e:
        return f"vector_search error: {type(e).__name__}: {e}"
    if not results:
        if filename:
            return f'No relevant documents found in "{filename}".'
        return "No relevant documents found."
    return _join_chunks(results)


@tool
def list_documents() -> str:
    """List the unique documents (sources) currently indexed in the vectorstore.

    Returns a bullet list of filenames. Use this when the user asks what
    documents are available before running a vector_search.
    """
    try:
        client = get_client()
        collection = client[DEFAULT_DB][DEFAULT_COLLECTION]
        filenames = sorted(
            {
                doc["source"]["filename"]
                for doc in collection.find(
                    {"source.filename": {"$exists": True}},
                    {"source.filename": 1},
                )
                if doc.get("source", {}).get("filename")
            }
        )
    except Exception as e:
        return f"list_documents error: {type(e).__name__}: {e}"
    if not filenames:
        return "No documents indexed yet."
    return "Indexed documents:\n" + "\n".join(f"- {name}" for name in filenames)


@tool
def get_document_outline(filename: str) -> str:
    """List every page of a document with a short preview of each page.

    Use this when the user references a specific document and you need to
    identify which page(s) contain the relevant material before calling
    search_by_page. Returns one line per page in the form `page N: <preview>`.

    Args:
        filename: exact source filename as returned by list_documents.
    """
    try:
        client = get_client()
        collection = client[DEFAULT_DB][DEFAULT_COLLECTION]
        # One chunk per page (smallest chunk_id per page is fine for preview).
        pipeline = [
            {"$match": {"source.filename": filename}},
            {"$sort": {"source.page": 1, "chunk_id": 1}},
            {"$group": {
                "_id": "$source.page",
                "preview": {"$first": "$content"},
            }},
            {"$sort": {"_id": 1}},
        ]
        rows = list(collection.aggregate(pipeline))
    except Exception as e:
        return f"get_document_outline error: {type(e).__name__}: {e}"
    if not rows:
        return f'No pages found for "{filename}". Check filename with list_documents.'
    lines = [f'Outline of "{filename}":']
    for row in rows:
        preview = (row.get("preview") or "").strip().replace("\n", " ")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        lines.append(f"  page {row['_id']}: {preview}")
    return "\n".join(lines)


@tool
def search_by_page(filename: str, page: int, k: int = 5) -> str:
    """Retrieve chunks from one specific page of a document.

    Use when the user cites a page number, or when you have identified a
    page via get_document_outline and need its full content.

    Args:
        filename: exact source filename as returned by list_documents.
        page: 1-based page number to retrieve.
        k: max chunks to return (1-20, default 5).
    """
    k = max(1, min(int(k), 20))
    try:
        client = get_client()
        collection = client[DEFAULT_DB][DEFAULT_COLLECTION]
        cursor = (
            collection
            .find({"source.filename": filename, "source.page": int(page)})
            .sort("chunk_id", 1)
            .limit(k)
        )
        chunks = [DocumentChunk.model_validate(d) for d in cursor]
    except Exception as e:
        return f"search_by_page error: {type(e).__name__}: {e}"
    if not chunks:
        return f'No chunks found for "{filename}" page {page}.'
    return _join_chunks(chunks)


@tool
def search_by_filename(filename: str, query: str, k: int = 5) -> str:
    """Search within a single document (filename-scoped retrieval).

    Runs the same vector + BM25 rerank as vector_search but restricts every
    candidate to the named document. Use when the user asks a question
    scoped to one specific document.

    Args:
        filename: exact source filename as returned by list_documents.
        query: natural-language question or keywords.
        k: number of chunks to return (1-20, default 5).
    """
    k = max(1, min(int(k), 20))
    try:
        results: List[DocumentChunk] = rerank_search(query=query, k=k, filename=filename)
    except Exception as e:
        return f"search_by_filename error: {type(e).__name__}: {e}"
    if not results:
        return f'No relevant chunks found in "{filename}" for that query.'
    return _join_chunks(results)
