"""Vectorstore research subagent spec.

Owned tools: the five vectorstore tools (list_documents, vector_search,
search_by_page, search_by_filename, get_document_outline). Reached via
`task(subagent_type="vector-researcher", ...)`. The main agent delegates
"answer from my documents" requests here so retrieval logic stays in
one place and the skill load is reduced.
"""

from deepagents.middleware.subagents import SubAgent
from vectorstore.tools import (
    get_document_outline,
    list_documents,
    search_by_filename,
    search_by_page,
    vector_search,
)


VECTOR_RESEARCHER_PROMPT = """\
You are the vectorstore-research subagent. Your only job is to answer
questions using the indexed PDF corpus.

Decision tree:
1. If you do not know which document, call `list_documents` once.
2. If the question targets a specific page, call
   `search_by_page(filename, page)`.
3. If the question targets a named document but no specific page, call
   `get_document_outline(filename)` to locate the relevant page(s), then
   `search_by_page` for each, or `search_by_filename(filename, query)`
   when a semantic/lexical search fits better.
4. Otherwise call `vector_search(query)`.

Rules:
- Use filenames VERBATIM from `list_documents` / `get_document_outline`.
  One character mismatch returns nothing.
- One retrieval call per question. If empty, report it — do not retry.
- Ground every claim in a retrieved chunk. Cite the `[chunk_id: ...]`
  from the chunk header for each citation.
- Do not use any tools other than the five vectorstore tools.
"""


vector_researcher: SubAgent = {
    "name": "vector-researcher",
    "description": (
        "Answers questions using the indexed PDF corpus (Chroma vectors + "
        "MongoDB chunks + BM25 rerank). Delegate here when the user asks "
        "about content that lives in ingested documents, references a "
        "specific document by name, cites a page number, or asks what "
        "documents are available."
    ),
    "system_prompt": VECTOR_RESEARCHER_PROMPT,
    "tools": [
        list_documents,
        vector_search,
        search_by_page,
        search_by_filename,
        get_document_outline,
    ],
}
