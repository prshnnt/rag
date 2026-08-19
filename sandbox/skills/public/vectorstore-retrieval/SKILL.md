---
name: vectorstore-retrieval
description: Retrieves and grounds answers in indexed documents using the vectorstore tools. Use when the user asks questions about content that lives in ingested PDFs, references a specific document by name, cites a page number, asks what documents are available, or wants to verify where a fact came from in the corpus.
---

# Vectorstore Retrieval Skill

The vectorstore holds chunked text from ingested PDFs (Chroma for vectors,
MongoDB for full chunk content, BM25 for lexical recall). Five tools are
available:

| Tool | Purpose |
|---|---|
| `list_documents` | List every indexed document filename. |
| `get_document_outline` | Per-page previews for one document. |
| `search_by_filename` | Semantic + lexical search scoped to one document. |
| `search_by_page` | Exact-page retrieval from one document. |
| `vector_search` | Whole-corpus semantic + lexical search. |

## Decision Tree

Pick the tool before calling anything else. Walk this in order:

1. **Do you know which document?**
   - No → call `list_documents` once. Use the returned filename(s) verbatim
     in subsequent calls. Do NOT guess filenames.
2. **Do you know which page?**
   - Yes → call `search_by_page(filename, page)` directly.
   - No, but the user references a named document → call
     `get_document_outline(filename)` to find the relevant page(s), then
     call `search_by_page(filename, page)` for each.
3. **The question is about the named document but not a specific page?**
   - Call `search_by_filename(filename, query)`.
4. **The question spans the whole corpus?**
   - Call `vector_search(query)`. Optionally pass `filename` if you
     narrowed it down.

## Rules

- **One retrieval call per user question.** If the first call returns
  chunks, answer from those chunks. If it returns "No relevant documents
  found." or "No chunks found", tell the user and stop — do not retry
  with a rephrased query.
- **Always copy filenames verbatim** from `list_documents` /
  `get_document_outline`. A single character mismatch returns nothing.
- **Ground every claim** in a chunk's content. Quote the chunk id
  (`[chunk_id: ...]`) when citing. If the chunks don't support the claim,
  say so — don't fill in from prior knowledge.
- **Page numbers are 1-based** and come from the `[source: ... (page N)]`
  header on each chunk.
- **Tool args:**
  - `query` is natural language; keywords also work.
  - `k` is 1-20; default 5. Raise k only when you need broader context.
  - `filename` must match exactly — no extensions added or stripped.

## Common Patterns

### "What documents do you have?"
```
list_documents()  →  bullet list of filenames
```

### "Summarize <doc>.pdf"
```
list_documents()                         # confirm filename
get_document_outline("<doc>.pdf")        # find page ranges
search_by_filename("<doc>.pdf", k=10)    # pull representative chunks
# then synthesize
```

### "What's on page 42 of <doc>.pdf?"
```
search_by_page("<doc>.pdf", page=42)
```

### "According to <doc>.pdf, what does Article 21 say?"
```
search_by_filename("<doc>.pdf", "Article 21", k=5)
# if empty, fall back to:
get_document_outline("<doc>.pdf")        # locate the article's page
search_by_page("<doc>.pdf", page=<n>)    # read the article body
```

### "What does the corpus say about X?"
```
vector_search("X", k=5)
```

## Anti-Patterns

- Calling `vector_search` three times with rephrased queries — the tool
  contract is "at most once per question".
- Hallucinating a filename. Always verify via `list_documents` or
  `get_document_outline`.
- Quoting chunks without their `chunk_id`. Citations must be traceable.
- Falling back to websearch when the answer exists in the corpus —
  check the vectorstore first.
