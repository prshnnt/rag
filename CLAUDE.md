# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

RAG (retrieval-augmented generation) agent scaffold. Python 3.14, managed with `uv`. Uses LangChain, ChromaDB, Ollama, and `deepagents` (LangChain DeepAgents framework).

## Commands

Project uses `uv` (not pip). Virtual env at `.venv/`.

```bash
# Install / sync deps
uv sync

# Run entry point (currently empty)
uv run python main.py

# Lint / format
uv run ruff check .
uv run ruff format .

# No test suite exists yet — add tests under a tests/ dir when introduced.
```

## Architecture

Directory layout:

- `main.py` — entry point (currently empty, intended to launch the agent)
- `docLoader/`
  - `loader.py` — single PDF loader. `load_pages(path, range_, fmt="text"|"Document")`
    returns one item per page in ascending order. `range_` is a list of
    inclusive `(start, end)` 1-indexed tuples (e.g. `[(1, 3), (7, 9)]`);
    `None` loads all pages. `PageFormat` enum selects the output shape.
    Also exposes `load_pdf_file_tool` — LangChain `@tool` wrapper for agent use.
- `ingestion/`
  - `pipeline.py` — `build_records(documents, source_tag)` produces
    `{"document": Document, "metadata": dict, "id": str}` records with
    deterministic SHA1 ids. `ingest(documents, store, ...)` batch-upserts
    into a `VectorStore`.
- `vectordatabase/`
  - `vectorstore.py` — `VectorStore(persist_directory, collection_name, embedding_fn)`
    wraps a persistent ChromaDB collection. Methods: `add_documents(records)`,
    `query(text, k, where)`, `count()`, `delete_collection()`. Query returns
    `QueryHit(document, score, id)`. Default embedding is Chroma's built-in;
    pass `embedding_fn=...` (e.g. `OllamaEmbeddings`) for production models.
- `docs/pdfs/` — source PDF corpus (currently `coi.pdf`)
- `rag/` — ChromaDB persistent store
  - `chroma.sqlite3` — vector index metadata
  - `<uuid>/` — segment data files (HNSW index, length, header, link lists, data)

## Data flow

1. `docs/pdfs/*.pdf` → `docLoader/loader.py::load_pages(...)` → list of `Document` (one per page)
2. `Document` list → `ingestion/pipeline.py::ingest(...)` → `{"document", "metadata", "id"}` records
3. Records → `vectordatabase/vectorstore.py::VectorStore.add_documents(...)` → ChromaDB (`rag/`)
4. Agent (`main.py`) uses `deepagents` + Ollama LLM, with `load_pdf_file_tool` and `VectorStore.query(...)` as retriever

## Key dependencies

- `chromadb` — vector store
- `langchain-community`, `langchain-core`, `langchain-ollama` — LangChain stack
- `deepagents` — agent framework
- `pypdf` — PDF parsing
- `pydantic` — schema/validation

## Environment

- `.env` holds `OLLAMA_API_KEY` (loaded via `python-dotenv`)
- Local Ollama server expected for inference

## Conventions

- Loaders and retrievers must be exposed as LangChain `@tool`s if callable by the agent
- Page numbers are 1-indexed in user-facing APIs
- Ingestion records use `{"document": Document, "metadata": dict, "id": str}` shape
- ChromaDB persistence path is `./rag/`; default collection name `documents`
