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
- `docLoader/` — document loaders exposed as LangChain tools
  - `pdfloader.py` — `pypdf`-based PDF page extractor. Exposes three functions (`load_pdf_as_pages`, `load_pdf_as_text`, `load_pdf_as_documents`) plus a `@tool`-decorated `load_pdf_file_tool` for agent invocation. Supports page-range and specific-page filtering (1-indexed).
  - `textloader.py` — plain-text loader. `load_text_file`, `load_text_as_documents`, and `load_text_file_tool` (LangChain tool).
- `ingestion/` — empty; intended for chunking/embedding/upsert pipeline
- `docs/pdfs/` — source PDF corpus (currently `coi.pdf`)
- `rag/` — ChromaDB persistent store
  - `chroma.sqlite3` — vector index metadata
  - `<uuid>/` — segment data files (HNSW index, length, header, link lists, data)
- `docLoader/parsed_chunks.json` — pre-parsed chunk output (intermediate cache)

## Data flow (intended)

1. `docs/pdfs/*.pdf` → `docLoader/pdfloader.py` → pages
2. Pages → `ingestion/` (chunk + embed) → ChromaDB (`rag/`)
3. Agent (`main.py`) uses `deepagents` + Ollama LLM, with `load_pdf_file_tool` / `load_text_file_tool` as retrievers

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

- Loaders must be exposed as LangChain `@tool`s if callable by the agent
- Page numbers are 1-indexed in user-facing APIs
- ChromaDB persistence path is `./rag/`
