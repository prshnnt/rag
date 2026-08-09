# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Textual TUI chat app (`main.py`) fronting a `deepagents` LangGraph agent (`agents/agent.py`) that has SQL tooling over a local Chinook sqlite DB, Redis-backed thread checkpointer, human-in-the-loop approval for SQL writes, a sandboxed filesystem backend, and a library of deepagents skills. Vectorstore/RAG scaffolding exists but is unwired placeholders.

## Commands

```bash
# Requires Python 3.14, uv, and a Redis server on localhost:6379.
uv sync
uv run main.py          # launches the TUI; needs redis + valid OLLAMA_API_KEY in .env
```

There is no test suite and no linter wired up in `pyproject.toml`. `uv.lock` is committed and pins all transitive deps (deepagents, langgraph-checkpoint-redis, textual, langchain-ollama, etc.).

## Required runtime services

- **Redis** at `redis://localhost:6379` — checkpointer state. Override via `MainAgent.create(redis_url=...)` or by editing the default in `agents/agent.py:95`.
- **Ollama Cloud** at `https://api.ollama.com` — model `gpt-oss:120b`. Auth via `OLLAMA_API_KEY` in `.env` (loaded by `python-dotenv` in `agents/agent.py:19`).
- **Tavily** — used by `tools/websearch.py`. Auth via `TAVILY_API_KEY` in `.env`.

`.env` currently has committed secrets (`OLLAMA_API_KEY`, `DAYTONA_API_KEY`, `TAVILY_API_KEY`). Rotate before sharing or pushing.

## Architecture

### Process model
- `main.py` runs the entire app on a single asyncio loop. `MainAgent` is async end-to-end because `AsyncRedisSaver` is async-only and bound to the loop that created it.
- Lifecycle: `build_agent()` → `MainAgent.create()` → `ChatApp(agent).run_async()` → `agent.aclose()` in `finally`.

### `main.py` — TUI
- `ChatApp(App)` — Header, VerticalScroll `#chat`, Input `#input-bar`, Footer. Holds one `MainAgent` and a `thread_id` (uuid4). Keybindings: `ctrl+n` new thread, `ctrl+q` quit. Slash commands: `/help`, `/bye`, `/quit`, `/new`, `/thread <id>`.
- `ApprovalScreen(ModalScreen)` — modal for gated tool calls. Buttons rendered conditionally on the tool's `allowed_decisions` (`approve`/`edit`/`reject`). `edit` opens a `TextArea`; invalid JSON notifies and keeps the modal open.
- `run_turn(message)` is `@work(exclusive=True)`. It loops: `render_stream(astream(...))` → if `approval_required`, `resolve_decisions(pending)` → `render_stream(aresume(decisions, ...))`. Disables input until the turn fully resolves.
- `render_stream` consumes `StreamChunk`s and renders `content` (Markdown), `tool_start` (system line), `error` (error line), `approval_required` (returns payload to caller).

### `agents/agent.py` — Agent
- `get_model()` → `ChatOllama(model="gpt-oss:120b", base_url="https://api.ollama.com", temperature=0)`.
- DB: `SQLDatabase.from_uri(f"sqlite:///{dir_of_agent.py}/chinook.db", sample_rows_in_table_info=3)`.
- Tools: `SQLDatabaseToolkit(db, llm).get_tools() + get_tools()`. `get_tools()` is the extension point for non-SQL tools — currently returns `[websearch, list_documents, vector_search]` (websearch via `tools/websearch.py`; `list_documents`/`vector_search` come from `vectorstore/tools.py`).
- Checkpointer: `AsyncRedisSaver.from_conn_string(redis_url)` entered as an async context manager + `await checkpointer.asetup()`.
- `create_deep_agent(...)` with:
  - `backend=FilesystemBackend(root_dir="./sandbox/", virtual_mode=True)`
  - `skills=["./skills/examples", "./skills/public"]` (resolved relative to process CWD, not the backend root — these point into `sandbox/skills/`)
  - `subagents=[]`
  - `memory=["./AGENTS.md"]` (currently empty placeholder file)
  - `checkpointer=checkpointer`
  - `interrupt_on=DEFAULT_INTERRUPT_ON`

### Stream protocol
`StreamType = Literal["start", "message_start", "content", "message_end", "tool_start", "tool_end", "approval_required", "end", "error"]`

Per turn: `start` → `(message_start → content* → message_end | tool_start → tool_end)*` → `end` OR `approval_required`. `error` can fire anywhere instead of the terminal. `approval_required` pauses the graph (state checkpointed); caller resumes via `aresume(decisions)`.

`StreamChunk` (pydantic): `thread_id`, `type`, `content?`, `tool?`, `skill?`, `data?`. For `approval_required`, `data` carries `{"action_requests": [...], "review_configs": [...]}`.

### Human-in-the-loop
`DEFAULT_INTERRUPT_ON` gates `sql_db_query` (`["approve", "edit", "reject"]`) and `websearch` (`["approve", "reject"]`). The other SQL toolkit tools (`sql_db_list_tables`, `sql_db_schema`, `sql_db_query_checker`) are read-only and run unguarded. `MainAgent._pending_interrupts(config)` reads `state.interrupts`, with a fallback that flattens per-task interrupts for older langgraph.

Decision shapes:
- `{"type": "approve"}`
- `{"type": "reject"}`
- `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}`

### Skills (`sandbox/skills/`)
Loaded by the agent from `./skills/examples` and `./skills/public` (relative to CWD). Skills can be either a directory with a `SKILL.md` (frontmatter `name`, `description`) or a single `.skill` file. The custom ones are `public/query-writing` and `public/schema-exploration`. The rest of `examples/` and `public/` are upstream skill packs (docx, pdf, xlsx, pptx, frontend-design, etc.) — most aren't useful against the Chinook DB.

### `schemas/document_schema.py`
Frozen Pydantic `DocumentMetadata` with UUIDs (`user_id`, `document_id`, `chunk_id`), `document_name`, `document_type ∈ {pdf, docx, txt, xlsx, pptx}`, `chunk_index: int >= 0`. For chunk provenance during RAG ingestion. No I/O wired up yet.

### Scaffolds (not implemented)
- `vectorstore/{injestion,pipline,reranker,retreiver,tools}.py` — all 0-byte placeholders. Note filename typos: `injestion.py`, `pipline.py`, `retreiver.py`. ChromaDB is in deps; `docs/pdfs/coi.pdf` is the input corpus.
- `agents/subagents/sql_agent.py` — empty placeholder.
- `AGENTS.md` — empty (deepagents `memory` param no-ops on it).
- `get_tools()` extension point in `agents/agent.py` for adding non-SQL tools beyond `websearch`.

### Tools
- `tools/websearch.py` — `tavily.search()` wrapped as a LangChain `@tool`. Loads `TAVILY_API_KEY` from `.env`.

### Sandbox
- `sandbox/` is the `FilesystemBackend` root with `virtual_mode=True`. Contains `conversation_history/`, `mnt/`, `node_modules/`, `package.json` (a `package-lock.json`-equipped node setup for sandbox scripts), and the `skills/` tree.
- `sandbox/package.json` exists for any JS-side sandbox tooling — check it before adding node deps.

## Common gotchas

- Skills/memory paths in `create_deep_agent(...)` are CWD-relative. Run from the repo root (`uv run main.py` from `D:\Workspace\rag`), not from a subdir.
- `MainAgent._pending_interrupts` checks for `state.interrupts` (newer langgraph) and falls back to per-task `interrupts` (older). Don't rely on only one.
- `async for event in self.agent.astream_events(...)` finishing normally does NOT mean the turn ended — a gated tool pauses the graph rather than erroring. Always check `_pending_interrupts` after the loop.
- `AsyncRedisSaver` must be entered as an async context manager and `asetup()`-ed before first use. Released via `aclose()` (called from `main.py` in a `finally`).
- `interrupt_on` does NOT need a per-tool config for read-only tools — `deepagents` treats absent keys as "no interrupt".
- `.env` is in `.gitignore` but the current tracked `.env` contains real keys. Don't assume `git status` flags it; the file is committed and needs rotation.
- `AGENTS.md` is empty, which means `memory=["./AGENTS.md"]` currently injects nothing into the agent's system prompt.

## Committing

Per project memory: do not include `Co-Authored-By: Claude` in commit messages.