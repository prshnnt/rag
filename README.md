# rag

Textual TUI chat app fronting a `deepagents` LangGraph agent. Agent has SQL tooling over a local Chinook sqlite DB, Redis-backed thread checkpointer, human-in-the-loop approval for SQL writes, sandboxed filesystem backend, and skills for schema exploration + query writing. Vectorstore/RAG scaffolding exists but files are empty placeholders.

## Stack

- Python `>=3.14`, `uv` for env/lock
- `deepagents >=0.6.8`, `langgraph-checkpoint-redis >=0.5.1`
- `langchain-ollama` (model: `gpt-oss:120b` via `https://api.ollama.com`)
- `langchain-community` `SQLDatabaseToolkit` over sqlite
- `textual >=8.2.8` (TUI)
- `chromadb`, `pypdf`, `fastapi`, `uvicorn`, `jinja2`, `pydantic v2` (listed but mostly unused yet)
- Redis required at `redis://localhost:6379` for checkpoints

## Layout

```
.
├── main.py                    # Textual TUI: ChatApp, ApprovalScreen modal, run_turn loop
├── pyproject.toml             # deps + python 3.14
├── uv.lock                    # locked deps
├── .env                       # OLLAMA_API_KEY (currently committed; rotate)
├── .gitignore                 # pycache, .venv, .env, *.db, chroma runtime, parsed chunks
├── .python-version            # 3.14
├── AGENTS.md                  # empty (deepagents memory hook)
├── agents/
│   ├── agent.py               # MainAgent: build_agent(), MainAgent.create/aclose, astream/aresume
│   ├── subagents/
│   │   └── sql_agent.py       # empty placeholder
│   └── chinook.db             # sqlite (~20KB) — local DB the agent queries
├── sandbox/
│   │                          # FilesystemBackend root_dir=./sandbox/ (virtual_mode=True)
│   └── skills/
│       ├── query-writing/SKILL.md       # skill: SELECT/JOIN/agg workflow, no DML
│       └── schema-exploration/SKILL.md  # skill: list tables, describe columns, FK mapping
├── schemas/
│   └── document_schema.py     # Pydantic DocumentMetadata (frozen)
├── docs/
│   └── pdfs/
│       └── coi.pdf            # 2.3MB reference PDF (not yet ingested)
├── vectorstore/               # all files empty placeholders
│   ├── injestion.py           # (sic — typo)
│   ├── pipline.py             # (sic — typo)
│   ├── reranker.py
│   ├── retreiver.py           # (sic — typo)
│   └── tools.py
```

## Entry point

`main.py` — run with `uv run main.py` (or `python main.py` inside the venv).

Flow:
1. `build_agent()` → `MainAgent.create()` (async).
2. `ChatApp(agent)` mounts Header, VerticalScroll `#chat`, Input `#input-bar`, Footer.
3. User submits message → `run_turn(message)` worker (exclusive) calls `agent.astream(message, thread_id=...)`.
4. Stream chunks render to chat; on `approval_required` chunk, main calls `aresume(decisions)` after modal decisions.

### Commands (in TUI)

- `/help`
- `/bye`, `/quit`
- `/new` — new thread (Ctrl+N)
- `/thread <id>` — switch thread
- Ctrl+Q — quit

### Bindings
- `ctrl+n` → `action_new_thread`
- `ctrl+q` → `quit`

## Stream protocol

`StreamType = Literal["start", "message_start", "content", "message_end", "tool_start", "tool_end", "approval_required", "end", "error"]`

Chunk lifecycle per turn:
```
start
  → (message_start → content* → message_end | tool_start → tool_end)*
  → end | approval_required
```
`error` can fire anywhere instead of terminal. `approval_required` pauses the graph; resume via `aresume()` with a `decisions` list.

## Agent (`agents/agent.py`)

`MainAgent` is async end-to-end because `AsyncRedisSaver` is async-only. Must run inside a single asyncio loop for the process lifetime.

### Construction
- `get_model()` → `ChatOllama(model="gpt-oss:120b", base_url="https://api.ollama.com", temperature=0)`
- DB: `sqlite:///{dir_of_agent.py}/chinook.db` via `SQLDatabase.from_uri(..., sample_rows_in_table_info=3)`
- Tools: `SQLDatabaseToolkit(db, llm).get_tools() + get_tools()` (currently `get_tools()` returns `[]`)
- Checkpointer: `AsyncRedisSaver.from_conn_string(redis_url)` entered + `asetup()`
- Agent built via `create_deep_agent(...)` with:
  - `backend=FilesystemBackend(root_dir="./sandbox/", virtual_mode=True)`
  - `skills=["./skills/"]`
  - `subagents=[]`
  - `memory=["./AGENTS.md"]`
  - `checkpointer=checkpointer`
  - `interrupt_on=DEFAULT_INTERRUPT_ON`

### HITL (Human-in-the-loop)
```python
DEFAULT_INTERRUPT_ON = {
    "sql_db_query": {"allowed_decisions": ["approve", "edit", "reject"]},
    # other SQL toolkit tools (list/schema/check) are read-only, not gated
}
```
Decision shapes per `action_request`:
- `{"type": "approve"}`
- `{"type": "reject"}`
- `{"type": "edit", "edited_action": {"name": ..., "args": {...}}}`

### Methods
- `await MainAgent.create(redis_url, interrupt_on=None) -> MainAgent`
- `await agent.aclose()` — releases Redis context.
- `agent.astream(message, thread_id="default")` → `AsyncGenerator[StreamChunk]`
- `agent.aresume(decisions, thread_id="default")` → `AsyncGenerator[StreamChunk]`
- `agent._pending_interrupts(config)` — reads `state.interrupts`, falls back to per-task `interrupts` for older langgraph.
- `build_agent()` — top-level convenience factory.

### `StreamChunk` (pydantic)
- `thread_id: str`
- `type: StreamType`
- `content: Optional[str]`
- `tool: Optional[str]`
- `skill: Optional[str]`
- `data: Optional[Dict[str, Any]]` — payload for `approval_required`: `{"action_requests": [...], "review_configs": [...]}`

## TUI internals (`main.py`)

### `ApprovalScreen(ModalScreen)`
- Args: `tool_name`, `args: dict`, `allowed: list`
- Renders title `⚠ Approval required: {tool_name}`
- If `"edit"` in `allowed` → `TextArea` with pretty-printed JSON args; else `Static` with the same.
- Buttons rendered conditionally on `allowed`:
  - `"approve"` → `Button("Approve", variant="success")` → dismisses `{"type": "approve"}`
  - `"edit"` → `Button("Save && Run", variant="primary")` → parses TextArea as JSON, dismisses `{"type": "edit", "edited_action": {"name": ..., "args": ...}}`. Invalid JSON → `notify(..., severity="error")`, stays open.
  - `"reject"` → `Button("Reject", variant="error")` → dismisses `{"type": "reject"}`

### `ChatApp(App)`
- Bound to a single `MainAgent` and a `thread_id` (uuid4 on init).
- CSS targets `#chat`, `.user-msg`, `.assistant-msg`, `.system-msg`, `.error-msg`, `#input-bar`.
- `render_stream(chunk_iter)` consumes chunks:
  - `content` → `start_assistant_message` (if first), then `append_assistant`
  - `tool_start` → `🔧 running {tool}…`
  - `error` → `⚠ error: {content}` (error-msg)
  - `approval_required` → returns pending payload to caller
  - returns `pending` (dict) or `None`
- `resolve_decisions(pending)` builds a `decisions` list (one per `action_requests` entry), looks up `allowed_decisions` from `review_configs`, awaits `push_screen_wait(ApprovalScreen(...))`. Defaults to `["approve", "reject"]` if tool not in `review_configs`. Missing decision → `{"type": "reject"}`.
- `run_turn(message)` is an `@work(exclusive=True)`. Disables input, then loops: `pending = render_stream(astream(...))`, while pending: `decisions = resolve_decisions(pending)`, `pending = render_stream(aresume(decisions, ...))`. Re-enables input in `finally`.

## Schemas (`schemas/document_schema.py`)

```python
DocumentType = Literal["pdf", "docx", "txt", "xlsx", "pptx"]

class DocumentMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)
    user_id: UUID
    document_id: UUID
    document_name: str
    document_type: DocumentType
    chunk_id: UUID
    chunk_index: int = Field(ge=0)
```

Frozen, used to track chunk provenance for ingestion. No I/O yet.

## Skills (`sandbox/skills/`)

Both files are deepagents `SKILL.md` (frontmatter `name`, `description`).

- `query-writing` — SELECT/JOIN/aggregation workflow. Prohibits DML (`INSERT`/`UPDATE`/`DELETE`/`DROP`). Encourages `write_todos` planning, table aliases, `LIMIT 5` default, no `SELECT *`.
- `schema-exploration` — uses `sql_db_list_tables`, `sql_db_schema`. Maps FK chains. References the Chinook tables: Artist, Album, Track, Genre, MediaType, Playlist, PlaylistTrack, Customer, Employee, Invoice, InvoiceLine.

## Vectorstore (`vectorstore/`)

All five files exist but are 0 bytes — scaffolds to fill:
- `injestion.py` (sic)
- `pipline.py` — ingestion pipeline (filename typo)
- `reranker.py`
- `retreiver.py` — retriever (filename typo)
- `tools.py`

ChromaDB is in deps; `docs/pdfs/coi.pdf` is the input corpus. Schema exists; wiring does not.

## Env / secrets

`.env` currently holds:
```
OLLAMA_API_KEY = 6448155a60cc4d24969bb66eafaffedc.-LVXIFS6U4dadyqGEy6o4er3
```
Key is committed to the repo. **Rotate it.** The key is loaded via `python-dotenv` in `agents/agent.py` and used by `ChatOllama` (despite the `OLLAMA_API_KEY` name, the client targets `https://api.ollama.com`).

Redis URL default: `redis://localhost:6379`. Pass override to `MainAgent.create(redis_url=...)`.

## Git state

- Branch: `master`
- Modified: `main.py`
- Untracked: `vectorstore/`
- Recent commits:
  - `c8da050` chore: add textual dep for tui work
  - `889f338` feat: thread-scoped state, message_start/end events, ipybox sandbox
  - `e2d04bf` refactor: async agent with redis checkpointer + thread-aware cli
  - `0728d1c` (skills added)
  - `90c2270` chore: ignore .db files and sandbox directory

## Known gaps / TODO signals

- `vectorstore/*` empty — no ingestion, retrieval, or reranking wired.
- `agents/subagents/sql_agent.py` empty.
- `AGENTS.md` empty (deepagents `memory` param points here, so it's currently a no-op).
- `sandbox/` is `FilesystemBackend` root in `virtual_mode=True`, but `memory=["./AGENTS.md"]` and `skills=["./skills/"]` are relative to process CWD, not the backend root — may resolve to the repo root depending on where the app is launched.
- `interrupt_on` only gates `sql_db_query`; read-only SQL tools (list/schema/check) run unguarded.
- `.env` secret committed — rotate.
- Filename typos in `vectorstore/pipline.py` and `vectorstore/retreiver.py`.
- `get_tools()` in `agents/agent.py` returns `[]` — extension point for custom tools.
- No tests present.

## How to run

```
# requires Python 3.14, uv, running redis on localhost:6379
uv sync
uv run main.py
```

In TUI: ask things like "list tables", "top 5 countries by revenue", etc. SQL writes will pause for approval/edit/reject.
