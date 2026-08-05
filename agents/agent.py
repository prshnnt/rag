import os
import uuid
from typing import Optional, Literal, Dict, Any, AsyncGenerator

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from deepagents.backends import FilesystemBackend , CompositeBackend
from deepagents.middleware import FilesystemMiddleware
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.types import Command
from langchain.messages import HumanMessage
from tools.websearch import websearch
from vectorstore.tools import list_documents , vector_search

load_dotenv()


def get_model(model_name: str = "gpt-oss:120b") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url="https://api.ollama.com",
        temperature=0,
    )


def get_tools() -> list:
    """Extra (non-SQL) tools for the agent. Add custom tools here."""
    return [websearch , list_documents , vector_search]


# Real lifecycle of a single turn, in order:
#   start -> (message_start -> content* -> message_end | tool_start -> tool_end)*
#     -> end | approval_required
# "error" can happen at any point instead of the terminal "end".
# "approval_required" means execution is PAUSED (state is checkpointed) and the
# turn is not over until the caller resumes it via `aresume()`.
StreamType = Literal[
    "start", "message_start", "content", "message_end",
    "tool_start", "tool_end", "approval_required", "end", "error",
]


class StreamChunk(BaseModel):
    """A chunk of streamed response."""

    thread_id: str = Field(..., description="Thread ID")
    type: StreamType = Field(..., description="Type of chunk, see StreamType")
    content: Optional[str] = Field(None, description="Text content for content/end chunks")
    tool: Optional[str] = Field(None, description="Tool name for tool_start/tool_end chunks")
    skill: Optional[str] = Field(None, description="Skill being invoked, if any")
    data: Optional[Dict[str, Any]] = Field(
        None,
        description=(
            "Structured payload for approval_required chunks: "
            "{'action_requests': [...], 'review_configs': [...]}"
        ),
    )


# Tools that require a human decision before they're allowed to execute.
# See https://docs.langchain.com/oss/python/deepagents/human-in-the-loop
DEFAULT_INTERRUPT_ON: Dict[str, Any] = {
    # sql_db_query can run arbitrary SQL (including writes) -> always gate it.
    "sql_db_query": {"allowed_decisions": ["approve", "edit", "reject"]},
    "websearch": {"allowed_decisions":["approve","reject"]}
    # Schema/table-listing/query-checking tools are read-only and safe to
    # leave unattended; deepagents treats an absent key as "no interrupt".
}


class MainAgent:
    """
    Wraps a deepagents graph with SQL tooling and Redis-backed persistence.

    Because AsyncRedisSaver is async-only, this class is async end-to-end.
    Construct it with `await MainAgent.create()` and tear it down with
    `await agent.aclose()` so the Redis connection is released cleanly.
    Everything must run inside a single asyncio event loop for the lifetime
    of the process (async Redis clients are bound to the loop that created
    them and cannot be reused across loops).
    """

    def __init__(self, model, agent, redis_cm, checkpointer):
        self.model = model
        self.agent = agent
        self._redis_cm = redis_cm
        self.checkpointer = checkpointer

    @classmethod
    async def create(
        cls,
        redis_url: str = "redis://localhost:6379",
        interrupt_on: Optional[Dict[str, Any]] = None,
    ) -> "MainAgent":
        model = get_model()

        base_dir = os.path.dirname(os.path.abspath(__file__))
        db_path = os.path.join(base_dir, "chinook.db")
        db = SQLDatabase.from_uri(f"sqlite:///{db_path}", sample_rows_in_table_info=3)
        toolkit = SQLDatabaseToolkit(db=db, llm=model)
        tools = toolkit.get_tools() + get_tools()

        # AsyncRedisSaver must be entered as an async context manager and
        # explicitly `asetup()` to create its indices before first use.
        redis_cm = AsyncRedisSaver.from_conn_string(redis_url)
        checkpointer = await redis_cm.__aenter__()
        await checkpointer.asetup()

        agent = create_deep_agent(
            model=model,
            tools=tools,
            backend=FilesystemBackend(root_dir="./sandbox/", virtual_mode=True),
            skills=["./skills/examples","./skills/public"],
            subagents=[],
            memory=["./AGENTS.md"],
            checkpointer=checkpointer,
            # Human-in-the-loop: pause before running gated tools until a
            # human calls MainAgent.aresume() with approve/edit/reject.
            interrupt_on=DEFAULT_INTERRUPT_ON if interrupt_on is None else interrupt_on,
        )

        return cls(model, agent, redis_cm, checkpointer)

    async def aclose(self):
        """Release the Redis connection. Call once at process shutdown."""
        if self._redis_cm is not None:
            await self._redis_cm.__aexit__(None, None, None)
            self._redis_cm = None

    async def _pending_interrupts(self, config: dict):
        """Return any interrupts left un-resolved after the last run, if any."""
        state = await self.agent.aget_state(config)
        interrupts = getattr(state, "interrupts", None)
        if interrupts:
            return interrupts
        # Older langgraph versions surface interrupts per-task instead of
        # top-level; fall back to flattening those.
        return tuple(
            i
            for task in getattr(state, "tasks", ())
            for i in getattr(task, "interrupts", ())
        )

    async def _run(self, graph_input, thread_id: str) -> AsyncGenerator[StreamChunk, None]:
        """Drive one graph execution (fresh message OR a Command(resume=...))
        and yield StreamChunks for it, ending in exactly one of:
        'end' (turn finished), 'approval_required' (paused, needs a human
        decision), or 'error'.
        """
        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        full_response = ""

        try:
            async for event in self.agent.astream_events(
                graph_input, config=config, version="v2"
            ):
                event_type = event.get("event", "")
                data = event.get("data", {})

                if event_type == "on_chat_model_start":
                    yield StreamChunk(type="message_start", thread_id=thread_id)

                elif event_type == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    text = getattr(chunk, "content", None) if chunk else None
                    if text:
                        full_response += text
                        yield StreamChunk(type="content", thread_id=thread_id, content=text)

                elif event_type == "on_chat_model_end":
                    yield StreamChunk(type="message_end", thread_id=thread_id)

                elif event_type == "on_tool_start":
                    yield StreamChunk(
                        type="tool_start",
                        thread_id=thread_id,
                        tool=event.get("name", "unknown"),
                    )

                elif event_type == "on_tool_end":
                    output = data.get("output")
                    output_text = getattr(output, "content", output)
                    yield StreamChunk(
                        type="tool_end",
                        thread_id=thread_id,
                        tool=event.get("name", "unknown"),
                        content=str(output_text),
                    )

        except Exception as e:
            yield StreamChunk(type="error", thread_id=thread_id, content=str(e))
            return

        # astream_events finishing "normally" doesn't mean the turn is done --
        # if a gated tool was called, the graph is paused and checkpointed
        # rather than errored. Check for that before declaring "end".
        interrupts = await self._pending_interrupts(config)
        if interrupts:
            yield StreamChunk(
                type="approval_required",
                thread_id=thread_id,
                data=interrupts[0].value,
            )
            return

        yield StreamChunk(type="end", thread_id=thread_id, content=full_response)

    async def astream(
        self, message: str, thread_id: str = "default"
    ) -> AsyncGenerator[StreamChunk, None]:
        """Start (or continue) a conversation turn with a new user message."""
        yield StreamChunk(type="start", thread_id=thread_id)
        async for chunk in self._run({"messages": [HumanMessage(content=message)]}, thread_id):
            yield chunk

    async def aresume(
        self, decisions: list, thread_id: str = "default"
    ) -> AsyncGenerator[StreamChunk, None]:
        """
        Resume a turn paused by an 'approval_required' chunk.

        `decisions` must have one entry per action_request from that chunk,
        in the same order, each one of:
          {"type": "approve"}
          {"type": "reject"}
          {"type": "edit", "edited_action": {"name": ..., "args": {...}}}
        """
        yield StreamChunk(type="start", thread_id=thread_id)
        async for chunk in self._run(Command(resume={"decisions": decisions}), thread_id):
            yield chunk


async def build_agent() -> MainAgent:
    """Convenience factory for callers (e.g. main.py) to construct the singleton."""
    return await MainAgent.create()