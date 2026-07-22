import os
import uuid
from typing import Optional, Literal, AsyncGenerator

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from deepagents.backends import FilesystemBackend
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langchain.messages import HumanMessage

load_dotenv()


def get_model(model_name: str = "gpt-oss:120b") -> ChatOllama:
    return ChatOllama(
        model=model_name,
        base_url="https://api.ollama.com",
        temperature=0,
    )


def get_tools() -> list:
    """Extra (non-SQL) tools for the agent. Add custom tools here."""
    return []


# Real lifecycle of a single turn, in order:
#   start -> (message_start -> content* -> message_end | tool_start -> tool_end)* -> end
# "error" can happen at any point instead of the terminal "end".
StreamType = Literal[
    "start", "message_start", "content", "message_end",
    "tool_start", "tool_end", "end", "error",
]


class StreamChunk(BaseModel):
    """A chunk of streamed response."""

    thread_id: str = Field(..., description="Thread ID")
    type: StreamType = Field(..., description="Type of chunk, see StreamType")
    content: Optional[str] = Field(None, description="Text content for content/end chunks")
    tool: Optional[str] = Field(None, description="Tool name for tool_start/tool_end chunks")
    skill: Optional[str] = Field(None, description="Skill being invoked, if any")


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
    async def create(cls, redis_url: str = "redis://localhost:6379") -> "MainAgent":
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
            skills=["./skills/"],
            subagents=[],
            memory=["./AGENTS.md"],
            checkpointer=checkpointer,
        )

        return cls(model, agent, redis_cm, checkpointer)

    async def aclose(self):
        """Release the Redis connection. Call once at process shutdown."""
        if self._redis_cm is not None:
            await self._redis_cm.__aexit__(None, None, None)
            self._redis_cm = None

    async def astream(
        self, message: str, thread_id: str = "default"
    ) -> AsyncGenerator[StreamChunk, None]:
        yield StreamChunk(type="start", thread_id=thread_id)

        config = {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        full_response = ""

        try:
            async for event in self.agent.astream_events(
                {"messages": [HumanMessage(content=message)]},
                config=config,
                version="v2",
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

        yield StreamChunk(type="end", thread_id=thread_id, content=full_response)


async def build_agent() -> MainAgent:
    """Convenience factory for callers (e.g. main.py) to construct the singleton."""
    return await MainAgent.create()