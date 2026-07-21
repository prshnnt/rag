import os
from typing import Optional , Dict , Any , List , Literal , AsyncGenerator, Generator
from dotenv import load_dotenv

from pydantic import BaseModel, Field

from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from deepagents.backends import FilesystemBackend
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langgraph.checkpoint.redis import AsyncRedisSaver
import asyncio
from langchain.messages import HumanMessage , AIMessage
import datetime
from datetime import timezone
load_dotenv()

def get_model(model_name="gpt-oss:120b"):
    return ChatOllama(
        model=model_name,
        base_url="https://api.ollama.com",
        temperature=0,
        headers={"Authorization": f"Bearer {os.getenv('OLLAMA_API_KEY', '')}"},
    )
def get_tools():
    return []

StreamType = Literal[""]
class StreamChunk(BaseModel): # thread_id , type , content , tool_name , 
    """A chunk of streamed response."""
    thread_id: str = Field(None, description="Thread ID")
    type: str = Field(..., description="Type of chunk: 'start', 'content', 'tool','skill','subagent', 'end', 'error'")
    content: Optional[str] = Field(None, description="Content for content chunks")
    tool: Optional[str] = Field(None, description="Tool name for tool_call chunks")
    skill: Optional[str] = Field(None , description="Skill which is being invoked")
    task: Optional[List[str]] = Field(None, description="list of subagents running.")

class MainAgent:
    def __init__(self):
        self.model = get_model()
        # self.checkpointer = AsyncRedisSaver(redis_url="redis://localhost:6379")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_path = os.path.join(base_dir, "chinook.db")
        self.db = SQLDatabase.from_uri(f"sqlite:///{self.db_path}", sample_rows_in_table_info=3)
        toolkit = SQLDatabaseToolkit(db=self.db, llm=self.model)
        sql_tools = toolkit.get_tools()

        self.tools = sql_tools + get_tools()

        self.agent = create_deep_agent(
            model=self.model,
            tools=self.tools,
            backend=FilesystemBackend(root_dir="./sandbox/", virtual_mode=True),
            skills=["./skills/"],
            subagents=[],
            memory=["./AGENTS.md"],
            # checkpointer=self.checkpointer
        )
    

    async def astream(self,message)-> AsyncGenerator[str, None]:
        agent = self.agent
        yield StreamChunk(
            type="start",
            # thread_id=thread_id
            )
        # config = {
        #     "configurable": {
        #         "thread_id": thread_id
        #     }
        # }
        full_response = ""
        try:
            async with asyncio.timeout(300):
                async for event in agent.astream_events(
                    {"messages": [HumanMessage(content=message)]},
                    # config=config,
                    version="v2"
                ):
                    event_type = event.get("event","")
                    event_data = event.get("data",{})
                    
                    if event_type == "on_chat_model_stream":
                        chunk = event_data.get("chunk",{})
                        if chunk and hasattr(chunk,"content") and chunk.content:
                            # full_response += chunk.content
                            yield StreamChunk(
                                type="content",
                                # thread_id=thread_id,
                                content=chunk.content,
                            )
                    
                    if event_type == "on_tool_start":
                        tool_name = event.get("name", "unknown")
                        yield StreamChunk(
                            type="tool",
                            # thread_id=thread_id,
                            tool=tool_name
                        )
                    
                    if event_type == "on_tool_end":
                        yield StreamChunk(
                            type="tool_output",
                            # thread_id=thread_id,
                            content=str(event_data.get("output")),
                        )
                    
                    if event_type == "on_chat_model_end":
                        yield StreamChunk(
                            type="end",
                            # thread_id=thread_id,
                            content=str(event_data.get("content")),
                        )
                    
                    if event_type == "on_chat_model_error":
                        yield StreamChunk(
                            type="error",
                            # thread_id=thread_id,
                            content=str(event_data.get("error")),
                        )
        except Exception as e:
            yield StreamChunk(
                type="error",
                # thread_id=thread_id,
                content=str(e),
            )


        yield StreamChunk(
            type='end',
            # thread_id=thread_id,
            content=full_response,
        )

    def stream(self, message) -> Generator[StreamChunk, None, None]:
        """Sync stream for CLI. Wraps langgraph agent.stream (v2) -> StreamChunk."""
        from langchain.messages import AIMessageChunk, ToolMessage

        yield StreamChunk(type="start")

        try:
            for chunk in self.agent.stream(
                {"messages": [{"role": "user", "content": message}]},
                stream_mode="messages",
                subgraphs=True,
                version="v2",
            ):
                kind = chunk.get("type")
                data = chunk.get("data")

                if kind == "messages":
                    token, _meta = data
                    if isinstance(token, AIMessageChunk):
                        if isinstance(token.content, str) and token.content:
                            yield StreamChunk(type="content", content=token.content)
                        for tc in token.tool_call_chunks or []:
                            if tc.get("name"):
                                yield StreamChunk(type="tool", tool=tc["name"])
                    elif isinstance(token, ToolMessage):
                        yield StreamChunk(
                            type="tool_output",
                            content=str(token.content),
                        )

        except Exception as e:
            yield StreamChunk(type="error", content=str(e))

        yield StreamChunk(type="end")


agent = MainAgent()