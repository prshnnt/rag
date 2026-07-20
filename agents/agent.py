import os
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama
from langgraph.checkpoint.redis import AsyncRedisSaver

def get_model(model_name="gpt-oss:120b"):
    return ChatOllama(
        model=model_name,
        base_url="https://api.ollama.com",
        temperature=0
    )
def get_tools():
    return []
def get_agent():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpointer = AsyncRedisSaver(redis_url="redis://localhost:6379")
    agent = create_deep_agent(
        model=get_model(),
        tools=get_tools(),
        backend=FilesystemBackend(root_dir=base_dir),
        skills=["./skills/"],
        subagents=[],
        memory=["./AGENTS.md"],
        checkpointer=checkpointer
    )
    return agent
