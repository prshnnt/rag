import os
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_ollama import ChatOllama

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
    agent = create_deep_agent(
        model=get_model(),
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        tools=get_tools(),
        subagents=[],
        backend=FilesystemBackend(root_dir=base_dir)
    )
    return agent