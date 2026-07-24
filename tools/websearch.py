import os
from typing import Literal
from tavily import TavilyClient
from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()

tavily_client = TavilyClient()

@tool
def websearch(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    return tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
