import asyncio
import sys
import uuid

from rich.console import Console
from rich.markdown import Markdown

from agents.agent import build_agent

console = Console()

HELP = """/help         show this
/bye, /quit   exit
/new          start a new thread
/thread <id>  switch to an existing thread id
"""


async def main():
    print("Type /help. Ctrl+C to quit.")
    agent = await build_agent()
    thread_id = str(uuid.uuid4())
    print(f"[thread: {thread_id}]")

    try:
        while True:
            try:
                q = await asyncio.to_thread(input, "USER > ")
                q = q.strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not q:
                continue
            if q == "/help":
                print(HELP)
                continue
            if q in ("/bye", "/quit"):
                break
            if q == "/new":
                thread_id = str(uuid.uuid4())
                print(f"[thread: {thread_id}]")
                continue
            if q.startswith("/thread"):
                parts = q.split(maxsplit=1)
                if len(parts) == 2 and parts[1].strip():
                    thread_id = parts[1].strip()
                    print(f"[thread: {thread_id}]")
                else:
                    print(f"[current thread: {thread_id}]")
                continue

            full_message = ""
            async for chunk in agent.astream(q, thread_id=thread_id):
                if chunk.type == "content" and chunk.content:
                    full_message += chunk.content
                elif chunk.type == "tool_start":
                    print(f"\n[tool: {chunk.tool}]", flush=True)
                elif chunk.type == "error":
                    print(f"\n[error: {chunk.content}]", file=sys.stderr, flush=True)

            if full_message:
                console.print(Markdown(full_message))
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())