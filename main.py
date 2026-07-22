import asyncio
import json
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

Some tool calls (e.g. SQL writes) pause for your approval before running.
When prompted, type one of: approve, edit, reject.
"""


async def render_stream(chunk_iter):
    """Consume a chunk stream: print tool events inline, return the
    accumulated text and any pending approval payload (or None)."""
    full_message = ""
    pending = None
    async for chunk in chunk_iter:
        if chunk.type == "content" and chunk.content:
            full_message += chunk.content
        elif chunk.type == "tool_start":
            print(f"\n[tool: {chunk.tool}]", flush=True)
        elif chunk.type == "error":
            print(f"\n[error: {chunk.content}]", file=sys.stderr, flush=True)
        elif chunk.type == "approval_required":
            pending = chunk.data
    return full_message, pending


async def prompt_for_decisions(pending: dict) -> list:
    """Ask the user, one gated tool call at a time, what to do with it."""
    action_requests = pending.get("action_requests", [])
    review_configs = {c.get("action_name"): c for c in pending.get("review_configs", [])}

    decisions = []
    for action in action_requests:
        name = action.get("name")
        args = action.get("args")
        allowed = review_configs.get(name, {}).get("allowed_decisions", ["approve", "reject"])

        print(f"\n[approval required] tool={name}")
        print(f"  args: {json.dumps(args)}")
        print(f"  allowed decisions: {', '.join(allowed)}")

        while True:
            choice = (await asyncio.to_thread(input, "decision > ")).strip().lower()
            if choice in allowed:
                break
            print(f"  please enter one of: {', '.join(allowed)}")

        if choice == "edit":
            raw = await asyncio.to_thread(
                input, f"  new args as JSON (blank = keep original) > "
            )
            new_args = args
            if raw.strip():
                try:
                    new_args = json.loads(raw)
                except json.JSONDecodeError:
                    print("  invalid JSON, keeping original args")
            decisions.append({"type": "edit", "edited_action": {"name": name, "args": new_args}})
        else:
            decisions.append({"type": choice})

    return decisions


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

            full_message, pending = await render_stream(agent.astream(q, thread_id=thread_id))

            while pending:
                if full_message:
                    console.print(Markdown(full_message))
                    full_message = ""
                decisions = await prompt_for_decisions(pending)
                full_message, pending = await render_stream(
                    agent.aresume(decisions, thread_id=thread_id)
                )

            if full_message:
                console.print(Markdown(full_message))
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())