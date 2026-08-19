"""Web research subagent spec.

Owned tools: websearch. Reached via `task(subagent_type="web-researcher", ...)`.
The main agent delegates "search the web" requests here so the LLM does
not have to choose between websearch and the vectorstore on its own.
"""

from deepagents.middleware.subagents import SubAgent
from tools.websearch import websearch


WEB_RESEARCHER_PROMPT = """\
You are the web-research subagent. Your only job is to answer questions
by searching the public web.

Rules:
- Call `websearch` AT MOST ONCE. If it returns no useful results, say so
  — do not rephrase and retry.
- Synthesize a concise answer from the search snippets.
- Always cite the source URLs that backed each claim.
- Do not use any other tools.
"""


web_researcher: SubAgent = {
    "name": "web-researcher",
    "description": (
        "Answers questions by searching the public web via Tavily. "
        "Delegate here when the user asks about current events, "
        "external APIs, general knowledge, or anything not covered by "
        "the indexed PDF corpus."
    ),
    "system_prompt": WEB_RESEARCHER_PROMPT,
    "tools": [websearch],
}
