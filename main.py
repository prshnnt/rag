import sys
from agents.agent import agent

HELP = """/help  show this
/bye   exit
/quit  exit
/new   new thread
/thread <id>  switch thread
"""

def main():
    print("Type /help. Ctrl+C to quit.")
    while True:
        try:
            q = input(f"USER > ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not q: continue
        if q == "/help": print(HELP); continue
        if q in ("/bye", "/quit"): break
        # if q == "/new":
        #     thread = f"t-{int(time.time())}"
        #     continue
        for chunk in agent.stream(q):
            if chunk.type == "content" and chunk.content:
                print(chunk.content, end="", flush=True)
            elif chunk.type == "tool":
                print(f"\n[tool: {chunk.tool}]", flush=True)
            elif chunk.type == "error":
                print(f"\n[error: {chunk.content}]", file=sys.stderr, flush=True)
        print()

if __name__ == "__main__":
    main()