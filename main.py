import asyncio
import json
import uuid

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Markdown, Static, TextArea

from agents.agent import MainAgent, build_agent

HELP = """\
**Commands**

- `/help` — show this
- `/bye`, `/quit` — exit
- `/new` — start a new thread
- `/thread <id>` — switch to an existing thread id

Some tool calls (e.g. SQL writes) pause for your approval. A dialog will pop
up asking you to **Approve**, **Edit** the arguments, or **Reject** it.
"""


class ApprovalScreen(ModalScreen):
    """Modal asking the user to approve/edit/reject a single gated tool call."""

    DEFAULT_CSS = """
    ApprovalScreen {
        align: center middle;
    }
    #approval-dialog {
        width: 76;
        height: auto;
        max-height: 80%;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    #approval-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }
    #args-area, #args-view {
        height: auto;
        max-height: 14;
        border: round $primary;
        margin-bottom: 1;
    }
    #approval-buttons {
        height: auto;
        align: center middle;
    }
    #approval-buttons Button {
        margin: 0 1;
    }
    """

    def __init__(self, tool_name: str, args: dict, allowed: list):
        super().__init__()
        self.tool_name = tool_name
        self.args = args
        self.allowed = allowed

    def compose(self) -> ComposeResult:
        with Vertical(id="approval-dialog"):
            yield Label(f"⚠  Approval required: {self.tool_name}", id="approval-title")
            if "edit" in self.allowed:
                yield TextArea(json.dumps(self.args, indent=2), id="args-area")
            else:
                yield Static(json.dumps(self.args, indent=2), id="args-view")
            with Horizontal(id="approval-buttons"):
                if "approve" in self.allowed:
                    yield Button("Approve", id="approve", variant="success")
                if "edit" in self.allowed:
                    yield Button("Save && Run", id="edit", variant="primary")
                if "reject" in self.allowed:
                    yield Button("Reject", id="reject", variant="error")

    @on(Button.Pressed, "#approve")
    def _approve(self) -> None:
        self.dismiss({"type": "approve"})

    @on(Button.Pressed, "#reject")
    def _reject(self) -> None:
        self.dismiss({"type": "reject"})

    @on(Button.Pressed, "#edit")
    def _edit(self) -> None:
        raw = self.query_one("#args-area", TextArea).text
        try:
            new_args = json.loads(raw)
        except json.JSONDecodeError:
            self.notify("Invalid JSON — fix it before saving.", severity="error")
            return
        self.dismiss({"type": "edit", "edited_action": {"name": self.tool_name, "args": new_args}})


class ChatApp(App):
    """A chat TUI for the Redis-backed deepagents SQL agent, with HITL approval."""

    CSS = """
    #chat {
        padding: 1 2;
    }
    #chat > * {
        margin-bottom: 1;
        width: 100%;
    }
    .user-msg {
        border: round $primary;
        padding: 0 1;
        background: $primary 10%;
    }
    .assistant-msg {
        border: round $secondary;
        padding: 0 1;
    }
    .system-msg {
        color: $text-muted;
        text-style: italic;
        padding: 0 1;
    }
    .error-msg {
        color: $error;
        text-style: bold;
        padding: 0 1;
    }
    #input-bar {
        dock: bottom;
    }
    """

    BINDINGS = [
        ("ctrl+n", "new_thread", "New thread"),
        ("ctrl+q", "quit", "Quit"),
    ]

    def __init__(self, agent: MainAgent):
        super().__init__()
        self.agent = agent
        self.thread_id = str(uuid.uuid4())
        self._current_markdown: Markdown | None = None
        self._current_text = ""

    def compose(self) -> ComposeResult:
        yield Header()
        yield VerticalScroll(id="chat")
        yield Input(placeholder="Type a message… /help for commands", id="input-bar")
        yield Footer()

    async def on_mount(self) -> None:
        self.title = "Deep Agent"
        self.sub_title = f"thread {self.thread_id[:8]}"
        await self.add_system(f"Ready. Thread `{self.thread_id}`. Type /help for commands.")
        self.query_one(Input).focus()

    # ---- chat log helpers -------------------------------------------------

    async def add_system(self, text: str, css_class: str = "system-msg") -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Static(text, classes=css_class))
        chat.scroll_end(animate=False)

    async def add_user(self, text: str) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        await chat.mount(Markdown(text, classes="user-msg"))
        chat.scroll_end(animate=False)

    async def start_assistant_message(self) -> None:
        chat = self.query_one("#chat", VerticalScroll)
        widget = Markdown("", classes="assistant-msg")
        await chat.mount(widget)
        self._current_markdown = widget
        self._current_text = ""
        chat.scroll_end(animate=False)

    async def append_assistant(self, text: str) -> None:
        self._current_text += text
        if self._current_markdown is not None:
            await self._current_markdown.update(self._current_text)
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)

    # ---- input handling -----------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return

        if text == "/help":
            await self.add_system(HELP)
            return
        if text in ("/bye", "/quit"):
            self.exit()
            return
        if text == "/new":
            await self.action_new_thread()
            return
        if text.startswith("/thread"):
            parts = text.split(maxsplit=1)
            if len(parts) == 2 and parts[1].strip():
                self.thread_id = parts[1].strip()
                self.sub_title = f"thread {self.thread_id[:8]}"
                await self.add_system(f"Switched to thread `{self.thread_id}`")
            else:
                await self.add_system(f"Current thread: `{self.thread_id}`")
            return

        await self.add_user(text)
        self.run_turn(text)

    async def action_new_thread(self) -> None:
        self.thread_id = str(uuid.uuid4())
        self.sub_title = f"thread {self.thread_id[:8]}"
        await self.add_system(f"New thread `{self.thread_id}`")

    # ---- streaming + approval loop ----------------------------------------

    @work(exclusive=True)
    async def run_turn(self, message: str) -> None:
        input_widget = self.query_one("#input-bar", Input)
        input_widget.disabled = True
        try:
            pending = await self.render_stream(self.agent.astream(message, thread_id=self.thread_id))
            while pending:
                decisions = await self.resolve_decisions(pending)
                pending = await self.render_stream(
                    self.agent.aresume(decisions, thread_id=self.thread_id)
                )
        finally:
            input_widget.disabled = False
            input_widget.focus()

    async def render_stream(self, chunk_iter) -> dict | None:
        """Consume a chunk stream, rendering it into the chat log. Returns
        the pending approval payload if the turn paused, else None."""
        pending = None
        started = False
        async for chunk in chunk_iter:
            if chunk.type == "content" and chunk.content:
                if not started:
                    await self.start_assistant_message()
                    started = True
                await self.append_assistant(chunk.content)
            elif chunk.type == "tool_start":
                await self.add_system(f"🔧 running {chunk.tool}…")
            elif chunk.type == "error":
                await self.add_system(f"⚠ error: {chunk.content}", css_class="error-msg")
            elif chunk.type == "approval_required":
                pending = chunk.data
        self._current_markdown = None
        return pending

    async def resolve_decisions(self, pending: dict) -> list:
        action_requests = pending.get("action_requests", [])
        review_configs = {c.get("action_name"): c for c in pending.get("review_configs", [])}

        decisions = []
        for action in action_requests:
            name = action.get("name")
            args = action.get("args") or {}
            allowed = review_configs.get(name, {}).get("allowed_decisions", ["approve", "reject"])
            decision = await self.push_screen_wait(ApprovalScreen(name, args, allowed))
            decisions.append(decision or {"type": "reject"})
        return decisions


async def main() -> None:
    agent = await build_agent()
    app = ChatApp(agent)
    try:
        await app.run_async()
    finally:
        await agent.aclose()


if __name__ == "__main__":
    asyncio.run(main())
