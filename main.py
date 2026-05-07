"""CLI entry point.

Usage:
    uv run python main.py                       # interactive mode
    uv run python main.py "your task here"      # one-shot mode
"""

from __future__ import annotations

import json
import sys
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt

from src.agent import Agent
from src.llm import LLMClient, LLMError

console = Console()


def _short_args(args: dict[str, Any], limit: int = 100) -> str:
    s = json.dumps(args, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _short_result(result: dict[str, Any] | None, limit: int = 200) -> str:
    if result is None:
        return ""
    s = json.dumps(result, ensure_ascii=False)
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _on_step(name: str,
             args: dict[str, Any],
             result: dict[str, Any] | None) -> None:
    """Pretty-print each tool call as it happens."""
    if name == "ask_user":
        return  # the prompt itself is the visible event
    ok = result is None or result.get("ok", True)
    color = "cyan" if ok else "red"
    console.print(
        f"[{color}]→ {name}[/]([dim]{_short_args(args)}[/]) "
        f"[dim]{_short_result(result)}[/]"
    )


def _ask_user(question: str) -> str:
    """Callback the agent uses for clarifying questions."""
    answer = Prompt.ask(f"[yellow]?[/] {question}")
    return answer.strip()


def _run_one(agent: Agent, request: str) -> None:
    console.print(Panel(request, title="Task", border_style="blue"))
    try:
        answer = agent.run(request)
    except LLMError as exc:
        console.print(f"[red]LLM error:[/] {exc}")
        return
    console.print(Panel(Markdown(answer),
                        title="Result", border_style="green"))


def main() -> int:
    load_dotenv()
    try:
        llm = LLMClient()
    except LLMError as exc:
        console.print(f"[red]Setup error:[/] {exc}")
        return 1

    agent = Agent(llm=llm, ask_user=_ask_user, on_step=_on_step)

    if len(sys.argv) > 1:
        _run_one(agent, " ".join(sys.argv[1:]))
        return 0

    console.print(Panel.fit(
        "Task Execution AI Agent\n"
        "Type your request, or 'exit' to quit.",
        border_style="magenta",
    ))
    while True:
        try:
            request = Prompt.ask("\n[bold magenta]you[/]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            return 0
        if not request:
            continue
        if request.lower() in {"exit", "quit", "q"}:
            return 0
        _run_one(agent, request)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
