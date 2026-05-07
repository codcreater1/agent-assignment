"""Core agent loop.

The agent is just: send messages → if the model calls a tool, run it
and feed the result back → otherwise we have the final answer.

The only special tool is ``ask_user``: we don't run it; we delegate to
a callback supplied by the caller (the CLI). This is what powers
clarifying questions.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable

from .llm import LLMClient, LLMError
from .prompts import build_system_prompt
from .tools import TOOL_FUNCTIONS, TOOL_SCHEMAS


# Schema for the pseudo-tool the agent uses to ask the user something.
ASK_USER_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "ask_user",
        "description": (
            "Ask the user a single clarifying question. Use ONLY when a "
            "required piece of information is missing and cannot be "
            "reasonably assumed."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {"type": "string",
                             "description": "The question to show the user."},
            },
            "required": ["question"],
        },
    },
}

ALL_TOOL_SCHEMAS = TOOL_SCHEMAS + [ASK_USER_SCHEMA]


StepCallback = Callable[[str, dict[str, Any], dict[str, Any] | None], None]


class Agent:
    """Single-user task execution agent."""

    def __init__(self,
                 llm: LLMClient,
                 ask_user: Callable[[str], str],
                 on_step: StepCallback | None = None,
                 max_steps: int | None = None) -> None:
        self.llm = llm
        self.ask_user = ask_user
        self.on_step = on_step or (lambda *_a, **_kw: None)
        self.max_steps = max_steps or int(os.getenv("AGENT_MAX_STEPS", "10"))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_request: str) -> str:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": build_system_prompt()},
            {"role": "user", "content": user_request},
        ]

        for _ in range(self.max_steps):
            try:
                msg = self.llm.chat(messages, tools=ALL_TOOL_SCHEMAS)
            except LLMError as exc:
                return f"LLM error: {exc}"

            messages.append(self._assistant_dict(msg))

            if not msg.tool_calls:
                return msg.content or "(empty response)"

            for tc in msg.tool_calls:
                tool_result = self._dispatch(tc.function.name,
                                             tc.function.arguments)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })

        return ("I hit the maximum number of steps without finishing. "
                "Try rephrasing the request or splitting it up.")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _assistant_dict(msg: Any) -> dict[str, Any]:
        """Convert an OpenAI assistant message back to the dict format
        we have to send on the next turn."""
        out: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        return out

    def _dispatch(self, name: str, raw_args: str) -> dict[str, Any]:
        try:
            args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            result = {"ok": False, "error": f"Invalid JSON arguments: {exc}"}
            self.on_step(name, {}, result)
            return result

        if name == "ask_user":
            question = args.get("question", "").strip()
            if not question:
                result = {"ok": False, "error": "question is required."}
                self.on_step(name, args, result)
                return result
            answer = self.ask_user(question)
            result = {"ok": True, "answer": answer}
            self.on_step(name, args, result)
            return result

        fn = TOOL_FUNCTIONS.get(name)
        if fn is None:
            result = {"ok": False, "error": f"Unknown tool '{name}'."}
            self.on_step(name, args, result)
            return result

        try:
            result = fn(**args)
        except TypeError as exc:
            result = {"ok": False, "error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # pragma: no cover - defensive
            result = {"ok": False,
                      "error": f"{name} crashed: {exc.__class__.__name__}: {exc}"}

        self.on_step(name, args, result)
        return result
