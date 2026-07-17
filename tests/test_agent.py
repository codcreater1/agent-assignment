"""Tests for the tool-calling loop in src/agent.py.

Uses a scripted fake LLM client instead of hitting a real provider, so
these run offline and deterministically in CI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.agent import Agent
from src.llm import LLMError


@dataclass
class FakeFunction:
    name: str
    arguments: str


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction


@dataclass
class FakeMessage:
    content: str | None
    tool_calls: list[FakeToolCall] = field(default_factory=list)


class ScriptedLLM:
    """Returns pre-scripted messages in order, one per ``.chat()`` call."""

    def __init__(self, script: list[FakeMessage]) -> None:
        self._script = list(script)
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages, tools=None, temperature=0.2):
        self.calls.append(messages)
        if not self._script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._script.pop(0)


def tool_call(name: str, args: dict[str, Any], call_id: str = "call_1") -> FakeToolCall:
    return FakeToolCall(id=call_id, function=FakeFunction(name=name, arguments=json.dumps(args)))


def test_final_answer_with_no_tool_calls():
    llm = ScriptedLLM([FakeMessage(content="All done, no tools needed.")])
    agent = Agent(llm=llm, ask_user=lambda q: "n/a", max_steps=5)
    result = agent.run("say hi")
    assert result == "All done, no tools needed."
    assert len(llm.calls) == 1


def test_single_tool_call_then_final_answer(monkeypatch):
    monkeypatch.setenv("AGENT_DATA_FILE", "/tmp/agent_test_single_tool.json")
    llm = ScriptedLLM([
        FakeMessage(content=None, tool_calls=[
            tool_call("search_service", {"category": "coworking", "city": "warsaw"})
        ]),
        FakeMessage(content="Found 4 coworking spaces."),
    ])
    agent = Agent(llm=llm, ask_user=lambda q: "n/a", max_steps=5)
    result = agent.run("find coworking spaces in warsaw")
    assert result == "Found 4 coworking spaces."
    # Second call to the LLM must include the tool result in the transcript.
    second_call_messages = llm.calls[1]
    tool_messages = [m for m in second_call_messages if m.get("role") == "tool"]
    assert len(tool_messages) == 1
    payload = json.loads(tool_messages[0]["content"])
    assert payload["ok"] is True


def test_ask_user_pseudo_tool_delegates_to_callback():
    llm = ScriptedLLM([
        FakeMessage(content=None, tool_calls=[
            tool_call("ask_user", {"question": "Which city?"})
        ]),
        FakeMessage(content="Booked in Warsaw."),
    ])
    seen_questions = []

    def fake_ask(q: str) -> str:
        seen_questions.append(q)
        return "Warsaw"

    agent = Agent(llm=llm, ask_user=fake_ask, max_steps=5)
    result = agent.run("book me a dentist")
    assert result == "Booked in Warsaw."
    assert seen_questions == ["Which city?"]


def test_unknown_tool_reports_error_without_crashing():
    llm = ScriptedLLM([
        FakeMessage(content=None, tool_calls=[
            tool_call("does_not_exist", {})
        ]),
        FakeMessage(content="Recovered."),
    ])
    agent = Agent(llm=llm, ask_user=lambda q: "n/a", max_steps=5)
    result = agent.run("do the impossible")
    assert result == "Recovered."
    tool_messages = [m for m in llm.calls[1] if m.get("role") == "tool"]
    payload = json.loads(tool_messages[0]["content"])
    assert payload["ok"] is False
    assert "Unknown tool" in payload["error"]


def test_malformed_tool_call_json_does_not_crash():
    llm = ScriptedLLM([
        FakeMessage(content=None, tool_calls=[
            FakeToolCall(id="call_1", function=FakeFunction(
                name="search_service", arguments="{not valid json"))
        ]),
        FakeMessage(content="Handled the bad arguments."),
    ])
    agent = Agent(llm=llm, ask_user=lambda q: "n/a", max_steps=5)
    result = agent.run("break it")
    assert result == "Handled the bad arguments."


def test_max_steps_gives_up_gracefully():
    # Every step returns another tool call, never a final answer.
    infinite = [
        FakeMessage(content=None, tool_calls=[
            tool_call("list_bookings", {}, call_id=f"c{i}")
        ])
        for i in range(10)
    ]
    llm = ScriptedLLM(infinite)
    agent = Agent(llm=llm, ask_user=lambda q: "n/a", max_steps=3)
    result = agent.run("loop forever")
    assert "maximum number of steps" in result
    assert len(llm.calls) == 3


def test_llm_error_surfaces_to_caller():
    class AlwaysFailsLLM:
        def chat(self, messages, tools=None, temperature=0.2):
            raise LLMError("provider is down")

    agent = Agent(llm=AlwaysFailsLLM(), ask_user=lambda q: "n/a", max_steps=3)
    result = agent.run("anything")
    assert "LLM error" in result
    assert "provider is down" in result


def test_tool_use_failed_retries_once_then_continues():
    llm = ScriptedLLM([])

    class FlakyThenGood:
        def __init__(self):
            self.attempts = 0

        def chat(self, messages, tools=None, temperature=0.2):
            self.attempts += 1
            if self.attempts == 1:
                raise LLMError("400 tool_use_failed: malformed function call")
            return FakeMessage(content="Recovered after retry.")

    flaky = FlakyThenGood()
    agent = Agent(llm=flaky, ask_user=lambda q: "n/a", max_steps=5)
    result = agent.run("trigger a malformed call")
    assert result == "Recovered after retry."
    assert flaky.attempts == 2
